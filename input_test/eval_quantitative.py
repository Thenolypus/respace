"""
Quantitative evaluation: OOB and MBL metrics for generated room layouts.

Walks eval/eval_set/*/model/Floor/unit_1/room/generated_scene.json,
runs eval_scene() on each non-bathroom room, and aggregates OOB/MBL
per model across all seeds, styles, and floor plans.

Usage:
  uv run python -m input_test.eval_quantitative
  uv run python -m input_test.eval_quantitative --eval-dir eval/eval_set
  uv run python -m input_test.eval_quantitative --eval-dir eval/eval_set --skip-mesh
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

from src.eval import eval_scene, eval_bounds


MODEL_NAMES = ["1_7b", "4b", "original"]


def discover_scenes(eval_dir):
    """Find all generated_scene.json files, grouped by model.

    Returns dict: model_name -> list of (scene_path, metadata_dict)
    where metadata_dict has keys: style_seed, floor, room_stem, room_type
    """
    eval_dir = Path(eval_dir)
    scenes = {m: [] for m in MODEL_NAMES}

    for style_seed_dir in sorted(eval_dir.iterdir()):
        if not style_seed_dir.is_dir():
            continue
        style_seed = style_seed_dir.name  # e.g. "scand_456"

        for model_dir in sorted(style_seed_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            if model_name not in scenes:
                continue

            for floor_dir in sorted(model_dir.iterdir()):
                if not floor_dir.is_dir():
                    continue
                floor = floor_dir.name

                for unit_dir in sorted(floor_dir.iterdir()):
                    if not unit_dir.is_dir():
                        continue

                    for room_dir in sorted(unit_dir.iterdir()):
                        if not room_dir.is_dir():
                            continue
                        scene_file = room_dir / "generated_scene.json"
                        if not scene_file.exists():
                            continue

                        room_stem = room_dir.name
                        # Extract room type from stem (last part after last underscore)
                        room_type = room_stem.rsplit("_", 1)[-1] if "_" in room_stem else "unknown"

                        scenes[model_name].append((
                            scene_file,
                            {
                                "style_seed": style_seed,
                                "floor": floor,
                                "unit": unit_dir.name,
                                "room_stem": room_stem,
                                "room_type": room_type,
                            }
                        ))

    return scenes


def evaluate_scene_file(scene_path, skip_mesh=False):
    """Load and evaluate a single scene. Returns metrics dict or None on failure."""
    with open(scene_path) as f:
        scene = json.load(f)

    if not eval_bounds(scene):
        print(f"  SKIP (invalid bounds): {scene_path}")
        return None

    objs = scene.get("objects")
    if not objs or len(objs) == 0:
        print(f"  SKIP (no objects): {scene_path}")
        return None

    if skip_mesh:
        return evaluate_scene_bbox_only(scene)

    metrics = eval_scene(scene, is_debug=False)
    return metrics


def evaluate_scene_bbox_only(scene):
    """Fast bbox-only OOB/MBL (no mesh voxelization)."""
    from src.eval import compute_oob, compute_bbl
    from src.utils import create_floor_plan_polygon

    bounds_top = scene.get("bounds_top")
    bounds_bottom = scene.get("bounds_bottom")
    floor_plan_polygon = create_floor_plan_polygon(bounds_bottom)
    objs = scene.get("objects", [])

    oobs = []
    for obj in objs:
        oob = compute_oob(obj, floor_plan_polygon, bounds_bottom, bounds_top)
        oobs.append(oob)

    bbls = []
    for i, obj_x in enumerate(objs):
        for obj_y in objs[i + 1:]:
            bbl = compute_bbl(obj_x, obj_y)
            bbls.append(bbl)

    total_oob = sum(oobs)
    total_mbl = sum(bbls)
    total_pbl = total_oob + total_mbl

    return {
        "total_oob_loss": total_oob,
        "total_mbl_loss": total_mbl,
        "total_pbl_loss": total_pbl,
        "is_valid_scene_pbl": total_pbl <= 0.1,
    }


def print_table(model_metrics):
    """Print a formatted comparison table."""
    print("\n" + "=" * 85)
    print(f"{'Model':<12} {'N':>4}  {'OOB (x1e3)':>14}  {'MBL (x1e3)':>14}  {'PBL (x1e3)':>14}  {'Valid%':>7}")
    print("-" * 85)

    for model_name in MODEL_NAMES:
        data = model_metrics.get(model_name)
        if not data or len(data["oob"]) == 0:
            print(f"{model_name:<12} {'--':>4}  {'--':>14}  {'--':>14}  {'--':>14}  {'--':>7}")
            continue

        n = len(data["oob"])
        oob_mean = np.mean(data["oob"]) * 1e3
        oob_std = np.std(data["oob"]) * 1e3
        mbl_mean = np.mean(data["mbl"]) * 1e3
        mbl_std = np.std(data["mbl"]) * 1e3
        pbl_mean = np.mean(data["pbl"]) * 1e3
        pbl_std = np.std(data["pbl"]) * 1e3
        valid_pct = np.mean(data["valid"]) * 100

        print(f"{model_name:<12} {n:>4}  {oob_mean:>6.2f} +/- {oob_std:<5.2f}  {mbl_mean:>6.2f} +/- {mbl_std:<5.2f}  {pbl_mean:>6.2f} +/- {pbl_std:<5.2f}  {valid_pct:>6.1f}%")

    print("=" * 85)


def print_per_floor_table(model_metrics_per_floor):
    """Print per-floor breakdown."""
    all_floors = sorted({f for floors in model_metrics_per_floor.values() for f in floors})

    for floor in all_floors:
        print(f"\n--- {floor} ---")
        print(f"{'Model':<12} {'N':>4}  {'OOB (x1e3)':>14}  {'MBL (x1e3)':>14}  {'Valid%':>7}")
        print("-" * 65)

        for model_name in MODEL_NAMES:
            data = model_metrics_per_floor.get(model_name, {}).get(floor)
            if not data or len(data["oob"]) == 0:
                print(f"{model_name:<12} {'--':>4}  {'--':>14}  {'--':>14}  {'--':>7}")
                continue

            n = len(data["oob"])
            oob_mean = np.mean(data["oob"]) * 1e3
            oob_std = np.std(data["oob"]) * 1e3
            mbl_mean = np.mean(data["mbl"]) * 1e3
            mbl_std = np.std(data["mbl"]) * 1e3
            valid_pct = np.mean(data["valid"]) * 100

            print(f"{model_name:<12} {n:>4}  {oob_mean:>6.2f} +/- {oob_std:<5.2f}  {mbl_mean:>6.2f} +/- {mbl_std:<5.2f}  {valid_pct:>6.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Quantitative OOB/MBL evaluation")
    parser.add_argument("--eval-dir", type=str, default="eval/eval_set",
                        help="Root directory of eval sets (default: eval/eval_set)")
    parser.add_argument("--skip-mesh", action="store_true",
                        help="Use bbox-only metrics (faster, less accurate)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save raw per-scene metrics to JSON file")
    args = parser.parse_args()

    if not Path(args.eval_dir).exists():
        print(f"ERROR: eval dir not found: {args.eval_dir}")
        sys.exit(1)

    print(f"Eval dir: {args.eval_dir}")
    print(f"Mode: {'bbox-only' if args.skip_mesh else 'mesh-based (voxelized)'}")

    scenes_by_model = discover_scenes(args.eval_dir)
    for m in MODEL_NAMES:
        total = len(scenes_by_model[m])
        non_bath = sum(1 for _, meta in scenes_by_model[m] if meta["room_type"] != "bathroom")
        print(f"  {m}: {total} scenes ({non_bath} non-bathroom)")

    # Aggregate storage
    model_metrics = {m: {"oob": [], "mbl": [], "pbl": [], "valid": []} for m in MODEL_NAMES}
    model_metrics_per_floor = {m: {} for m in MODEL_NAMES}
    all_per_scene = []  # for JSON export

    for model_name in MODEL_NAMES:
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*70}")

        for scene_path, meta in scenes_by_model[model_name]:
            # Skip bathrooms - rule-based, same across all models
            if meta["room_type"] == "bathroom":
                continue

            short_path = f"{meta['style_seed']}/{model_name}/{meta['floor']}/{meta['room_stem']}"
            print(f"  {short_path} ... ", end="", flush=True)

            metrics = evaluate_scene_file(scene_path, skip_mesh=args.skip_mesh)
            if metrics is None:
                print("SKIPPED")
                continue

            oob = metrics["total_oob_loss"]
            mbl = metrics["total_mbl_loss"]
            pbl = metrics["total_pbl_loss"]
            valid = metrics["is_valid_scene_pbl"]

            print(f"OOB={oob:.5f}  MBL={mbl:.5f}  valid={valid}")

            model_metrics[model_name]["oob"].append(oob)
            model_metrics[model_name]["mbl"].append(mbl)
            model_metrics[model_name]["pbl"].append(pbl)
            model_metrics[model_name]["valid"].append(valid)

            # Per-floor aggregation
            floor = meta["floor"]
            if floor not in model_metrics_per_floor[model_name]:
                model_metrics_per_floor[model_name][floor] = {"oob": [], "mbl": [], "pbl": [], "valid": []}
            model_metrics_per_floor[model_name][floor]["oob"].append(oob)
            model_metrics_per_floor[model_name][floor]["mbl"].append(mbl)
            model_metrics_per_floor[model_name][floor]["pbl"].append(pbl)
            model_metrics_per_floor[model_name][floor]["valid"].append(valid)

            all_per_scene.append({
                "model": model_name,
                **meta,
                "oob": oob,
                "mbl": mbl,
                "pbl": pbl,
                "valid": valid,
            })

    # Print results
    print("\n\n" + "#" * 85)
    print("# AGGREGATE RESULTS (all seeds, styles, floors)")
    print("#" * 85)
    print_table(model_metrics)

    print("\n\n" + "#" * 85)
    print("# PER-FLOOR BREAKDOWN")
    print("#" * 85)
    print_per_floor_table(model_metrics_per_floor)

    # Save raw metrics
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {}
        for model_name in MODEL_NAMES:
            data = model_metrics[model_name]
            if len(data["oob"]) == 0:
                continue
            summary[model_name] = {
                "n_scenes": len(data["oob"]),
                "oob_mean": float(np.mean(data["oob"])),
                "oob_std": float(np.std(data["oob"])),
                "mbl_mean": float(np.mean(data["mbl"])),
                "mbl_std": float(np.std(data["mbl"])),
                "pbl_mean": float(np.mean(data["pbl"])),
                "pbl_std": float(np.std(data["pbl"])),
                "valid_ratio": float(np.mean(data["valid"])),
            }

        output = {
            "summary": summary,
            "per_scene": all_per_scene,
        }

        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved raw metrics to: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
