"""
Evaluation orchestrator: runs the same units across multiple SG-LLM models
with a shared vanilla LLM stage for consistency.

Input structure (eval_units/):
  eval_units/
    FloorName/
      unit_1/    <-- flat directory of room JSONs
      unit_2/

Output structure (eval/):
  eval/
    vanilla_cache/
      FloorName__unit_1/
        room_stem_vanilla.json
    model_name/
      FloorName/
        unit_1/
          room_stem/
            generated_scene.json
            cross_scene_retrieval.json

Usage:
  ATTN_IMPLEMENTATION=sdpa uv run python -m input_test.orchestrate_eval \
      --eval-units-dir eval_units --seed 42

  ATTN_IMPLEMENTATION=sdpa uv run python -m input_test.orchestrate_eval \
      --eval-units-dir eval_units --seed 42 \
      --style-prompt "modern scandinavian" --no-fill-ratio
"""

import os
import gc
import json
import sys
import shutil
import argparse
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

from src.respace import ReSpace, ORI_VANILLA_MODEL_ID
from src.sample import AssetRetrievalModule
from src.bathroom_layout import generate_bathroom_layout
from src.utils import set_seeds

# Import shared constants and helpers from orchestrate_unit
from input_test.orchestrate_unit import (
    MODEL_ID,
    ENV_FILE,
    ENV_FILE_BATHROOM,
    N_BON_SGLLM,
    N_BON_ASSETS,
    K_FEW_SHOT_SAMPLES,
    ORI_N_BON_ASSETS,
    DO_PROP_SAMPLING,
    DO_ICL,
    DO_CLASS_LABELS,
    USE_VLLM,
    DEFAULT_DATASET_ROOM_TYPE,
    RETRIEVAL_LAMBD,
    RETRIEVAL_SIGMA,
    RETRIEVAL_TEMP,
    RETRIEVAL_TOP_P,
    RETRIEVAL_TOP_K,
    RETRIEVAL_SIZE_THRESHOLD,
    SUPPORTED_ROOM_TYPES,
    DATASET_ROOM_TYPE_MAP,
    normalize_room_type,
    order_rooms_living_first,
    render_topdown_bboxes,
    _load_env,
    _make_retrieval_module,
    _run_retrieval_for_rooms,
    _run_original_retrieval_for_rooms,
)

# ============================================================================ #
# Model definitions                                                             #
# ============================================================================ #

EVAL_MODELS = [
    {
        "name": "1_7b",
        "checkpoint": "ckpts/qwen3_1_7b/old/checkpoint-best",
        "arch": True,
    },
    {
        "name": "4b",
        "checkpoint": "ckpts/qwen3_4b/old/checkpoint-best",
        "arch": True,
    },
    {
        "name": "original",
        "checkpoint": MODEL_ID,
        "arch": False,
        "ori_method": True,
    },
]

# ============================================================================ #
# Unit discovery                                                                #
# ============================================================================ #


def discover_eval_units(eval_units_dir):
    """Discover all floor/unit combos under eval_units_dir.

    Expected structure:
      eval_units_dir/FloorName/unit_X/  (contains room JSONs)

    Returns list of (floor_name, unit_name, unit_path).
    """
    eval_units_dir = Path(eval_units_dir)
    units = []
    for floor_dir in sorted(eval_units_dir.iterdir()):
        if not floor_dir.is_dir():
            continue
        for unit_dir in sorted(floor_dir.iterdir()):
            if not unit_dir.is_dir():
                continue
            # Check that it has room JSONs
            room_files = list(unit_dir.glob("*.json"))
            if not room_files:
                print(f"SKIP: {floor_dir.name}/{unit_dir.name} (no .json files)")
                continue
            units.append((floor_dir.name, unit_dir.name, unit_dir))
    return units


def discover_rooms_from_dir(unit_dir):
    """Discover rooms from a flat directory of room JSONs.

    Returns list of (json_path, canonical_room_type, scene_dict).
    """
    room_files = sorted(Path(unit_dir).glob("*.json"))
    room_entries = []
    for rf in room_files:
        with open(rf) as f:
            scene = json.load(f)
        raw_type = scene.get("room_type", "unknown")
        canonical = normalize_room_type(raw_type)
        if canonical is None:
            print(f"SKIP: {rf.name} (room_type={raw_type}, not in {SUPPORTED_ROOM_TYPES})")
            continue
        room_entries.append((rf, canonical, scene))
    return room_entries


# ============================================================================ #
# Vanilla LLM cache                                                             #
# ============================================================================ #


def get_vanilla_cache_dir(eval_output_dir, floor_name, unit_name):
    return Path(eval_output_dir) / "vanilla_cache" / f"{floor_name}__{unit_name}"


def load_vanilla_cache(cache_dir, room_stem):
    cache_file = Path(cache_dir) / f"{room_stem}_vanilla.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def save_vanilla_cache(cache_dir, room_stem, vanilla_commands):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = Path(cache_dir) / f"{room_stem}_vanilla.json"
    with open(cache_file, "w") as f:
        json.dump(vanilla_commands, f, indent=2)
    print(f"  Saved vanilla cache: {cache_file}")


# ============================================================================ #
# Stage 1: Layout generation with vanilla cache support                         #
# ============================================================================ #


def run_stage1_with_cache(room_entries, output_dir, model_path, style_prompt,
                          use_fill_ratio, include_openings, vanilla_cache_dir,
                          is_first_model, n_bon_assets=N_BON_ASSETS):
    """Generate layouts. On first model, runs vanilla LLM and caches commands.
    On subsequent models, uses cached commands and only runs SG-LLM placement.

    Returns list of (room_stem, room_type, generated_scene, room_output_dir).
    """
    print(f"\n{'='*70}")
    print("STAGE 1: Layout Generation")
    if not is_first_model:
        print("  (using cached vanilla commands)")
    print(f"{'='*70}")

    results = []

    # Separate bathrooms
    bathroom_entries = [(jp, rt, sc) for jp, rt, sc in room_entries if rt == "bathroom"]
    respace_entries = [(jp, rt, sc) for jp, rt, sc in room_entries if rt != "bathroom"]

    # --- Bathrooms (rule-based) ---
    for json_path, room_type, scene in bathroom_entries:
        stem = json_path.stem
        room_output = output_dir / stem
        room_output.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Generating bathroom layout: {json_path.name} -> {room_output} ---")
        result_scene = generate_bathroom_layout(scene)
        n_objects = len(result_scene.get("objects", []))
        if n_objects == 0:
            print(f"  FAILED: no fixtures placed for {json_path.name}")
            continue

        print(f"  Placed {n_objects} fixtures")
        scene_out = room_output / "generated_scene.json"
        with open(scene_out, "w") as f:
            json.dump(result_scene, f, indent=2)
        print(f"  Saved: {scene_out}")
        render_topdown_bboxes(result_scene, room_output)
        results.append((stem, room_type, result_scene, room_output))

    # --- Other rooms (ReSpace) ---
    if respace_entries:
        print(f"\nInitializing ReSpace (dataset_room_type={DEFAULT_DATASET_ROOM_TYPE})")
        respace = ReSpace(
            model_id=model_path,
            env_file=ENV_FILE,
            dataset_room_type=DEFAULT_DATASET_ROOM_TYPE,
            use_gpu=True,
            n_bon_sgllm=N_BON_SGLLM,
            n_bon_assets=n_bon_assets,
            do_prop_sampling_for_prompt=DO_PROP_SAMPLING,
            do_icl_for_prompt=DO_ICL,
            do_class_labels_for_prompt=DO_CLASS_LABELS,
            k_few_shot_samples=K_FEW_SHOT_SAMPLES,
            use_vllm=USE_VLLM,
            include_openings=include_openings,
        )

        for json_path, room_type, scene in respace_entries:
            stem = json_path.stem
            room_output = output_dir / stem
            room_output.mkdir(parents=True, exist_ok=True)

            print(f"\n--- Generating layout: {json_path.name} -> {room_output} ---")
            print(f"  Room type: {room_type}")

            if is_first_model:
                # Run vanilla LLM and cache the commands
                result = respace.generate_full_scene(
                    room_type=room_type,
                    scene_bounds_only=scene,
                    pth_viz_output=room_output,
                    style_prompt=style_prompt,
                    use_fill_ratio=use_fill_ratio,
                    return_vanilla_commands=True,
                )
                if len(result) == 3:
                    result_scene, is_success, vanilla_cmds = result
                else:
                    result_scene, is_success = result
                    vanilla_cmds = None

                if not is_success:
                    print(f"  FAILED: generation unsuccessful for {json_path.name}")
                    continue

                if vanilla_cmds is not None:
                    save_vanilla_cache(vanilla_cache_dir, stem, vanilla_cmds)
            else:
                # Load cached vanilla commands
                cached = load_vanilla_cache(vanilla_cache_dir, stem)
                if cached is None:
                    print(f"  ERROR: no vanilla cache found for {stem}, skipping")
                    continue

                result_scene, is_success = respace.generate_full_scene(
                    room_type=room_type,
                    scene_bounds_only=scene,
                    pth_viz_output=room_output,
                    style_prompt=style_prompt,
                    use_fill_ratio=use_fill_ratio,
                    cached_vanilla_commands=cached,
                )

                if not is_success:
                    print(f"  FAILED: generation unsuccessful for {json_path.name}")
                    continue

            n_objects = len(result_scene.get("objects", []))
            print(f"  Generated {n_objects} objects")
            for i, obj in enumerate(result_scene["objects"]):
                print(f"    [{i}] {obj.get('desc', 'unknown'):40s}  pos={obj['pos']}  size={obj['size']}")

            scene_out = room_output / "generated_scene.json"
            with open(scene_out, "w") as f:
                json.dump(result_scene, f, indent=2)
            print(f"  Saved: {scene_out}")
            render_topdown_bboxes(result_scene, room_output)
            results.append((stem, room_type, result_scene, room_output))

        del respace
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nStage 1 complete: {len(results)} room(s) generated.")
    return results


# ============================================================================ #
# Stage 2: Asset retrieval (reused from orchestrate_unit)                       #
# ============================================================================ #


def run_stage2(stage1_results, style_prompt, lambda_style, stochastic, ori_sample=False):
    """Cross-scene style-coherent asset retrieval."""
    print(f"\n{'='*70}")
    if ori_sample:
        print("STAGE 2: Original Asset Retrieval (description + size)")
    else:
        print("STAGE 2: Cross-scene Style-coherent Asset Retrieval")
    print(f"{'='*70}")

    if not stage1_results:
        print("No rooms to process.")
        return []

    dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dvc}")

    default_entries = [(s, rt, sc, ro) for s, rt, sc, ro in stage1_results if rt != "bathroom"]
    bathroom_entries = [(s, rt, sc, ro) for s, rt, sc, ro in stage1_results if rt == "bathroom"]

    unit_style_embeds = []
    results = []

    if default_entries:
        print(f"\n--- Asset retrieval for {len(default_entries)} non-bathroom room(s) using {ENV_FILE} ---")
        _load_env(ENV_FILE)
        retrieval = _make_retrieval_module(dvc)
        if ori_sample:
            res = _run_original_retrieval_for_rooms(
                default_entries, retrieval, stochastic,
                room_idx_offset=0,
            )
        else:
            res = _run_retrieval_for_rooms(
                default_entries, retrieval, unit_style_embeds,
                style_prompt, lambda_style, stochastic,
                room_idx_offset=0,
            )
        results.extend(res)
        del retrieval
        torch.cuda.empty_cache()
        gc.collect()

    if bathroom_entries:
        print(f"\n--- Asset retrieval for {len(bathroom_entries)} bathroom room(s) using {ENV_FILE_BATHROOM} ---")
        _load_env(ENV_FILE_BATHROOM)
        retrieval_bath = _make_retrieval_module(dvc)
        if ori_sample:
            res = _run_original_retrieval_for_rooms(
                bathroom_entries, retrieval_bath, stochastic,
                room_idx_offset=len(default_entries),
            )
        else:
            res = _run_retrieval_for_rooms(
                bathroom_entries, retrieval_bath, unit_style_embeds,
                style_prompt, lambda_style, stochastic,
                room_idx_offset=len(default_entries),
            )
        results.extend(res)
        del retrieval_bath
        torch.cuda.empty_cache()
        gc.collect()

    _load_env(ENV_FILE)

    print(f"\nStage 2 complete: {len(results)} room(s) with assets retrieved.")
    return results


# ============================================================================ #
# Main                                                                          #
# ============================================================================ #


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation: run same units across multiple SG-LLM models."
    )
    parser.add_argument("--eval-units-dir", type=str, required=True,
                        help="Directory containing FloorName/unit_X/ subdirectories with room JSONs")
    parser.add_argument("--output-dir", type=str, default="eval",
                        help="Output directory (default: eval/)")
    parser.add_argument("--style-prompt", type=str, default=None,
                        help="Style prompt for vanilla LLM and cross-scene retrieval")
    parser.add_argument("--no-fill-ratio", action="store_true",
                        help="Disable fill_ratio adjustment for non-rectangular rooms")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--lambda-style", type=float, default=0.1,
                        help="Weight for style coherence term (default: 0.1)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic sampling for asset retrieval (default: greedy)")
    parser.add_argument("--vanilla-cache-dir", type=str, default=None,
                        help="Path to existing vanilla_cache/ dir to reuse (skips vanilla LLM generation)") # eval/vanilla_cache
    args = parser.parse_args()

    use_fill_ratio = not args.no_fill_ratio
    eval_output_dir = Path(args.output_dir)

    if args.seed is not None:
        set_seeds(args.seed)
        print(f"Seed: {args.seed}")

    # Discover units
    units = discover_eval_units(args.eval_units_dir)
    if not units:
        print(f"ERROR: no units found under {args.eval_units_dir}")
        sys.exit(1)

    print(f"EVAL MODE: {args.eval_units_dir}")
    print(f"Found {len(units)} unit(s):")
    for floor_name, unit_name, unit_path in units:
        print(f"  {floor_name}/{unit_name} -> {unit_path}")
    print(f"\nModels ({len(EVAL_MODELS)}):")
    for m in EVAL_MODELS:
        print(f"  {m['name']}: {m['checkpoint']} (arch={m['arch']})")

    # Process each unit across all models
    for floor_name, unit_name, unit_path in units:
        print(f"\n{'#'*70}")
        print(f"# UNIT: {floor_name}/{unit_name}")
        print(f"{'#'*70}")

        room_entries = discover_rooms_from_dir(unit_path)
        if not room_entries:
            print(f"SKIP: {floor_name}/{unit_name} (no supported rooms)")
            continue

        room_entries = order_rooms_living_first(room_entries)
        if args.vanilla_cache_dir:
            vanilla_cache_dir = Path(args.vanilla_cache_dir) / f"{floor_name}__{unit_name}"
        else:
            vanilla_cache_dir = get_vanilla_cache_dir(eval_output_dir, floor_name, unit_name)

        print(f"Rooms ({len(room_entries)}):")
        for i, (rf, rt, _) in enumerate(room_entries):
            tag = " <-- style anchor" if i == 0 else ""
            print(f"  [{i}] {rf.name} ({rt}){tag}")

        for model_idx, model_cfg in enumerate(EVAL_MODELS):
            model_name = model_cfg["name"]
            model_path = model_cfg["checkpoint"]
            include_openings = model_cfg["arch"]
            ori_method = model_cfg.get("ori_method", False)
            is_first_model = (model_idx == 0) and not args.vanilla_cache_dir

            # Override settings for original method
            if ori_method:
                model_use_fill_ratio = False
                model_style_prompt = None
                model_n_bon_assets = ORI_N_BON_ASSETS
            else:
                model_use_fill_ratio = use_fill_ratio
                model_style_prompt = args.style_prompt
                model_n_bon_assets = N_BON_ASSETS

            print(f"\n{'='*70}")
            print(f"MODEL: {model_name} ({model_path})")
            print(f"  arch={include_openings}, first_model={is_first_model}, ori_method={ori_method}")
            print(f"{'='*70}")

            # Reset seed before each model so SG-LLM placement is reproducible per model
            if args.seed is not None:
                set_seeds(args.seed)

            model_output_dir = eval_output_dir / model_name / floor_name / unit_name
            model_output_dir.mkdir(parents=True, exist_ok=True)

            # Copy metadata.json from source floor dir into the model's floor dir
            src_metadata = unit_path.parent / "metadata.json"
            dst_metadata = eval_output_dir / model_name / floor_name / "metadata.json"
            if src_metadata.exists() and not dst_metadata.exists():
                shutil.copy2(src_metadata, dst_metadata)
                print(f"  Copied metadata: {dst_metadata}")

            # Stage 1
            stage1_results = run_stage1_with_cache(
                room_entries, model_output_dir, model_path,
                model_style_prompt, model_use_fill_ratio, include_openings,
                vanilla_cache_dir, is_first_model,
                n_bon_assets=model_n_bon_assets,
            )

            # Re-order: living first for style anchor
            living = []
            dining = []
            rest = []
            for entry in stage1_results:
                stem, rt, scene, room_output = entry
                if "livingroom" in rt.lower():
                    living.append(entry)
                elif "diningroom" in rt.lower():
                    dining.append(entry)
                else:
                    rest.append(entry)
            stage1_ordered = living + dining + rest

            # Stage 2
            stage2_results = run_stage2(
                stage1_ordered, model_style_prompt, args.lambda_style,
                args.stochastic, ori_sample=ori_method,
            )

            print(f"\n  {model_name}: {len(stage2_results)} rooms completed")
            for stem, room_type, scene, room_output, params in stage2_results:
                n_objs = len(scene.get("objects", []))
                print(f"    {stem} ({room_type}): {n_objs} objects")

    print(f"\n{'#'*70}")
    print(f"# EVALUATION COMPLETE")
    print(f"# Output: {eval_output_dir}")
    print(f"{'#'*70}")
    print("\nDone!")


if __name__ == "__main__":
    main()
