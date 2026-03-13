"""
Orchestrator: End-to-end pipeline for a unit of rooms.

Takes a FloorPlan-Cleaner output directory (with metadata.json) or a flat
directory of room JSONs, and runs:
  Stage 1  Layout generation   (ReSpace.generate_full_scene per room)
  Stage 2  Cross-scene asset retrieval  (style-coherent, living room anchor)
  Stage 3  3D rendering        (top-down + diagonal JPGs, opt-in)

FloorPlan-Cleaner output mode (has metadata.json):
  ATTN_IMPLEMENTATION=sdpa uv run python -m input_test.orchestrate_unit \
      --floorplan-dir input_test/01OG_2ROOMS --unit 1

  ATTN_IMPLEMENTATION=sdpa uv run python -m input_test.orchestrate_unit \
      --floorplan-dir input_test/01OG_2ROOMS --unit 1 \
      --style-prompt "modern scandinavian" --match-room-type

Manual / flat directory mode (no metadata.json, just room JSONs):
  ATTN_IMPLEMENTATION=sdpa uv run python -m input_test.orchestrate_unit \
      --unit-dir input_test/17feb/unit_1

  ATTN_IMPLEMENTATION=sdpa uv run python -m input_test.orchestrate_unit \
      --unit-dir input_test/17feb/unit_1 \
      --style-prompt "modern scandinavian" --render

Batch mode (process all floors and all units under a parent directory):
  ATTN_IMPLEMENTATION=sdpa uv run python -m input_test.orchestrate_unit \
      --batch-dir input_test/3d_front/full-fill --no-fill-ratio

  ATTN_IMPLEMENTATION=sdpa uv run python -m input_test.orchestrate_unit \
      --batch-dir input_test/3d_front/full-fill \
      --style-prompt "modern scandinavian" --render
"""

import os
import gc
import json
import sys
import time
import argparse
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

from src.respace import ReSpace, ORI_VANILLA_MODEL_ID
from src.sample import AssetRetrievalModule
from src.bathroom_layout import generate_bathroom_layout
from src.utils import set_seeds

def _fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


# ============================================================================ #
# CONFIGURATION                                                                 #
# ============================================================================ #

MODEL_ID = "gradient-spaces/respace-sg-llm-1.5b"
ENV_FILE = ".env"
ENV_FILE_BATHROOM = ".env_heg"

# ReSpace layout generation params
N_BON_SGLLM = 8
N_BON_ASSETS = 1
K_FEW_SHOT_SAMPLES = 2

# Original ReSpace method params (--ori-method)
ORI_N_BON_ASSETS = 1
DO_PROP_SAMPLING = True
DO_ICL = True
DO_CLASS_LABELS = True
USE_VLLM = False
DEFAULT_DATASET_ROOM_TYPE = "all"

# Asset retrieval params
RETRIEVAL_LAMBD = 0.5
RETRIEVAL_SIGMA = 0.05
RETRIEVAL_TEMP = 0.2
RETRIEVAL_TOP_P = 0.95
RETRIEVAL_TOP_K = 20
RETRIEVAL_SIZE_THRESHOLD = 0.5

# Supported room types (others are skipped).
# Matching is substring-based: "livingroom/diningroom" matches "livingroom".
# Bathrooms use rule-based generation (src.bathroom_layout), not ReSpace.
SUPPORTED_ROOM_TYPES = {"livingroom", "diningroom", "bedroom", "bathroom"}

# Dining rooms have no dedicated dataset split, so we use livingroom stats
# for the SG-LLM (Stage 2). Stage 1 still sees "diningroom" in the prompt.
DATASET_ROOM_TYPE_MAP = {
    "diningroom": "livingroom",
}


def normalize_room_type(raw_type):
    """Map a raw room_type string to a supported canonical type.

    Handles compound types from FloorPlan-Cleaner like "livingroom/diningroom".
    Checks more specific types first (diningroom before livingroom) to avoid
    "livingroom" matching "livingdiningroom".
    Returns the matched canonical type, or None if unsupported.
    """
    raw_lower = raw_type.lower().replace(" ", "")
    # Check diningroom before livingroom so "livingdiningroom" doesn't match "livingroom"
    if "diningroom" in raw_lower and "livingroom" not in raw_lower:
        return "diningroom"
    for supported in SUPPORTED_ROOM_TYPES:
        if supported in raw_lower:
            return supported
    return None

# ============================================================================ #
# Helpers                                                                       #
# ============================================================================ #


def _opening_distance_to_wall(opening, bounds_bottom):
    """Compute the minimum distance from an opening's center to the nearest wall edge."""
    from shapely.geometry import Point, LineString
    ox, oz = opening["pos"][0], opening["pos"][2]
    pts = [(v[0], v[2]) for v in bounds_bottom]
    min_dist = float("inf")
    for i in range(len(pts)):
        seg = LineString([pts[i], pts[(i + 1) % len(pts)]])
        min_dist = min(min_dist, seg.distance(Point(ox, oz)))
    return min_dist


def _opening_bbox_verts(opening):
    """Compute the 4 XZ corners of an opening's bounding box (unrotated)."""
    ox, oz = opening["pos"][0], opening["pos"][2]
    sw, sd = opening["size"][0], opening["size"][2]
    hw, hd = sw / 2.0, sd / 2.0
    return [
        (ox - hw, oz - hd),
        (ox + hw, oz - hd),
        (ox + hw, oz + hd),
        (ox - hw, oz + hd),
    ]


def render_topdown_bboxes(scene, output_path):
    """Render a 2D top-down view of the floor plan with furniture bounding boxes."""
    objects = scene.get("objects", [])
    openings = scene.get("openings", [])

    fig, (ax, ax_legend) = plt.subplots(1, 2, figsize=(16, 10),
                                         gridspec_kw={"width_ratios": [3, 1]})

    floor_verts = [(v[0], v[2]) for v in scene["bounds_bottom"]]
    floor_poly = MplPolygon(floor_verts, closed=True, fill=True,
                            facecolor="#f5deb3", edgecolor="black", linewidth=2)
    ax.add_patch(floor_poly)

    # --- Draw openings (doors and windows) ---
    bounds = scene["bounds_bottom"]
    for opening in openings:
        verts = _opening_bbox_verts(opening)
        d_wall = _opening_distance_to_wall(opening, bounds)
        ox, oz = opening["pos"][0], opening["pos"][2]

        if opening["type"] == "door":
            color = "red"
            label = f"door\nd={d_wall:.2f}"
        else:
            color = "blue"
            label = f"window\nd={d_wall:.2f}"

        opening_poly = MplPolygon(verts, closed=True, fill=True,
                                  facecolor=(*plt.cm.colors.to_rgba(color)[:3], 0.3),
                                  edgecolor=color, linewidth=1.5, zorder=4)
        ax.add_patch(opening_poly)
        ax.plot(ox, oz, "X", color=color, markersize=12, zorder=5, markeredgewidth=2)
        ax.annotate(label, (ox, oz), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 10), textcoords="offset points", color=color,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.2",
                    facecolor="white", edgecolor=color, alpha=0.8))

    colors = plt.cm.tab10.colors
    legend_entries = []

    for i, obj in enumerate(objects):
        pos = obj["pos"]
        size = obj["size"]
        rot = obj.get("rot", [0, 0, 0, 1])

        w, d = size[0], size[2]
        cx, cz = pos[0], pos[2]

        qx, qy, qz, qw = rot
        yaw = np.arctan2(2 * (qw * qy + qx * qz), 1 - 2 * (qy**2 + qz**2))

        corners = np.array([
            [-w/2, -d/2],
            [ w/2, -d/2],
            [ w/2,  d/2],
            [-w/2,  d/2],
        ])
        cos_a, sin_a = np.cos(-yaw), np.sin(-yaw)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = corners @ rot_mat.T
        rotated[:, 0] += cx
        rotated[:, 1] += cz

        color = colors[i % len(colors)]
        bbox_poly = MplPolygon(rotated, closed=True, fill=True,
                               facecolor=(*color, 0.4), edgecolor=color, linewidth=1.5)
        ax.add_patch(bbox_poly)

        ax.text(cx, cz, str(i), fontsize=8, ha="center", va="center",
                color="black", fontweight="bold")

        desc = obj.get("desc", f"obj_{i}")
        legend_entries.append((i, color, desc))

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Z (meters)")
    ax.set_title(f"Top-down layout: {scene.get('room_type', 'room')} ({len(objects)} objects)")
    ax.grid(True, alpha=0.3)

    ax_legend.axis("off")
    ax_legend.set_title("Legend", fontsize=12, fontweight="bold")

    if legend_entries:
        col_labels = ["#", "Color", "Description"]
        max_desc_chars = 30
        cell_text = [
            [str(idx), "", "\n".join(textwrap.wrap(desc, max_desc_chars))]
            for idx, _, desc in legend_entries
        ]
        cell_colors = [["white", (*c, 0.4), "white"] for _, c, _ in legend_entries]

        table = ax_legend.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellColours=cell_colors,
            colColours=["#dddddd"] * 3,
            loc="upper center",
            cellLoc="left",
            colWidths=[0.08, 0.1, 0.82],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)

        for (row, col), cell in table.get_celld().items():
            n_lines = cell.get_text().get_text().count("\n") + 1
            cell.set_height(0.05 * n_lines)

    out_file = Path(output_path) / "floorplan_bboxes.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved bbox plot: {out_file}")


def order_rooms_living_first(room_entries):
    """Sort room entries so living/dining rooms come first (style anchor).

    Each entry is a tuple of (json_path, room_type, scene_dict).
    Living rooms first, then dining rooms, then the rest.
    """
    living = []
    dining = []
    rest = []
    for entry in room_entries:
        rt = entry[1].lower()
        if "livingroom" in rt or "living_room" in rt:
            living.append(entry)
        elif "diningroom" in rt or "dining_room" in rt:
            dining.append(entry)
        else:
            rest.append(entry)
    if not living and not dining:
        print("WARNING: No living/dining room found. Using first room as style anchor.")
    return living + dining + rest


def render_3d_scene(scene_with_assets, output_path, filename):
    """Render the scene with retrieved 3D assets."""
    from src.viz import render_scene_and_export
    print(f"  Rendering 3D scene: {filename} ...")
    render_scene_and_export(
        scene_with_assets,
        filename=filename,
        pth_output=output_path,
        resolution=(1024, 1024),
        show_bboxes=False,
        show_assets=True,
        use_dynamic_zoom=True,
    )
    print(f"  Saved renders: {output_path}/top/ and {output_path}/diag/")


# ============================================================================ #
# Stage 1: Layout generation                                                    #
# ============================================================================ #


def _generate_single_room(respace, json_path, room_type, scene, room_output, style_prompt=None, use_fill_ratio=True, unit_room_types=None):
    """Generate layout for a single non-bathroom room via ReSpace.

    Returns (result_scene, success_bool).
    """
    print(f"\n--- Generating layout: {json_path.name} -> {room_output} ---")
    print(f"  Room type: {room_type}")
    print(f"  Boundary vertices: {len(scene['bounds_bottom'])}")

    result_scene, is_success = respace.generate_full_scene(
        room_type=room_type,
        scene_bounds_only=scene,
        pth_viz_output=room_output,
        style_prompt=style_prompt,
        use_fill_ratio=use_fill_ratio,
        unit_room_types=unit_room_types,
    )

    if not is_success:
        print(f"  FAILED: generation unsuccessful for {json_path.name}")
        return None, False

    n_objects = len(result_scene.get("objects", []))
    print(f"  Generated {n_objects} objects")
    for i, obj in enumerate(result_scene["objects"]):
        print(f"    [{i}] {obj.get('desc', 'unknown'):40s}  pos={obj['pos']}  size={obj['size']}")

    return result_scene, True


def _generate_bathroom(json_path, scene, room_output):
    """Generate layout for a bathroom via rule-based placement.

    Returns (result_scene, success_bool).
    """
    print(f"\n--- Generating bathroom layout: {json_path.name} -> {room_output} ---")
    print(f"  Boundary vertices: {len(scene['bounds_bottom'])}")
    print(f"  Openings: {len(scene.get('openings', []))}")

    result_scene = generate_bathroom_layout(scene)

    n_objects = len(result_scene.get("objects", []))
    if n_objects == 0:
        print(f"  FAILED: no fixtures placed for {json_path.name}")
        return None, False

    print(f"  Placed {n_objects} fixtures")
    for i, obj in enumerate(result_scene["objects"]):
        print(f"    [{i}] {obj.get('desc', 'unknown'):40s}  pos={obj['pos']}  size={obj['size']}")

    return result_scene, True


def run_stage1(room_entries, output_dir, model_path, match_room_type, style_prompt=None, use_fill_ratio=True, include_openings=False, vanilla_model_id=None, n_bon_assets=N_BON_ASSETS):
    """Generate furniture layouts for each room.

    Bathrooms use the rule-based generator; all other rooms use ReSpace.
    Returns list of (room_stem, room_type, generated_scene, room_output_dir).
    """
    print(f"\n{'='*70}")
    print("STAGE 1: Layout Generation")
    print(f"{'='*70}")

    _stage1_start = time.time()
    results = []

    # Build unit-wide room type list (non-bathroom) for cross-room awareness
    unit_room_types = [rt for _, rt, _ in room_entries if rt != "bathroom"]
    if len(unit_room_types) > 1:
        print(f"  Unit room context: {unit_room_types}")

    # Separate bathrooms from rooms that need ReSpace
    bathroom_entries = [(jp, rt, sc) for jp, rt, sc in room_entries if rt == "bathroom"]
    respace_entries = [(jp, rt, sc) for jp, rt, sc in room_entries if rt != "bathroom"]

    # --- Bathrooms (rule-based, no model needed) ---
    for json_path, room_type, scene in bathroom_entries:
        stem = json_path.stem
        room_output = output_dir / stem
        room_output.mkdir(parents=True, exist_ok=True)

        result_scene, ok = _generate_bathroom(json_path, scene, room_output)
        if not ok:
            continue

        scene_out = room_output / "generated_scene.json"
        with open(scene_out, "w") as f:
            json.dump(result_scene, f, indent=2)
        print(f"  Saved: {scene_out}")

        render_topdown_bboxes(result_scene, room_output)
        results.append((stem, room_type, result_scene, room_output))

    # --- Other rooms (ReSpace model) ---
    if respace_entries:
        if match_room_type:
            rooms_by_type = {}
            for json_path, room_type, scene in respace_entries:
                rooms_by_type.setdefault(room_type, []).append((json_path, room_type, scene))

            for room_type, entries in rooms_by_type.items():
                dataset_rt = DATASET_ROOM_TYPE_MAP.get(room_type, room_type)
                print(f"\nInitializing ReSpace for room type: {room_type} (dataset: {dataset_rt})")
                respace = ReSpace(
                    model_id=model_path,
                    env_file=ENV_FILE,
                    dataset_room_type=dataset_rt,
                    use_gpu=True,
                    n_bon_sgllm=N_BON_SGLLM,
                    n_bon_assets=n_bon_assets,
                    do_prop_sampling_for_prompt=DO_PROP_SAMPLING,
                    do_icl_for_prompt=DO_ICL,
                    do_class_labels_for_prompt=DO_CLASS_LABELS,
                    k_few_shot_samples=K_FEW_SHOT_SAMPLES,
                    use_vllm=USE_VLLM,
                    include_openings=include_openings,
                    vanilla_model_id=vanilla_model_id,
                )

                for json_path, rt, scene in entries:
                    stem = json_path.stem
                    room_output = output_dir / stem
                    room_output.mkdir(parents=True, exist_ok=True)

                    result_scene, ok = _generate_single_room(respace, json_path, rt, scene, room_output, style_prompt=style_prompt, use_fill_ratio=use_fill_ratio, unit_room_types=unit_room_types)
                    if not ok:
                        continue

                    scene_out = room_output / "generated_scene.json"
                    with open(scene_out, "w") as f:
                        json.dump(result_scene, f, indent=2)
                    print(f"  Saved: {scene_out}")

                    render_topdown_bboxes(result_scene, room_output)
                    results.append((stem, rt, result_scene, room_output))

                del respace
                torch.cuda.empty_cache()
                gc.collect()
        else:
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
                vanilla_model_id=vanilla_model_id,
            )

            for json_path, room_type, scene in respace_entries:
                stem = json_path.stem
                room_output = output_dir / stem
                room_output.mkdir(parents=True, exist_ok=True)

                result_scene, ok = _generate_single_room(respace, json_path, room_type, scene, room_output, style_prompt=style_prompt, use_fill_ratio=use_fill_ratio, unit_room_types=unit_room_types)
                if not ok:
                    continue

                scene_out = room_output / "generated_scene.json"
                with open(scene_out, "w") as f:
                    json.dump(result_scene, f, indent=2)
                print(f"  Saved: {scene_out}")

                render_topdown_bboxes(result_scene, room_output)
                results.append((stem, room_type, result_scene, room_output))

            del respace
            torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.time() - _stage1_start
    print(f"\nStage 1 complete: {len(results)} room(s) generated. [{_fmt_time(elapsed)}]")
    return results


# ============================================================================ #
# Stage 2: Cross-scene style-coherent asset retrieval                           #
# ============================================================================ #


def _load_env(env_file):
    """Clear asset-related env vars and reload from the given .env file."""
    asset_keys = [
        "PTH_ASSETS_METADATA", "PTH_ASSETS_METADATA_SCALED",
        "PTH_ASSETS_METADATA_SIMPLE_DESCS", "PTH_ASSETS_METADATA_PROMPTS",
        "PTH_ASSETS_EMBED", "PTH_ASSETS_EMBED_STYLE", "PTH_ASSETS_EMBED_CATEGORY",
        "PTH_3DFUTURE_ASSETS",
    ]
    for k in asset_keys:
        os.environ.pop(k, None)
    load_dotenv(env_file, override=True)
    print(f"  Loaded env: {env_file}")


def _make_retrieval_module(dvc):
    """Create a fresh AssetRetrievalModule using current os.environ."""
    return AssetRetrievalModule(
        lambd=RETRIEVAL_LAMBD,
        sigma=RETRIEVAL_SIGMA,
        temp=RETRIEVAL_TEMP,
        top_p=RETRIEVAL_TOP_P,
        top_k=RETRIEVAL_TOP_K,
        asset_size_threshold=RETRIEVAL_SIZE_THRESHOLD,
        dvc=dvc,
        do_print=True,
    )


def _run_original_retrieval_for_rooms(entries, retrieval, stochastic, room_idx_offset):
    """Run original asset retrieval (description + size only) for a list of rooms."""
    results = []
    mode = "stochastic" if stochastic else "greedy"

    for local_idx, (stem, room_type, scene, room_output) in enumerate(entries):
        room_idx = room_idx_offset + local_idx

        print(f"\n{'#'*70}")
        print(f"# Room [{room_idx}]: {stem}")
        print(f"# Type: {room_type}")
        print(f"# mode={mode} (original sampling)")
        print(f"{'#'*70}")

        n_objs = len(scene.get("objects", []))
        if n_objs == 0:
            print("  No objects in scene, skipping.")
            continue

        result = retrieval.sample_all_assets(
            scene,
            is_greedy_sampling=not stochastic,
        )

        for i, obj in enumerate(result.get("objects", [])):
            desc = obj.get("sampled_asset_desc", obj.get("desc", "N/A"))
            print(f"  [{i}] {desc[:60]}")

        params = {
            "mode": "original",
            "greedy": not stochastic,
            "room_order": room_idx,
        }

        retrieval_out = {
            "original_sample": result,
            "params": params,
        }
        out_file = room_output / "original_retrieval.json"
        with open(out_file, "w") as f:
            json.dump(retrieval_out, f, indent=2)
        print(f"  Saved: {out_file}")

        results.append((stem, room_type, result, room_output, params))

    return results


def _run_retrieval_for_rooms(entries, retrieval, unit_style_embeds, style_prompt,
                             lambda_style, stochastic, room_idx_offset):
    """Run asset retrieval on a list of (stem, room_type, scene, room_output) entries.

    Mutates unit_style_embeds in-place.  Returns list of result tuples.
    """
    results = []
    mode = "stochastic" if stochastic else "greedy"

    for local_idx, (stem, room_type, scene, room_output) in enumerate(entries):
        room_idx = room_idx_offset + local_idx
        is_anchor = (room_idx == 0)

        print(f"\n{'#'*70}")
        print(f"# Room [{room_idx}]: {stem}")
        print(f"# Type: {room_type}")
        print(f"# lambda_style={lambda_style}, mode={mode}")
        if is_anchor:
            print(f"# Role: STYLE ANCHOR")
        else:
            print(f"# Role: CROSS-SCENE BIASED ({len(unit_style_embeds)} embeddings from prior rooms)")
        if style_prompt:
            print(f"# user_prompt=\"{style_prompt}\"")
        print(f"{'#'*70}")

        n_objs = len(scene.get("objects", []))
        if n_objs == 0:
            print("  No objects in scene, skipping.")
            continue

        initial_embeds = None if is_anchor else list(unit_style_embeds)

        result, room_style_embeds = retrieval.sample_all_assets_style_coherent_cross_scene(
            scene,
            lambda_style=lambda_style,
            is_greedy_sampling=not stochastic,
            user_prompt=style_prompt,
            initial_style_embeds=initial_embeds,
            use_category_only=False,
        )

        # Collect only new embeddings from this room
        n_prior = len(initial_embeds) if initial_embeds else 0
        if style_prompt is not None:
            n_prior += 1  # user prompt embed was prepended
        new_embeds = room_style_embeds[n_prior:]
        unit_style_embeds.extend(new_embeds)

        print(f"\n  Room contributed {len(new_embeds)} new style embeddings")
        print(f"  Unit-wide total: {len(unit_style_embeds)} style embeddings")

        # Print object summary
        for i, obj in enumerate(result.get("objects", [])):
            desc = obj.get("sampled_asset_desc", obj.get("desc", "N/A"))
            print(f"  [{i}] {desc[:60]}")

        # Save retrieval results
        params = {
            "lambda_style": lambda_style,
            "greedy": not stochastic,
            "user_prompt": style_prompt,
            "room_order": room_idx,
            "is_anchor": is_anchor,
            "n_prior_style_embeds": n_prior - (1 if style_prompt else 0),
            "n_new_style_embeds": len(new_embeds),
            "n_total_unit_style_embeds": len(unit_style_embeds),
        }

        retrieval_out = {
            "cross_scene_style": result,
            "params": params,
        }
        out_file = room_output / "cross_scene_retrieval.json"
        with open(out_file, "w") as f:
            json.dump(retrieval_out, f, indent=2)
        print(f"  Saved: {out_file}")

        results.append((stem, room_type, result, room_output, params))

    return results


def run_stage2(stage1_results, style_prompt, lambda_style, stochastic, ori_sample=False):
    """Retrieve assets for each room.

    When ori_sample=True, uses the original sampling (description + size only).
    Otherwise uses cross-scene style-coherent retrieval.
    Bathrooms use assets from .env_heg; all other rooms use .env.
    Returns list of (room_stem, room_type, sampled_scene, room_output_dir, params).
    """
    print(f"\n{'='*70}")
    if ori_sample:
        print("STAGE 2: Original Asset Retrieval (description + size)")
    else:
        print("STAGE 2: Cross-scene Style-coherent Asset Retrieval")
    print(f"{'='*70}")

    _stage2_start = time.time()

    if not stage1_results:
        print("No rooms to process.")
        return []

    dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dvc}")

    # Split into non-bathroom (default assets) and bathroom (custom assets)
    default_entries = [(s, rt, sc, ro) for s, rt, sc, ro in stage1_results if rt != "bathroom"]
    bathroom_entries = [(s, rt, sc, ro) for s, rt, sc, ro in stage1_results if rt == "bathroom"]

    unit_style_embeds = []
    results = []

    # --- Non-bathroom rooms (use .env) ---
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

    # --- Bathroom rooms (use .env_heg) ---
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

    # Restore default env
    _load_env(ENV_FILE)

    elapsed = time.time() - _stage2_start
    print(f"\nStage 2 complete: {len(results)} room(s) with assets retrieved. [{_fmt_time(elapsed)}]")
    print(f"Total style embeddings accumulated: {len(unit_style_embeds)}")
    return results


# ============================================================================ #
# Stage 3: Rendering                                                            #
# ============================================================================ #


def run_stage3(stage2_results):
    """Render 3D scenes with retrieved assets."""
    print(f"\n{'='*70}")
    print("STAGE 3: 3D Rendering")
    print(f"{'='*70}")

    _stage3_start = time.time()
    for stem, room_type, scene, room_output, params in stage2_results:
        if not scene.get("bounds_bottom"):
            print(f"  WARNING: Skipping render for {stem} -- no bounds_bottom")
            continue

        render_path = room_output / "render"
        render_path.mkdir(parents=True, exist_ok=True)
        render_3d_scene(scene, render_path, stem)

    elapsed = time.time() - _stage3_start
    print(f"\nStage 3 complete. [{_fmt_time(elapsed)}]")


# ============================================================================ #
# Main                                                                          #
# ============================================================================ #


def discover_rooms_from_metadata(floorplan_dir, unit_id):
    """Discover rooms from FloorPlan-Cleaner metadata.json.

    Returns list of (json_path, canonical_room_type, scene_dict).
    """
    metadata_path = floorplan_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Find the requested unit
    unit_entry = None
    for u in metadata["units"]:
        if u["unit_id"] == unit_id:
            unit_entry = u
            break

    if unit_entry is None:
        available = [u["unit_id"] for u in metadata["units"]]
        print(f"ERROR: unit {unit_id} not found in metadata. Available: {available}")
        sys.exit(1)

    room_entries = []
    for room in unit_entry["rooms"]:
        raw_type = room["room_type"]
        canonical = normalize_room_type(raw_type)

        json_path = floorplan_dir / room["output_file"]
        if not json_path.exists():
            print(f"WARNING: room file missing: {json_path}")
            continue

        if canonical is None:
            print(f"SKIP: {json_path.name} (room_type={raw_type}, not in {SUPPORTED_ROOM_TYPES})")
            continue

        with open(json_path) as f:
            scene = json.load(f)

        room_entries.append((json_path, canonical, scene))

    return room_entries


def discover_all_units_from_batch_dir(batch_dir):
    """Discover all floor folders and their units under a parent directory.

    Scans for subdirectories containing metadata.json, then yields
    (floorplan_dir, unit_id) for every unit found.
    Returns list of (floorplan_dir, unit_id, unit_output_dir).
    """
    batch_dir = Path(batch_dir)
    floor_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir()])
    units = []
    for floor_dir in floor_dirs:
        metadata_path = floor_dir / "metadata.json"
        if not metadata_path.exists():
            print(f"SKIP: {floor_dir.name} (no metadata.json)")
            continue
        with open(metadata_path) as f:
            metadata = json.load(f)
        for u in metadata["units"]:
            uid = u["unit_id"]
            unit_output = floor_dir / f"unit_{uid}" / "output"
            units.append((floor_dir, uid, unit_output))
    return units


def discover_rooms_from_dir(unit_dir):
    """Discover rooms from a flat directory of room JSONs.

    Returns list of (json_path, canonical_room_type, scene_dict).
    """
    room_files = sorted(unit_dir.glob("*.json"))
    if not room_files:
        print(f"ERROR: no .json files found in {unit_dir}")
        sys.exit(1)

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


def run_unit_pipeline(room_entries, output_dir, model_path, match_room_type,
                      style_prompt, lambda_style, stochastic, use_fill_ratio,
                      ori_method, do_render, source_label, include_openings=False):
    """Run the full pipeline (stage 1-3) for a single unit.

    When ori_method=True, replicates the original ReSpace paper:
      - Vanilla LLM = Meta-Llama-3.1-8B-Instruct
      - No fill ratio, no openings, no style prompt
      - Original asset sampling (desc + size only), n_bon_assets=1
    Returns list of stage2 result tuples.
    """
    # Override settings when replicating original method
    if ori_method:
        vanilla_model_id = ORI_VANILLA_MODEL_ID
        use_fill_ratio = False
        include_openings = False
        style_prompt = None
        n_bon_assets = ORI_N_BON_ASSETS
        print("[ori-method] Replicating original ReSpace pipeline:")
        print(f"  Vanilla LLM: {vanilla_model_id}")
        print(f"  Fill ratio: disabled")
        print(f"  Architecture openings: disabled")
        print(f"  Style prompt: disabled")
        print(f"  n_bon_assets: {n_bon_assets}")
        print(f"  Asset retrieval: original (desc + size only)")
    else:
        vanilla_model_id = None
        n_bon_assets = N_BON_ASSETS

    print(f"\n{'='*70}")
    print(f"UNIT: {source_label}")
    print(f"{'='*70}")

    _pipeline_start = time.time()
    room_entries = order_rooms_living_first(room_entries)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Style prompt: {style_prompt or '(none)'}")
    print(f"Render 3D: {do_render}")
    print(f"Rooms ({len(room_entries)}):")
    for i, (rf, rt, _) in enumerate(room_entries):
        tag = " <-- style anchor" if i == 0 else ""
        print(f"  [{i}] {rf.name} ({rt}){tag}")

    stage1_results = run_stage1(room_entries, output_dir, model_path, match_room_type, style_prompt=style_prompt, use_fill_ratio=use_fill_ratio, include_openings=include_openings, vanilla_model_id=vanilla_model_id, n_bon_assets=n_bon_assets)

    # Re-order stage1 results: living room first, then dining, then rest
    living = []
    dining = []
    rest = []
    for entry in stage1_results:
        stem, rt, scene, room_output = entry
        if "livingroom" in rt.lower() or "living_room" in rt.lower():
            living.append(entry)
        elif "diningroom" in rt.lower() or "dining_room" in rt.lower():
            dining.append(entry)
        else:
            rest.append(entry)
    stage1_ordered = living + dining + rest

    stage2_results = run_stage2(stage1_ordered, style_prompt, lambda_style, stochastic, ori_sample=ori_method)

    if do_render:
        run_stage3(stage2_results)

    total_elapsed = time.time() - _pipeline_start
    print(f"\n{'#'*70}")
    print(f"# UNIT COMPLETE: {source_label}")
    print(f"# Output: {output_dir}")
    print(f"# Rooms processed: {len(stage2_results)}")
    print(f"# Total time: {_fmt_time(total_elapsed)}")
    print(f"{'#'*70}")

    for stem, room_type, scene, room_output, params in stage2_results:
        n_objs = len(scene.get("objects", []))
        summary = f"  {stem} ({room_type}): {n_objs} objects"
        if "is_anchor" in params:
            summary += f", anchor={params['is_anchor']}, new_embeds={params['n_new_style_embeds']}"
        summary += f", mode={params.get('mode', 'cross-scene')}"
        print(summary)

    return stage2_results


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate full pipeline: layout -> asset retrieval -> render."
    )

    # Input source (pick one)
    input_group = parser.add_argument_group("input source (pick one)")
    input_group.add_argument("--floorplan-dir", type=str, default=None,
                             help="FloorPlan-Cleaner output directory (contains metadata.json)")
    input_group.add_argument("--unit", type=int, default=None,
                             help="Unit ID to process (required with --floorplan-dir)")
    input_group.add_argument("--unit-dir", type=str, default=None,
                             help="Flat directory of room JSONs (no metadata.json needed)")
    input_group.add_argument("--batch-dir", type=str, default=None,
                             help="Parent directory containing multiple floor folders (each with metadata.json). Processes all floors and all units.")

    # Pipeline options
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Defaults to <source-dir>/output/")
    parser.add_argument("--style-prompt", type=str, default=None,
                        help="Style prompt for cross-scene coherence (e.g. 'modern scandinavian')")
    parser.add_argument("--lambda-style", type=float, default=0.1,
                        help="Weight for style coherence term (default: 0.1)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to local checkpoint directory")
    parser.add_argument("--match-room-type", action="store_true",
                        help="Init separate ReSpace per room type for type-specific generation")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic sampling for asset retrieval (default: greedy)")
    parser.add_argument("--no-fill-ratio", action="store_true",
                        help="Disable fill_ratio adjustment for non-rectangular rooms")
    parser.add_argument("--ori-method", action="store_true",
                        help="Replicate original ReSpace paper: Llama-3.1-8B vanilla LLM, "
                             "no fill ratio, no architecture openings, no style prompt, "
                             "original asset sampling (desc + size only), n_bon_assets=1")
    parser.add_argument("--render", action="store_true",
                        help="Enable Stage 3 (3D rendering with assets). Off by default.")
    parser.add_argument("--arch", action="store_true",
                        help="Include architectural openings (doors/windows) in the scene graph sent to the SG-LLM. "
                             "Requires a model trained with openings data.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    model_path = args.checkpoint if args.checkpoint else MODEL_ID
    use_fill_ratio = not args.no_fill_ratio

    if args.seed is not None:
        set_seeds(args.seed)
        print(f"Seed: {args.seed}")

    # ---------------------------------------------------------------------- #
    # Validate input mode                                                      #
    # ---------------------------------------------------------------------- #

    modes_given = sum([
        bool(args.floorplan_dir),
        bool(args.unit_dir),
        bool(args.batch_dir),
    ])
    if modes_given != 1:
        print("ERROR: provide exactly one of --floorplan-dir + --unit, --unit-dir, or --batch-dir.")
        sys.exit(1)
    if args.floorplan_dir and args.unit is None:
        print("ERROR: --unit is required when using --floorplan-dir.")
        sys.exit(1)

    # ---------------------------------------------------------------------- #
    # Batch mode                                                               #
    # ---------------------------------------------------------------------- #

    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
        if not batch_dir.is_dir():
            print(f"ERROR: batch directory not found: {batch_dir}")
            sys.exit(1)

        units = discover_all_units_from_batch_dir(batch_dir)
        if not units:
            print(f"ERROR: no units found under {batch_dir}")
            sys.exit(1)

        print(f"BATCH MODE: {batch_dir}")
        print(f"Found {len(units)} unit(s) across {len(set(fp for fp, _, _ in units))} floor(s):")
        for fp_dir, uid, unit_out in units:
            print(f"  {fp_dir.name} / unit_{uid} -> {unit_out}")

        all_results = []
        for fp_dir, uid, unit_out in units:
            source_label = f"{fp_dir.name}/unit_{uid}"
            room_entries = discover_rooms_from_metadata(fp_dir, uid)
            if not room_entries:
                print(f"SKIP: {source_label} (no supported rooms)")
                continue

            output_dir = Path(args.output_dir) / fp_dir.name / f"unit_{uid}" / "output" if args.output_dir else unit_out

            results = run_unit_pipeline(
                room_entries, output_dir, model_path, args.match_room_type,
                args.style_prompt, args.lambda_style, args.stochastic,
                use_fill_ratio, args.ori_method, args.render, source_label,
                include_openings=args.arch,
            )
            all_results.append((source_label, results))

        print(f"\n{'#'*70}")
        print(f"# BATCH COMPLETE")
        print(f"# Directory: {batch_dir}")
        print(f"# Units processed: {len(all_results)}")
        print(f"{'#'*70}")
        for label, results in all_results:
            n_rooms = len(results)
            total_objs = sum(len(sc.get("objects", [])) for _, _, sc, _, _ in results)
            print(f"  {label}: {n_rooms} rooms, {total_objs} total objects")

        print("\nDone!")
        return

    # ---------------------------------------------------------------------- #
    # Single-unit mode                                                         #
    # ---------------------------------------------------------------------- #

    if args.floorplan_dir:
        floorplan_dir = Path(args.floorplan_dir)
        if not floorplan_dir.is_dir():
            print(f"ERROR: floorplan directory not found: {floorplan_dir}")
            sys.exit(1)
        source_label = f"{floorplan_dir.name}/unit_{args.unit}"
        room_entries = discover_rooms_from_metadata(floorplan_dir, args.unit)
        default_output = floorplan_dir / f"unit_{args.unit}" / "output"
    else:
        unit_dir = Path(args.unit_dir)
        if not unit_dir.is_dir():
            print(f"ERROR: unit directory not found: {unit_dir}")
            sys.exit(1)
        source_label = str(unit_dir)
        room_entries = discover_rooms_from_dir(unit_dir)
        default_output = unit_dir / "output"

    if not room_entries:
        print(f"ERROR: no supported rooms found.")
        print(f"Supported types: {SUPPORTED_ROOM_TYPES}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else default_output

    run_unit_pipeline(
        room_entries, output_dir, model_path, args.match_room_type,
        args.style_prompt, args.lambda_style, args.stochastic,
        use_fill_ratio, args.ori_method, args.render, source_label,
        include_openings=args.arch,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
