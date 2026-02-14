"""
Test script: Generate furniture layouts from custom floorplans in batch.

Directory structure:
  input_test/
    <unit_folder>/          e.g. "unit_1"
      room_a.json
      room_b.json
      ...

Output structure (written inside the unit folder):
  input_test/
    <unit_folder>/
      <room_a>/
        generated_scene.json
        floorplan_bboxes.png
      <room_b>/
        generated_scene.json
        floorplan_bboxes.png

Usage:
  ATTN_IMPLEMENTATION=sdpa uv run python -m input_test.test_custom_floorplan --unit unit_1
  ATTN_IMPLEMENTATION=sdpa uv run python -m input_test.test_custom_floorplan --unit unit_1 --match-room-type
"""

import json
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

from src.respace import ReSpace

# ============================================================================ #
# CONFIGURATION -- tweak these to improve layout quality                       #
# ============================================================================ #

MODEL_ID = "gradient-spaces/respace-sg-llm-1.5b"
ENV_FILE = ".env"

# Best-of-N for the SG-LLM placement model. Higher = more candidates, better
# spatial layouts but slower. Default 8, try 16-32 for small/tricky rooms.
N_BON_SGLLM = 8

# Best-of-N for asset retrieval. Higher = better size/style match.
# 1 = greedy (fast), 4-8 = stochastic (better quality).
N_BON_ASSETS = 4

# Number of few-shot (ICL) examples injected into the vanilla LLM prompt.
# More examples = stronger guidance on what furniture belongs in a given room
# size. 0 disables ICL entirely. Try 3-5 for better results.
K_FEW_SHOT_SAMPLES = 2

# Whether to sample object count proportionally to floor area (True) or
# uniformly within the valid range for that floor area bin (False).
DO_PROP_SAMPLING = True

# Whether to include in-context learning examples in the vanilla LLM prompt.
DO_ICL = True

# Whether to constrain the vanilla LLM to known furniture class labels.
DO_CLASS_LABELS = True

# Use vLLM for inference (requires vllm installed and enough GPU memory).
USE_VLLM = False

# Default dataset_room_type when --match-room-type is NOT used.
# "all" uses the full mixed dataset; "bedroom", "livingroom", etc. restricts
# the training stats and few-shot examples to that room type only.
DEFAULT_DATASET_ROOM_TYPE = "all"

# ============================================================================ #


def render_topdown_bboxes(scene, output_path):
    """Render a 2D top-down view of the floor plan with furniture bounding boxes."""
    objects = scene.get("objects", [])

    fig, (ax, ax_legend) = plt.subplots(1, 2, figsize=(16, 10),
                                         gridspec_kw={"width_ratios": [3, 1]})

    floor_verts = [(v[0], v[2]) for v in scene["bounds_bottom"]]
    floor_poly = MplPolygon(floor_verts, closed=True, fill=True,
                            facecolor="#f5deb3", edgecolor="black", linewidth=2)
    ax.add_patch(floor_poly)

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

        # Place a number label on the bbox instead of the full description
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

    # Build legend table on the right panel
    ax_legend.axis("off")
    ax_legend.set_title("Legend", fontsize=12, fontweight="bold")

    if legend_entries:
        col_labels = ["#", "Color", "Description"]
        cell_text = [[str(idx), "", desc] for idx, _, desc in legend_entries]
        cell_colors = [["white", (*c, 0.4), "white"] for _, c, _ in legend_entries]

        table = ax_legend.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellColours=cell_colors,
            colColours=["#dddddd"] * 3,
            loc="upper center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)

    out_file = output_path / "floorplan_bboxes.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {out_file}")


def process_room(respace, room_json_path, output_dir):
    """Run inference on a single room JSON and save results."""
    with open(room_json_path) as f:
        scene = json.load(f)

    room_type = scene.get("room_type", "room")
    print(f"  Room type: {room_type}")
    print(f"  Boundary vertices: {len(scene['bounds_bottom'])}")

    output_dir.mkdir(parents=True, exist_ok=True)

    result_scene, is_success = respace.generate_full_scene(
        room_type=room_type,
        scene_bounds_only=scene,
        pth_viz_output=output_dir,
    )

    if not is_success:
        print(f"  FAILED: generation unsuccessful for {room_json_path.name}")
        return False

    n_objects = len(result_scene.get("objects", []))
    print(f"  Generated {n_objects} objects")
    for i, obj in enumerate(result_scene["objects"]):
        print(f"    [{i}] {obj.get('desc', 'unknown'):40s}  pos={obj['pos']}  size={obj['size']}")

    scene_out = output_dir / "generated_scene.json"
    with open(scene_out, "w") as f:
        json.dump(result_scene, f, indent=2)
    print(f"  Saved scene JSON to {scene_out}")

    render_topdown_bboxes(result_scene, output_dir)
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch generate furniture layouts from room JSONs.")
    parser.add_argument("--unit", required=True,
                        help="Name of the unit subfolder inside input_test/ (e.g. 'unit_1')")
    parser.add_argument("--match-room-type", action="store_true",
                        help="Re-initialize ReSpace per room type so dataset_room_type matches "
                             "each room's type (bedroom, livingroom, etc.). Slower but produces "
                             "room-type-specific few-shot examples and class labels.")
    args = parser.parse_args()

    base_dir = Path("input_test")
    unit_dir = base_dir / args.unit
    if not unit_dir.is_dir():
        print(f"ERROR: unit directory not found: {unit_dir}")
        sys.exit(1)

    room_files = sorted(unit_dir.glob("*.json"))
    if not room_files:
        print(f"ERROR: no .json files found in {unit_dir}")
        sys.exit(1)

    print(f"Found {len(room_files)} room(s) in {unit_dir}:")
    for f in room_files:
        print(f"  - {f.name}")

    if args.match_room_type:
        # Group rooms by room_type, init a separate ReSpace per type
        rooms_by_type = {}
        for rf in room_files:
            with open(rf) as f:
                rt = json.load(f).get("room_type", "all")
            rooms_by_type.setdefault(rt, []).append(rf)

        for room_type, files in rooms_by_type.items():
            print(f"\n{'=' * 60}")
            print(f"Initializing ReSpace for room type: {room_type}")
            print(f"{'=' * 60}")
            respace = ReSpace(
                model_id=MODEL_ID,
                env_file=ENV_FILE,
                dataset_room_type=room_type,
                use_gpu=True,
                n_bon_sgllm=N_BON_SGLLM,
                n_bon_assets=N_BON_ASSETS,
                do_prop_sampling_for_prompt=DO_PROP_SAMPLING,
                do_icl_for_prompt=DO_ICL,
                do_class_labels_for_prompt=DO_CLASS_LABELS,
                k_few_shot_samples=K_FEW_SHOT_SAMPLES,
                use_vllm=USE_VLLM,
            )

            for rf in files:
                stem = rf.stem
                output_dir = unit_dir / stem
                print(f"\n--- Processing: {rf.name} -> {output_dir} ---")
                process_room(respace, rf, output_dir)
    else:
        print(f"\nInitializing ReSpace (dataset_room_type={DEFAULT_DATASET_ROOM_TYPE})...")
        respace = ReSpace(
            model_id=MODEL_ID,
            env_file=ENV_FILE,
            dataset_room_type=DEFAULT_DATASET_ROOM_TYPE,
            use_gpu=True,
            n_bon_sgllm=N_BON_SGLLM,
            n_bon_assets=N_BON_ASSETS,
            do_prop_sampling_for_prompt=DO_PROP_SAMPLING,
            do_icl_for_prompt=DO_ICL,
            do_class_labels_for_prompt=DO_CLASS_LABELS,
            k_few_shot_samples=K_FEW_SHOT_SAMPLES,
            use_vllm=USE_VLLM,
        )

        for rf in room_files:
            stem = rf.stem
            output_dir = unit_dir / stem
            print(f"\n--- Processing: {rf.name} -> {output_dir} ---")
            process_room(respace, rf, output_dir)

    print("\nAll done!")


if __name__ == "__main__":
    main()