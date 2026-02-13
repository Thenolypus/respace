"""
Test script: Generate furniture layout from a custom floorplan and visualize as bounding boxes.
Usage: ATTN_IMPLEMENTATION=sdpa uv run python -m input_test.test_custom_floorplan
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

from src.respace import ReSpace


def render_topdown_bboxes(scene, output_path):
    """Render a 2D top-down view of the floor plan with furniture bounding boxes."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Draw floor boundary (use x,z from bounds_bottom)
    floor_verts = [(v[0], v[2]) for v in scene["bounds_bottom"]]
    floor_poly = MplPolygon(floor_verts, closed=True, fill=True,
                            facecolor="#f5deb3", edgecolor="black", linewidth=2, label="Floor")
    ax.add_patch(floor_poly)

    # Color palette for objects
    colors = plt.cm.tab10.colors

    for i, obj in enumerate(scene.get("objects", [])):
        pos = obj["pos"]      # [x, y, z] - y is up
        size = obj["size"]    # [width, height, depth]
        rot = obj.get("rot", [0, 0, 0, 1])  # quaternion [x, y, z, w]

        # Top-down: use x,z plane. Size: width=size[0], depth=size[2]
        w, d = size[0], size[2]
        cx, cz = pos[0], pos[2]

        # Compute yaw from quaternion (rotation around y-axis)
        # yaw = atan2(2*(w*y + x*z), 1 - 2*(y^2 + z^2)) but for y-axis rotation:
        qx, qy, qz, qw = rot
        yaw = np.arctan2(2 * (qw * qy + qx * qz), 1 - 2 * (qy**2 + qz**2))
        angle_deg = np.degrees(yaw)

        # Create rotated rectangle corners
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

        # Label
        desc = obj.get("desc", f"obj_{i}")
        ax.text(cx, cz, desc, fontsize=7, ha="center", va="center",
                color="black", fontweight="bold")

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Z (meters)")
    ax.set_title(f"Top-down layout: {scene.get('room_type', 'room')} ({len(scene.get('objects', []))} objects)")
    ax.grid(True, alpha=0.3)

    out_file = output_path / "floorplan_bboxes.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {out_file}")


def main():
    input_path = Path("input_test/unit_1_room_1_livingroom.json")
    output_path = Path("input_test/3og_example/output_livingroom")
    output_path.mkdir(parents=True, exist_ok=True)

    # Load custom floorplan
    with open(input_path) as f:
        scene = json.load(f)

    print(f"Loaded floorplan: {input_path}")
    print(f"Room type: {scene['room_type']}")
    print(f"Boundary vertices: {len(scene['bounds_bottom'])}")

    # Initialize ReSpace pipeline
    print("Initializing ReSpace pipeline...")
    respace = ReSpace(
        model_id="gradient-spaces/respace-sg-llm-1.5b",
        env_file=".env",
        dataset_room_type="all",
        use_gpu=True,
        n_bon_sgllm=8,
        n_bon_assets=4,
    )

    # Generate full scene from custom boundaries
    print("Generating furniture layout...")
    result_scene, is_success = respace.generate_full_scene(
        room_type=scene["room_type"],
        scene_bounds_only=scene,
        pth_viz_output=output_path,
    )

    if not is_success:
        print("FAILED: Scene generation was not successful.")
        sys.exit(1)

    n_objects = len(result_scene.get("objects", []))
    print(f"Generated {n_objects} objects")

    # Print placed objects
    for i, obj in enumerate(result_scene["objects"]):
        print(f"  [{i}] {obj.get('desc', 'unknown'):30s}  pos={obj['pos']}  size={obj['size']}")

    # Save generated scene JSON
    scene_out_path = output_path / "generated_scene.json"
    with open(scene_out_path, "w") as f:
        json.dump(result_scene, f, indent=2)
    print(f"Saved scene JSON to {scene_out_path}")

    # Render 2D top-down bounding box visualization (no OpenGL/EGL needed)
    print("Rendering bounding box visualization...")
    render_topdown_bboxes(result_scene, output_path)

    print("Done!")


if __name__ == "__main__":
    main()
