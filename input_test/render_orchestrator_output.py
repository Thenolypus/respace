"""
Render 3D scenes from orchestrator outputs.

Takes a directory of orchestrator outputs (from orchestrate_unit.py) and
renders each room's cross_scene_retrieval.json using the 3D asset renderer.

Usage:
  uv run python -m input_test.render_orchestrator_output \
      --output-dir input_test/01OG_2ROOMS/unit_2/output

  uv run python -m input_test.render_orchestrator_output \
      --output-dir input_test/01OG_2ROOMS/unit_2/output \
      --resolution 1024
"""

import json
import argparse
from pathlib import Path

from src.viz import render_scene_and_export


def render_room(scene, render_path, filename, resolution):
    """Render a single scene with retrieved 3D assets."""
    print(f"  Rendering: {filename} ...")
    render_scene_and_export(
        scene,
        filename=filename,
        pth_output=render_path,
        resolution=(resolution, resolution),
        show_bboxes=False,
        show_assets=True,
        use_dynamic_zoom=True,
    )
    print(f"  Saved: {render_path}/top/ and {render_path}/diag/")


def main():
    parser = argparse.ArgumentParser(
        description="Render 3D scenes from orchestrator output directory."
    )
    parser.add_argument("--output-dir", required=True,
                        help="Orchestrator output directory (e.g. input_test/01OG_2ROOMS/unit_2/output)")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Render resolution (default: 1024)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"ERROR: directory not found: {output_dir}")
        return

    # Find all cross_scene_retrieval.json files
    retrieval_files = sorted(output_dir.glob("*/cross_scene_retrieval.json"))
    if not retrieval_files:
        print(f"ERROR: no cross_scene_retrieval.json found in {output_dir}/*/")
        return

    print(f"Found {len(retrieval_files)} room(s) to render:")
    for f in retrieval_files:
        print(f"  - {f}")

    for retrieval_path in retrieval_files:
        room_name = retrieval_path.parent.name
        room_output = retrieval_path.parent

        with open(retrieval_path) as f:
            data = json.load(f)

        scene = data["cross_scene_style"]

        if not scene.get("bounds_bottom"):
            print(f"WARNING: Skipping {room_name} -- no bounds_bottom")
            continue

        print(f"\n--- {room_name} ---")
        render_path = room_output / "render"
        render_path.mkdir(parents=True, exist_ok=True)
        render_room(scene, render_path, room_name, args.resolution)

    print("\nDone!")


if __name__ == "__main__":
    main()
