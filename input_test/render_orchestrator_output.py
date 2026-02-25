"""
Render 3D scenes from orchestrator outputs.

Takes a directory of orchestrator outputs (from orchestrate_unit.py) and
renders each room's cross_scene_retrieval.json using the 3D asset renderer.
Also generates a 360-degree rotating video of the entire unit.

Usage:
  uv run python -m input_test.render_orchestrator_output \
      --output-dir input_test/01OG_2ROOMS/unit_2/output

  uv run python -m input_test.render_orchestrator_output \
      --output-dir input_test/01OG_2ROOMS/unit_2/output \
      --resolution 1024

  uv run python -m input_test.render_orchestrator_output \
      --output-dir input_test/01OG_2ROOMS/unit_2/output \
      --skip-rooms
"""

import json
import re
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

from src.viz import render_scene_and_export, create_360_video_unit


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


def render_unit_360(output_dir, room_scenes, resolution):
    """Render a 360-degree video of the entire unit using all room scenes."""
    # Derive paths: output_dir is like .../03OG_PLAN/unit_1/output
    unit_dir = output_dir.parent          # .../03OG_PLAN/unit_1
    floorplan_dir = unit_dir.parent       # .../03OG_PLAN
    metadata_path = floorplan_dir / "metadata.json"

    if not metadata_path.exists():
        print(f"WARNING: metadata.json not found at {metadata_path}, skipping unit 360 video")
        return

    # Extract unit_id from directory name (e.g. "unit_2" -> 2)
    match = re.search(r"unit_(\d+)", unit_dir.name)
    if not match:
        print(f"WARNING: could not parse unit_id from {unit_dir.name}, skipping unit 360 video")
        return
    unit_id = int(match.group(1))

    render_path = output_dir / "unit_360"
    render_path.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Unit {unit_id} 360 video ---")
    video_path = create_360_video_unit(
        metadata_path=str(metadata_path),
        floorplan_dir=str(floorplan_dir),
        unit_id=unit_id,
        pth_output=render_path,
        resolution=(resolution, resolution),
        room_scenes=room_scenes,
    )
    print(f"  Saved: {video_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render 3D scenes from orchestrator output directory."
    )
    parser.add_argument("--output-dir", required=True,
                        help="Orchestrator output directory (e.g. input_test/01OG_2ROOMS/unit_2/output)")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Render resolution (default: 1024)")
    parser.add_argument("--skip-rooms", action="store_true",
                        help="Skip per-room renders, only generate unit 360 video")
    parser.add_argument("--skip-unit-360", action="store_true",
                        help="Skip unit 360 video generation")
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

    # Collect all room scenes (needed for both per-room renders and unit 360)
    room_scenes = {}
    for retrieval_path in retrieval_files:
        room_name = retrieval_path.parent.name

        with open(retrieval_path) as f:
            data = json.load(f)

        scene = data["cross_scene_style"]

        if not scene.get("bounds_bottom"):
            print(f"WARNING: Skipping {room_name} -- no bounds_bottom")
            continue

        room_scenes[room_name] = scene

        if not args.skip_rooms:
            print(f"\n--- {room_name} ---")
            room_output = retrieval_path.parent
            render_path = room_output / "render"
            render_path.mkdir(parents=True, exist_ok=True)
            render_room(scene, render_path, room_name, args.resolution)

    # Generate unit 360 video
    if not args.skip_unit_360 and room_scenes:
        render_unit_360(output_dir, room_scenes, args.resolution)

    print("\nDone!")


if __name__ == "__main__":
    main()
