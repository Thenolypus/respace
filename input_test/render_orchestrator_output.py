"""
Render 3D scenes from orchestrator outputs.

Takes a directory of orchestrator outputs (from orchestrate_unit.py) and
renders each room's cross_scene_retrieval.json using the 3D asset renderer.
Also generates a 360-degree rotating video of the entire unit.

Single-unit mode:
  uv run python -m input_test.render_orchestrator_output \
      --output-dir input_test/01OG_2ROOMS/unit_2/output

  uv run python -m input_test.render_orchestrator_output \
      --output-dir input_test/01OG_2ROOMS/unit_2/output \
      --resolution 1024

  uv run python -m input_test.render_orchestrator_output \
      --output-dir input_test/01OG_2ROOMS/unit_2/output \
      --skip-rooms

Batch mode (render all units under a parent directory):
  uv run python -m input_test.render_orchestrator_output \
      --batch-dir input_test/3d_front/full-fill

  uv run python -m input_test.render_orchestrator_output \
      --batch-dir input_test/3d_front/full-fill \
      --resolution 1024 --skip-rooms

Eval mode (retrieval JSONs live directly under unit dirs, no 'output' subfolder):
  uv run python -m input_test.render_orchestrator_output \
      --batch-dir input_test/3d_front/1_7b --eval
"""

import os
import json
import re
import shutil
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

from src.viz import render_scene_and_export, create_360_video_unit

ENV_FILE = ".env"
ENV_FILE_BATHROOM = ".env_heg"

ASSET_ENV_KEYS = [
    "PTH_ASSETS_METADATA", "PTH_ASSETS_METADATA_SCALED",
    "PTH_ASSETS_METADATA_SIMPLE_DESCS", "PTH_ASSETS_METADATA_PROMPTS",
    "PTH_ASSETS_EMBED", "PTH_ASSETS_EMBED_STYLE",
    "PTH_3DFUTURE_ASSETS",
]


def _load_env(env_file):
    """Clear asset-related env vars and reload from the given .env file."""
    for k in ASSET_ENV_KEYS:
        os.environ.pop(k, None)
    load_dotenv(env_file, override=True)


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


def render_unit_360(output_dir, room_scenes, resolution, video_duration=12.0, eval_mode=False, zoom=1.0):
    """Render a 360-degree video of the entire unit using all room scenes."""
    # Derive paths: in normal mode output_dir is .../floor/unit_1/output,
    # in eval mode output_dir is .../floor/unit_1 (no 'output' subfolder).
    if eval_mode:
        unit_dir = output_dir                 # .../floor/unit_1
        floorplan_dir = unit_dir.parent       # .../floor
    else:
        unit_dir = output_dir.parent          # .../floor/unit_1
        floorplan_dir = unit_dir.parent       # .../floor
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

    def _switch_env_for_room(room):
        env_file = ENV_FILE_BATHROOM if room.get("room_type") == "bathroom" else ENV_FILE
        _load_env(env_file)

    print(f"\n--- Unit {unit_id} 360 video ---")
    video_path = create_360_video_unit(
        metadata_path=str(metadata_path),
        floorplan_dir=str(floorplan_dir),
        unit_id=unit_id,
        pth_output=render_path,
        resolution=(resolution, resolution),
        room_scenes=room_scenes,
        pre_room_hook=_switch_env_for_room,
        video_duration=video_duration,
        zoom=zoom,
    )
    print(f"  Saved: {video_path}")
    return video_path


def discover_unit_outputs_from_batch_dir(batch_dir, eval_mode=False):
    """Discover all unit output directories under a parent directory.

    Scans for <floor>/<unit_N>/output/ directories that contain at least one
    cross_scene_retrieval.json file.  When eval_mode is True, looks directly
    under <floor>/<unit_N>/ instead of requiring an 'output' subfolder.
    Returns list of (output_dir, label) tuples.
    """
    batch_dir = Path(batch_dir)
    units = []
    for floor_dir in sorted(batch_dir.iterdir()):
        if not floor_dir.is_dir():
            continue
        for unit_dir in sorted(floor_dir.iterdir()):
            if not unit_dir.is_dir() or not unit_dir.name.startswith("unit_"):
                continue
            if eval_mode:
                search_dir = unit_dir
            else:
                search_dir = unit_dir / "output"
                if not search_dir.is_dir():
                    continue
            retrieval_files = list(search_dir.glob("*/cross_scene_retrieval.json"))
            if not retrieval_files:
                retrieval_files = list(search_dir.glob("*/original_retrieval.json"))
            if retrieval_files:
                label = f"{floor_dir.name}/{unit_dir.name}"
                units.append((search_dir, label))
    return units


def render_single_unit(output_dir, resolution, skip_rooms, skip_unit_360, video_duration=12.0, eval_mode=False, zoom=1.0):
    """Render all rooms in a single unit output directory.

    Returns (n_rooms, video_path) where video_path may be None.
    """
    # Find all retrieval JSON files (cross_scene or original)
    retrieval_files = sorted(output_dir.glob("*/cross_scene_retrieval.json"))
    if not retrieval_files:
        retrieval_files = sorted(output_dir.glob("*/original_retrieval.json"))
    if not retrieval_files:
        print(f"  No cross_scene_retrieval.json or original_retrieval.json found in {output_dir}/*/")
        return 0, None

    print(f"  Found {len(retrieval_files)} room(s) to render:")
    for f in retrieval_files:
        print(f"    - {f}")

    # Collect all room scenes (needed for both per-room renders and unit 360)
    room_scenes = {}
    for retrieval_path in retrieval_files:
        room_name = retrieval_path.parent.name

        with open(retrieval_path) as f:
            data = json.load(f)

        if "cross_scene_style" in data:
            scene = data["cross_scene_style"]
        elif "original_sample" in data:
            scene = data["original_sample"]
        else:
            print(f"  WARNING: Skipping {room_name} -- no recognized scene key")
            continue

        if not scene.get("bounds_bottom"):
            print(f"  WARNING: Skipping {room_name} -- no bounds_bottom")
            continue

        room_scenes[room_name] = scene

        if not skip_rooms:
            print(f"\n  --- {room_name} ---")
            room_type = scene.get("room_type", "")
            env_file = ENV_FILE_BATHROOM if room_type == "bathroom" else ENV_FILE
            _load_env(env_file)
            room_output = retrieval_path.parent
            render_path = room_output / "render"
            render_path.mkdir(parents=True, exist_ok=True)
            render_room(scene, render_path, room_name, resolution)

    # Generate unit 360 video
    video_path = None
    if not skip_unit_360 and room_scenes:
        video_path = render_unit_360(output_dir, room_scenes, resolution, video_duration, eval_mode=eval_mode, zoom=zoom)

    return len(room_scenes), video_path


def main():
    parser = argparse.ArgumentParser(
        description="Render 3D scenes from orchestrator output directory."
    )

    # Input source (pick one)
    input_group = parser.add_argument_group("input source (pick one)")
    input_group.add_argument("--output-dir", type=str, default=None,
                        help="Orchestrator output directory (e.g. input_test/01OG_2ROOMS/unit_2/output)")
    input_group.add_argument("--batch-dir", type=str, default=None,
                        help="Parent directory containing multiple floor folders. "
                             "Renders all units found under <batch-dir>/<floor>/unit_*/output/.")

    parser.add_argument("--resolution", type=int, default=1024,
                        help="Render resolution (default: 1024)")
    parser.add_argument("--skip-rooms", action="store_true",
                        help="Skip per-room renders, only generate unit 360 video")
    parser.add_argument("--skip-unit-360", action="store_true",
                        help="Skip unit 360 video generation")
    parser.add_argument("--video-duration", type=float, default=12.0,
                        help="Duration of 360 video in seconds (default: 12.0, slower rotation)")
    parser.add_argument("--zoom", type=float, default=1.0,
                        help="Zoom factor for 360 video camera (default: 1.0, higher = closer)")
    parser.add_argument("--eval", action="store_true",
                        help="Eval mode: look for cross_scene_retrieval.json directly under "
                             "unit directories instead of inside an 'output' subfolder.")
    args = parser.parse_args()

    # Validate input mode
    modes_given = sum([bool(args.output_dir), bool(args.batch_dir)])
    if modes_given != 1:
        print("ERROR: provide exactly one of --output-dir or --batch-dir.")
        return

    # ---------------------------------------------------------------------- #
    # Batch mode                                                               #
    # ---------------------------------------------------------------------- #

    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
        if not batch_dir.is_dir():
            print(f"ERROR: batch directory not found: {batch_dir}")
            return

        units = discover_unit_outputs_from_batch_dir(batch_dir, eval_mode=args.eval)
        if not units:
            print(f"ERROR: no unit outputs found under {batch_dir}")
            return

        print(f"BATCH MODE: {batch_dir}")
        print(f"Found {len(units)} unit(s) to render:")
        for output_dir, label in units:
            print(f"  {label} -> {output_dir}")

        # Central directory for all 360 videos
        renders_dir = batch_dir / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)

        total_rooms = 0
        for output_dir, label in units:
            print(f"\n{'='*70}")
            print(f"UNIT: {label}")
            print(f"{'='*70}")
            n_rooms, video_path = render_single_unit(output_dir, args.resolution,
                                                      args.skip_rooms, args.skip_unit_360,
                                                      args.video_duration, eval_mode=args.eval,
                                                      zoom=args.zoom)
            total_rooms += n_rooms

            # Copy 360 video to central renders/ directory
            # label is e.g. "ComplexApt/unit_1" -> "ComplexApt_unit_1_360.mp4"
            if video_path and Path(video_path).exists():
                dest_name = label.replace("/", "_") + "_360.mp4"
                dest_path = renders_dir / dest_name
                shutil.copy2(video_path, dest_path)
                print(f"  Copied to: {dest_path}")

        print(f"\n{'#'*70}")
        print(f"# BATCH COMPLETE")
        print(f"# Directory: {batch_dir}")
        print(f"# Units rendered: {len(units)}")
        print(f"# Total rooms: {total_rooms}")
        print(f"# Renders: {renders_dir}")
        print(f"{'#'*70}")
        print("\nDone!")
        return

    # ---------------------------------------------------------------------- #
    # Single-unit mode                                                         #
    # ---------------------------------------------------------------------- #

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"ERROR: directory not found: {output_dir}")
        return

    render_single_unit(output_dir, args.resolution, args.skip_rooms, args.skip_unit_360, args.video_duration, eval_mode=args.eval, zoom=args.zoom)
    print("\nDone!")


if __name__ == "__main__":
    main()
