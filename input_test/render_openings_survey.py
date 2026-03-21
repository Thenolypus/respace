"""
Render 360-degree videos of individual openings (doors/windows) for survey.

Generates three videos showing what each opening type looks like:
  1. A normal-sized door (red box)
  2. A large-sized door (red box, wider)
  3. A window (blue box)

Each scene includes walls and floor for context, matching the colour scheme
used in the full orchestrator renders.

Usage:
  uv run python -m input_test.render_openings_survey --output-dir input_test/survey_openings
"""

import os
import copy
import math
import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh
import pyrender
from tqdm import tqdm

from src.viz import (
    create_floor_slab,
    create_wall_meshes,
    create_opening_meshes,
    create_pyrender_scene_from_trimesh,
    render_frame_at_angle,
)

# ── Scene definitions ──────────────────────────────────────────────────────── #
# A small rectangular room (4m x 4m, 2.6m tall) used as backdrop for each
# opening.  Each scene contains walls + floor + one opening only.

ROOM_HEIGHT = 2.6
WALL_HEIGHT = ROOM_HEIGHT * 0.25  # same scale as unit 360 renders

BOUNDS_BOTTOM = [
    [ 2.0, 0.0, -2.0],
    [ 2.0, 0.0,  2.0],
    [-2.0, 0.0,  2.0],
    [-2.0, 0.0, -2.0],
]

SCENES = {
    "normal_door": {
        "label": "Normal Door",
        "openings": [
            {
                "type": "door",
                "pos": [0.0, 1.05, -2.1],      # centered on back wall
                "size": [0.9, 2.1, 0.1],        # ~90cm wide, standard height
            },
        ],
    },
    "large_door": {
        "label": "Large Door",
        "openings": [
            {
                "type": "door",
                "pos": [0.0, 1.05, -2.1],
                "size": [2.0, 2.1, 0.1],        # ~200cm wide
            },
        ],
    },
    "window": {
        "label": "Window",
        "openings": [
            {
                "type": "window",
                "pos": [0.0, 1.6, -2.1],        # higher on wall
                "size": [1.5, 1.4, 0.46],        # 150cm wide, 140cm tall
            },
        ],
    },
}


# ── Rendering ──────────────────────────────────────────────────────────────── #

def build_scene(openings):
    """Build a trimesh scene with floor, walls, and the given openings."""
    ts = trimesh.Scene()

    # Floor
    floor = create_floor_slab(BOUNDS_BOTTOM)
    ts.add_geometry(floor)

    # Walls
    for wall in create_wall_meshes(BOUNDS_BOTTOM, WALL_HEIGHT):
        ts.add_geometry(wall)

    # Opening meshes
    for mesh in create_opening_meshes(openings):
        ts.add_geometry(mesh)

    return ts


def render_360_video(trimesh_scene, video_path, resolution=(1024, 1024),
                     fps=30, video_duration=8.0):
    """Render a 360-degree rotating video of the given trimesh scene."""
    bounds = np.array(BOUNDS_BOTTOM)
    x_span = bounds[:, 0].max() - bounds[:, 0].min()
    z_span = bounds[:, 2].max() - bounds[:, 2].min()
    scene_span = (x_span, ROOM_HEIGHT, z_span)
    camera_height = max(x_span, z_span, 6.0)
    bg_color = np.array([240, 240, 240]) / 255.0

    total_frames = int(fps * video_duration)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        resolution,
    )

    for frame_idx in tqdm(range(total_frames), desc=f"  Rendering"):
        angle = (frame_idx / total_frames) * 360
        frame = render_frame_at_angle(
            trimesh_scene, angle, resolution, camera_height, scene_span, bg_color
        )
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()


# ── Main ───────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description="Render 360 videos of individual openings for the survey."
    )
    parser.add_argument(
        "--output-dir", type=str, default="input_test/survey_openings",
        help="Directory to write the videos (default: input_test/survey_openings)",
    )
    parser.add_argument(
        "--resolution", type=int, default=1024,
        help="Render resolution (default: 1024)",
    )
    parser.add_argument(
        "--duration", type=float, default=8.0,
        help="Video duration in seconds (default: 8.0)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, cfg in SCENES.items():
        print(f"\n--- {cfg['label']} ---")
        ts = build_scene(cfg["openings"])
        video_path = output_dir / f"{name}_360.mp4"
        render_360_video(
            ts, video_path,
            resolution=(args.resolution, args.resolution),
            video_duration=args.duration,
        )
        print(f"  Saved: {video_path}")

    print(f"\nDone! All videos in: {output_dir}")


if __name__ == "__main__":
    main()
