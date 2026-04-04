"""
Visualization test for opening metrics (DAO / WBO).

Renders top-down views showing:
  - Room floor plan (beige)
  - Furniture bounding boxes (colored, with index labels)
  - Door arc sweep zones (red, semi-transparent)
  - Window clearance zones (blue, semi-transparent)
  - Intersection regions highlighted (magenta for door, cyan for window)
  - Per-object overlap values annotated

Outputs one PNG per scene into eval/viz_openings/.

Usage:
  uv run python -m input_test.viz_openings_test
  uv run python -m input_test.viz_openings_test --scene eval/eval_set/scand_456/1_7b/ComplexHouse/unit_1/unit_1_room_4_livingroom/generated_scene.json
  uv run python -m input_test.viz_openings_test --all
"""

import argparse
import json
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, FancyArrowPatch
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon

from src.eval import get_xz_bbox_from_obj
from input_test.eval_openings import (
    make_door_arc_polygon,
    make_window_clearance_polygon,
    compute_opening_overlap,
    find_nearest_wall_segment,
    get_door_width,
)

#DEFAULT_TEST_SCENES = [
#    # One per floor plan type from scand_456/1_7b
#    "eval/eval_set/scand_456/1_7b/ComplexHouse/unit_1/unit_1_room_4_livingroom/generated_scene.json",
#    "eval/eval_set/scand_456/1_7b/ComplexHouse/unit_1/unit_1_room_1_bedroom/generated_scene.json",
#    "eval/eval_set/scand_456/1_7b/ComplexApt/unit_1/unit_1_room_1_livingroom/generated_scene.json",
#    "eval/eval_set/scand_456/1_7b/ComplexApt/unit_1/unit_1_room_4_bedroom/generated_scene.json",
#    "eval/eval_set/scand_456/1_7b/SimpleApartment/unit_1/unit_1_room_1_bedroom/generated_scene.json",
#    "eval/eval_set/scand_456/1_7b/SimpleApartment/unit_1/unit_1_room_2_livingroom/generated_scene.json",
#]

DEFAULT_TEST_SCENES = [
    # One per floor plan type from scand_456/1_7b
    "eval/eval_set/earthy_123/1_7b/SimpleApartment/unit_1/unit_1_room_2_livingroom/generated_scene.json",
    "eval/eval_set/earthy_123/4b/SimpleApartment/unit_1/unit_1_room_2_livingroom/generated_scene.json",
    "eval/eval_set/earthy_123/original/SimpleApartment/unit_1/unit_1_room_2_livingroom/generated_scene.json",
]

OUTPUT_DIR = Path("eval/viz_openings")


def shapely_poly_to_mpl_verts(poly):
    """Convert a Shapely Polygon exterior to matplotlib-compatible vertices."""
    if poly.is_empty:
        return []
    x, y = poly.exterior.xy
    return list(zip(x, y))


def get_obj_xz_verts(obj):
    """Get the rotated XZ bounding box corners for an object."""
    bbox_poly, _, _, _ = get_xz_bbox_from_obj(obj)
    x, y = bbox_poly.exterior.xy
    return list(zip(x, y))


def render_scene_openings(scene, output_path, title=""):
    """Render a single scene with opening zones and intersection highlights."""
    objects = scene.get("objects", [])
    openings = scene.get("openings", [])
    bounds_bottom = scene.get("bounds_bottom", [])

    if not bounds_bottom:
        print(f"  SKIP: no bounds_bottom")
        return

    doors = [o for o in openings if o["type"] == "door"]
    windows = [o for o in openings if o["type"] == "window"]

    # --- Build all zone polygons ---
    door_zones = []
    for door in doors:
        try:
            arc_poly, y_start, y_end = make_door_arc_polygon(door, bounds_bottom)
            door_zones.append((door, arc_poly, y_start, y_end))
        except Exception as e:
            print(f"  Warning: door arc failed: {e}")

    window_zones = []
    for window in windows:
        try:
            win_poly, y_start, y_end = make_window_clearance_polygon(window, bounds_bottom)
            window_zones.append((window, win_poly, y_start, y_end))
        except Exception as e:
            print(f"  Warning: window zone failed: {e}")

    # --- Compute per-object overlaps ---
    obj_door_overlaps = []  # per object: list of (door_idx, overlap_vol)
    obj_window_overlaps = []
    for obj in objects:
        d_overlaps = []
        for di, (door, arc_poly, y_start, y_end) in enumerate(door_zones):
            vol = compute_opening_overlap(obj, arc_poly, y_start, y_end)
            if vol > 0:
                d_overlaps.append((di, vol))
        obj_door_overlaps.append(d_overlaps)

        w_overlaps = []
        for wi, (window, win_poly, y_start, y_end) in enumerate(window_zones):
            vol = compute_opening_overlap(obj, win_poly, y_start, y_end)
            if vol > 0:
                w_overlaps.append((wi, vol))
        obj_window_overlaps.append(w_overlaps)

    # --- Plot ---
    fig, (ax, ax_info) = plt.subplots(1, 2, figsize=(20, 12),
                                       gridspec_kw={"width_ratios": [3, 1]})

    # Floor plan
    floor_verts = [(v[0], v[2]) for v in bounds_bottom]
    floor_poly = MplPolygon(floor_verts, closed=True, fill=True,
                            facecolor="#f5deb3", edgecolor="black", linewidth=2, zorder=1)
    ax.add_patch(floor_poly)

    # Door arc zones
    for di, (door, arc_poly, y_start, y_end) in enumerate(door_zones):
        verts = shapely_poly_to_mpl_verts(arc_poly)
        if verts:
            patch = MplPolygon(verts, closed=True, fill=True,
                               facecolor=(1, 0, 0, 0.15), edgecolor="red",
                               linewidth=1.5, linestyle="--", zorder=3)
            ax.add_patch(patch)

        # Mark hinge point
        _, _, wall_dir_unit, inward_normal = find_nearest_wall_segment(door, bounds_bottom)
        door_width = get_door_width(door)
        hinge = np.array([door["pos"][0], door["pos"][2]]) + wall_dir_unit * (door_width / 2.0)
        ax.plot(hinge[0], hinge[1], "o", color="darkred", markersize=8, zorder=6)

        # Arrow showing inward normal
        ax.annotate("", xy=(hinge[0] + inward_normal[0] * 0.3, hinge[1] + inward_normal[1] * 0.3),
                    xytext=(hinge[0], hinge[1]),
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=2), zorder=6)

        # Label
        ox, oz = door["pos"][0], door["pos"][2]
        ax.annotate(f"D{di}\n{door_width:.2f}m",
                    (ox, oz), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points", color="red",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="red", alpha=0.9), zorder=7)

    # Window clearance zones
    for wi, (window, win_poly, y_start, y_end) in enumerate(window_zones):
        verts = shapely_poly_to_mpl_verts(win_poly)
        if verts:
            patch = MplPolygon(verts, closed=True, fill=True,
                               facecolor=(0, 0, 1, 0.15), edgecolor="blue",
                               linewidth=1.5, linestyle="--", zorder=3)
            ax.add_patch(patch)

        ox, oz = window["pos"][0], window["pos"][2]
        win_width = max(window["size"][0], window["size"][2])
        ax.annotate(f"W{wi}\n{win_width:.2f}m",
                    (ox, oz), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points", color="blue",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="blue", alpha=0.9), zorder=7)

    # Objects
    tab_colors = plt.cm.tab10.colors
    for i, obj in enumerate(objects):
        color = tab_colors[i % len(tab_colors)]
        verts = get_obj_xz_verts(obj)

        has_door_overlap = len(obj_door_overlaps[i]) > 0
        has_window_overlap = len(obj_window_overlaps[i]) > 0

        # Draw object bbox
        edge_color = color
        edge_width = 1.5
        if has_door_overlap or has_window_overlap:
            edge_color = "magenta" if has_door_overlap else "cyan"
            edge_width = 3.0

        bbox_patch = MplPolygon(verts, closed=True, fill=True,
                                facecolor=(*color, 0.4), edgecolor=edge_color,
                                linewidth=edge_width, zorder=4)
        ax.add_patch(bbox_patch)

        # Draw intersection regions
        bbox_shapely, _, _, _ = get_xz_bbox_from_obj(obj)
        for di, vol in obj_door_overlaps[i]:
            _, arc_poly, _, _ = door_zones[di]
            isect = bbox_shapely.intersection(arc_poly)
            if not isect.is_empty:
                try:
                    isect_verts = shapely_poly_to_mpl_verts(isect)
                    if isect_verts:
                        isect_patch = MplPolygon(isect_verts, closed=True, fill=True,
                                                 facecolor=(1, 0, 1, 0.5), edgecolor="magenta",
                                                 linewidth=2, zorder=5)
                        ax.add_patch(isect_patch)
                except Exception:
                    pass

        for wi, vol in obj_window_overlaps[i]:
            _, win_poly, _, _ = window_zones[wi]
            isect = bbox_shapely.intersection(win_poly)
            if not isect.is_empty:
                try:
                    isect_verts = shapely_poly_to_mpl_verts(isect)
                    if isect_verts:
                        isect_patch = MplPolygon(isect_verts, closed=True, fill=True,
                                                 facecolor=(0, 1, 1, 0.5), edgecolor="cyan",
                                                 linewidth=2, zorder=5)
                        ax.add_patch(isect_patch)
                except Exception:
                    pass

        # Object index label
        cx, cz = obj["pos"][0], obj["pos"][2]
        ax.text(cx, cz, str(i), fontsize=9, ha="center", va="center",
                color="black", fontweight="bold", zorder=8)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Z (meters)")
    room_type = scene.get("room_type", "room")
    ax.set_title(f"{title}\n{room_type} | {len(objects)} objects | {len(doors)} doors | {len(windows)} windows")
    ax.grid(True, alpha=0.3)

    # --- Info panel ---
    ax_info.axis("off")

    lines = []
    lines.append("LEGEND")
    lines.append("-" * 40)
    lines.append("Red dashed = Door arc sweep zone")
    lines.append("Blue dashed = Window clearance zone")
    lines.append("Magenta fill = Door intersection")
    lines.append("Cyan fill = Window intersection")
    lines.append("Red dot = Door hinge point")
    lines.append("Arrow = Inward normal direction")
    lines.append("")
    lines.append("OBJECTS")
    lines.append("-" * 40)

    total_dao = 0.0
    total_wbo = 0.0

    for i, obj in enumerate(objects):
        desc = obj.get("desc", f"obj_{i}")
        if len(desc) > 35:
            desc = desc[:32] + "..."

        d_overlaps = obj_door_overlaps[i]
        w_overlaps = obj_window_overlaps[i]

        flag = ""
        if d_overlaps:
            for di, vol in d_overlaps:
                flag += f" [DAO D{di}: {vol:.5f}m3]"
                total_dao += vol
        if w_overlaps:
            for wi, vol in w_overlaps:
                flag += f" [WBO W{wi}: {vol:.5f}m3]"
                total_wbo += vol

        if flag:
            lines.append(f"[{i}] {desc}")
            lines.append(f"    {flag.strip()}")
        else:
            lines.append(f"[{i}] {desc}")

    lines.append("")
    lines.append("=" * 40)
    lines.append(f"Total DAO: {total_dao:.5f} m3 ({total_dao*1e3:.2f} x1e-3)")
    lines.append(f"Total WBO: {total_wbo:.5f} m3 ({total_wbo*1e3:.2f} x1e-3)")
    lines.append(f"Combined:  {(total_dao+total_wbo):.5f} m3")

    # Door details
    lines.append("")
    lines.append("DOORS")
    lines.append("-" * 40)
    for di, (door, arc_poly, y_start, y_end) in enumerate(door_zones):
        w = get_door_width(door)
        lines.append(f"D{di}: width={w:.2f}m  arc_area={arc_poly.area:.3f}m2  y=[{y_start:.2f}, {y_end:.2f}]")

    if window_zones:
        lines.append("")
        lines.append("WINDOWS")
        lines.append("-" * 40)
        for wi, (window, win_poly, y_start, y_end) in enumerate(window_zones):
            w = max(window["size"][0], window["size"][2])
            lines.append(f"W{wi}: width={w:.2f}m  zone_area={win_poly.area:.3f}m2  y=[{y_start:.2f}, {y_end:.2f}]")

    text = "\n".join(lines)
    ax_info.text(0.02, 0.98, text, transform=ax_info.transAxes,
                 fontsize=7, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize opening overlap metrics")
    parser.add_argument("--scene", type=str, default=None,
                        help="Path to a single generated_scene.json to visualize")
    parser.add_argument("--all", action="store_true",
                        help="Visualize all default test scenes")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.scene:
        scenes = [Path(args.scene)]
    elif args.all:
        scenes = [Path(s) for s in DEFAULT_TEST_SCENES]
    else:
        scenes = [Path(s) for s in DEFAULT_TEST_SCENES]

    print(f"Visualizing {len(scenes)} scene(s) -> {output_dir}/")

    for scene_path in scenes:
        if not scene_path.exists():
            print(f"  SKIP (not found): {scene_path}")
            continue

        with open(scene_path) as f:
            scene = json.load(f)

        # Build a readable title from path
        parts = scene_path.parts
        # Try to extract floor/room info
        title = "/".join(parts[-5:-1]) if len(parts) >= 5 else scene_path.stem

        out_name = title.replace("/", "__").replace("\\", "__") + ".png"
        out_path = output_dir / out_name

        print(f"\n  {title}")
        render_scene_openings(scene, out_path, title=title)


if __name__ == "__main__":
    main()
