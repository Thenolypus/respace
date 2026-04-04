"""
Opening-awareness evaluation: Door Arc Overlap (DAO) and Window Block (WBO) metrics.

For each generated scene, computes:
  - DAO: volume of furniture overlapping with door swing arcs (quarter-circle on floor)
  - WBO: volume of furniture overlapping with window clearance zones

Both metrics use 2D XZ intersection (like MBL) projected to 3D via height overlap.
The overlap is proportional (partial intersections, not binary).

Usage:
  uv run python -m input_test.eval_openings
  uv run python -m input_test.eval_openings --eval-dir eval/eval_set
  uv run python -m input_test.eval_openings --output-json eval/metrics/openings.json
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon
from shapely.ops import unary_union

from src.eval import get_xz_bbox_from_obj


MODEL_NAMES = ["1_7b", "4b", "original"]

# Door arc resolution: number of points on the quarter-circle
ARC_RESOLUTION = 64


def get_door_width(opening):
    """Get the door width (the wall-spanning dimension)."""
    sx, sz = opening["size"][0], opening["size"][2]
    # The thin axis is the wall-normal direction; the wide axis is the door width
    if sx < sz:
        return sz
    else:
        return sx


def get_wall_normal_axis(opening):
    """Determine which axis is wall-normal (thin) and which is wall-parallel.

    Returns (thin_axis_idx, span_axis_idx) where idx is 0 for X, 2 for Z.
    thin_axis_idx: the axis perpendicular to the wall (small dimension)
    span_axis_idx: the axis along the wall (door width dimension)
    """
    sx, sz = opening["size"][0], opening["size"][2]
    if sx < sz:
        return 0, 2  # thin in X, spans Z
    else:
        return 2, 0  # thin in Z, spans X


def find_nearest_wall_segment(opening, bounds_bottom):
    """Find which wall the opening sits on.

    Returns (wall_start_2d, wall_end_2d, wall_direction_unit, inward_normal_unit)
    where inward_normal points into the room.
    """
    from shapely.geometry import Point, LineString

    ox, oz = opening["pos"][0], opening["pos"][2]
    pts = [(v[0], v[2]) for v in bounds_bottom]

    min_dist = float("inf")
    best_seg_idx = 0
    for i in range(len(pts)):
        seg = LineString([pts[i], pts[(i + 1) % len(pts)]])
        d = seg.distance(Point(ox, oz))
        if d < min_dist:
            min_dist = d
            best_seg_idx = i

    p1 = np.array(pts[best_seg_idx])
    p2 = np.array(pts[(best_seg_idx + 1) % len(pts)])

    wall_dir = p2 - p1
    wall_len = np.linalg.norm(wall_dir)
    if wall_len < 1e-9:
        return p1, p2, np.array([1, 0]), np.array([0, 1])

    wall_dir_unit = wall_dir / wall_len

    # Two candidate normals
    n1 = np.array([-wall_dir_unit[1], wall_dir_unit[0]])
    n2 = np.array([wall_dir_unit[1], -wall_dir_unit[0]])

    # Pick the one pointing inward (toward room centroid)
    centroid = np.mean(pts, axis=0)
    wall_mid = (p1 + p2) / 2
    to_centroid = centroid - wall_mid
    inward_normal = n1 if np.dot(n1, to_centroid) > 0 else n2

    return p1, p2, wall_dir_unit, inward_normal


def make_door_arc_polygon(opening, bounds_bottom):
    """Create a quarter-circle arc polygon representing the door swing area.

    The arc is on the XZ plane (floor). The hinge is at one edge of the door
    on the wall, and the door swings 90 degrees inward into the room.
    Arc radius = door width.

    Returns a Shapely Polygon (2D XZ) and the height range (y_start, y_end).
    """
    ox, oz = opening["pos"][0], opening["pos"][2]
    door_width = get_door_width(opening)
    door_height = opening["size"][1]

    _, _, wall_dir_unit, inward_normal = find_nearest_wall_segment(opening, bounds_bottom)

    radius = door_width
    # Door hinge is at one edge of the opening along the wall
    hinge = np.array([ox, oz]) + wall_dir_unit * (door_width / 2.0)

    # The arc sweeps from the wall direction (closed position) to 90 degrees
    # inward. We compute the start angle and sweep.
    # Closed position: door lies along the wall, pointing from hinge toward the
    # other edge of the opening
    closed_dir = -wall_dir_unit  # from hinge back toward opening center
    open_dir = inward_normal     # fully open, perpendicular into room

    # Generate arc points from closed to open position (90 degree sweep)
    start_angle = np.arctan2(closed_dir[1], closed_dir[0])
    end_angle = np.arctan2(open_dir[1], open_dir[0])

    # Ensure we sweep the shorter arc (should be ~90 degrees)
    angle_diff = end_angle - start_angle
    # Normalize to [-pi, pi]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    # If the sweep is negative (clockwise), that's fine for the polygon

    angles = np.linspace(start_angle, start_angle + angle_diff, ARC_RESOLUTION)

    arc_points = [(hinge[0] + radius * np.cos(a), hinge[1] + radius * np.sin(a)) for a in angles]
    # Close the polygon: hinge -> arc points -> hinge
    polygon_points = [(hinge[0], hinge[1])] + arc_points + [(hinge[0], hinge[1])]

    arc_poly = Polygon(polygon_points)
    if not arc_poly.is_valid:
        arc_poly = arc_poly.buffer(0)

    # Door arc extends from floor level up to door height
    # Door y_start is at floor (pos[1] - height/2 for centered, or 0)
    # Based on the data: pos[1]=1.05, size[1]=2.1 -> door goes from 0 to 2.1
    y_start = opening["pos"][1] - door_height / 2.0
    y_end = opening["pos"][1] + door_height / 2.0

    return arc_poly, y_start, y_end


def make_window_clearance_polygon(opening, bounds_bottom, clearance_depth=0.6):
    """Create a rectangular clearance zone in front of a window.

    The zone extends `clearance_depth` meters inward from the window into the room,
    spanning the full window width.

    Returns a Shapely Polygon (2D XZ) and the height range (y_start, y_end).
    """
    ox, oz = opening["pos"][0], opening["pos"][2]
    window_height = opening["size"][1]

    thin_axis, span_axis = get_wall_normal_axis(opening)
    window_width = opening["size"][span_axis]

    _, _, wall_dir_unit, inward_normal = find_nearest_wall_segment(opening, bounds_bottom)

    center = np.array([ox, oz])
    half_w = window_width / 2.0

    # Rectangle corners: along wall +/- half_width, extend inward by clearance_depth
    c1 = center - wall_dir_unit * half_w
    c2 = center + wall_dir_unit * half_w
    c3 = c2 + inward_normal * clearance_depth
    c4 = c1 + inward_normal * clearance_depth

    poly = Polygon([(c1[0], c1[1]), (c2[0], c2[1]), (c3[0], c3[1]), (c4[0], c4[1])])
    if not poly.is_valid:
        poly = poly.buffer(0)

    # Window height range
    y_start = opening["pos"][1] - window_height / 2.0
    y_end = opening["pos"][1] + window_height / 2.0

    return poly, y_start, y_end


def compute_opening_overlap(obj, zone_poly, zone_y_start, zone_y_end, epsilon=1e-7):
    """Compute the 3D overlap volume between an object's bbox and an opening zone.

    Same approach as compute_bbl: 2D XZ intersection area * Y-axis height overlap.
    Returns overlap volume in cubic meters.
    """
    bbox_obj, obj_height, obj_y_start, obj_y_end = get_xz_bbox_from_obj(obj)

    intersection = bbox_obj.intersection(zone_poly)
    if intersection.is_empty:
        return 0.0
    area = intersection.area
    if area < epsilon:
        return 0.0

    # Y-axis overlap
    y_overlap_start = max(obj_y_start, zone_y_start)
    y_overlap_end = min(obj_y_end, zone_y_end)
    overlap_height = max(0, y_overlap_end - y_overlap_start)

    volume = area * overlap_height
    if volume < epsilon:
        return 0.0

    return volume


def eval_scene_openings(scene):
    """Evaluate door arc overlap (DAO) and window block overlap (WBO) for a scene.

    Returns dict with:
      - total_dao: total door arc overlap volume (m^3)
      - total_wbo: total window block overlap volume (m^3)
      - n_doors: number of doors evaluated
      - n_windows: number of windows evaluated
      - per_door_dao: list of per-door total overlap
      - per_window_wbo: list of per-window total overlap
    """
    openings = scene.get("openings", [])
    objects = scene.get("objects", [])
    bounds_bottom = scene.get("bounds_bottom", [])

    if not openings or not objects or not bounds_bottom:
        return {
            "total_dao": 0.0, "total_wbo": 0.0,
            "n_doors": 0, "n_windows": 0,
            "per_door_dao": [], "per_window_wbo": [],
        }

    doors = [o for o in openings if o["type"] == "door"]
    windows = [o for o in openings if o["type"] == "window"]

    per_door_dao = []
    for door in doors:
        try:
            arc_poly, y_start, y_end = make_door_arc_polygon(door, bounds_bottom)
        except Exception as e:
            print(f"  Warning: failed to create door arc: {e}")
            per_door_dao.append(0.0)
            continue

        door_total = 0.0
        for obj in objects:
            overlap = compute_opening_overlap(obj, arc_poly, y_start, y_end)
            door_total += overlap
        per_door_dao.append(door_total)

    per_window_wbo = []
    for window in windows:
        try:
            win_poly, y_start, y_end = make_window_clearance_polygon(window, bounds_bottom)
        except Exception as e:
            print(f"  Warning: failed to create window zone: {e}")
            per_window_wbo.append(0.0)
            continue

        win_total = 0.0
        for obj in objects:
            overlap = compute_opening_overlap(obj, win_poly, y_start, y_end)
            win_total += overlap
        per_window_wbo.append(win_total)

    return {
        "total_dao": sum(per_door_dao),
        "total_wbo": sum(per_window_wbo),
        "n_doors": len(doors),
        "n_windows": len(windows),
        "per_door_dao": per_door_dao,
        "per_window_wbo": per_window_wbo,
    }


def discover_scenes(eval_dir):
    """Find all generated_scene.json files, grouped by model."""
    eval_dir = Path(eval_dir)
    scenes = {m: [] for m in MODEL_NAMES}

    for style_seed_dir in sorted(eval_dir.iterdir()):
        if not style_seed_dir.is_dir():
            continue
        style_seed = style_seed_dir.name

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


def print_table(model_metrics):
    """Print formatted comparison table."""
    print("\n" + "=" * 90)
    print(f"{'Model':<12} {'N':>4}  {'DAO (x1e3)':>14}  {'WBO (x1e3)':>14}  {'DAO+WBO (x1e3)':>16}")
    print("-" * 90)

    for model_name in MODEL_NAMES:
        data = model_metrics.get(model_name)
        if not data or len(data["dao"]) == 0:
            print(f"{model_name:<12} {'--':>4}  {'--':>14}  {'--':>14}  {'--':>16}")
            continue

        n = len(data["dao"])
        dao_mean = np.mean(data["dao"]) * 1e3
        dao_std = np.std(data["dao"]) * 1e3
        wbo_mean = np.mean(data["wbo"]) * 1e3
        wbo_std = np.std(data["wbo"]) * 1e3
        total = np.array(data["dao"]) + np.array(data["wbo"])
        total_mean = np.mean(total) * 1e3
        total_std = np.std(total) * 1e3

        print(f"{model_name:<12} {n:>4}  {dao_mean:>6.2f} +/- {dao_std:<5.2f}  {wbo_mean:>6.2f} +/- {wbo_std:<5.2f}  {total_mean:>7.2f} +/- {total_std:<5.2f}")

    print("=" * 90)


def print_per_floor_table(model_metrics_per_floor):
    """Print per-floor breakdown."""
    all_floors = sorted({f for floors in model_metrics_per_floor.values() for f in floors})

    for floor in all_floors:
        print(f"\n--- {floor} ---")
        print(f"{'Model':<12} {'N':>4}  {'DAO (x1e3)':>14}  {'WBO (x1e3)':>14}")
        print("-" * 55)

        for model_name in MODEL_NAMES:
            data = model_metrics_per_floor.get(model_name, {}).get(floor)
            if not data or len(data["dao"]) == 0:
                print(f"{model_name:<12} {'--':>4}  {'--':>14}  {'--':>14}")
                continue

            n = len(data["dao"])
            dao_mean = np.mean(data["dao"]) * 1e3
            dao_std = np.std(data["dao"]) * 1e3
            wbo_mean = np.mean(data["wbo"]) * 1e3
            wbo_std = np.std(data["wbo"]) * 1e3

            print(f"{model_name:<12} {n:>4}  {dao_mean:>6.2f} +/- {dao_std:<5.2f}  {wbo_mean:>6.2f} +/- {wbo_std:<5.2f}")


def main():
    parser = argparse.ArgumentParser(description="Opening-awareness evaluation (DAO/WBO)")
    parser.add_argument("--eval-dir", type=str, default="eval/eval_set",
                        help="Root directory of eval sets (default: eval/eval_set)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save raw per-scene metrics to JSON file")
    parser.add_argument("--window-clearance", type=float, default=0.6,
                        help="Window clearance depth in meters (default: 0.6)")
    args = parser.parse_args()

    if not Path(args.eval_dir).exists():
        print(f"ERROR: eval dir not found: {args.eval_dir}")
        sys.exit(1)

    print(f"Eval dir: {args.eval_dir}")
    print(f"Window clearance depth: {args.window_clearance}m")

    scenes_by_model = discover_scenes(args.eval_dir)
    for m in MODEL_NAMES:
        total = len(scenes_by_model[m])
        non_bath = sum(1 for _, meta in scenes_by_model[m] if meta["room_type"] != "bathroom")
        print(f"  {m}: {total} scenes ({non_bath} non-bathroom)")

    model_metrics = {m: {"dao": [], "wbo": []} for m in MODEL_NAMES}
    model_metrics_per_floor = {m: {} for m in MODEL_NAMES}
    all_per_scene = []

    for model_name in MODEL_NAMES:
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*70}")

        for scene_path, meta in scenes_by_model[model_name]:
            if meta["room_type"] == "bathroom":
                continue

            short_path = f"{meta['style_seed']}/{model_name}/{meta['floor']}/{meta['room_stem']}"
            print(f"  {short_path} ... ", end="", flush=True)

            with open(scene_path) as f:
                scene = json.load(f)

            if not scene.get("openings") or not scene.get("objects"):
                print("SKIP (no openings or objects)")
                continue

            metrics = eval_scene_openings(scene)
            dao = metrics["total_dao"]
            wbo = metrics["total_wbo"]

            print(f"DAO={dao:.5f} ({metrics['n_doors']} doors)  WBO={wbo:.5f} ({metrics['n_windows']} windows)")

            model_metrics[model_name]["dao"].append(dao)
            model_metrics[model_name]["wbo"].append(wbo)

            floor = meta["floor"]
            if floor not in model_metrics_per_floor[model_name]:
                model_metrics_per_floor[model_name][floor] = {"dao": [], "wbo": []}
            model_metrics_per_floor[model_name][floor]["dao"].append(dao)
            model_metrics_per_floor[model_name][floor]["wbo"].append(wbo)

            all_per_scene.append({
                "model": model_name,
                **meta,
                "dao": dao,
                "wbo": wbo,
                "n_doors": metrics["n_doors"],
                "n_windows": metrics["n_windows"],
                "per_door_dao": metrics["per_door_dao"],
                "per_window_wbo": metrics["per_window_wbo"],
            })

    # Print results
    print("\n\n" + "#" * 90)
    print("# AGGREGATE RESULTS: Opening Awareness (all seeds, styles, floors)")
    print("#" * 90)
    print_table(model_metrics)

    print("\n\n" + "#" * 90)
    print("# PER-FLOOR BREAKDOWN")
    print("#" * 90)
    print_per_floor_table(model_metrics_per_floor)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {}
        for model_name in MODEL_NAMES:
            data = model_metrics[model_name]
            if len(data["dao"]) == 0:
                continue
            summary[model_name] = {
                "n_scenes": len(data["dao"]),
                "dao_mean": float(np.mean(data["dao"])),
                "dao_std": float(np.std(data["dao"])),
                "wbo_mean": float(np.mean(data["wbo"])),
                "wbo_std": float(np.std(data["wbo"])),
            }

        output = {"summary": summary, "per_scene": all_per_scene}
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved raw metrics to: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
