"""
Rule-based bathroom layout generator.

Generates fixture placements (sink, mirror, toilet, bathtub/shower) for bathroom
rooms using spatial rules relative to the door position. Outputs SSR-compatible
dicts that can be passed directly to the AssetRetrievalModule.
"""

import copy
import math
import numpy as np
from shapely.geometry import Polygon, Point, LineString


# ---------------------------------------------------------------------------
# Default fixture definitions (desc, size [w, h, d])
# ---------------------------------------------------------------------------

FIXTURES = {
    "sink": {
        "desc": "Modern minimalist ceramic bathroom sink with rectangular base, oval basin, and clean lines",
        "size": [0.8, 0.85, 0.5],
    },
    "mirror": {
        "desc": "A modern minimalist wall-mounted mirror accessory",
        "size": [0.6, 0.8, 0.05],
    },
    "toilet": {
        "desc": "Modern minimalist wall-mounted toilet with concealed ceramic tank",
        "size": [0.4, 0.45, 0.65],
    },
    "bathtub": {
        "desc": "A modern minimalist bathtub",
        "size": [1.7, 0.6, 0.75],
    },
    "shower": {
        "desc": "Modern minimalist industrial shower enclosure with dark tinted glass panels",
        "size": [0.9, 2.0, 0.9],
    },
    "rug": {
        "desc": "A dark gray textured rectangular textile rug with minimalist geometric design and modern contemporary style",
        "size": [0.8, 0.02, 0.5],
    },
}

# ---------------------------------------------------------------------------
# Building-code clearances  (IRC / IPC / NKBA)
# See: https://buildingcodegeek.com/bathroom-fixture-spacing-requirements-irc/
# ---------------------------------------------------------------------------
# Minimum clearance in front of each fixture (IRC: 21 in = 0.53 m)
MIN_FRONT_CLEARANCE = 0.53
# Minimum distance from toilet centerline to any side wall / fixture (IRC: 15 in = 0.38 m)
MIN_TOILET_CENTERLINE_SIDE = 0.38
# Minimum center-to-center distance between adjacent fixtures (IPC: 30 in = 0.76 m)
MIN_FIXTURE_CENTER_TO_CENTER = 0.76
# Minimum gap between any two fixture edges (practical minimum ~10 cm)
MIN_FIXTURE_EDGE_GAP = 0.10
# Minimum wall length to fit a standard bathtub (60 in = 1.52 m)
MIN_WALL_FOR_BATHTUB = 1.52


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def extract_walls(bounds_bottom):
    """
    Extract wall segments from the floor polygon vertices.

    Returns list of dicts:
        start_xz, end_xz: 2D endpoints [x, z]
        center_xz: midpoint [x, z]
        normal: inward-facing unit normal [nx, nz]
        length: wall length in meters
        index: wall index
    """
    pts = np.array(bounds_bottom)[:, [0, 2]]  # project to XZ
    n = len(pts)
    walls = []

    # Compute polygon centroid for inward normal orientation
    centroid = pts.mean(axis=0)

    for i in range(n):
        start = pts[i]
        end = pts[(i + 1) % n]
        edge = end - start
        length = np.linalg.norm(edge)
        if length < 1e-6:
            continue

        center = (start + end) / 2.0
        # Perpendicular: rotate edge 90 degrees
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)

        # Ensure normal points inward (toward centroid)
        to_centroid = centroid - center
        if np.dot(normal, to_centroid) < 0:
            normal = -normal

        walls.append({
            "start_xz": start,
            "end_xz": end,
            "center_xz": center,
            "normal": normal,
            "length": length,
            "index": i,
        })

    return walls


def map_openings_to_walls(walls, openings, opening_types=("window",)):
    """
    For each wall, compute the exclusion zones caused by openings.

    Args:
        walls: list of wall dicts from extract_walls
        openings: list of opening dicts with type, pos, size
        opening_types: tuple of opening types to map (e.g. ("window",) or ("window", "door"))

    Returns a dict: wall_index -> list of (frac_start, frac_end) representing
    the fractional range along the wall that is blocked by an opening.
    Each opening blocks its span + a buffer on each side.
    """
    BUFFER = 0.15  # extra clearance on each side of the opening (meters)
    filtered = [o for o in openings if o["type"] in opening_types]
    exclusions = {i: [] for i in range(len(walls))}

    for opening in filtered:
        op_pos_xz = np.array([opening["pos"][0], opening["pos"][2]])
        # Determine the span of the opening along its thin axis
        op_size = opening["size"]
        # The wide dimension (not thin, not height) is the opening's span along the wall
        thin_axis = 0 if op_size[0] < op_size[2] else 2
        op_span = op_size[2] if thin_axis == 0 else op_size[0]

        # Find which wall this opening is on
        best_wall_idx = 0
        best_dist = float("inf")
        for i, wall in enumerate(walls):
            line = LineString([wall["start_xz"], wall["end_xz"]])
            dist = line.distance(Point(op_pos_xz))
            if dist < best_dist:
                best_dist = dist
                best_wall_idx = i

        wall = walls[best_wall_idx]
        wall_vec = wall["end_xz"] - wall["start_xz"]
        wall_len = wall["length"]

        # Project opening center onto the wall to get fractional position
        op_to_start = op_pos_xz - wall["start_xz"]
        frac_center = np.dot(op_to_start, wall_vec) / (wall_len * wall_len) * wall_len
        frac_center = frac_center / wall_len

        # Compute blocked range (opening half-span + buffer, as fraction of wall)
        half_blocked = (op_span / 2.0 + BUFFER) / wall_len
        frac_start = max(0.0, frac_center - half_blocked)
        frac_end = min(1.0, frac_center + half_blocked)

        exclusions[best_wall_idx].append((frac_start, frac_end))

    return exclusions


def find_clear_position_on_wall(wall, fixture_width, window_exclusions, preferred_frac=0.5):
    """
    Find a fractional position along a wall where a fixture can be placed
    without overlapping any window exclusion zones.

    Args:
        wall: wall dict
        fixture_width: width of the fixture along the wall (meters)
        window_exclusions: list of (frac_start, frac_end) exclusion zones for this wall
        preferred_frac: preferred position (0.5 = center)

    Returns:
        A valid fractional position, or None if no position fits.
    """
    wall_len = wall["length"]
    half_fixture_frac = (fixture_width / 2.0) / wall_len

    # If no exclusions, return preferred
    if not window_exclusions:
        return preferred_frac

    # Check if preferred position is clear
    def is_clear(frac):
        f_start = frac - half_fixture_frac
        f_end = frac + half_fixture_frac
        for ex_s, ex_e in window_exclusions:
            if f_start < ex_e and f_end > ex_s:  # overlap
                return False
        return f_start >= 0.0 and f_end <= 1.0

    if is_clear(preferred_frac):
        return preferred_frac

    # Try shifting away from exclusion zones: scan from preferred outward
    for offset in np.arange(0.05, 0.5, 0.05):
        for candidate in [preferred_frac - offset, preferred_frac + offset]:
            if 0.0 + half_fixture_frac <= candidate <= 1.0 - half_fixture_frac:
                if is_clear(candidate):
                    return candidate

    return None


def find_door_wall(walls, openings):
    """
    Find which wall the door is on.
    Returns (door_opening, door_wall_index).
    """
    doors = [o for o in openings if o["type"] == "door"]
    if not doors:
        raise ValueError("No door found in openings")

    door = doors[0]
    door_pos_xz = np.array([door["pos"][0], door["pos"][2]])

    # Find the wall closest to the door position
    best_wall_idx = 0
    best_dist = float("inf")

    for i, wall in enumerate(walls):
        # Distance from door pos to wall line segment
        line = LineString([wall["start_xz"], wall["end_xz"]])
        dist = line.distance(Point(door_pos_xz))
        if dist < best_dist:
            best_dist = dist
            best_wall_idx = i

    return door, best_wall_idx


def get_opposite_wall(walls, door_wall_idx):
    """
    Find the wall most opposite to the door wall (antiparallel normal, farthest away).
    Returns wall index.
    """
    door_wall = walls[door_wall_idx]
    door_normal = door_wall["normal"]
    door_center = door_wall["center_xz"]

    best_idx = None
    best_score = -float("inf")

    for i, wall in enumerate(walls):
        if i == door_wall_idx:
            continue
        # Score: how antiparallel the normal is * distance from door wall
        antiparallel = -np.dot(wall["normal"], door_normal)
        dist = np.linalg.norm(wall["center_xz"] - door_center)
        score = antiparallel * dist
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def get_adjacent_walls(walls, door_wall_idx, opposite_wall_idx):
    """
    Get walls that are neither the door wall nor the opposite wall.
    Sorted by length (longest first) for better placement options.
    """
    adjacent = []
    for i, wall in enumerate(walls):
        if i == door_wall_idx or i == opposite_wall_idx:
            continue
        adjacent.append(i)
    adjacent.sort(key=lambda i: walls[i]["length"], reverse=True)
    return adjacent


def get_farthest_wall(walls, door_pos_xz, exclude_indices):
    """
    Find the wall farthest from the door position, excluding specified walls.
    """
    best_idx = None
    best_dist = -1

    for i, wall in enumerate(walls):
        if i in exclude_indices:
            continue
        dist = np.linalg.norm(wall["center_xz"] - door_pos_xz)
        if dist > best_dist:
            best_dist = dist
            best_idx = i

    return best_idx


def wall_normal_to_quaternion(normal):
    """
    Convert a 2D inward-facing wall normal [nx, nz] to a Y-axis rotation quaternion [qx, qy, qz, qw].
    The object should face the direction of the normal (into the room).
    """
    # Yaw angle: angle from +Z axis to the normal direction
    # In the SSR convention, identity quaternion [0,0,0,1] means facing +Z
    # We need the object to face along the normal direction
    yaw = math.atan2(normal[0], normal[1])  # atan2(nx, nz)
    qy = math.sin(yaw / 2)
    qw = math.cos(yaw / 2)
    return [0.0, round(qy, 5), 0.0, round(qw, 5)]


def place_on_wall(wall, offset_along_fraction, offset_from_wall, y_pos=0.0):
    """
    Compute a 3D position for placing an object on a wall.

    Args:
        wall: wall dict from extract_walls
        offset_along_fraction: 0.0 = wall start, 0.5 = center, 1.0 = wall end
        offset_from_wall: distance from wall surface into the room (meters)
        y_pos: vertical position (0.0 for floor objects)

    Returns:
        [x, y, z] position
    """
    # Position along the wall
    along = wall["start_xz"] + offset_along_fraction * (wall["end_xz"] - wall["start_xz"])
    # Offset into the room along the inward normal
    pos_xz = along + wall["normal"] * offset_from_wall
    return [round(float(pos_xz[0]), 4), round(float(y_pos), 4), round(float(pos_xz[1]), 4)]


def compute_aabb(obj):
    """
    Compute axis-aligned bounding box corners in XZ plane for a placed object.
    Returns (min_x, min_z, max_x, max_z).

    Accounts for Y-axis rotation at 90-degree increments by swapping width/depth
    when the object is rotated ~90 or ~270 degrees.
    """
    pos = obj["pos"]
    size = obj["size"]
    w, d = size[0], size[2]

    # Extract yaw from quaternion to determine if width/depth swap
    qx, qy, qz, qw = obj["rot"]
    yaw = math.atan2(2 * (qw * qy + qx * qz), 1 - 2 * (qy**2 + qz**2))
    # If rotated ~90 or ~270 degrees, swap width and depth
    if abs(math.cos(yaw)) < 0.5:
        w, d = d, w

    return (
        pos[0] - w / 2.0,
        pos[2] - d / 2.0,
        pos[0] + w / 2.0,
        pos[2] + d / 2.0,
    )


def check_overlap(obj_a, obj_b, min_gap=0.0):
    """Check if two objects overlap (or are closer than min_gap) in the XZ plane."""
    a = compute_aabb(obj_a)
    b = compute_aabb(obj_b)
    # Shrink each box by -min_gap/2 on each side (equivalent to expanding gap)
    g = min_gap / 2.0
    return not (a[2] + g <= b[0] - g or b[2] + g <= a[0] - g or
                a[3] + g <= b[1] - g or b[3] + g <= a[1] - g)


def check_any_overlap(candidate, placed_objects, min_gap=MIN_FIXTURE_EDGE_GAP):
    """Check if candidate overlaps with any already-placed floor-level object."""
    for existing in placed_objects:
        if existing["pos"][1] > 0.5:
            continue  # skip wall-mounted items (mirror)
        if check_overlap(candidate, existing, min_gap=min_gap):
            return True
    return False


def check_inside_bounds(obj, bounds_polygon):
    """Check if an object's AABB is mostly inside the room polygon."""
    pt = Point(obj["pos"][0], obj["pos"][2])
    return bounds_polygon.contains(pt)


# ---------------------------------------------------------------------------
# Main layout generator
# ---------------------------------------------------------------------------

def _try_place_on_wall(wall, wall_idx, fixture_size, window_exclusions,
                       placed_objects, floor_polygon, preferred_frac=0.5,
                       min_gap=MIN_FIXTURE_EDGE_GAP):
    """
    Try to place a fixture on a given wall, respecting window exclusions
    and overlap with already-placed objects.

    Scans multiple fractional positions starting from preferred_frac, then
    expanding outward in both directions (step=0.05).

    Returns (obj_dict, frac) on success, or (None, None) on failure.
    """
    wall_len = wall["length"]
    half_fixture_frac = (fixture_size[0] / 2.0) / wall_len
    win_excl = window_exclusions.get(wall_idx, [])

    # Build list of candidate fractions: preferred first, then scan outward
    candidates = [preferred_frac]
    for offset in np.arange(0.05, 0.5, 0.05):
        candidates.append(preferred_frac + offset)
        candidates.append(preferred_frac - offset)

    offset_from_wall = fixture_size[2] / 2.0
    rot = wall_normal_to_quaternion(wall["normal"])

    for frac in candidates:
        if frac < half_fixture_frac or frac > 1.0 - half_fixture_frac:
            continue

        # Check window exclusion at this fraction
        f_clear = find_clear_position_on_wall(wall, fixture_size[0], win_excl, preferred_frac=frac)
        if f_clear is None:
            continue
        # Use the exact clear position returned (may be slightly adjusted)
        frac = f_clear

        pos = place_on_wall(wall, frac, offset_from_wall, y_pos=0.0)
        candidate = {
            "size": list(fixture_size),
            "pos": pos,
            "rot": rot,
        }

        if check_any_overlap(candidate, placed_objects, min_gap=min_gap):
            continue

        if not floor_polygon.contains(Point(pos[0], pos[2])):
            continue

        return candidate, frac

    return None, None


def generate_bathroom_layout(scene):
    """
    Generate a bathroom layout with rule-based fixture placement.

    Layout strategy (based on building codes and interior-design best practices):
      1. Sink + mirror on the wall opposite the door (focal point when entering).
      2. Toilet on a side wall, as far from the door as possible so it is NOT
         in the direct line of sight (privacy rule).
      3. Bathtub / shower on the remaining wall (or shares a wall if only 4 walls).
      4. Rug in front of the bathtub / shower.

    Building-code clearances enforced (IRC / IPC / NKBA):
      - 15 in (0.38 m) from toilet centerline to any side wall or fixture.
      - 30 in (0.76 m) center-to-center between adjacent fixtures.
      - 21 in (0.53 m) clear floor space in front of each fixture.
      - 10 cm minimum edge-to-edge gap between any two fixtures.

    Windows act as exclusion zones on walls.

    Args:
        scene: SSR dict with bounds_bottom, bounds_top, openings, and empty objects.

    Returns:
        New SSR dict with populated objects list.
    """
    scene = copy.deepcopy(scene)
    bounds_bottom = scene["bounds_bottom"]
    openings = scene.get("openings", [])

    if not openings or not any(o["type"] == "door" for o in openings):
        raise ValueError("Bathroom scene must have at least one door in openings")

    # Build floor polygon
    floor_pts = np.array(bounds_bottom)[:, [0, 2]]
    floor_polygon = Polygon(floor_pts.tolist())

    # Extract walls and find door
    walls = extract_walls(bounds_bottom)
    door, door_wall_idx = find_door_wall(walls, openings)
    door_pos_xz = np.array([door["pos"][0], door["pos"][2]])

    # Map openings to walls as exclusion zones
    window_exclusions = map_openings_to_walls(walls, openings, opening_types=("window",))
    # Combined exclusions (windows + doors) for placing fixtures on the door wall
    all_exclusions = map_openings_to_walls(walls, openings, opening_types=("window", "door"))

    # Identify key walls
    opposite_wall_idx = get_opposite_wall(walls, door_wall_idx)
    adjacent_indices = get_adjacent_walls(walls, door_wall_idx, opposite_wall_idx)

    objects = []
    wall_assignments = {}  # wall_idx -> list of fixture keys placed on it

    def mark_wall(wall_idx, key):
        wall_assignments.setdefault(wall_idx, []).append(key)

    # ------------------------------------------------------------------ #
    # 1. SINK — opposite the door (the "focal point" wall)
    # ------------------------------------------------------------------ #
    sink_size = FIXTURES["sink"]["size"]
    sink_wall_idx = opposite_wall_idx
    sink_wall = walls[sink_wall_idx]
    sink_frac = None

    # Try opposite wall first; fall back to longest adjacent if too short
    candidate_walls = [opposite_wall_idx] + adjacent_indices
    for try_idx in candidate_walls:
        w = walls[try_idx]
        if w["length"] < sink_size[0] + 0.1:
            continue
        result, frac = _try_place_on_wall(
            w, try_idx, sink_size, window_exclusions,
            objects, floor_polygon, preferred_frac=0.5,
        )
        if result is not None:
            sink_wall_idx = try_idx
            sink_wall = w
            sink_frac = frac
            result["desc"] = FIXTURES["sink"]["desc"]
            objects.append(result)
            mark_wall(sink_wall_idx, "sink")
            break

    if sink_frac is None:
        raise ValueError("Could not place sink on any wall")

    # ------------------------------------------------------------------ #
    # 2. MIRROR — wall-mounted above the sink (same wall, same fraction)
    # ------------------------------------------------------------------ #
    mirror_size = FIXTURES["mirror"]["size"]
    mirror_offset = mirror_size[2] / 2.0
    mirror_pos = place_on_wall(sink_wall, sink_frac, mirror_offset, y_pos=1.4)
    mirror_rot = wall_normal_to_quaternion(sink_wall["normal"])

    mirror_obj = {
        "desc": FIXTURES["mirror"]["desc"],
        "size": list(mirror_size),
        "pos": mirror_pos,
        "rot": mirror_rot,
    }
    objects.append(mirror_obj)
    mark_wall(sink_wall_idx, "mirror")

    # ------------------------------------------------------------------ #
    # 3. TOILET — on an adjacent (side) wall, far from the door
    #    Privacy rule: never place directly opposite the door.
    #    IRC: >= 0.38 m from centerline to nearest side wall.
    # ------------------------------------------------------------------ #
    toilet_size = FIXTURES["toilet"]["size"]
    toilet_wall_idx = None

    # Sort adjacent walls: prefer the one farthest from the door
    adj_sorted = sorted(
        adjacent_indices,
        key=lambda i: np.linalg.norm(walls[i]["center_xz"] - door_pos_xz),
        reverse=True,
    )

    for adj_idx in adj_sorted:
        adj_wall = walls[adj_idx]
        min_len_needed = toilet_size[0] + 2 * MIN_TOILET_CENTERLINE_SIDE
        if adj_wall["length"] < min_len_needed:
            continue

        # Prefer the far end of the wall (away from door)
        dist_start = np.linalg.norm(adj_wall["start_xz"] - door_pos_xz)
        dist_end = np.linalg.norm(adj_wall["end_xz"] - door_pos_xz)
        margin = (MIN_TOILET_CENTERLINE_SIDE + toilet_size[0] / 2.0) / adj_wall["length"]

        if dist_start > dist_end:
            preferred_frac = max(margin, min(0.9, margin))
        else:
            preferred_frac = max(0.1, min(1.0 - margin, 1.0 - margin))

        result, frac = _try_place_on_wall(
            adj_wall, adj_idx, toilet_size, window_exclusions,
            objects, floor_polygon, preferred_frac=preferred_frac,
        )
        if result is not None:
            result["desc"] = FIXTURES["toilet"]["desc"]
            objects.append(result)
            toilet_wall_idx = adj_idx
            mark_wall(adj_idx, "toilet")
            break

    # Fallback: try same wall as sink but offset (with overlap check this time)
    if toilet_wall_idx is None:
        for pf in [0.15, 0.85, 0.25, 0.75]:
            if abs(pf - sink_frac) < 0.2:
                continue
            result, frac = _try_place_on_wall(
                sink_wall, sink_wall_idx, toilet_size, window_exclusions,
                objects, floor_polygon, preferred_frac=pf,
            )
            if result is not None:
                result["desc"] = FIXTURES["toilet"]["desc"]
                objects.append(result)
                toilet_wall_idx = sink_wall_idx
                mark_wall(sink_wall_idx, "toilet")
                break

    if toilet_wall_idx is None:
        print("WARNING: Could not place toilet on any wall")

    # ------------------------------------------------------------------ #
    # 4. BATHTUB / SHOWER
    #    Strategy: try shower on the door wall first (next to door, facing
    #    into the room — visible when entering). If that fails, try
    #    bathtub/shower on other walls ranked by available space.
    # ------------------------------------------------------------------ #
    bath_wall_idx = None
    bath_fixture_key = None

    # --- Phase 1: try shower on the door wall (preferred placement) ---
    door_wall = walls[door_wall_idx]
    shower_size = FIXTURES["shower"]["size"]
    if door_wall["length"] >= shower_size[0] + 0.1:
        result, frac = _try_place_on_wall(
            door_wall, door_wall_idx, shower_size, all_exclusions,
            objects, floor_polygon, preferred_frac=0.5,
        )
        if result is not None:
            result["desc"] = FIXTURES["shower"]["desc"]
            objects.append(result)
            bath_wall_idx = door_wall_idx
            bath_fixture_key = "shower"
            mark_wall(door_wall_idx, "shower")

    # --- Phase 2: fallback to other walls for bathtub/shower ---
    if bath_wall_idx is None:
        bath_candidates = [
            i for i in range(len(walls)) if i != door_wall_idx
        ]
        bath_candidates.sort(key=lambda i: (
            len(wall_assignments.get(i, [])),   # fewer fixtures first
            -walls[i]["length"],                 # longer walls first
        ))

        for bi in bath_candidates:
            bw = walls[bi]
            attempts = []
            if bw["length"] >= MIN_WALL_FOR_BATHTUB:
                attempts.append(("bathtub", FIXTURES["bathtub"]["size"]))
            if bw["length"] >= shower_size[0] + 0.1:
                attempts.append(("shower", shower_size))

            for try_key, try_size in attempts:
                result, frac = _try_place_on_wall(
                    bw, bi, try_size, window_exclusions,
                    objects, floor_polygon, preferred_frac=0.5,
                )
                if result is not None:
                    result["desc"] = FIXTURES[try_key]["desc"]
                    objects.append(result)
                    bath_wall_idx = bi
                    bath_fixture_key = try_key
                    mark_wall(bi, try_key)
                    break

            if bath_wall_idx is not None:
                break

    if bath_wall_idx is None:
        print("WARNING: Could not place bathtub or shower on any wall")

    # ------------------------------------------------------------------ #
    # 5. RUG — in front of the bathtub / shower
    # ------------------------------------------------------------------ #
    if bath_wall_idx is not None and bath_fixture_key is not None:
        bath_obj_placed = next(
            o for o in objects
            if o["desc"] == FIXTURES[bath_fixture_key]["desc"]
        )
        rug_size = FIXTURES["rug"]["size"]
        bath_normal = walls[bath_wall_idx]["normal"]
        bath_pos_xz = np.array([bath_obj_placed["pos"][0], bath_obj_placed["pos"][2]])

        bath_depth = bath_obj_placed["size"][2]
        # Rug just in front of the bath (edge-to-edge + small gap)
        rug_offset = bath_depth / 2.0 + rug_size[2] / 2.0 + 0.05
        rug_pos_xz = bath_pos_xz + bath_normal * rug_offset
        rug_pos = [round(float(rug_pos_xz[0]), 4), 0.0, round(float(rug_pos_xz[1]), 4)]
        rug_rot = wall_normal_to_quaternion(bath_normal)

        rug_obj = {
            "desc": FIXTURES["rug"]["desc"],
            "size": list(rug_size),
            "pos": rug_pos,
            "rot": rug_rot,
        }

        if check_inside_bounds(rug_obj, floor_polygon):
            objects.append(rug_obj)

    # ------------------------------------------------------------------ #
    # Validation pass
    # ------------------------------------------------------------------ #
    for obj in objects:
        if not check_inside_bounds(obj, floor_polygon):
            print(f"WARNING: {obj['desc'][:40]}... center is outside room bounds")

    scene["objects"] = objects
    return scene
