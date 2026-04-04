"""
Extract door and window openings from original 3D-FRONT scenes
and add them to the SSR dataset scenes in dataset-ssr3dfront/scenes/.

The openings are derived from mesh entries with type "Door" or "Window"
in the original 3D-FRONT JSON files. Position and size are computed from
the bounding box of the mesh vertex data (xyz flat array).

Doors on shared walls may only be children of one adjacent room, so we
check ALL door/window meshes in the scene for spatial proximity to the
target room's walls.
"""

import json
import os
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path


def compute_bbox_from_xyz(xyz_flat):
    """Compute bounding box center and size from flat xyz vertex list."""
    vertices = np.array(xyz_flat).reshape(-1, 3)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = ((mins + maxs) / 2.0).tolist()
    size = (maxs - mins).tolist()
    return center, size


def get_door_window_meshes(scene_data):
    """Extract all Door and Window meshes with computed bounding boxes."""
    meshes = []
    mesh_uids = set()
    for mesh in scene_data.get("mesh", []):
        if mesh.get("type") in ("Door", "Window"):
            xyz = mesh.get("xyz", [])
            if len(xyz) < 9:  # need at least 3 vertices
                continue
            center, size = compute_bbox_from_xyz(xyz)
            # Skip threshold/flat pieces (very small height, likely door bottom piece)
            if size[1] < 0.1:
                continue
            mesh_uids.add(mesh.get("uid"))
            meshes.append({
                "uid": mesh.get("uid"),
                "type": mesh.get("type").lower(),  # "door" or "window"
                "center": center,
                "size": [round(s, 2) for s in size],
            })
    return meshes, mesh_uids


def get_room_children_refs(scene_data, room_instanceid):
    """Get set of mesh UIDs that are children of a specific room."""
    refs = set()
    for room in scene_data.get("scene", {}).get("room", []):
        if room.get("instanceid") == room_instanceid:
            for child in room.get("children", []):
                refs.add(child.get("ref"))
    return refs


def is_on_room_wall(mesh_center, mesh_size, room_bounds, tolerance=0.3):
    """Check if a door/window mesh is on or near one of the room's walls.

    Room bounds are the original (pre-shift) bounds_bottom as Nx3 array.
    We check if the mesh center is within tolerance of any wall segment.
    """
    bounds_xz = room_bounds[:, [0, 2]]
    n = len(bounds_xz)
    mesh_x, mesh_z = mesh_center[0], mesh_center[2]

    for i in range(n):
        p1 = bounds_xz[i]
        p2 = bounds_xz[(i + 1) % n]

        # Check if mesh center is near this wall segment
        # Wall segment from p1 to p2
        seg = p2 - p1
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-6:
            continue

        # Project mesh point onto segment line
        t = np.dot(np.array([mesh_x, mesh_z]) - p1, seg) / (seg_len ** 2)
        # Allow some extension beyond segment endpoints for corner cases
        t_clamped = np.clip(t, -0.1, 1.1)
        closest = p1 + t_clamped * seg
        dist = np.linalg.norm(np.array([mesh_x, mesh_z]) - closest)

        if dist < tolerance:
            return True

    return False


def compute_room_center(bounds_bottom):
    """Compute the XZ center used to shift the room (same as in 04_training_dataset_stage_1_json.py)."""
    bounds = np.array(bounds_bottom)
    bounds_min_xz = np.min(bounds, axis=0)
    bounds_max_xz = np.max(bounds, axis=0)
    center_xz = bounds_min_xz + (bounds_max_xz - bounds_min_xz) / 2.0
    return np.array([center_xz[0], 0, center_xz[2]])


def find_openings_for_room(scene_data, room_instanceid, original_bounds_bottom):
    """Find all door/window openings relevant to a room.

    Checks both direct children and spatially proximate meshes from other rooms.
    """
    all_dw_meshes, dw_uids = get_door_window_meshes(scene_data)

    if not all_dw_meshes:
        return []

    # Get direct children refs for this room
    direct_refs = get_room_children_refs(scene_data, room_instanceid)

    # Also check all rooms for their door/window children near our walls
    original_bounds = np.array(original_bounds_bottom)

    openings = []
    seen_uids = set()

    for mesh in all_dw_meshes:
        uid = mesh["uid"]
        if uid in seen_uids:
            continue

        # Include if: direct child of room, OR spatially on room wall
        is_direct = uid in direct_refs
        is_proximate = is_on_room_wall(mesh["center"], mesh["size"], original_bounds)

        if is_direct or is_proximate:
            seen_uids.add(uid)
            openings.append(mesh)

    return openings


def extract_original_bounds_from_3dfront(scene_data, room_instanceid):
    """Extract original room bounds from the floor mesh vertices in the 3D-FRONT scene."""
    # Build mesh lookup for fast access
    mesh_lookup = {m.get("uid"): m for m in scene_data.get("mesh", [])}

    for room in scene_data.get("scene", {}).get("room", []):
        if room.get("instanceid") == room_instanceid:
            for child in room.get("children", []):
                ref = child.get("ref")
                mesh = mesh_lookup.get(ref)
                if mesh and mesh.get("type") == "Floor":
                    xyz = mesh.get("xyz", [])
                    if len(xyz) >= 9:
                        vertices = np.array(xyz).reshape(-1, 3)
                        # Round to 1 decimal to match 03_extract_corners_for_rooms.py
                        vertices = np.round(vertices, decimals=1)
                        return vertices.tolist()

    return None


def process_ssr_scene(ssr_scene_path, pth_3dfront_scenes, scene_cache):
    """Process a single SSR scene file and add openings data."""
    ssr_scene = json.load(open(ssr_scene_path))

    # Extract orig_scene_uid from filename: {orig_scene_uid}-{room_uid}.json
    filename = os.path.basename(ssr_scene_path)
    # The orig_scene_uid is a UUID (36 chars), followed by '-', then room_uid
    orig_scene_uid = filename[:36]
    room_id = ssr_scene.get("room_id")

    if not room_id:
        return False, "no room_id"

    # Load original 3D-FRONT scene (with caching)
    if orig_scene_uid not in scene_cache:
        scene_path = os.path.join(pth_3dfront_scenes, f"{orig_scene_uid}.json")
        if not os.path.isfile(scene_path):
            return False, f"original scene not found: {scene_path}"
        scene_cache[orig_scene_uid] = json.load(open(scene_path))

    scene_data = scene_cache[orig_scene_uid]

    # Get original (pre-shift) bounds from SSR scene's shifted bounds + floor mesh center
    # The SSR bounds_bottom are the simplified room polygon (already center-shifted).
    # We recover the original bounds by adding back the center computed from floor mesh.
    floor_bounds = extract_original_bounds_from_3dfront(scene_data, room_id)
    if floor_bounds is None:
        return False, f"could not find floor mesh for {room_id}"

    # Compute center from the rounded floor mesh (matches original preprocessing)
    center_3d = compute_room_center(floor_bounds)

    if np.any(np.isnan(center_3d)):
        return False, "NaN center"

    # Recover original room polygon by un-shifting the SSR bounds
    ssr_bounds = ssr_scene.get("bounds_bottom")
    if ssr_bounds is None:
        return False, "no bounds_bottom in SSR scene"
    original_room_polygon = (np.array(ssr_bounds) + center_3d).tolist()

    # Find all openings for this room using the room polygon for wall proximity
    openings_raw = find_openings_for_room(scene_data, room_id, original_room_polygon)

    # Apply center shift and format openings
    openings = []
    for opening in openings_raw:
        shifted_center = np.array(opening["center"]) - center_3d
        openings.append({
            "type": opening["type"],
            "pos": [round(c, 2) for c in shifted_center.tolist()],
            "size": opening["size"],
        })

    ssr_scene["openings"] = openings

    with open(ssr_scene_path, "w") as f:
        json.dump(ssr_scene, f, indent=4)

    return True, len(openings)


def main():
    load_dotenv(".env")

    pth_3dfront_scenes = os.getenv("PTH_3DFRONT_SCENES")
    scenes_dir = Path("dataset-ssr3dfront/scenes")

    all_files = sorted([f for f in scenes_dir.glob("*.json") if f.stem[0].isalnum()])
    print(f"Processing {len(all_files)} SSR scenes...")

    scene_cache = {}
    stats = {"success": 0, "failed": 0, "total_openings": 0, "no_openings": 0}

    for ssr_path in tqdm(all_files):
        success, result = process_ssr_scene(str(ssr_path), pth_3dfront_scenes, scene_cache)

        if success:
            stats["success"] += 1
            stats["total_openings"] += result
            if result == 0:
                stats["no_openings"] += 1
        else:
            stats["failed"] += 1
            # Uncomment for debugging:
            # print(f"  FAILED {ssr_path.name}: {result}")

    print(f"\nDone!")
    print(f"  Processed: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total openings added: {stats['total_openings']}")
    print(f"  Scenes with 0 openings: {stats['no_openings']}")


if __name__ == "__main__":
    main()
