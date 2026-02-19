"""
Inject door/window openings into existing SSR-3DFRONT dataset scenes.

This script reads the published SSR dataset (already processed with bounds + objects),
extracts door/window bounding boxes from the raw 3D-FRONT scene JSONs, recovers the
centering offset by matching furniture positions, and injects centered openings into
each SSR scene file.

Usage (from repo root):
    uv run python src/preprocessing/3d-front/inject_openings_into_ssr.py

Requires:
    - Raw 3D-FRONT scene JSONs at PTH_3DFRONT_SCENES (from .env)
    - SSR dataset at dataset-ssr3dfront/scenes/
"""

import json
import os
import copy
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from shapely.geometry import Polygon, Point


def extract_openings_from_meshes(scene):
	"""Extract door/window bounding boxes from raw 3D-FRONT scene mesh data."""
	openings = []
	for mesh in scene.get("mesh", []):
		mtype = mesh.get("type", "")
		if mtype not in ("Door", "Window"):
			continue
		xyz = mesh.get("xyz", [])
		if len(xyz) < 9:
			continue
		pts = np.array(xyz).reshape(-1, 3)
		# skip flat threshold quads (all y-values are 0)
		if np.allclose(pts[:, 1], 0.0, atol=1e-3):
			continue
		bbox_min = pts.min(axis=0)
		bbox_max = pts.max(axis=0)
		center = ((bbox_min + bbox_max) / 2.0).tolist()
		size = (bbox_max - bbox_min).tolist()
		openings.append({
			"type": mtype.lower(),
			"pos": [round(c, 2) for c in center],
			"size": [round(s, 2) for s in size],
		})
	return openings


def recover_centering_offset(ssr_scene, raw_scene):
	"""Recover the center_3d offset by matching furniture jids between SSR and raw."""
	room_id = ssr_scene.get("room_id")

	# Build uid -> jid lookup from raw furniture
	raw_furniture_jids = {}
	for furn in raw_scene.get("furniture", []):
		uid = furn.get("uid")
		jid = furn.get("jid")
		if uid and jid:
			raw_furniture_jids[uid] = jid

	# Find the matching room in the raw scene
	target_room = None
	for room in raw_scene.get("scene", {}).get("room", []):
		if room.get("instanceid") == room_id:
			target_room = room
			break

	if target_room is None:
		return None

	# Build jid -> raw_pos mapping for this room's children
	raw_objs_by_jid = {}
	for child in target_room.get("children", []):
		ref = child.get("ref")
		pos = child.get("pos")
		if ref and pos and ref in raw_furniture_jids:
			if not all(p == 0 for p in pos):
				jid = raw_furniture_jids[ref]
				raw_objs_by_jid.setdefault(jid, []).append(pos)

	# Match SSR objects by jid and compute offset
	offsets = []
	for ssr_obj in ssr_scene.get("objects", []):
		jid = ssr_obj.get("jid")
		if not jid:
			continue
		# handle scaled jids like "abc-(1.0)-(1.0)-(1.0)"
		base_jid = jid.split("-(")[0] if "-(" in jid else jid
		candidates = raw_objs_by_jid.get(base_jid, []) or raw_objs_by_jid.get(jid, [])
		if candidates:
			raw_pos = np.array(candidates[0])
			ssr_pos = np.array(ssr_obj["pos"])
			offset = raw_pos - ssr_pos
			offset[1] = 0.0  # y centering is always 0
			offsets.append(offset)

	if not offsets:
		return None

	# Use median for robustness
	center_3d = np.median(offsets, axis=0)
	return center_3d


def assign_openings_to_room(openings, bounds_bottom, tolerance=0.5):
	"""Return openings whose XZ center is within tolerance of the room polygon boundary."""
	coords_2d = [(b[0], b[2]) for b in bounds_bottom]
	polygon = Polygon(coords_2d)
	if not polygon.is_valid:
		return []
	boundary = polygon.exterior
	room_openings = []
	for opening in openings:
		px, _, pz = opening["pos"]
		dist = boundary.distance(Point(px, pz))
		if dist <= tolerance:
			room_openings.append(opening)
	return room_openings


def process_ssr_scene(pth_ssr_scene, pth_3dfront_scenes, raw_scene_cache):
	"""Process a single SSR scene file: inject openings."""
	filename = os.path.basename(pth_ssr_scene)
	# filename format: <orig_scene_uid>-<room_uid>.json
	# orig_scene_uid is first 5 UUID segments (36 chars)
	orig_scene_uid = filename[:36]

	# Load raw 3D-FRONT scene (with caching)
	if orig_scene_uid not in raw_scene_cache:
		pth_raw = os.path.join(pth_3dfront_scenes, f"{orig_scene_uid}.json")
		if not os.path.exists(pth_raw):
			return None, "raw_not_found"
		with open(pth_raw) as f:
			raw_scene_cache[orig_scene_uid] = json.load(f)
	raw_scene = raw_scene_cache[orig_scene_uid]

	# Load SSR scene
	with open(pth_ssr_scene) as f:
		ssr_scene = json.load(f)

	# Extract all openings from raw scene
	all_openings = extract_openings_from_meshes(raw_scene)
	if not all_openings:
		ssr_scene["openings"] = []
		return ssr_scene, "no_openings_in_raw"

	# Recover centering offset
	center_3d = recover_centering_offset(ssr_scene, raw_scene)
	if center_3d is None:
		ssr_scene["openings"] = []
		return ssr_scene, "offset_recovery_failed"

	# Center the openings
	centered_openings = []
	for opening in all_openings:
		centered = copy.deepcopy(opening)
		pos = np.array(centered["pos"]) - center_3d
		centered["pos"] = [round(p, 2) for p in pos.tolist()]
		centered_openings.append(centered)

	# Assign to room by proximity to bounds
	room_openings = assign_openings_to_room(
		centered_openings, ssr_scene["bounds_bottom"]
	)

	ssr_scene["openings"] = room_openings
	return ssr_scene, "ok"


def main():
	load_dotenv(".env")

	pth_3dfront_scenes = os.getenv("PTH_3DFRONT_SCENES")
	pth_ssr_scenes = "dataset-ssr3dfront/scenes"
	pth_output = "dataset-ssr3dfront-openings/scenes"

	os.makedirs(pth_output, exist_ok=True)

	all_files = sorted([
		f for f in os.listdir(pth_ssr_scenes)
		if f.endswith(".json") and not f.startswith(".")
	])

	# Skip already-processed files for resume support
	already_done = set(
		f for f in os.listdir(pth_output)
		if f.endswith(".json") and not f.startswith(".")
	) if os.path.exists(pth_output) else set()

	remaining = [f for f in all_files if f not in already_done]
	print(f"Processing {len(all_files)} SSR scenes ({len(already_done)} already done, {len(remaining)} remaining)...")

	raw_scene_cache = {}
	MAX_CACHE_SIZE = 100  # limit cached raw scenes to avoid OOM
	stats = {"ok": 0, "no_openings_in_raw": 0, "offset_recovery_failed": 0, "raw_not_found": 0, "error": 0}
	total_openings = 0

	for filename in tqdm(remaining):
		try:
			# Evict oldest cache entries if too large
			if len(raw_scene_cache) > MAX_CACHE_SIZE:
				oldest_key = next(iter(raw_scene_cache))
				del raw_scene_cache[oldest_key]

			pth_in = os.path.join(pth_ssr_scenes, filename)
			result_scene, status = process_ssr_scene(pth_in, pth_3dfront_scenes, raw_scene_cache)
			stats[status] += 1

			if result_scene is not None:
				n_openings = len(result_scene.get("openings", []))
				total_openings += n_openings

				pth_out = os.path.join(pth_output, filename)
				with open(pth_out, "w") as f:
					json.dump(result_scene, f, indent=4)
		except Exception as exc:
			print(f"\nError processing {filename}: {exc}")
			stats["error"] += 1

	print(f"\nDone!")
	print(f"  Total scenes: {len(all_files)}")
	print(f"  OK (with or without openings): {stats['ok'] + stats['no_openings_in_raw']}")
	print(f"  - Scenes with openings injected: {stats['ok']}")
	print(f"  - Scenes with no openings in raw: {stats['no_openings_in_raw']}")
	print(f"  Offset recovery failed: {stats['offset_recovery_failed']}")
	print(f"  Raw scene not found: {stats['raw_not_found']}")
	print(f"  Total openings injected: {total_openings}")
	print(f"  Avg openings per scene: {total_openings / max(1, stats['ok']):.1f}")


if __name__ == "__main__":
	main()