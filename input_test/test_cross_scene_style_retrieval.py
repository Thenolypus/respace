"""
Test script: Cross-scene (unit-wide) style-coherent asset retrieval.

Extends per-room style coherence to the entire unit. The living room is
always processed first and establishes the unit-wide style anchor. Its
accumulated style embeddings are then forwarded to every subsequent room,
so that from room 2 onwards even object[0] receives the 3-metric blend
(semantic + size + style).

Usage:
  uv run python -m input_test.test_cross_scene_style_retrieval --unit 13feb_settings
  uv run python -m input_test.test_cross_scene_style_retrieval --unit 13feb_settings --lambda-style 0.3
  uv run python -m input_test.test_cross_scene_style_retrieval --unit 13feb_settings --render
  uv run python -m input_test.test_cross_scene_style_retrieval --unit 13feb_settings --user-prompt "modern industrial dark metal"
"""

import os
import json
import argparse
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

from src.sample import AssetRetrievalModule

ENV_FILE = ".env"
ENV_FILE_BATHROOM = ".env_heg"


# --------------------------------------------------------------------------- #
# Output helpers                                                               #
# --------------------------------------------------------------------------- #

def _parse_field(val):
	"""Parse a metadata field that may be a list or a comma-separated string."""
	if val is None:
		return set()
	if isinstance(val, str):
		return {v.lower().strip() for v in val.split(",") if v.strip()}
	return {v.lower().strip() for v in val if v}


def get_asset_style_fields(jid, metadata, metadata_scaled):
	"""Return (color_set, style_set, material_set) for a given asset JID."""
	asset = metadata.get(jid)
	if asset is None:
		scaled = metadata_scaled.get(jid)
		if scaled is None:
			return set(), set(), set()
		orig_jid = scaled.get("jid")
		asset = metadata.get(orig_jid, {})

	colors = _parse_field(asset.get("color"))
	styles = _parse_field(asset.get("style"))
	materials = _parse_field(asset.get("material"))
	return colors, styles, materials


def print_scene_summary(scene, label):
	"""Print a compact summary of a sampled scene."""
	metadata = json.load(open("data/metadata/model_info_3dfuture_assets.json"))
	metadata_scaled = json.load(open("data/metadata/model_info_3dfuture_assets_scaled.json"))

	print(f"\n{'='*90}")
	print(f"  {label}")
	print(f"{'='*90}")
	for i, obj in enumerate(scene.get("objects", [])):
		jid = obj.get("sampled_asset_jid", "N/A")
		colors, styles, materials = get_asset_style_fields(jid, metadata, metadata_scaled)
		print(f"[{i}] {obj.get('sampled_asset_desc', 'N/A')[:60]}")
		print(f"     Colors: {sorted(colors)}  Styles: {sorted(styles)}")
	print()


def render_scene(scene_with_assets, output_path, filename):
	"""Render the scene with retrieved 3D assets."""
	from src.viz import render_scene_and_export
	print(f"Rendering scene: {filename} ...")
	render_scene_and_export(
		scene_with_assets,
		filename=filename,
		pth_output=output_path,
		resolution=(1024, 1024),
		show_bboxes=False,
		show_assets=True,
		use_dynamic_zoom=True,
	)
	print(f"Saved renders to {output_path}/top/ and {output_path}/diag/")


# --------------------------------------------------------------------------- #
# Room ordering                                                                #
# --------------------------------------------------------------------------- #

def _load_env(env_file):
	"""Clear asset-related env vars and reload from the given .env file."""
	asset_keys = [
		"PTH_ASSETS_METADATA", "PTH_ASSETS_METADATA_SCALED",
		"PTH_ASSETS_METADATA_SIMPLE_DESCS", "PTH_ASSETS_METADATA_PROMPTS",
		"PTH_ASSETS_EMBED", "PTH_ASSETS_EMBED_STYLE", "PTH_ASSETS_EMBED_CATEGORY",
		"PTH_3DFUTURE_ASSETS",
	]
	for k in asset_keys:
		os.environ.pop(k, None)
	load_dotenv(env_file, override=True)
	print(f"  Loaded env: {env_file}")


def _is_bathroom(scene_path):
	"""Check if a scene file belongs to a bathroom room."""
	name = scene_path.parent.name.lower()
	if "bathroom" in name:
		return True
	with open(scene_path) as f:
		scene = json.load(f)
	room_type = scene.get("room_type", "").lower()
	return "bathroom" in room_type


def order_scenes_living_room_first(scene_files):
	"""
	Sort scene files so that living room comes first.
	Looks for 'livingroom' or 'living_room' in the parent folder name.
	"""
	living = []
	rest = []
	for f in scene_files:
		name = f.parent.name.lower()
		if "livingroom" in name or "living_room" in name:
			living.append(f)
		else:
			rest.append(f)
	if not living:
		print("WARNING: No living room found in unit. Using first room as style anchor.")
		return list(scene_files)
	return living + rest


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
	parser = argparse.ArgumentParser(
		description="Cross-scene (unit-wide) style-coherent asset retrieval."
	)
	parser.add_argument("--unit", required=True,
						help="Unit subfolder inside input_test/ (e.g. '13feb_settings')")
	parser.add_argument("--lambda-style", type=float, default=0.05,
						help="Weight for style coherence term (default: 0.2)")
	parser.add_argument("--render", action="store_true",
						help="Render 3D scenes with retrieved assets")
	parser.add_argument("--stochastic", action="store_true",
						help="Use stochastic sampling instead of greedy")
	parser.add_argument("--user-prompt", type=str, default=None,
						help="Optional style prompt to anchor all selections (e.g. 'modern industrial dark metal')")
	parser.add_argument("--full-desc", action="store_true",
						help="Use full descriptions for semantic query (no category stripping)")
	parser.add_argument("--category-full", action="store_true",
						help="Category query against full-desc embeddings (legacy, noisy)")
	parser.add_argument("--ori-sample", action="store_true",
						help="Use original asset sampling (description + size only, no style coherence)")
	args = parser.parse_args()

	base_dir = Path("input_test")
	unit_dir = base_dir / args.unit

	# Find all generated_scene.json files
	scene_files = sorted(unit_dir.glob("*/generated_scene.json"))
	if not scene_files:
		print(f"ERROR: No generated_scene.json found in {unit_dir}/*/")
		print("Run test_custom_floorplan.py first to generate layouts.")
		return

	# Order: living room first, then the rest
	scene_files = order_scenes_living_room_first(scene_files)

	print(f"Found {len(scene_files)} generated scene(s) (ordered living room first):")
	for i, f in enumerate(scene_files):
		tag = " <-- style anchor" if i == 0 else ""
		print(f"  [{i}] {f}{tag}")

	dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"\nDevice: {dvc}")

	# Split scenes into non-bathroom and bathroom groups
	default_scenes = []
	bathroom_scenes = []
	for scene_path in scene_files:
		if _is_bathroom(scene_path):
			bathroom_scenes.append(scene_path)
		else:
			default_scenes.append(scene_path)

	# Process non-bathroom first, then bathroom (mirrors orchestrate_unit.py)
	ordered_groups = []
	if default_scenes:
		ordered_groups.append((ENV_FILE, default_scenes))
	if bathroom_scenes:
		ordered_groups.append((ENV_FILE_BATHROOM, bathroom_scenes))

	print(f"\nNon-bathroom rooms: {len(default_scenes)}, Bathroom rooms: {len(bathroom_scenes)}")

	# Unit-wide style context -- accumulates across rooms
	unit_style_embeds = []
	all_results = {}
	mode = "stochastic" if args.stochastic else "greedy"
	room_idx = 0

	for env_file, group_scenes in ordered_groups:
		print(f"\n--- Loading assets from {env_file} for {len(group_scenes)} room(s) ---")
		_load_env(env_file)

		retrieval = AssetRetrievalModule(
			lambd=0.5,
			sigma=0.05,
			temp=0.2,
			top_p=0.95,
			top_k=20,
			asset_size_threshold=0.5,
			dvc=dvc,
			do_print=True,
		)

		for scene_path in group_scenes:
			room_name = scene_path.parent.name
			is_anchor = (room_idx == 0)

			if args.full_desc:
				sem_mode = "full-desc"
			elif args.category_full:
				sem_mode = "category-full (legacy)"
			else:
				sem_mode = "category-only"
			print(f"\n{'#'*90}")
			print(f"# Room [{room_idx}]: {room_name}")
			print(f"# Source: {scene_path}")
			print(f"# Assets: {env_file}")
			if args.ori_sample:
				print(f"# Mode: ORIGINAL (description + size only)")
			else:
				print(f"# lambda_style={args.lambda_style}, semantic={sem_mode}")
				if is_anchor:
					print(f"# Role: STYLE ANCHOR (living room)")
				else:
					print(f"# Role: CROSS-SCENE BIASED ({len(unit_style_embeds)} embeddings from prior rooms)")
				if args.user_prompt:
					print(f"# user_prompt=\"{args.user_prompt}\"")
			print(f"{'#'*90}")

			with open(scene_path) as f:
				scene = json.load(f)

			n_objs = len(scene.get("objects", []))
			if n_objs == 0:
				print("  No objects in scene, skipping.")
				continue

			if args.ori_sample:
				# Original sampling: description + size only, no style coherence
				result = retrieval.sample_all_assets(
					scene,
					is_greedy_sampling=not args.stochastic,
				)

				print_scene_summary(result, f"{room_name} - original sampling ({mode})")

				room_output = scene_path.parent

				room_results = {
					"cross_scene_style": result,
					"params": {
						"mode": "original",
						"greedy": not args.stochastic,
						"room_order": room_idx,
					},
				}
				out_file = room_output / "cross_scene_retrieval.json"
				with open(out_file, "w") as f:
					json.dump(room_results, f, indent=2)
				print(f"Saved to {out_file}")
			else:
				# Cross-scene style-coherent retrieval
				# For the anchor room, no initial style embeds.
				# For subsequent rooms, pass the accumulated unit style.
				initial_embeds = None if is_anchor else list(unit_style_embeds)

				result, room_style_embeds = retrieval.sample_all_assets_style_coherent_cross_scene(
					scene,
					lambda_style=args.lambda_style,
					is_greedy_sampling=not args.stochastic,
					user_prompt=args.user_prompt,
					initial_style_embeds=initial_embeds,
				)

				# Collect only the NEW embeddings from this room (skip the ones we passed in)
				n_prior = len(initial_embeds) if initial_embeds else 0
				if args.user_prompt is not None:
					n_prior += 1  # user prompt embed was prepended
				new_embeds = room_style_embeds[n_prior:]
				unit_style_embeds.extend(new_embeds)

				print(f"\n  >> Room contributed {len(new_embeds)} new style embeddings")
				print(f"  >> Unit-wide total: {len(unit_style_embeds)} style embeddings")

				print_scene_summary(result, f"{room_name} - cross-scene style ({mode})")

				room_output = scene_path.parent

				room_results = {
					"cross_scene_style": result,
					"params": {
						"lambda_style": args.lambda_style,
						"greedy": not args.stochastic,
						"user_prompt": args.user_prompt,
						"room_order": room_idx,
						"is_anchor": is_anchor,
						"n_prior_style_embeds": n_prior,
						"n_new_style_embeds": len(new_embeds),
						"n_total_unit_style_embeds": len(unit_style_embeds),
					},
				}
				out_file = room_output / "cross_scene_retrieval.json"
				with open(out_file, "w") as f:
					json.dump(room_results, f, indent=2)
				print(f"Saved to {out_file}")

			all_results[room_name] = room_results

			# Render
			if args.render:
				if not scene.get("bounds_bottom"):
					print("WARNING: Skipping render -- scene has no bounds_bottom")
					room_idx += 1
					continue
				render_path = room_output / "render"
				render_path.mkdir(parents=True, exist_ok=True)
				render_scene(result, render_path, f"cross_style_{room_name}")

			room_idx += 1

		del retrieval

	# Restore default env
	_load_env(ENV_FILE)

	# Unit summary
	print(f"\n{'#'*90}")
	print(f"# UNIT SUMMARY")
	print(f"# Total rooms processed: {len(all_results)}")
	print(f"# Total style embeddings accumulated: {len(unit_style_embeds)}")
	print(f"{'#'*90}")
	for name, res in all_results.items():
		p = res["params"]
		print(f"  {name}: order={p['room_order']}, anchor={p['is_anchor']}, "
			  f"prior={p['n_prior_style_embeds']}, new={p['n_new_style_embeds']}")

	print("\nDone!")


if __name__ == "__main__":
	main()