"""
Test script: Standalone test for the AssetRetrievalModule.
Initializes only the retrieval module (SigLIP + embedding catalog) and tests
asset retrieval given object descriptions and sizes.

Usage:
  Fixed queries:          uv run python -m input_test.test_retrieval
  From layout output:     uv run python -m input_test.test_retrieval --from-layout
  With 3D visualization:  uv run python -m input_test.test_retrieval --from-layout --render
"""

import json
import copy
import argparse
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

from src.sample import AssetRetrievalModule


# --------------------------------------------------------------------------- #
# Toggle: switch between fixed test queries or loading from layout outputs     #
# --------------------------------------------------------------------------- #

# Fixed test queries: (description, [width, height, depth])
TEST_QUERIES = [
	("a rectangular wooden dining table", [1.6, 0.75, 0.9]),
	("a modern fabric sofa", [2.2, 0.85, 0.9]),
	("a small bedside nightstand with drawers", [0.5, 0.55, 0.4]),
	("a king size bed with upholstered headboard", [2.0, 1.2, 2.1]),
	("a swivel office chair", [0.6, 1.0, 0.6]),
	("a tall bookshelf", [0.8, 1.8, 0.3]),
	("a round coffee table", [0.8, 0.45, 0.8]),
	("a floor lamp", [0.4, 1.6, 0.4]),
]

# Paths to generated scene JSONs from the layout test
LAYOUT_SCENE_PATHS = [
	Path("input_test/3og_example/output_livingroom/generated_scene.json")
	#Path("input_test/unit_1_example/output_bedroom_2/generated_scene.json"),
	#Path("input_test/unit_1_example/output_bedroom_3/generated_scene.json"),
]


def build_scene_from_fixed_queries():
	"""Build a scene dict from the hardcoded TEST_QUERIES."""
	return {
		"objects": [
			{"desc": desc, "size": size}
			for desc, size in TEST_QUERIES
		]
	}


def load_scenes_from_layouts():
	"""Load generated scenes and strip existing retrieval fields so we can re-retrieve."""
	scenes = []
	for pth in LAYOUT_SCENE_PATHS:
		if not pth.exists():
			print(f"WARNING: layout file not found, skipping: {pth}")
			continue
		with open(pth) as f:
			scene = json.load(f)

		# Store original retrieval results for comparison, then strip them
		for obj in scene.get("objects", []):
			obj["original_sampled_asset_jid"] = obj.pop("sampled_asset_jid", None)
			obj["original_sampled_asset_desc"] = obj.pop("sampled_asset_desc", None)
			obj["original_sampled_asset_size"] = obj.pop("sampled_asset_size", None)
			obj.pop("uuid", None)
			obj.pop("jid", None)

		scenes.append((pth, scene))
	return scenes


def print_results(result_scene, label, original_available=False):
	print(f"\n--- {label} ---\n")
	for i, obj in enumerate(result_scene["objects"]):
		print(f"[{i}] Query: {obj['desc']}")
		print(f"    Query size:    {obj['size']}")
		print(f"    Retrieved JID: {obj.get('sampled_asset_jid', 'N/A')}")
		print(f"    Retrieved desc: {obj.get('sampled_asset_desc', 'N/A')}")
		print(f"    Retrieved size: {obj.get('sampled_asset_size', 'N/A')}")
		if original_available and obj.get("original_sampled_asset_jid"):
			match = obj.get("sampled_asset_jid") == obj.get("original_sampled_asset_jid")
			print(f"    Original JID:  {obj['original_sampled_asset_jid']}  {'(MATCH)' if match else '(DIFFERENT)'}")
		print()


def render_scene(scene_with_assets, output_path, filename):
	"""Render the scene with retrieved 3D assets using the existing viz pipeline (OpenGL/EGL)."""
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


def run_retrieval(retrieval, scene, output_path, tag="", original_available=False, do_render=False):
	n_objs = len(scene.get("objects", []))
	print(f"\nRunning retrieval for {n_objs} objects...\n")
	print("=" * 80)

	# Greedy
	result_greedy = retrieval.sample_all_assets(scene, is_greedy_sampling=True)
	print("=" * 80)
	print_results(result_greedy, f"GREEDY RESULTS {tag}", original_available)

	# Stochastic
	print("=" * 80)
	print("Running stochastic retrieval...\n")
	result_stochastic = retrieval.sample_all_assets(scene, is_greedy_sampling=False)
	print("=" * 80)
	print_results(result_stochastic, f"STOCHASTIC RESULTS {tag}", original_available)

	# Save
	results = {"greedy": result_greedy, "stochastic": result_stochastic}
	suffix = f"_{tag}" if tag else ""
	out_file = output_path / f"retrieval_results{suffix}.json"
	with open(out_file, "w") as f:
		json.dump(results, f, indent=2)
	print(f"Saved results to {out_file}")

	# Render 3D scenes (requires bounds_bottom from a real layout)
	if do_render:
		if not scene.get("bounds_bottom"):
			print("WARNING: Skipping render — scene has no bounds_bottom (use --from-layout)")
			return
		render_path = output_path / f"render{suffix}"
		render_path.mkdir(parents=True, exist_ok=True)
		render_scene(result_greedy, render_path, f"greedy{suffix}")
		render_scene(result_stochastic, render_path, f"stochastic{suffix}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--from-layout", action="store_true",
						help="Load objects from layout test outputs instead of fixed queries")
	parser.add_argument("--render", action="store_true",
						help="Render 3D scenes with retrieved assets (requires EGL/display)")
	args = parser.parse_args()

	output_path = Path("input_test/output_retrieval")
	output_path.mkdir(parents=True, exist_ok=True)

	dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {dvc}")

	print("Initializing AssetRetrievalModule...")
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

	if args.from_layout:
		scenes = load_scenes_from_layouts()
		if not scenes:
			print("ERROR: No layout scenes found. Run test_custom_floorplan.py first.")
			return
		for pth, scene in scenes:
			tag = pth.parent.name  # e.g. "output_bedroom_3"
			room_type = scene.get("room_type", "unknown")
			print(f"\n{'#' * 80}")
			print(f"# Layout: {pth}  (room_type={room_type})")
			print(f"{'#' * 80}")
			run_retrieval(retrieval, scene, output_path, tag=tag, original_available=True, do_render=args.render)
	else:
		scene = build_scene_from_fixed_queries()
		run_retrieval(retrieval, scene, output_path, do_render=args.render)

	print("\nDone!")


if __name__ == "__main__":
	main()
