"""
Test script: Style-coherent asset retrieval via SigLIP2 embeddings.

Takes previously generated layouts (from test_custom_floorplan.py) and
re-samples assets with style coherence. The first object uses the full
SG-LLM description (style seed); subsequent objects use category-only
descriptions and are biased toward matching the established style via
cosine similarity on pre-computed style-only SigLIP2 embeddings.

Usage:
  uv run python -m input_test.test_style_coherent_retrieval --unit 13feb_settings
  uv run python -m input_test.test_style_coherent_retrieval --unit 13feb_settings --lambda-style 0.3
  uv run python -m input_test.test_style_coherent_retrieval --unit 13feb_settings --render
  uv run python -m input_test.test_style_coherent_retrieval --unit 13feb_settings --user-prompt "modern industrial dark metal"
"""

import json
import argparse
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

from src.sample import AssetRetrievalModule


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
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
	parser = argparse.ArgumentParser(
		description="Style-coherent asset retrieval from pre-generated layouts."
	)
	parser.add_argument("--unit", required=True,
						help="Unit subfolder inside input_test/ (e.g. '13feb_settings')")
	parser.add_argument("--lambda-style", type=float, default=0.2,
						help="Weight for style coherence term (default: 0.2)")
	parser.add_argument("--render", action="store_true",
						help="Render 3D scenes with retrieved assets")
	parser.add_argument("--stochastic", action="store_true",
						help="Use stochastic sampling instead of greedy")
	parser.add_argument("--user-prompt", type=str, default=None,
						help="Optional style prompt to anchor all selections (e.g. 'modern industrial dark metal')")
	args = parser.parse_args()

	base_dir = Path("input_test")
	unit_dir = base_dir / args.unit
	output_dir = base_dir / f"output_style_retrieval_{args.unit}"
	output_dir.mkdir(parents=True, exist_ok=True)

	# Find all generated_scene.json files from previous layout runs
	scene_files = sorted(unit_dir.glob("*/generated_scene.json"))
	if not scene_files:
		print(f"ERROR: No generated_scene.json found in {unit_dir}/*/")
		print("Run test_custom_floorplan.py first to generate layouts.")
		return

	print(f"Found {len(scene_files)} generated scene(s):")
	for f in scene_files:
		print(f"  - {f}")

	dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"\nDevice: {dvc}")

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

	for scene_path in scene_files:
		room_name = scene_path.parent.name
		print(f"\n{'#'*90}")
		print(f"# Processing: {scene_path}")
		print(f"# Room: {room_name}")
		print(f"# lambda_style={args.lambda_style}")
		if args.user_prompt:
			print(f"# user_prompt=\"{args.user_prompt}\"")
		print(f"{'#'*90}")

		with open(scene_path) as f:
			scene = json.load(f)

		n_objs = len(scene.get("objects", []))
		if n_objs == 0:
			print("  No objects in scene, skipping.")
			continue

		# Run style-coherent sampling
		result = retrieval.sample_all_assets_style_coherent(
			scene,
			lambda_style=args.lambda_style,
			is_greedy_sampling=not args.stochastic,
			user_prompt=args.user_prompt,
		)

		# Print summary
		mode = "stochastic" if args.stochastic else "greedy"
		print_scene_summary(result, f"{room_name} - style-coherent ({mode})")

		# Also run baseline (original system) for comparison
		print(f"\n--- Baseline (original system, no style bias) ---")
		baseline = retrieval.sample_all_assets(scene, is_greedy_sampling=not args.stochastic)
		print_scene_summary(baseline, f"{room_name} - baseline ({mode})")

		# Save results
		room_output = output_dir / room_name
		room_output.mkdir(parents=True, exist_ok=True)

		results = {
			"style_coherent": result,
			"baseline": baseline,
			"params": {
				"lambda_style": args.lambda_style,
				"greedy": not args.stochastic,
				"user_prompt": args.user_prompt,
			},
		}
		out_file = room_output / "retrieval_comparison.json"
		with open(out_file, "w") as f:
			json.dump(results, f, indent=2)
		print(f"Saved comparison to {out_file}")

		# Render
		if args.render:
			if not scene.get("bounds_bottom"):
				print("WARNING: Skipping render -- scene has no bounds_bottom")
				continue
			render_path = room_output / "render"
			render_path.mkdir(parents=True, exist_ok=True)
			render_scene(result, render_path, f"style_coherent_{room_name}")
			render_scene(baseline, render_path, f"baseline_{room_name}")

	print("\nDone!")


if __name__ == "__main__":
	main()
