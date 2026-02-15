"""
Test script: Style-coherent asset retrieval.

Takes previously generated layouts (from test_custom_floorplan.py), strips
color/style from descriptions, and re-samples assets with a style coherence
bias. The first object establishes the "style seed"; subsequent objects are
biased toward matching the established style via Jaccard similarity over the
metadata color/style/material fields.

Usage:
  uv run python -m input_test.test_style_coherent_retrieval --unit 13feb_settings
  uv run python -m input_test.test_style_coherent_retrieval --unit 13feb_settings --lambda-style 0.3
  uv run python -m input_test.test_style_coherent_retrieval --unit 13feb_settings --render
"""

import json
import copy
import argparse
import torch
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

from src.sample import AssetRetrievalModule


# --------------------------------------------------------------------------- #
# Style similarity helpers                                                     #
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


def get_asset_category(jid, metadata, metadata_scaled, simple_descs):
	"""Get the furniture category for an asset (e.g. 'sofa', 'lamp', 'table')."""
	asset = metadata.get(jid)
	if asset is None:
		scaled = metadata_scaled.get(jid)
		if scaled is None:
			return "unknown"
		orig_jid = scaled.get("jid")
		asset = metadata.get(orig_jid, {})

	summary = asset.get("summary", "")
	return simple_descs.get(summary, "unknown")


def jaccard(set_a, set_b):
	"""Jaccard similarity between two sets. Returns 0 if both empty."""
	if not set_a and not set_b:
		return 0.0
	union = set_a | set_b
	if len(union) == 0:
		return 0.0
	return len(set_a & set_b) / len(union)


def compute_style_similarities(
	candidate_jids,
	selected_jids,
	metadata,
	metadata_scaled,
	simple_descs=None,
	query_category=None,
	w_color=0.5,
	w_style=0.3,
	w_material=0.2,
):
	"""
	Compute style similarity for every candidate against already-selected assets.

	If query_category is provided, only compares against selected assets of the
	same category (e.g. lamp vs lamp, sofa vs sofa). If no same-category assets
	have been selected yet, returns uniform scores (no bias).

	Returns a numpy array of shape (n_candidates,).
	"""
	if not selected_jids:
		return np.ones(len(candidate_jids), dtype=np.float32)

	# Filter selected assets to same category if requested
	if query_category and simple_descs:
		same_cat_jids = [
			j for j in selected_jids
			if get_asset_category(j, metadata, metadata_scaled, simple_descs) == query_category
		]
	else:
		same_cat_jids = selected_jids

	# No same-category assets selected yet -- no style bias for this object
	if not same_cat_jids:
		return np.ones(len(candidate_jids), dtype=np.float32)

	# Pre-fetch style fields for the comparison set
	selected_fields = [get_asset_style_fields(j, metadata, metadata_scaled) for j in same_cat_jids]

	scores = np.zeros(len(candidate_jids), dtype=np.float32)
	for i, cand_jid in enumerate(candidate_jids):
		c_colors, c_styles, c_materials = get_asset_style_fields(cand_jid, metadata, metadata_scaled)

		sim_sum = 0.0
		for s_colors, s_styles, s_materials in selected_fields:
			sim = (
				w_color * jaccard(c_colors, s_colors)
				+ w_style * jaccard(c_styles, s_styles)
				+ w_material * jaccard(c_materials, s_materials)
			)
			sim_sum += sim
		scores[i] = sim_sum / len(selected_fields)

	return scores


# --------------------------------------------------------------------------- #
# Description stripping                                                        #
# --------------------------------------------------------------------------- #

def strip_desc_to_category(obj):
	"""
	Strip a full asset description to just the furniture category.

	Strategy (in priority order):
	1. Use the 'prompt' field if present (e.g. "floor standing lamp") -- this is
	   the short category-level label from the SG-LLM pipeline.
	2. Look up the 'desc' in the simple_descs mapping.
	3. Fall back to the original desc unchanged.
	"""
	# 1. Use prompt field (always a short category-level string)
	prompt = obj.get("prompt")
	if prompt:
		return prompt

	# 2. Try simple_descs mapping
	desc = obj.get("desc", "")
	if not hasattr(strip_desc_to_category, "_map"):
		try:
			import os
			pth = os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS",
							"data/metadata/model_info_3dfuture_assets_simple_descs.json")
			with open(pth) as f:
				strip_desc_to_category._map = json.load(f)
		except Exception:
			strip_desc_to_category._map = {}

	if desc in strip_desc_to_category._map:
		return strip_desc_to_category._map[desc]

	# 3. Fallback
	return desc


# --------------------------------------------------------------------------- #
# Style-coherent sampling                                                      #
# --------------------------------------------------------------------------- #

def sample_scene_with_style_coherence(
	retrieval,
	scene,
	lambda_style=0.2,
	is_greedy_sampling=True,
	do_strip_descs=True,
	do_print=True,
	category_constrained=True,
):
	"""
	Sample assets for a scene with autoregressive style coherence.

	1. For each object (in order):
	   a. Compute semantic + size similarities (existing system).
	   b. Compute style similarity vs. already-selected assets of the same
	      furniture category (if category_constrained=True).
	   c. Blend: final = lam_sem * sem + lam_size * size + lam_style * style
	      where lam_sem + lam_size are re-scaled from the original lambda
	      to sum to (1 - lambda_style).
	   d. Sample from the blended distribution.
	2. The first object has no style bias (no prior selections).
	"""
	sampled_scene = copy.deepcopy(scene)
	sampled_scene["objects"] = []
	selected_jids = []

	metadata = retrieval.all_assets_metadata
	metadata_scaled = retrieval.all_assets_metadata_scaled
	all_jids = retrieval.all_jids_catalog

	# Load simple_descs for category lookup
	simple_descs = {}
	if category_constrained:
		try:
			import os
			pth = os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS",
							"data/metadata/model_info_3dfuture_assets_simple_descs.json")
			with open(pth) as f:
				simple_descs = json.load(f)
		except Exception:
			print("WARNING: Could not load simple_descs, disabling category-constrained style")
			category_constrained = False

	orig_lambd = retrieval.lambd.item()

	for obj_idx, obj in enumerate(scene.get("objects", [])):
		desc = obj.get("desc", "")
		size = obj.get("size", [])

		# Optionally strip to category-only description
		query_desc = strip_desc_to_category(obj) if do_strip_descs else desc

		# Determine query category early so we can print it
		query_category = None
		if category_constrained:
			query_category = simple_descs.get(desc, None)
			if query_category is None:
				query_category = simple_descs.get(obj.get("sampled_asset_desc", ""), None)

		if do_print:
			print(f"\n{'='*90}")
			print(f"OBJECT [{obj_idx}]")
			print(f"  Original desc: \"{desc}\"")
			if do_strip_descs:
				print(f"  Stripped desc:  \"{query_desc}\"")
			print(f"  Size: {size}")
			print(f"  Selected so far: {len(selected_jids)} assets")
			if category_constrained:
				print(f"  Query category: {query_category or 'unknown'}")
			print(f"{'='*90}")

		# Step 1: Get semantic + size similarities from existing system
		query_embeds = retrieval.get_text_embeddings([query_desc])
		semantic_sims = retrieval.compute_semantic_similarities(query_embeds).squeeze(1)  # (n_assets,)

		query_size_t = torch.tensor([size])
		if retrieval.accelerator:
			query_size_t = query_size_t.to(retrieval.accelerator.device)
		else:
			query_size_t = query_size_t.to(retrieval.dvc)
		size_sims = retrieval.compute_size_similarities(query_size_t).squeeze(1)  # (n_assets,)

		# Step 2: Compute style similarity vs. already-selected assets
		style_scores = compute_style_similarities(
			all_jids, selected_jids, metadata, metadata_scaled,
			simple_descs=simple_descs if category_constrained else None,
			query_category=query_category,
		)
		style_sims = torch.tensor(style_scores, device=semantic_sims.device)

		# Check if style bias is active (not uniform 1.0 scores)
		style_is_active = not (style_sims == 1.0).all().item()

		# Step 3: Blend
		if not style_is_active:
			# No style bias: first object or no same-category assets selected yet
			weighted_sims = orig_lambd * semantic_sims + (1 - orig_lambd) * size_sims
			effective_style_weight = 0.0
		else:
			# Re-scale original sem/size weights to fit within (1 - lambda_style)
			remaining = 1.0 - lambda_style
			lam_sem = orig_lambd * remaining
			lam_size = (1 - orig_lambd) * remaining
			weighted_sims = lam_sem * semantic_sims + lam_size * size_sims + lambda_style * style_sims
			effective_style_weight = lambda_style

		# Step 4: Apply top-k / top-p and sample
		# We temporarily override the retrieval module's internals
		retrieval._last_semantic_sims = semantic_sims.unsqueeze(1)
		retrieval._last_size_sims = size_sims.unsqueeze(1)
		retrieval._last_weighted_sims = weighted_sims.unsqueeze(1)

		probs = retrieval.compute_final_probabilities(weighted_sims.unsqueeze(1))  # (1, n_assets)
		probs = probs.squeeze(0)  # (n_assets,)

		# Print top candidates with style breakdown
		if do_print:
			n_top = min(10, retrieval.top_k)
			idxs_top = torch.argsort(probs, descending=True)[:n_top]
			n_nonzero = (probs > 0).sum().item()
			print(f"Candidates surviving top-k/top-p: {n_nonzero}")
			print(f"Weights: sem={orig_lambd * (1-effective_style_weight):.3f}, "
				  f"size={(1-orig_lambd) * (1-effective_style_weight):.3f}, "
				  f"style={effective_style_weight:.3f}")
			print()
			print(f"{'Rank':<5} {'Prob':>8} {'Sem':>8} {'Size':>8} {'Style':>8} {'Blend':>8}  "
				  f"{'Colors':<20} {'Styles':<25} Description")
			print(f"{'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  "
				  f"{'-'*20} {'-'*25} {'-'*50}")

			for rank, idx in enumerate(idxs_top):
				jid = all_jids[idx.item()]
				colors, styles, materials = get_asset_style_fields(jid, metadata, metadata_scaled)

				asset = metadata.get(jid)
				if asset is None:
					asset = metadata_scaled.get(jid)
					orig_jid = asset.get("jid")
					orig_asset = metadata.get(orig_jid)
					asset_desc = orig_asset.get("summary")
				else:
					asset_desc = asset.get("summary")

				prob_val = probs[idx].item()
				sem_val = semantic_sims[idx].item()
				size_val = size_sims[idx].item()
				style_val = style_sims[idx].item()
				blend_val = weighted_sims[idx].item()

				colors_str = ",".join(sorted(colors))[:19]
				styles_str = ",".join(sorted(styles))[:24]

				print(f"#{rank+1:<4} {prob_val:>8.4f} {sem_val:>8.4f} {size_val:>8.4f} "
					  f"{style_val:>8.4f} {blend_val:>8.4f}  "
					  f"{colors_str:<20} {styles_str:<25} {asset_desc[:50]}")
			print()

		# Sample
		if is_greedy_sampling:
			_, idx_sampled = torch.max(probs, dim=0)
		else:
			idx_sampled = torch.multinomial(probs, num_samples=1).squeeze()

		jid_sampled = all_jids[idx_sampled.item()]

		# Build the new object
		asset = metadata.get(jid_sampled)
		scale_sampled = None
		if asset is None:
			asset_scaled = metadata_scaled.get(jid_sampled)
			size_sampled = asset_scaled.get("size")
			scale_sampled = asset_scaled.get("scale")
			orig_jid = asset_scaled.get("jid")
			orig_asset = metadata.get(orig_jid)
			desc_sampled = orig_asset.get("summary")
		else:
			desc_sampled = asset.get("summary")
			size_sampled = asset.get("size")

		new_obj = copy.deepcopy(obj)
		new_obj.update({
			"sampled_asset_jid": jid_sampled,
			"sampled_asset_desc": desc_sampled,
			"sampled_asset_size": size_sampled,
			"uuid": __import__("uuid").uuid4().hex,
		})
		if scale_sampled is not None:
			new_obj["scale"] = scale_sampled

		sampled_scene["objects"].append(new_obj)
		selected_jids.append(jid_sampled)

		if do_print:
			sel_colors, sel_styles, sel_materials = get_asset_style_fields(jid_sampled, metadata, metadata_scaled)
			print(f"  >> SELECTED: {desc_sampled[:70]}")
			print(f"     Colors: {sorted(sel_colors)}  Styles: {sorted(sel_styles)}  Materials: {sorted(sel_materials)}")

	return sampled_scene


# --------------------------------------------------------------------------- #
# Output helpers                                                               #
# --------------------------------------------------------------------------- #

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
	parser.add_argument("--no-strip", action="store_true",
						help="Don't strip descriptions to categories (use full descs)")
	parser.add_argument("--render", action="store_true",
						help="Render 3D scenes with retrieved assets")
	parser.add_argument("--stochastic", action="store_true",
						help="Use stochastic sampling instead of greedy")
	parser.add_argument("--no-category-constraint", action="store_true",
						help="Compare style against ALL selected assets, not just same category")
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
		do_print=False,  # we handle printing ourselves
	)

	for scene_path in scene_files:
		room_name = scene_path.parent.name
		print(f"\n{'#'*90}")
		print(f"# Processing: {scene_path}")
		print(f"# Room: {room_name}")
		print(f"# lambda_style={args.lambda_style}, strip_descs={not args.no_strip}, category_constrained={not args.no_category_constraint}")
		print(f"{'#'*90}")

		with open(scene_path) as f:
			scene = json.load(f)

		n_objs = len(scene.get("objects", []))
		if n_objs == 0:
			print("  No objects in scene, skipping.")
			continue

		# Run style-coherent sampling
		result = sample_scene_with_style_coherence(
			retrieval,
			scene,
			lambda_style=args.lambda_style,
			is_greedy_sampling=not args.stochastic,
			do_strip_descs=not args.no_strip,
			do_print=True,
			category_constrained=not args.no_category_constraint,
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
				"strip_descs": not args.no_strip,
				"greedy": not args.stochastic,
				"category_constrained": not args.no_category_constraint,
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
