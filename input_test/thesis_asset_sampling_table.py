"""
Thesis figure: Asset sampling scoring breakdown for one room.

Runs the style-coherent autoregressive asset retrieval on a single
generated room and exports per-object top-k scoring tables as JSON
(for LaTeX/programmatic use) and as formatted console output.

Usage:
  uv run python -m input_test.thesis_asset_sampling_table
  uv run python -m input_test.thesis_asset_sampling_table --scene eval/eval_set/scand_456/1_7b/SimpleApartment/unit_1/unit_1_room_2_livingroom/generated_scene.json
  uv run python -m input_test.thesis_asset_sampling_table --lambda-style 0.1
  uv run python -m input_test.thesis_asset_sampling_table --top-n 5
"""

import os
import json
import copy
import argparse
import torch
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

from src.sample import AssetRetrievalModule


# Default scene to use
DEFAULT_SCENE = (
    "eval/eval_set/scand_456/1_7b/SimpleApartment/unit_1/"
    "unit_1_room_2_livingroom/generated_scene.json"
)


def get_asset_metadata_fields(jid, metadata, metadata_scaled):
    """Return color, style, material, and summary for a given asset JID."""
    asset = metadata.get(jid)
    if asset is None:
        scaled = metadata_scaled.get(jid)
        if scaled is None:
            return {}, ""
        orig_jid = scaled.get("jid")
        asset = metadata.get(orig_jid, {})

    def parse(val):
        if val is None:
            return []
        if isinstance(val, str):
            return [v.strip() for v in val.split(",") if v.strip()]
        return [v.strip() for v in val if v]

    return {
        "color": parse(asset.get("color")),
        "style": parse(asset.get("style")),
        "material": parse(asset.get("material")),
        "size": asset.get("size", []),
    }, asset.get("summary", "")


def run_sampling_with_diagnostics(scene, retrieval, lambda_style, top_n):
    """
    Run style-coherent sampling and capture per-object scoring tables.

    Returns a list of dicts, one per object, each containing:
      - obj_idx, query_desc, query_size, style_bias_active, n_style_embeds
      - candidates: list of top-n candidates with all score components
      - selected: the sampled asset info
    """
    metadata = json.load(open(os.getenv("PTH_ASSETS_METADATA")))
    metadata_scaled = json.load(open(os.getenv("PTH_ASSETS_METADATA_SCALED")))

    sampled_scene = copy.deepcopy(scene)
    sampled_scene["objects"] = []
    selected_style_embeds = []
    desc_size_map = {}

    orig_lambd = retrieval.lambd.item()
    all_tables = []

    for obj_idx, obj in enumerate(scene.get("objects", [])):
        desc = obj.get("desc", "")
        size = obj.get("size", [])
        prompt = obj.get("prompt", "")
        is_first_object = (obj_idx == 0)

        query_desc = desc

        # Compute similarities
        query_embeds = retrieval.get_text_embeddings([query_desc])
        semantic_sims = retrieval.compute_semantic_similarities(query_embeds).squeeze(1)

        query_size_t = torch.tensor([size]).to(retrieval.dvc)
        size_sims = retrieval.compute_size_similarities(query_size_t).squeeze(1)

        # Blend
        use_style_bias = len(selected_style_embeds) > 0 and obj_idx > 0
        if use_style_bias:
            style_embeds_tensor = torch.stack(selected_style_embeds, dim=0)
            style_sims = retrieval.compute_style_similarity(style_embeds_tensor)

            remaining = 1.0 - lambda_style
            lam_sem = orig_lambd * remaining
            lam_size = (1 - orig_lambd) * remaining
            weighted_sims = lam_sem * semantic_sims + lam_size * size_sims + lambda_style * style_sims
        else:
            style_sims = None
            lam_sem = orig_lambd
            lam_size = 1 - orig_lambd
            weighted_sims = orig_lambd * semantic_sims + (1 - orig_lambd) * size_sims

        # Store intermediates for create_sampled_obj
        retrieval._last_semantic_sims = semantic_sims.unsqueeze(1)
        retrieval._last_size_sims = size_sims.unsqueeze(1)
        retrieval._last_weighted_sims = weighted_sims.unsqueeze(1)
        retrieval._last_style_sims = style_sims.unsqueeze(1) if style_sims is not None else None

        # Probability computation
        probs = retrieval.compute_final_probabilities(weighted_sims.unsqueeze(1))
        probs = probs.squeeze(0)

        # Extract top-n candidates
        n_nonzero = (probs > 0).sum().item()
        idxs_top = torch.argsort(probs, descending=True)[:top_n]

        candidates = []
        for rank, idx in enumerate(idxs_top):
            jid = retrieval.all_jids_catalog[idx.item()]
            meta, summary = get_asset_metadata_fields(jid, metadata, metadata_scaled)

            candidate = {
                "rank": rank + 1,
                "jid": jid,
                "description": summary,
                "asset_size": meta.get("size", []),
                "color": meta.get("color", []),
                "style": meta.get("style", []),
                "material": meta.get("material", []),
                "prob": round(probs[idx].item(), 6),
                "sem_score": round(semantic_sims[idx].item(), 6),
                "size_score": round(size_sims[idx].item(), 6),
                "blend_score": round(weighted_sims[idx].item(), 6),
            }
            if style_sims is not None:
                candidate["style_score"] = round(style_sims[idx].item(), 6)
            candidates.append(candidate)

        # Greedy sample (pick rank 1)
        _, idx_sampled = torch.max(probs, dim=0)
        jid_sampled = retrieval.all_jids_catalog[idx_sampled.item()]

        # Build sampled object
        asset = metadata.get(jid_sampled)
        scale = None
        if asset is None:
            asset_s = metadata_scaled.get(jid_sampled)
            sampled_size = asset_s.get("size")
            scale = asset_s.get("scale")
            orig_jid = asset_s.get("jid")
            sampled_desc = metadata.get(orig_jid, {}).get("summary", "")
        else:
            sampled_desc = asset.get("summary", "")
            sampled_size = asset.get("size")

        new_obj = copy.deepcopy(obj)
        new_obj.update({
            "sampled_asset_jid": jid_sampled,
            "sampled_asset_desc": sampled_desc,
            "sampled_asset_size": sampled_size,
        })
        if scale is not None:
            new_obj["scale"] = scale
        sampled_scene["objects"].append(new_obj)

        # Track style embedding for next objects
        idx_in_catalog = retrieval.jid_to_idx[jid_sampled]
        selected_style_embeds.append(retrieval.all_style_embeds_catalog[idx_in_catalog])

        # Build table entry
        weights = {
            "lambda_semantic": round(lam_sem, 4),
            "lambda_size": round(lam_size, 4),
        }
        if use_style_bias:
            weights["lambda_style"] = round(lambda_style, 4)

        table_entry = {
            "obj_idx": obj_idx,
            "query_description": desc,
            "query_prompt": prompt,
            "query_size": size,
            "style_bias_active": use_style_bias,
            "n_style_embeds": len(selected_style_embeds) - 1,  # before this object
            "blend_weights": weights,
            "n_candidates_after_filtering": n_nonzero,
            "candidates": candidates,
            "selected_jid": jid_sampled,
            "selected_description": sampled_desc,
        }
        all_tables.append(table_entry)

    return all_tables, sampled_scene


def print_tables(all_tables):
    """Print thesis-quality formatted tables to console."""
    for t in all_tables:
        print(f"\n{'='*120}")
        print(f"Object [{t['obj_idx']}]: \"{t['query_prompt']}\"")
        print(f"  Full description: \"{t['query_description'][:90]}...\"" if len(t['query_description']) > 90 else f"  Full description: \"{t['query_description']}\"")
        print(f"  Target size: {t['query_size']}")
        style_str = f"ACTIVE ({t['n_style_embeds']} prior embeddings)" if t['style_bias_active'] else "INACTIVE (first object)"
        print(f"  Style bias: {style_str}")
        w = t['blend_weights']
        weight_parts = [f"sem={w['lambda_semantic']}", f"size={w['lambda_size']}"]
        if 'lambda_style' in w:
            weight_parts.append(f"style={w['lambda_style']}")
        print(f"  Blend weights: {', '.join(weight_parts)}")
        print(f"  Candidates after top-k/top-p: {t['n_candidates_after_filtering']}")
        print()

        has_style = 'style_score' in t['candidates'][0] if t['candidates'] else False

        # Header
        if has_style:
            print(f"  {'Rank':<5} {'Prob':>8} {'Sem':>8} {'Size':>8} {'Style':>8} {'Blend':>8}  {'Asset Size':<28} Description")
            print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*28} {'-'*50}")
        else:
            print(f"  {'Rank':<5} {'Prob':>8} {'Sem':>8} {'Size':>8} {'Blend':>8}  {'Asset Size':<28} Description")
            print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*28} {'-'*50}")

        for c in t['candidates']:
            size_str = str([round(s, 3) for s in c['asset_size']]) if c['asset_size'] else "N/A"
            desc_short = c['description'][:65]
            if has_style:
                print(f"  #{c['rank']:<4} {c['prob']:>8.4f} {c['sem_score']:>8.4f} {c['size_score']:>8.4f} {c['style_score']:>8.4f} {c['blend_score']:>8.4f}  {size_str:<28} {desc_short}")
            else:
                print(f"  #{c['rank']:<4} {c['prob']:>8.4f} {c['sem_score']:>8.4f} {c['size_score']:>8.4f} {c['blend_score']:>8.4f}  {size_str:<28} {desc_short}")

        print(f"\n  >> Selected: \"{t['selected_description'][:80]}\"")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Export asset sampling scoring tables for thesis documentation."
    )
    parser.add_argument("--scene", type=str, default=DEFAULT_SCENE,
                        help=f"Path to generated_scene.json (default: {DEFAULT_SCENE})")
    parser.add_argument("--lambda-style", type=float, default=0.1,
                        help="Style coherence weight (default: 0.1)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top candidates to show per object (default: 10)")
    parser.add_argument("--output", type=str, default="input_test/thesis_sampling_tables.json",
                        help="Output JSON path")
    args = parser.parse_args()

    scene_path = Path(args.scene)
    if not scene_path.exists():
        print(f"ERROR: Scene file not found: {scene_path}")
        return

    with open(scene_path) as f:
        scene = json.load(f)

    n_objs = len(scene.get("objects", []))
    print(f"Scene: {scene_path}")
    print(f"Room type: {scene.get('room_type')}")
    print(f"Objects: {n_objs}")
    print(f"lambda_style: {args.lambda_style}")
    print(f"top_n: {args.top_n}")

    dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dvc}")

    retrieval = AssetRetrievalModule(
        lambd=0.5,
        sigma=0.05,
        temp=0.2,
        top_p=0.95,
        top_k=20,
        asset_size_threshold=0.5,
        dvc=dvc,
        do_print=False,
    )

    print("\nRunning asset sampling with scoring diagnostics...\n")
    all_tables, sampled_scene = run_sampling_with_diagnostics(
        scene, retrieval, args.lambda_style, args.top_n
    )

    # Print to console
    print_tables(all_tables)

    # Save JSON
    output = {
        "scene_path": str(scene_path),
        "room_type": scene.get("room_type"),
        "n_objects": n_objs,
        "params": {
            "lambd": 0.5,
            "sigma": 0.05,
            "temp": 0.2,
            "top_k": 20,
            "top_p": 0.95,
            "lambda_style": args.lambda_style,
            "sampling": "greedy",
        },
        "objects": all_tables,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved scoring tables to: {out_path}")

    # Also save the re-sampled scene
    scene_out = out_path.parent / "thesis_sampled_scene.json"
    with open(scene_out, "w") as f:
        json.dump(sampled_scene, f, indent=2)
    print(f"Saved re-sampled scene to: {scene_out}")


if __name__ == "__main__":
    main()
