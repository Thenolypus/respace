"""
Pre-compute style-only SigLIP2 embeddings for all assets.

For each asset JID, constructs a style string from its metadata fields
(color, style, material) and embeds it with SigLIP2. The resulting
embeddings are saved to a pickle file with the same structure as the
full-description embeddings (jids, sizes, embeds).

The style strings intentionally exclude the furniture category so that
cross-category style comparison is meaningful (e.g. a "modern black metal"
lamp can match a "modern black metal" table).

Usage:
  uv run python -m src.preprocessing.3d-front.07_compute_style_embeds
"""

import torch
import numpy as np
import pickle
import json
from transformers import AutoTokenizer, SiglipTextModel
from tqdm import tqdm
from dotenv import load_dotenv
import os


def parse_field(val):
	"""Parse a metadata field that may be a list or a comma-separated string."""
	if val is None:
		return []
	if isinstance(val, str):
		return [v.strip() for v in val.split(",") if v.strip()]
	if isinstance(val, list):
		return [v.strip() for v in val if isinstance(v, str) and v.strip()]
	return []


def build_style_string(asset):
	"""Build a style-only string from an asset's color, style, and material fields."""
	colors = parse_field(asset.get("color"))
	styles = parse_field(asset.get("style"))
	materials = parse_field(asset.get("material"))

	parts = colors + styles + materials
	if not parts:
		return ""
	return " ".join(parts).lower()


def get_batch_embeds(texts, batch_size=32, device='cuda'):
	embeds = []
	siglip_model.to(device)
	siglip_model.eval()

	for i in tqdm(range(0, len(texts), batch_size)):
		batch_texts = texts[i:i + batch_size]
		inputs = siglip_tokenizer(batch_texts, padding="max_length", max_length=64, return_tensors="pt", truncation=True, return_attention_mask=True)
		inputs = {k: v.to(device) for k, v in inputs.items()}

		with torch.no_grad():
			outputs = siglip_model(**inputs)
			pooled_output = outputs.pooler_output
			embeds.append(pooled_output.cpu().numpy())

	return np.vstack(embeds)


# **********************************************************************************************************

load_dotenv(".env")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

all_assets_metadata = json.load(open(os.getenv("PTH_ASSETS_METADATA")))
all_assets_metadata_scaled = json.load(open(os.getenv("PTH_ASSETS_METADATA_SCALED")))

siglip_model = SiglipTextModel.from_pretrained("google/siglip2-so400m-patch14-384")
siglip_tokenizer = AutoTokenizer.from_pretrained("google/siglip2-so400m-patch14-384")

all_jids = []
all_sizes = []
all_style_strings = []
empty_count = 0

print("len:", len(all_assets_metadata.items()))
print("len:", len(all_assets_metadata_scaled.items()))

for key, val in tqdm(all_assets_metadata.items()):
	all_jids.append(key)
	all_sizes.append([round(elem, 2) for elem in val.get("size")])
	style_str = build_style_string(val)
	if not style_str:
		empty_count += 1
	all_style_strings.append(style_str)

print(f"Assets with empty style strings: {empty_count}/{len(all_style_strings)}")

# Show some examples
for i in range(min(5, len(all_style_strings))):
	jid = all_jids[i]
	asset = all_assets_metadata[jid]
	print(f"  [{i}] {asset.get('summary', '')[:60]}")
	print(f"       style string: \"{all_style_strings[i]}\"")

all_embeds = get_batch_embeds(all_style_strings, device=device)

# Handle scaled assets: reuse the original asset's style embedding
for key, val in tqdm(all_assets_metadata_scaled.items()):
	idx_orig_asset = all_jids.index(val.get("jid"))
	embed_orig_asset = all_embeds[idx_orig_asset]

	all_jids.append(key)
	all_sizes.append([round(elem, 2) for elem in val.get("size")])
	all_embeds = np.vstack((all_embeds, embed_orig_asset))

all_sizes = np.array(all_sizes)

model_info_style_embeds = {
	"jids": all_jids,
	"sizes": all_sizes,
	"embeds": all_embeds
}

output_path = os.getenv("PTH_ASSETS_EMBED_STYLE")
with open(output_path, 'wb') as fp:
	pickle.dump(model_info_style_embeds, fp)

print(f"Saved style embeddings to {output_path}")
print(f"Shape: {all_embeds.shape}")
