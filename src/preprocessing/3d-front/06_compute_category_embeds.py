"""
Pre-compute category-only SigLIP2 embeddings for all assets.

For each asset JID, looks up its summary in the simple_descs mapping
(full description -> category word, e.g. "sofa", "table", "lamp") and
embeds that category string with SigLIP2.

This produces a catalog where both the stored embeddings AND the runtime
queries are category-only, avoiding the noise from comparing a short
category query against full-description embeddings.

Usage:
  uv run python -m src.preprocessing.3d-front.06_compute_category_embeds
"""

import torch
import numpy as np
import pickle
import json
from transformers import AutoTokenizer, SiglipTextModel
from tqdm import tqdm
from dotenv import load_dotenv
import os


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
simple_descs = json.load(open(os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS")))

siglip_model = SiglipTextModel.from_pretrained("google/siglip2-so400m-patch14-384")
siglip_tokenizer = AutoTokenizer.from_pretrained("google/siglip2-so400m-patch14-384")

all_jids = []
all_sizes = []
all_category_strings = []
missing_count = 0

print("len:", len(all_assets_metadata.items()))
print("len:", len(all_assets_metadata_scaled.items()))

for key, val in tqdm(all_assets_metadata.items()):
	all_jids.append(key)
	all_sizes.append([round(elem, 2) for elem in val.get("size")])

	summary = val.get("summary", "")
	category = simple_descs.get(summary)
	if category is None:
		missing_count += 1
		# fall back to the full summary if no category mapping exists
		category = summary
	all_category_strings.append(category)

print(f"Assets without category mapping: {missing_count}/{len(all_category_strings)}")

# Show some examples
for i in range(min(10, len(all_category_strings))):
	jid = all_jids[i]
	asset = all_assets_metadata[jid]
	print(f"  [{i}] {asset.get('summary', '')[:60]}")
	print(f"       category: \"{all_category_strings[i]}\"")

all_embeds = get_batch_embeds(all_category_strings, device=device)

# Handle scaled assets: reuse the original asset's category embedding
for key, val in tqdm(all_assets_metadata_scaled.items()):
	idx_orig_asset = all_jids.index(val.get("jid"))
	embed_orig_asset = all_embeds[idx_orig_asset]

	all_jids.append(key)
	all_sizes.append([round(elem, 2) for elem in val.get("size")])
	all_embeds = np.vstack((all_embeds, embed_orig_asset))

all_sizes = np.array(all_sizes)

model_info_category_embeds = {
	"jids": all_jids,
	"sizes": all_sizes,
	"embeds": all_embeds
}

output_path = os.getenv("PTH_ASSETS_EMBED_CATEGORY")
with open(output_path, 'wb') as fp:
	pickle.dump(model_info_category_embeds, fp)

print(f"Saved category embeddings to {output_path}")
print(f"Shape: {all_embeds.shape}")
