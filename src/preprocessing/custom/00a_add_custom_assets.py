"""
Adds new custom assets to an existing organized 3D-FUTURE-style dataset.

Reads existing model_info.json and model_info_mapping.json from the output dir,
deduplicates by (category_folder, original_name), and only adds truly new assets.

Input structure (same as 00_organize_custom_assets.py):
  new_assets/
  ├── corridor/
  │   └── new_mirror.glb
  ├── dining/
  │   └── new_chair.glb

Usage:
  uv run python src/preprocessing/custom/00a_add_custom_assets.py \
    --input_dir /path/to/new_assets \
    --dataset_dir /home/seant/MA_Repos/shared_dataset/hegias_assets

After this, re-run steps 01-04 as normal — they skip already-processed assets.
"""

import argparse
import json
import shutil
import uuid
from pathlib import Path


def load_existing(dataset_dir: Path):
	"""Load existing model_info and mapping, or return empty defaults."""
	model_info_path = dataset_dir / "model_info.json"
	mapping_path = dataset_dir / "model_info_mapping.json"

	if model_info_path.exists():
		with open(model_info_path) as f:
			model_info = json.load(f)
	else:
		model_info = []

	if mapping_path.exists():
		with open(mapping_path) as f:
			mapping = json.load(f)
	else:
		mapping = {}

	return model_info, mapping


def get_existing_keys(mapping: dict) -> set:
	"""Build a set of (category_folder, original_name) from existing mapping for dedup."""
	keys = set()
	for entry in mapping.values():
		keys.add((entry["category_folder"], entry["original_name"]))
	return keys


def main():
	parser = argparse.ArgumentParser(description="Add new assets to existing organized dataset")
	parser.add_argument("--input_dir", type=str, required=True, help="Path to folder with new .glb files in category subfolders")
	parser.add_argument("--dataset_dir", type=str, required=True, help="Path to existing organized dataset (hegias_assets)")
	args = parser.parse_args()

	input_dir = Path(args.input_dir)
	dataset_dir = Path(args.dataset_dir)

	if not dataset_dir.exists():
		raise FileNotFoundError(f"Dataset dir does not exist: {dataset_dir}")

	model_info, mapping = load_existing(dataset_dir)
	existing_keys = get_existing_keys(mapping)

	print(f"Existing dataset: {len(model_info)} assets")

	# Collect new .glb files
	glb_files = sorted(input_dir.rglob("*.glb"))
	print(f"Found {len(glb_files)} .glb files in {input_dir}")

	added = 0
	skipped = 0

	for glb_path in glb_files:
		rel = glb_path.relative_to(input_dir)
		category = rel.parent.as_posix()
		original_name = glb_path.stem

		key = (category, original_name)
		if key in existing_keys:
			print(f"  SKIP (duplicate): {category}/{original_name}.glb")
			skipped += 1
			continue

		model_id = str(uuid.uuid4())

		# Copy to UUID folder
		dest_dir = dataset_dir / model_id
		dest_dir.mkdir(exist_ok=True)
		shutil.copy2(glb_path, dest_dir / "raw_model.glb")

		# Append to model_info
		model_info.append({
			"model_id": model_id,
			"category": category,
		})

		# Append to mapping
		mapping[model_id] = {
			"original_path": str(glb_path),
			"original_name": original_name,
			"category_folder": category,
		}

		existing_keys.add(key)
		added += 1
		print(f"  ADD: {category}/{original_name}.glb -> {model_id}/raw_model.glb")

	# Write updated files
	with open(dataset_dir / "model_info.json", "w") as f:
		json.dump(model_info, f, indent=4)

	with open(dataset_dir / "model_info_mapping.json", "w") as f:
		json.dump(mapping, f, indent=4)

	print(f"\nDone. Added {added}, skipped {skipped} duplicates.")
	print(f"Total assets now: {len(model_info)}")


if __name__ == "__main__":
	main()
