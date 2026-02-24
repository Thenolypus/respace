"""
Organizes custom assets from category-based folder structure into 3D-FUTURE UUID-based structure.

Input structure:
  custom_assets/
  ├── corridor/
  │   └── mirror.glb
  │   └── bench.glb
  ├── dining/
  │   └── chair.glb

Output structure:
  <output_dir>/
  ├── <uuid-1>/
  │   └── raw_model.glb
  ├── <uuid-2>/
  │   └── raw_model.glb
  ├── model_info.json          (list format, input for 02_get_sg_description_gpt4.py)
  ├── model_info_mapping.json  (uuid -> original path + category, for your reference)

uv run python src/preprocessing/custom/00_organize_custom_assets.py \
  --input_dir /home/seant/MA_Repos/shared_dataset/hegias_assets_raw \
  --output_dir /home/seant/MA_Repos/shared_dataset/hegias_assets

"""

import argparse
import json
import shutil
import uuid
from pathlib import Path


def main():
	parser = argparse.ArgumentParser(description="Organize custom assets into 3D-FUTURE structure")
	parser.add_argument("--input_dir", type=str, required=True, help="Path to custom_assets/ folder")
	parser.add_argument("--output_dir", type=str, required=True, help="Path to output 3D-FUTURE-style folder")
	args = parser.parse_args()

	input_dir = Path(args.input_dir)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# Collect all .glb files
	glb_files = sorted(input_dir.rglob("*.glb"))
	print(f"Found {len(glb_files)} .glb files in {input_dir}")

	model_info_list = []  # list format for 02_get_sg_description_gpt4.py
	mapping = {}  # uuid -> original info, for reference

	for glb_path in glb_files:
		# Category = parent folder name relative to input_dir
		rel = glb_path.relative_to(input_dir)
		category = rel.parent.as_posix()  # e.g. "corridor" or "dining"
		original_name = glb_path.stem  # e.g. "mirror", "bench"

		model_id = str(uuid.uuid4())

		# Create UUID folder and copy
		dest_dir = output_dir / model_id
		dest_dir.mkdir(exist_ok=True)
		dest_file = dest_dir / "raw_model.glb"
		shutil.copy2(glb_path, dest_file)

		# model_info.json entry (matches 3D-FUTURE format expected by 02_*)
		model_info_list.append({
			"model_id": model_id,
			"category": category,  # use folder name as category hint for GPT-4
		})

		mapping[model_id] = {
			"original_path": str(glb_path),
			"original_name": original_name,
			"category_folder": category,
		}

		print(f"  {category}/{original_name}.glb -> {model_id}/raw_model.glb")

	# Write model_info.json (list format, consumed by 02_get_sg_description_gpt4.py)
	with open(output_dir / "model_info.json", "w") as f:
		json.dump(model_info_list, f, indent=4)

	# Write mapping for your reference
	with open(output_dir / "model_info_mapping.json", "w") as f:
		json.dump(mapping, f, indent=4)

	print(f"\nDone. {len(glb_files)} assets organized into {output_dir}")
	print(f"  model_info.json        - {len(model_info_list)} entries (input for description script)")
	print(f"  model_info_mapping.json - UUID to original path mapping")


if __name__ == "__main__":
	main()
