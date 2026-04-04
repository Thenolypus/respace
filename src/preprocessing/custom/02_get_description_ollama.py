"""
Step 2: Extract structured metadata from asset images using Qwen3-VL via Ollama.

Prerequisites:
  - Run organize_custom_assets.py first to create UUID folder structure
  - Each asset folder must have an image.jpg (rendered preview)
  - Ollama running with: ollama run qwen3-vl:8b

Usage:
  python 02_get_description_ollama.py --assets_dir <path_to_organized_assets> --output <path_to_output_json>
  uv run python src/preprocessing/custom/02_get_description_ollama.py --assets_dir /home/seant/MA_Repos/shared_dataset/hegias_assets --output /home/seant/MA_Repos/shared_dataset/model_info_custom_assets.json
  uv run python 02_get_description_ollama.py --assets_dir /home/seant/MA_Repos/shared_dataset/hegias_assets --output /home/seant/MA_Repos/shared_dataset/model_info_custom_assets.json

Output:
  model_info_custom_assets.json (same format as model_info_3dfuture_assets.json)
"""

import argparse
import base64
import json
import numpy as np
import trimesh
from pathlib import Path
from openai import OpenAI
import httpx
from tqdm import tqdm


OLLAMA_BASE_URL = "http://192.168.1.180:11434/v1"
OLLAMA_MODEL = "qwen3-vl:8b"

VISION_PROMPT = (
	"Please provide a concise JSON object of the furniture item in the image using "
	"'style', 'color', 'material', 'characteristics', and 'summary' as keys. "
	"Describe the style, noting any blends of design elements. "
	"Specify the materials used for different components (if applicable). "
	"List the key characteristics, including the shape, design features, and any distinctive elements or decorative accents. "
	"If there are multiple values for a key, use a list of strings. DO NOT build a nested JSON. "
	"The summary compactly captures the essence of the furniture's style, functionality, and aesthetic appeal, "
	"emphasizing its unique attributes. This description should clearly differentiate this piece from others "
	"while succinctly capturing its essential properties and we will use it for object retrieval, "
	"so it should be as accurate as possible, keyword-heavy, but just be one extremely short sentence. "
	"You are an interior designer EXPERT. Only output the JSON as a plain string and nothing else."
)


def encode_image(img_path: str) -> str:
	with open(img_path, "rb") as f:
		return base64.b64encode(f.read()).decode("utf-8")


def compute_bounding_box_sizes(scene) -> list:
	if isinstance(scene, trimesh.Scene):
		all_bounds = np.array([geom.bounds for geom in scene.geometry.values()])
		min_bounds = np.min(all_bounds[:, 0, :], axis=0)
		max_bounds = np.max(all_bounds[:, 1, :], axis=0)
	elif isinstance(scene, trimesh.Trimesh):
		min_bounds, max_bounds = scene.bounds
	else:
		raise ValueError("Input is neither trimesh.Scene nor trimesh.Trimesh")

	bbox_size = (max_bounds - min_bounds).tolist()
	return [round(x, 2) for x in bbox_size]


def get_vision_response(client: OpenAI, base64_image: str, category_hint: str = None) -> dict:
	prompt = VISION_PROMPT
	if category_hint:
		prompt = VISION_PROMPT.replace(
			"You are an interior designer EXPERT.",
			f"Hint: It's a {category_hint}. You are an interior designer EXPERT."
		)

	response = client.chat.completions.create(
		model=OLLAMA_MODEL,
		messages=[{
			"role": "user",
			"content": [
				{"type": "text", "text": prompt},
				{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
			],
		}],
		temperature=0.3,
	)

	answer = response.choices[0].message.content.strip()

	# Strip markdown code fences if present
	if answer.startswith("```"):
		lines = answer.split("\n")
		lines = [l for l in lines if not l.startswith("```")]
		answer = "\n".join(lines)

	# Strip <think>...</think> blocks (Qwen thinking tokens), including unclosed ones
	import re
	answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
	answer = re.sub(r"<think>.*", "", answer, flags=re.DOTALL).strip()

	if not answer:
		raise ValueError(f"Model returned empty/non-JSON response")

	# Try to extract JSON object if model added extra text
	json_match = re.search(r"\{.*\}", answer, flags=re.DOTALL)
	if json_match:
		answer = json_match.group()

	return json.loads(answer)


def main():
	parser = argparse.ArgumentParser(description="Extract asset metadata using Qwen3-VL via Ollama")
	parser.add_argument("--assets_dir", type=str, required=True, help="Path to organized assets (with UUID folders)")
	parser.add_argument("--output", type=str, required=True, help="Output JSON path")
	args = parser.parse_args()

	assets_dir = Path(args.assets_dir)

	# Load model_info.json (created by organize_custom_assets.py)
	model_info_path = assets_dir / "model_info.json"
	assets = json.load(open(model_info_path))
	print(f"Loaded {len(assets)} assets from {model_info_path}")

	# Load existing output if resuming
	output_path = Path(args.output)
	if output_path.exists():
		metadata = json.load(open(output_path))
		print(f"Resuming: {len(metadata)} already processed")
	else:
		metadata = {}

	client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=httpx.Timeout(300.0))

	for asset in tqdm(assets, desc="Processing assets"):
		model_id = asset["model_id"]

		if model_id in metadata:
			continue

		img_path = assets_dir / model_id / "image.jpg"
		if not img_path.exists():
			print(f"SKIP {model_id}: no image.jpg found")
			continue

		glb_path = assets_dir / model_id / "raw_model.glb"

		try:
			base64_image = encode_image(str(img_path))
			result = get_vision_response(client, base64_image, asset.get("category"))

			# Compute bounding box from mesh (more accurate than vision estimate)
			mesh = trimesh.load(str(glb_path))
			result["size"] = compute_bounding_box_sizes(mesh)

			metadata[model_id] = result

			# Save after each asset for resume support
			with open(output_path, "w") as f:
				json.dump(metadata, f, indent=4)

		except Exception as e:
			print(f"FAILED {model_id}: {e}")

	print(f"\nDone. {len(metadata)} assets processed -> {output_path}")


if __name__ == "__main__":
	main()
