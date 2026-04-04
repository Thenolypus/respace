"""
Step 3: Simplify full asset descriptions to single-word category labels using Qwen3 via Ollama.

Prerequisites:
  - Run 02_get_description_ollama.py first
  - Ollama running with: ollama run qwen3:8b

Usage:
  uv run python 03_simplify_descs_ollama.py --input <model_info_custom_assets.json> --output <model_info_custom_assets_simple_descs.json>
  uv run python src/preprocessing/custom/03_simplify_descs_ollama.py --input /home/seant/MA_Repos/shared_dataset/model_info_custom_assets.json --output /home/seant/MA_Repos/shared_dataset/model_info_custom_assets_simple_descs.json
  uv run python 03_simplify_descs_ollama.py --input /home/seant/MA_Repos/shared_dataset/model_info_custom_assets.json --output /home/seant/MA_Repos/shared_dataset/model_info_custom_assets_simple_descs.json

Output:
  model_info_custom_assets_simple_descs.json (maps summary string -> single-word category)
"""

import argparse
import json
from openai import OpenAI
from tqdm import tqdm


OLLAMA_BASE_URL = "http://192.168.1.180:11434/v1"
OLLAMA_MODEL = "qwen3:8b"

SIMPLIFY_PROMPT = (
	"The following description describes a furniture or object. "
	"Please extract the main subject such that your final output is only one word. "
	"If possible, use one of the labels from the NYU40 labels from ScanNetV2. "
	"Otherwise, just use your best judgment. Only return the label in lowercase and nothing else.\n"
	"Text: '{}'"
)


def get_simple_desc(client: OpenAI, description: str) -> str:
	response = client.chat.completions.create(
		model=OLLAMA_MODEL,
		messages=[{
			"role": "user",
			"content": SIMPLIFY_PROMPT.format(description),
		}],
		temperature=0.3,
	)
	return response.choices[0].message.content.strip().lower()


def main():
	parser = argparse.ArgumentParser(description="Simplify descriptions to category labels via Ollama")
	parser.add_argument("--input", type=str, required=True, help="Path to model_info_custom_assets.json")
	parser.add_argument("--output", type=str, required=True, help="Output simple descs JSON path")
	args = parser.parse_args()

	with open(args.input, "r") as f:
		metadata = json.load(f)

	# Load existing output if resuming
	try:
		with open(args.output, "r") as f:
			simple_descs = json.load(f)
		print(f"Resuming: {len(simple_descs)} already processed")
	except FileNotFoundError:
		simple_descs = {}

	client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

	for key, entry in tqdm(metadata.items(), desc="Simplifying descriptions"):
		summary = entry.get("summary")
		if not summary or summary in simple_descs:
			continue

		try:
			label = get_simple_desc(client, summary)
			simple_descs[summary] = label

			with open(args.output, "w") as f:
				json.dump(simple_descs, f, indent=4)

		except Exception as e:
			print(f"FAILED for '{summary[:50]}...': {e}")

	print(f"\nDone. {len(simple_descs)} entries -> {args.output}")


if __name__ == "__main__":
	main()
