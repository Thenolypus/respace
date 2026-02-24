"""
Step 4: Generate 10 prompt variations per asset using Qwen3 via Ollama.

Prerequisites:
  - Run 02_get_description_ollama.py first
  - Ollama running with: ollama run qwen3:8b

Usage:
  python 04_get_prompts_ollama.py --input <model_info_custom_assets.json> --output <model_info_custom_assets_prompts.json>
uv run python src/preprocessing/custom/04_get_prompts_ollama.py --input /home/seant/MA_Repos/shared_dataset/model_info_custom_assets.json --output /home/seant/MA_Repos/shared_dataset/model_info_custom_assets_prompts.json

Output:
  model_info_custom_assets_prompts.json (maps asset JID -> list of 10 prompt strings)
"""

import argparse
import json
import re
from openai import OpenAI
from tqdm import tqdm


OLLAMA_BASE_URL = "http://192.168.1.180:11434/v1"
OLLAMA_MODEL = "qwen3:8b"

N_PROMPTS_PER_OBJ = 10

PROMPT_PREFIX = (
	"The list below contains a sentence referencing to a single piece of furniture. "
	"Your task it to create a list of 10 short descriptions that vary in length. "
	"Each description refers to the subject with a maximum of 3-4 additional descriptive words "
	"that reference the colour, style, shape, etc. All your sentences should be in 'noun phrase'. "
	"You MUST include a variety of lengths in your descriptions, ensuring a few samples are very short "
	"(1-2 words max) and others are longer (4-5 words). Have at least one sample with only one word, "
	"except if you need to be more specific for the subject, e.g. use 'Coffee Table', not just 'Table', "
	"if present. Use mostly basic properties such as colour or material, but also include a few creative "
	"and diverse versions to increase robustness in our ML training dataset.\n\n"
	"The sentence is:\n\n"
)

PROMPT_POSTFIX = (
	f"\nYou MUST output EXACTLY a JSON array of {N_PROMPTS_PER_OBJ} strings. "
	"The output MUST start with [ and end with ]. Example format: [\"Chair\", \"Blue Chair\", \"Modern Blue Chair\"]. "
	"You MUST always point to the referenced object above and not hallucinate other furniture or be overly generic "
	"by using 'furniture' or 'piece'. Every list contains the descriptions in increasing word length. "
	"Do NOT output anything else. No markdown, no explanation, no numbered lists. ONLY the JSON array."
)


MAX_RETRIES = 3


def parse_response(answer: str) -> list:
	# Strip <think>...</think> blocks (Qwen thinking tokens), including unclosed ones
	answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
	answer = re.sub(r"<think>.*", "", answer, flags=re.DOTALL).strip()

	# Strip markdown code fences if present
	if answer.startswith("```"):
		lines = answer.split("\n")
		lines = [l for l in lines if not l.startswith("```")]
		answer = "\n".join(lines)

	if not answer:
		raise ValueError("Model returned empty/non-JSON response")

	# Extract JSON array if model added extra text
	json_match = re.search(r"\[.*\]", answer, flags=re.DOTALL)
	if json_match:
		answer = json_match.group()
	elif answer.startswith('"'):
		# Model output comma-separated strings without brackets, wrap them
		answer = f"[{answer}]"

	parsed = json.loads(answer)

	if not isinstance(parsed, list) or len(parsed) != N_PROMPTS_PER_OBJ:
		raise ValueError(f"Expected list of {N_PROMPTS_PER_OBJ}, got {type(parsed).__name__} len={len(parsed) if isinstance(parsed, list) else 'N/A'}")

	return [p.lower() for p in parsed]


def get_prompts(client: OpenAI, summary: str, jid: str) -> list:
	full_prompt = f"{PROMPT_PREFIX}- {summary}\n{PROMPT_POSTFIX}"

	for attempt in range(1, MAX_RETRIES + 1):
		response = client.chat.completions.create(
			model=OLLAMA_MODEL,
			messages=[{
				"role": "user",
				"content": full_prompt,
			}],
			temperature=0.7,
		)

		raw_answer = response.choices[0].message.content.strip()

		try:
			return parse_response(raw_answer)
		except Exception as e:
			preview = raw_answer[:200].replace('\n', ' ')
			print(f"  RETRY {attempt}/{MAX_RETRIES} {jid}: {e} | Raw: {preview}")
			if attempt == MAX_RETRIES:
				raise


def main():
	parser = argparse.ArgumentParser(description="Generate prompt variations via Ollama")
	parser.add_argument("--input", type=str, required=True, help="Path to model_info_custom_assets.json")
	parser.add_argument("--output", type=str, required=True, help="Output prompts JSON path")
	args = parser.parse_args()

	with open(args.input, "r") as f:
		metadata = json.load(f)

	# Load existing output if resuming
	try:
		with open(args.output, "r") as f:
			prompts = json.load(f)
		print(f"Resuming: {len(prompts)} already processed")
	except FileNotFoundError:
		prompts = {}

	client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

	for jid, entry in tqdm(metadata.items(), desc="Generating prompts"):
		if jid in prompts:
			continue

		summary = entry.get("summary")
		if not summary:
			continue

		try:
			prompts[jid] = get_prompts(client, summary, jid)

			with open(args.output, "w") as f:
				json.dump(prompts, f, indent=4)

		except Exception as e:
			print(f"FAILED {jid}: {e}")

	print(f"\nDone. {len(prompts)} assets -> {args.output}")


if __name__ == "__main__":
	main()
