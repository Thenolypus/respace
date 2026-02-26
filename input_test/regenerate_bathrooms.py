"""
Re-run generate_bathroom_layout on all bathroom generated_scene.json files
under input_test/3d_front/ to pick up updated FIXTURES (sizes, descriptions)
and remove mirror/rug fixtures.

Usage:
    uv run python -m input_test.regenerate_bathrooms
    uv run python -m input_test.regenerate_bathrooms --dry-run
"""

import json
import argparse
from pathlib import Path

from src.bathroom_layout import generate_bathroom_layout


def main():
    parser = argparse.ArgumentParser(description="Regenerate all bathroom layouts.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing files")
    args = parser.parse_args()

    base = Path("input_test/3d_front")
    bathroom_scenes = sorted(base.rglob("*bathroom*/generated_scene.json"))

    if not bathroom_scenes:
        print("No bathroom generated_scene.json files found.")
        return

    print(f"Found {len(bathroom_scenes)} bathroom scene(s):\n")
    for p in bathroom_scenes:
        print(f"  {p}")
    print()

    for scene_path in bathroom_scenes:
        print(f"{'='*70}")
        print(f"Processing: {scene_path}")

        with open(scene_path) as f:
            scene = json.load(f)

        # Keep only the structural data needed for layout generation
        input_scene = {
            "room_type": scene.get("room_type", "bathroom"),
            "bounds_top": scene["bounds_top"],
            "bounds_bottom": scene["bounds_bottom"],
            "openings": scene.get("openings", []),
            "objects": [],  # clear old objects so layout generator starts fresh
        }

        try:
            result = generate_bathroom_layout(input_scene)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        n_objects = len(result.get("objects", []))
        print(f"  Placed {n_objects} fixtures:")
        for i, obj in enumerate(result["objects"]):
            print(f"    [{i}] {obj['desc'][:50]:50s}  size={obj['size']}")

        if args.dry_run:
            print("  (dry-run, not writing)")
        else:
            with open(scene_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Written: {scene_path}")

        print()

    print("Done!")


if __name__ == "__main__":
    main()
