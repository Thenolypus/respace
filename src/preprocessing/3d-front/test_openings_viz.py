"""
Visualize openings injected into SSR scenes.

Draws the room boundary polygon (from bounds_bottom) and overlays
door/window openings as colored rectangles, plus furniture as dots.
Outputs PNGs to a specified directory for visual inspection.

Usage (from repo root):
    uv run python src/preprocessing/3d-front/test_openings_viz.py [--n 20] [--out viz_openings]
"""

import json
import os
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from shapely.geometry import Polygon, Point


def draw_scene(scene, title, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # -- Room boundary (XZ plane) --
    bounds = scene["bounds_bottom"]
    coords_2d = [(b[0], b[2]) for b in bounds]
    poly = Polygon(coords_2d)
    if not poly.is_valid:
        poly = poly.buffer(0)

    # Draw filled polygon
    xs, zs = poly.exterior.xy
    ax.fill(xs, zs, alpha=0.08, color="gray")
    ax.plot(xs, zs, "k-", linewidth=2, label="Room boundary")

    # Mark boundary vertices
    for i, (x, z) in enumerate(coords_2d):
        ax.plot(x, z, "ks", markersize=6)
        ax.annotate(f"v{i}", (x, z), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color="black")

    # -- Furniture (objects) --
    for obj in scene.get("objects", []):
        px, py, pz = obj["pos"]
        sx, sy, sz = obj["size"]
        # Draw as small rectangle (XZ footprint)
        rect = plt.Rectangle(
            (px - sx / 2, pz - sz / 2), sx, sz,
            linewidth=0.8, edgecolor="steelblue", facecolor="steelblue",
            alpha=0.25
        )
        ax.add_patch(rect)
        ax.plot(px, pz, "o", color="steelblue", markersize=3)

    # -- Openings --
    colors = {"door": "red", "window": "blue"}
    for opening in scene.get("openings", []):
        px, py, pz = opening["pos"]
        sx, sy, sz = opening["size"]
        otype = opening["type"]
        color = colors.get(otype, "green")

        # Draw opening footprint as rectangle
        rect = plt.Rectangle(
            (px - sx / 2, pz - sz / 2), sx, sz,
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.35
        )
        ax.add_patch(rect)
        ax.plot(px, pz, "x", color=color, markersize=10, markeredgewidth=2)

        # Distance from boundary
        dist = poly.exterior.distance(Point(px, pz))
        ax.annotate(
            f"{otype}\nd={dist:.2f}",
            (px, pz), textcoords="offset points",
            xytext=(8, 8), fontsize=7, color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8)
        )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="steelblue", alpha=0.25, label="Furniture"),
        mpatches.Patch(facecolor="red", alpha=0.35, label="Door"),
        mpatches.Patch(facecolor="blue", alpha=0.35, label="Window"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of scenes to visualize")
    parser.add_argument("--out", type=str, default="viz_openings", help="Output directory")
    parser.add_argument("--scenes-dir", type=str, default="dataset-ssr3dfront-openings/scenes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)

    files = sorted([f for f in os.listdir(args.scenes_dir) if f.endswith(".json")])

    # Split into with/without openings, sample from both
    with_openings = []
    without_openings = []
    for f in files:
        with open(os.path.join(args.scenes_dir, f)) as fp:
            d = json.load(fp)
        if d.get("openings"):
            with_openings.append((f, d))
        else:
            without_openings.append((f, d))

    # Sample: mostly scenes with openings, a few without
    n_with = min(args.n - 3, len(with_openings))
    n_without = min(3, len(without_openings))
    sampled = random.sample(with_openings, n_with) + random.sample(without_openings, n_without)
    random.shuffle(sampled)

    print(f"Visualizing {len(sampled)} scenes ({n_with} with openings, {n_without} without)...")

    # Also run validation checks
    n_far = 0
    far_threshold = 0.6

    for fname, scene in sampled:
        n_openings = len(scene.get("openings", []))
        room_type = scene.get("room_type", "?")
        title = f"{fname[:40]}...\nroom={room_type}  openings={n_openings}"
        out_path = os.path.join(args.out, fname.replace(".json", ".png"))
        draw_scene(scene, title, out_path)

        # Validate distances
        bounds = scene["bounds_bottom"]
        coords_2d = [(b[0], b[2]) for b in bounds]
        poly = Polygon(coords_2d)
        if poly.is_valid:
            for op in scene.get("openings", []):
                px, pz = op["pos"][0], op["pos"][2]
                dist = poly.exterior.distance(Point(px, pz))
                if dist > far_threshold:
                    n_far += 1
                    print(f"  WARNING: {fname[:40]}... {op['type']} dist={dist:.2f} > {far_threshold}")

    print(f"\nDone. PNGs saved to {args.out}/")
    if n_far:
        print(f"  {n_far} openings found > {far_threshold}m from boundary (check these)")
    else:
        print(f"  All openings within {far_threshold}m of boundary")


if __name__ == "__main__":
    main()
