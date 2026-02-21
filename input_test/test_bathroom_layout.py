"""
Test script: Standalone test for the rule-based bathroom layout generator.
Generates fixture placements for bathroom rooms and optionally runs asset retrieval.

Usage:
  Layout only:          uv run python -m input_test.test_bathroom_layout
  With asset retrieval:  uv run python -m input_test.test_bathroom_layout --with-retrieval
  With 3D rendering:     uv run python -m input_test.test_bathroom_layout --with-retrieval --render
"""

import json
import argparse
import math
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")

from src.bathroom_layout import generate_bathroom_layout


# --------------------------------------------------------------------------- #
# Input bathroom scenes                                                        #
# --------------------------------------------------------------------------- #

BATHROOM_SETTINGS_DIR = Path("input_test/bathroom_settings")
OUTPUT_DIR = Path("input_test/output_bathroom")


def _opening_distance_to_wall(opening, bounds_bottom):
    """Compute the minimum distance from an opening's center to the nearest wall edge."""
    from shapely.geometry import Point, LineString
    ox, oz = opening["pos"][0], opening["pos"][2]
    pts = [(v[0], v[2]) for v in bounds_bottom]
    min_dist = float("inf")
    for i in range(len(pts)):
        seg = LineString([pts[i], pts[(i + 1) % len(pts)]])
        min_dist = min(min_dist, seg.distance(Point(ox, oz)))
    return min_dist


def _opening_bbox_verts(opening):
    """Compute the 4 XZ corners of an opening's bounding box (unrotated)."""
    ox, oz = opening["pos"][0], opening["pos"][2]
    sw, sd = opening["size"][0], opening["size"][2]
    hw, hd = sw / 2.0, sd / 2.0
    return [
        (ox - hw, oz - hd),
        (ox + hw, oz - hd),
        (ox + hw, oz + hd),
        (ox - hw, oz + hd),
    ]


def render_topdown_bboxes(scene, output_path, filename="floorplan_bboxes"):
    """Render a 2D top-down view with numbered colored furniture boxes + legend table,
    and viz_openings-style door/window representations."""
    objects = scene.get("objects", [])
    bounds = scene["bounds_bottom"]
    openings = scene.get("openings", [])

    fig, (ax, ax_legend) = plt.subplots(1, 2, figsize=(16, 10),
                                         gridspec_kw={"width_ratios": [3, 1]})

    # --- Draw floor polygon with vertex labels ---
    floor_verts = [(v[0], v[2]) for v in bounds]
    floor_poly = MplPolygon(floor_verts, closed=True, fill=True,
                            facecolor="#f5deb3", edgecolor="black", linewidth=2)
    ax.add_patch(floor_poly)

    for i, (vx, vz) in enumerate(floor_verts):
        ax.plot(vx, vz, "s", color="black", markersize=8, zorder=6)
        ax.annotate(f"v{i}", (vx, vz), fontsize=8, ha="left", va="top",
                    xytext=(4, -4), textcoords="offset points", fontweight="bold")

    # --- Draw openings (viz_openings style: colored rect + X marker + distance label) ---
    for opening in openings:
        verts = _opening_bbox_verts(opening)
        d_wall = _opening_distance_to_wall(opening, bounds)
        ox, oz = opening["pos"][0], opening["pos"][2]

        if opening["type"] == "door":
            color = "red"
            label = f"door\nd={d_wall:.2f}"
        else:
            color = "blue"
            label = f"window\nd={d_wall:.2f}"

        opening_poly = MplPolygon(verts, closed=True, fill=True,
                                  facecolor=(*plt.cm.colors.to_rgba(color)[:3], 0.3),
                                  edgecolor=color, linewidth=1.5, zorder=4)
        ax.add_patch(opening_poly)
        ax.plot(ox, oz, "X", color=color, markersize=12, zorder=5, markeredgewidth=2)
        ax.annotate(label, (ox, oz), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 10), textcoords="offset points", color=color,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.2",
                    facecolor="white", edgecolor=color, alpha=0.8))

    # --- Draw furniture (colored numbered boxes) ---
    colors = plt.cm.tab10.colors
    legend_entries = []

    for i, obj in enumerate(objects):
        pos = obj["pos"]
        size = obj["size"]
        rot = obj.get("rot", [0, 0, 0, 1])

        w, d = size[0], size[2]
        cx, cz = pos[0], pos[2]

        qx, qy, qz, qw = rot
        yaw = np.arctan2(2 * (qw * qy + qx * qz), 1 - 2 * (qy**2 + qz**2))

        corners = np.array([
            [-w/2, -d/2],
            [ w/2, -d/2],
            [ w/2,  d/2],
            [-w/2,  d/2],
        ])
        cos_a, sin_a = np.cos(-yaw), np.sin(-yaw)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = corners @ rot_mat.T
        rotated[:, 0] += cx
        rotated[:, 1] += cz

        color = colors[i % len(colors)]
        bbox_poly = MplPolygon(rotated, closed=True, fill=True,
                               facecolor=(*color, 0.4), edgecolor=color, linewidth=1.5, zorder=3)
        ax.add_patch(bbox_poly)

        ax.text(cx, cz, str(i), fontsize=8, ha="center", va="center",
                color="black", fontweight="bold")

        desc = obj.get("desc", f"obj_{i}")
        legend_entries.append((i, color, desc))

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Z (meters)")
    ax.set_title(f"Top-down layout: {scene.get('room_type', 'room')} ({len(objects)} objects)")
    ax.grid(True, alpha=0.3)

    # --- Build legend table on the right panel ---
    ax_legend.axis("off")
    ax_legend.set_title("Legend", fontsize=12, fontweight="bold")

    if legend_entries:
        col_labels = ["#", "Color", "Description"]
        max_desc_chars = 30
        cell_text = [
            [str(idx), "", "\n".join(textwrap.wrap(desc, max_desc_chars))]
            for idx, _, desc in legend_entries
        ]
        cell_colors = [["white", (*c, 0.4), "white"] for _, c, _ in legend_entries]

        table = ax_legend.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellColours=cell_colors,
            colColours=["#dddddd"] * 3,
            loc="upper center",
            cellLoc="left",
            colWidths=[0.08, 0.1, 0.82],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)

        for (row, col), cell in table.get_celld().items():
            n_lines = cell.get_text().get_text().count("\n") + 1
            cell.set_height(0.05 * n_lines)

    out_file = output_path / f"{filename}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved visualization to {out_file}")


def print_layout_result(scene, label):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    # Room info
    pts = scene["bounds_bottom"]
    xs = [p[0] for p in pts]
    zs = [p[2] for p in pts]
    room_w = max(xs) - min(xs)
    room_d = max(zs) - min(zs)
    print(f"  Room: {room_w:.1f}m x {room_d:.1f}m  ({len(pts)} vertices)")

    openings = scene.get("openings", [])
    for o in openings:
        print(f"  Opening: {o['type']} at pos={o['pos']}")

    print(f"\n  Placed {len(scene['objects'])} fixtures:")
    for i, obj in enumerate(scene["objects"]):
        # Extract yaw from quaternion for readability
        qx, qy, qz, qw = obj["rot"]
        yaw_deg = math.degrees(math.atan2(2 * (qw * qy + qx * qz), 1 - 2 * (qy**2 + qz**2)))

        print(f"\n  [{i}] {obj['desc']}")
        print(f"      size: [{obj['size'][0]:.2f}, {obj['size'][1]:.2f}, {obj['size'][2]:.2f}]")
        print(f"      pos:  [{obj['pos'][0]:.3f}, {obj['pos'][1]:.3f}, {obj['pos'][2]:.3f}]")
        print(f"      rot:  [{qx:.5f}, {qy:.5f}, {qz:.5f}, {qw:.5f}]  (yaw={yaw_deg:.1f} deg)")

    print()


def run_retrieval_on_scene(retrieval, scene, output_path, tag, do_render=False):
    """Run asset retrieval on a bathroom layout and optionally render."""
    import copy
    print(f"\nRunning asset retrieval for {tag}...")
    print("=" * 70)

    result = retrieval.sample_all_assets(scene, is_greedy_sampling=True)
    print("=" * 70)

    print(f"\n  Asset retrieval results for {tag}:")
    for i, obj in enumerate(result["objects"]):
        print(f"  [{i}] {obj['desc'][:60]}...")
        print(f"      Sampled: {obj.get('sampled_asset_desc', 'N/A')[:60]}...")
        print(f"      JID:     {obj.get('sampled_asset_jid', 'N/A')}")
        print(f"      Size:    {obj.get('sampled_asset_size', 'N/A')}")
        print()

    # Save retrieval result
    out_file = output_path / f"{tag}_retrieved.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved retrieval result to {out_file}")

    if do_render:
        if not result.get("bounds_bottom"):
            print("  WARNING: Skipping render -- scene has no bounds_bottom")
            return
        from src.viz import render_scene_and_export
        render_path = output_path / f"render_{tag}"
        render_path.mkdir(parents=True, exist_ok=True)
        print(f"  Rendering {tag}...")
        render_scene_and_export(
            result,
            filename=tag,
            pth_output=str(render_path),
            resolution=(1024, 1024),
            show_bboxes=False,
            show_assets=True,
            use_dynamic_zoom=True,
        )
        print(f"  Saved renders to {render_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-retrieval", action="store_true",
                        help="Also run asset retrieval on generated layouts")
    parser.add_argument("--render", action="store_true",
                        help="Render 3D scenes (requires --with-retrieval and EGL/display)")
    parser.add_argument("--input-dir", type=str, default=str(BATHROOM_SETTINGS_DIR),
                        help="Directory with bathroom SSR input JSONs")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all bathroom JSONs
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"ERROR: No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} bathroom input(s) in {input_dir}")

    # Generate layouts
    results = []
    for json_path in json_files:
        tag = json_path.stem
        print(f"\nProcessing: {json_path.name}")

        with open(json_path) as f:
            scene = json.load(f)

        result = generate_bathroom_layout(scene)
        print_layout_result(result, f"{tag} ({json_path.name})")

        # Save layout result
        out_file = OUTPUT_DIR / f"{tag}_layout.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved layout to {out_file}")

        # Render 2D top-down bbox visualization
        render_topdown_bboxes(result, OUTPUT_DIR, filename=f"{tag}_bboxes")

        results.append((tag, result))

    # Optional: asset retrieval
    if args.with_retrieval:
        import torch
        from src.sample import AssetRetrievalModule

        dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nDevice: {dvc}")
        print("Initializing AssetRetrievalModule...")

        retrieval = AssetRetrievalModule(
            lambd=0.5,
            sigma=0.05,
            temp=0.2,
            top_p=0.95,
            top_k=20,
            asset_size_threshold=0.5,
            dvc=dvc,
            do_print=True,
        )

        for tag, scene in results:
            run_retrieval_on_scene(retrieval, scene, OUTPUT_DIR, tag, do_render=args.render)

    print("\nDone!")


if __name__ == "__main__":
    main()
