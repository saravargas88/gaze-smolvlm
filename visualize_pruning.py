"""
visualize_pruning.py
====================
Visualise which SmolVLM tiles are kept vs. pruned for a given frame and gaze point.

No model loading needed — works directly from images and the gaze CSV.

Output: one PNG per frame, saved to results/viz/<take>/

Usage:
    uv run python visualize_pruning.py                        # all frames, both takes
    uv run python visualize_pruning.py --takes iiith_cooking_108_5
    uv run python visualize_pruning.py --frame frame_0001.jpg --take iiith_cooking_108_5
    uv run python visualize_pruning.py --n_frames 8

Layout of each output image
----------------------------
  A 1×4 strip of the same frame for keep_ratio ∈ {1.00, 0.75, 0.50, 0.25}.

  Tile colours:
    • Blue   — global tile (tile 0, always kept)
    • Green  — kept local tile
    • Red    — pruned local tile
  Gaze point: cyan crosshair + filled circle
  Each tile shows its Gaussian score (0.00 – 1.00)
"""

import argparse
import math
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from gaze.pruner import TilePruner

# ── Constants ────────────────────────────────────────────────────────────────
ALL_TAKES   = ["iiith_cooking_108_5", "nus_cpr_27_3"]
DATA_ROOT   = "data/egoexo_data/takes"
ARIA_W      = 1408
ARIA_H      = 1408
KEEP_RATIOS = [1.00, 0.75, 0.50, 0.25]
SIGMA       = 0.3   # Gaussian spread used by TilePruner


# ── Tile geometry ─────────────────────────────────────────────────────────────

def infer_tile_grid(img_w: int, img_h: int, n_local: int = 12) -> tuple[int, int]:
    """
    Return (grid_cols, grid_rows) for the local tile layout.

    SmolVLM maximises tile count up to max_image_tiles, then picks the
    (cols, rows) factorisation of n_local whose aspect ratio best matches
    the image.  For 796×448 with n_local=12 this gives (4, 3).

    Args:
        n_local: total number of local tiles (default 12, confirmed for 796×448).
                 Override with --n_local_tiles if you run on different footage.
    """
    ratio = img_w / img_h
    best, best_diff = (1, n_local), float("inf")
    for cols in range(1, n_local + 1):
        if n_local % cols == 0:
            rows = n_local // cols
            diff = abs(cols / rows - ratio)
            if diff < best_diff:
                best_diff = diff
                best = (cols, rows)
    return best   # (cols, rows)


def tile_boxes_on_image(img_w: int, img_h: int, grid_cols: int, grid_rows: int):
    """
    Return list of (x0, y0, x1, y1) pixel boxes — one per local tile,
    in row-major order — covering the full image.

    Tile (row=r, col=c) is local tile index  r * grid_cols + c  (0-based).
    In TilePruner's image_hidden_states, tile 0 is global and local tiles
    start at index 1, so local tile i → hidden-states index i+1.
    """
    boxes = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            x0 = round(c * img_w / grid_cols)
            y0 = round(r * img_h / grid_rows)
            x1 = round((c + 1) * img_w / grid_cols)
            y1 = round((r + 1) * img_h / grid_rows)
            boxes.append((x0, y0, x1, y1))
    return boxes   # length = grid_cols * grid_rows


# ── Scoring (mirrors TilePruner.tile_scores but with the actual grid) ─────────

def score_tiles(grid_cols: int, grid_rows: int,
                gaze_x: float, gaze_y: float,
                sigma: float = SIGMA) -> np.ndarray:
    """
    Gaussian score for each local tile given the actual (non-square) grid.

    Returns shape (grid_rows, grid_cols) array of scores in [0, 1].
    """
    scores = np.zeros((grid_rows, grid_cols))
    for r in range(grid_rows):
        for c in range(grid_cols):
            # Tile centre in normalised image coordinates
            cx = (c + 0.5) / grid_cols
            cy = (r + 0.5) / grid_rows
            dist_sq = (cx - gaze_x) ** 2 + (cy - gaze_y) ** 2
            scores[r, c] = math.exp(-dist_sq / (2 * sigma ** 2))
    return scores


def select_tiles(scores_2d: np.ndarray, keep_ratio: float,
                 min_local: int = 2) -> list[int]:
    """
    Return flat 0-based local tile indices to KEEP (row-major).
    Always returns at least min_local tiles.
    """
    n_local = scores_2d.size
    n_keep  = max(min_local, int(round(n_local * keep_ratio)))
    flat    = scores_2d.ravel()
    order   = np.argsort(flat)[::-1]   # descending score
    return sorted(order[:n_keep].tolist())


# ── Single-frame visualisation ────────────────────────────────────────────────

def visualise_frame(
    image_path: str,
    gaze_x_norm: float,
    gaze_y_norm: float,
    out_path: str,
    keep_ratios: list[float] = KEEP_RATIOS,
    n_local: int = 12,
):
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    img_arr = np.array(img)

    grid_cols, grid_rows = infer_tile_grid(img_w, img_h, n_local=n_local)
    n_local = grid_cols * grid_rows
    boxes   = tile_boxes_on_image(img_w, img_h, grid_cols, grid_rows)
    scores  = score_tiles(grid_cols, grid_rows, gaze_x_norm, gaze_y_norm)

    n_ratios = len(keep_ratios)
    fig, axes = plt.subplots(
        1, n_ratios,
        figsize=(5 * n_ratios, 5),
        dpi=120,
    )
    if n_ratios == 1:
        axes = [axes]

    gaze_px_x = gaze_x_norm * img_w
    gaze_px_y = gaze_y_norm * img_h

    for ax, kr in zip(axes, keep_ratios):
        ax.imshow(img_arr)
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)   # y-axis: 0 at top
        ax.axis("off")

        # ── Global tile border (full image outline) ───────────────────────────
        ax.add_patch(mpatches.FancyBboxPatch(
            (2, 2), img_w - 4, img_h - 4,
            boxstyle="square,pad=0",
            linewidth=2.5, edgecolor="#1e90ff",
            facecolor="none", zorder=3,
        ))
        ax.text(6, 14, "G", fontsize=7, color="#1e90ff",
                fontweight="bold", zorder=4)

        # ── Local tile overlays ───────────────────────────────────────────────
        if kr < 1.0:
            kept_idx = select_tiles(scores, keep_ratio=kr)
        else:
            kept_idx = list(range(n_local))   # keep all

        for flat_i, (x0, y0, x1, y1) in enumerate(boxes):
            bw, bh = x1 - x0, y1 - y0
            is_kept = flat_i in kept_idx
            score   = scores.ravel()[flat_i]

            facecolor = (0.0, 0.9, 0.2, 0.25) if is_kept else (1.0, 0.1, 0.1, 0.30)
            edgecolor = "#00cc44"              if is_kept else "#cc0000"

            ax.add_patch(mpatches.Rectangle(
                (x0, y0), bw, bh,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.5, zorder=2,
            ))

            # Tile score label
            ax.text(
                x0 + bw / 2, y0 + bh / 2,
                f"{score:.2f}",
                fontsize=7, ha="center", va="center",
                color="white",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="black", alpha=0.45, linewidth=0),
                zorder=5,
            )

            # Tile number (top-left corner of tile)
            ax.text(
                x0 + 4, y0 + 11,
                str(flat_i + 1),          # 1-based local index
                fontsize=6, color="white", alpha=0.8, zorder=5,
            )

        # ── Gaze point ───────────────────────────────────────────────────────
        cs = 18   # crosshair half-size in px
        ax.plot([gaze_px_x - cs, gaze_px_x + cs], [gaze_px_y, gaze_px_y],
                color="cyan", lw=1.5, zorder=6)
        ax.plot([gaze_px_x, gaze_px_x], [gaze_px_y - cs, gaze_px_y + cs],
                color="cyan", lw=1.5, zorder=6)
        ax.scatter([gaze_px_x], [gaze_px_y], s=50, color="cyan",
                   zorder=7, edgecolors="black", linewidths=0.8)

        # ── Title ─────────────────────────────────────────────────────────────
        if kr == 1.0:
            n_kept_local = n_local
            label = "Baseline (keep_ratio=1.00)"
        else:
            n_kept_local = len(kept_idx)
            label = f"keep_ratio={kr:.2f}"
        total_tiles = 1 + n_local        # global + all local
        kept_tiles  = 1 + n_kept_local   # global always kept
        ax.set_title(
            f"{label}\n{kept_tiles}/{total_tiles} tiles  "
            f"({kept_tiles * 64} / {total_tiles * 64} vis-tokens)",
            fontsize=9,
        )

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor="#1e90ff", alpha=0.7, label="global tile (always kept)"),
        mpatches.Patch(facecolor="#00cc44", alpha=0.5, label="kept local tile"),
        mpatches.Patch(facecolor="#cc0000", alpha=0.5, label="pruned local tile"),
        mpatches.Patch(facecolor="cyan",    alpha=0.8, label="gaze point"),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.03),
        frameon=True,
    )

    fname = os.path.splitext(os.path.basename(image_path))[0]
    fig.suptitle(
        f"{fname}  ·  gaze ({gaze_x_norm:.3f}, {gaze_y_norm:.3f})  ·  "
        f"grid {grid_cols}×{grid_rows} local tiles  σ={SIGMA}",
        fontsize=10, y=1.01,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── Per-take runner ───────────────────────────────────────────────────────────

def run_take(take: str, args):
    take_dir   = os.path.join(DATA_ROOT, take)
    frames_dir = os.path.join(take_dir, "frames")
    gaze_csv   = os.path.join(take_dir, "eye_gaze", "general_eye_gaze_2d.csv")

    if not os.path.isdir(frames_dir):
        print(f"[SKIP] {take}: no frames directory found at {frames_dir}")
        return

    gaze_df = pd.read_csv(gaze_csv).set_index("frame_num")

    all_frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))

    # Filter to a specific frame if requested
    if args.frame:
        if args.frame in all_frames:
            sampled = [args.frame]
        else:
            print(f"[WARN] {args.frame} not found in {frames_dir}")
            return
    elif args.n_frames == 0 or args.n_frames >= len(all_frames):
        sampled = all_frames
    else:
        step    = max(1, len(all_frames) // args.n_frames)
        sampled = all_frames[::step][: args.n_frames]

    print(f"\nTake: {take}  ({len(sampled)} frames → results/viz/{take}/)")

    for fname in sampled:
        frame_idx = int(fname.replace("frame_", "").replace(".jpg", ""))
        frame_num = (frame_idx - 1) * 10

        if frame_num in gaze_df.index:
            gx_px = gaze_df.loc[frame_num, "x"]
            gy_px = gaze_df.loc[frame_num, "y"]
        else:
            gx_px, gy_px = ARIA_W / 2, ARIA_H / 2

        gx_norm = float(min(max(gx_px / ARIA_W, 0.0), 1.0))
        gy_norm = float(min(max(gy_px / ARIA_H, 0.0), 1.0))

        image_path = os.path.join(frames_dir, fname)
        out_name   = fname.replace(".jpg", "_pruning.png")
        out_path   = os.path.join("results", "viz", take, out_name)

        visualise_frame(
            image_path=image_path,
            gaze_x_norm=gx_norm,
            gaze_y_norm=gy_norm,
            out_path=out_path,
            keep_ratios=args.keep_ratios,
            n_local=args.n_local_tiles,
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--takes",  nargs="+", default=ALL_TAKES)
    parser.add_argument("--take",   default=None,
                        help="Shorthand for --takes with a single take")
    parser.add_argument("--frame",  default=None,
                        help="Visualise a specific frame only, e.g. frame_0001.jpg")
    parser.add_argument("--n_frames", type=int, default=0,
                        help="Number of frames to sample; 0 = all (default)")
    parser.add_argument("--keep_ratios", nargs="+", type=float,
                        default=KEEP_RATIOS)
    parser.add_argument("--n_local_tiles", type=int, default=12,
                        help="Number of local tiles (default 12 for 796×448 frames)")
    args = parser.parse_args()

    if args.take:
        args.takes = [args.take]

    for take in args.takes:
        run_take(take, args)

    print("\nDone. Open results/viz/ to inspect the tile visualisations.")


if __name__ == "__main__":
    main()
