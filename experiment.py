"""
Preliminary experiment: Gaze-Guided Tile Pruning on EgoExo4D
=============================================================
Take : iiith_cooking_108_5  (cooking activity, 64 extracted frames)
Gaze : general_eye_gaze_2d.csv  (x, y pixel coords in ego-camera space)

For each sampled frame we run SmolVLM-256M under 4 conditions:
  keep_ratio = 1.00  → baseline, all 17 tiles
  keep_ratio = 0.75  → drop 4 of 16 local tiles
  keep_ratio = 0.50  → drop 8 of 16 local tiles
  keep_ratio = 0.25  → drop 12 of 16 local tiles  (keep only 4 + global)

We measure total decode time and ms/token, and record the model's answer.

Results are saved to results/preliminary_experiment.csv and a summary
table is printed to stdout.

Usage:
    uv run python experiment.py
    uv run python experiment.py --n_frames 4 --prompt "Describe what you see."
"""

import argparse
import csv
import os
import time

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

from gaze.pruner import TilePruner

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TAKE          = "iiith_cooking_108_5"
DATA_ROOT     = "data/egoexo_data/takes"
MODEL_ID      = "HuggingFaceTB/SmolVLM-256M-Instruct"

# SmolVLM-256M tile layout (fixed by architecture)
N_TILES_TOTAL    = 17    # 1 global + 4×4 local
TOKENS_PER_TILE  = 64   # 8×8 after pixel shuffle
TOTAL_VIS_TOKENS = N_TILES_TOTAL * TOKENS_PER_TILE  # 1 088

# Aria RGB camera native resolution — gaze 2D coords are in this space
# (the extracted JPGs are 796×448 but gaze is projected onto the full 1408×1408 sensor)
ARIA_W = 1408
ARIA_H = 1408

KEEP_RATIOS = [1.0, 0.75, 0.50, 0.25]

RESULTS_PATH = "results/preliminary_experiment.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_id: str, device: str):
    print(f"Loading {model_id} on {device} …")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    print(f"  {TOTAL_VIS_TOKENS} visual tokens/image "
          f"({N_TILES_TOTAL} tiles × {TOKENS_PER_TILE} tok)\n")
    return processor, model


def build_inputs(processor, image: Image.Image, prompt: str, device: str) -> dict:
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(text=text, images=[image], return_tensors="pt").to(device)


def run_one(
    processor, model, device: str,
    image: Image.Image, prompt: str,
    keep_ratio: float,
    gaze_x: float, gaze_y: float,
    max_new_tokens: int = 64,
) -> dict:
    """Run a single forward pass; return timing + answer dict."""
    inputs = build_inputs(processor, image, prompt, device)
    image_token_id = model.config.image_token_id

    if keep_ratio < 1.0:
        # ── Vision encoder ──────────────────────────────────────────────────
        with torch.no_grad():
            image_hidden_states = model.model.get_image_features(
                pixel_values=inputs["pixel_values"].to(torch.float16),
                pixel_attention_mask=inputs["pixel_attention_mask"],
            ).pooler_output                       # (n_tiles, 64, 576)

        # ── Tile selection by gaze ───────────────────────────────────────────
        pruner = TilePruner(n_local_tiles_side=4, keep_ratio=keep_ratio)
        pruned_states, new_ids, new_mask, kept = pruner.prune(
            image_hidden_states=image_hidden_states,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_token_id=image_token_id,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
        )
        gen_inputs = dict(
            input_ids=new_ids,
            attention_mask=new_mask,
            image_hidden_states=pruned_states.to(device),
        )
        seq_len    = new_ids.shape[-1]
        tiles_kept = len(kept)
    else:
        gen_inputs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            pixel_attention_mask=inputs["pixel_attention_mask"],
        )
        seq_len    = inputs["input_ids"].shape[-1]
        tiles_kept = N_TILES_TOTAL

    # ── Generate ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed = time.perf_counter() - t0

    n_out  = outputs.shape[-1] - seq_len
    answer = processor.decode(outputs[0][seq_len:], skip_special_tokens=True)

    return {
        "tiles_kept":   tiles_kept,
        "tiles_total":  N_TILES_TOTAL,
        "vis_tok_kept": tiles_kept * TOKENS_PER_TILE,
        "vis_tok_total": TOTAL_VIS_TOKENS,
        "output_tokens": n_out,
        "total_s":       round(elapsed, 4),
        "ms_per_token":  round(elapsed * 1000 / max(n_out, 1), 2),
        "answer":        answer.strip(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_frames",      type=int,   default=8,
                        help="Number of frames to sample (default 8)")
    parser.add_argument("--max_new_tokens", type=int,  default=64)
    parser.add_argument("--prompt", default="What is the person doing in this scene?")
    args = parser.parse_args()

    take_dir  = os.path.join(DATA_ROOT, TAKE)
    frames_dir = os.path.join(take_dir, "frames")
    gaze_csv   = os.path.join(take_dir, "eye_gaze", "general_eye_gaze_2d.csv")

    # ── Load gaze data ───────────────────────────────────────────────────────
    gaze_df = pd.read_csv(gaze_csv).set_index("frame_num")
    print(f"Gaze CSV: {len(gaze_df)} rows  |  x ∈ [{gaze_df['x'].min():.0f}, {gaze_df['x'].max():.0f}]"
          f"  y ∈ [{gaze_df['y'].min():.0f}, {gaze_df['y'].max():.0f}]")
    print(f"Normalising gaze by Aria sensor resolution {ARIA_W}×{ARIA_H}\n")

    # ── Sample frames ────────────────────────────────────────────────────────
    # Extracted files are frame_0001.jpg … frame_0064.jpg
    # frame_NNNN.jpg → gaze frame_num = NNNN - 1
    all_frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    step = max(1, len(all_frames) // args.n_frames)
    sampled = all_frames[::step][: args.n_frames]
    print(f"Sampling {len(sampled)} / {len(all_frames)} frames: {sampled}\n")

    # ── Load model ───────────────────────────────────────────────────────────
    device = get_device()
    processor, model = load_model(MODEL_ID, device)

    # ── Run experiment ───────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    fieldnames = [
        "frame", "frame_num", "gaze_x_px", "gaze_y_px",
        "gaze_x_norm", "gaze_y_norm",
        "keep_ratio", "tiles_kept", "tiles_total",
        "vis_tok_kept", "vis_tok_total",
        "output_tokens", "total_s", "ms_per_token", "answer",
    ]
    csvfile = open(RESULTS_PATH, "w", newline="")
    writer  = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    print(f"{'Frame':<20} {'keep':>5} {'tiles':>6} {'vis_tok':>8} "
          f"{'time_s':>7} {'ms/tok':>8}  answer")
    print("-" * 100)

    for fname in sampled:
        frame_idx = int(fname.replace("frame_", "").replace(".jpg", ""))
        # Gaze CSV is at 10 fps; frames were extracted at 1 fps (every 10th gaze frame)
        # frame_0001 → gaze frame_num 0, frame_0002 → 10, frame_0003 → 20, etc.
        frame_num = (frame_idx - 1) * 10

        # Gaze lookup — clamp to [0, 1] in case coords exceed frame bounds
        if frame_num in gaze_df.index:
            gx_px = gaze_df.loc[frame_num, "x"]
            gy_px = gaze_df.loc[frame_num, "y"]
        else:
            gx_px, gy_px = FRAME_W / 2, FRAME_H / 2   # fallback: center

        gx_norm = float(min(max(gx_px / ARIA_W, 0.0), 1.0))
        gy_norm = float(min(max(gy_px / ARIA_H, 0.0), 1.0))

        image = Image.open(os.path.join(frames_dir, fname)).convert("RGB")

        for kr in KEEP_RATIOS:
            res = run_one(
                processor, model, device, image, args.prompt,
                keep_ratio=kr, gaze_x=gx_norm, gaze_y=gy_norm,
                max_new_tokens=args.max_new_tokens,
            )
            label = "baseline" if kr == 1.0 else f"gaze_{kr:.0%}"
            print(f"{fname:<20} {kr:>5.2f} {res['tiles_kept']:>6}/{res['tiles_total']}"
                  f" {res['vis_tok_kept']:>5}/{res['vis_tok_total']}"
                  f" {res['total_s']:>7.3f}s {res['ms_per_token']:>7.1f}ms"
                  f"  {res['answer'][:60]}")

            writer.writerow({
                "frame":        fname,
                "frame_num":    frame_num,
                "gaze_x_px":    round(gx_px, 1),
                "gaze_y_px":    round(gy_px, 1),
                "gaze_x_norm":  round(gx_norm, 4),
                "gaze_y_norm":  round(gy_norm, 4),
                "keep_ratio":   kr,
                **res,
            })
            csvfile.flush()

        print()   # blank line between frames

    csvfile.close()

    # ── Summary table ─────────────────────────────────────────────────────────
    df = pd.read_csv(RESULTS_PATH)
    summary = (
        df.groupby("keep_ratio")[["vis_tok_kept", "total_s", "ms_per_token"]]
        .mean()
        .sort_index(ascending=False)
        .rename(columns={
            "vis_tok_kept": "avg_vis_tokens",
            "total_s":      "avg_time_s",
            "ms_per_token": "avg_ms_per_tok",
        })
    )
    summary["speedup_vs_baseline"] = (
        summary["avg_time_s"].iloc[-1] / summary["avg_time_s"]
    ).round(2)
    print("\n===== SUMMARY (averaged over frames) =====")
    print(summary.to_string())
    print(f"\nFull results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
