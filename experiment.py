"""
Preliminary experiment: Gaze-Guided Tile Pruning on EgoExo4D
=============================================================
Supported takes:
  iiith_cooking_108_5   (cooking activity, 64 extracted frames)
  nus_cpr_27_3          (CPR activity,     50 extracted frames)

For each sampled frame we run SmolVLM-256M under 4 conditions:
  keep_ratio = 1.00  → baseline, all tiles kept
  keep_ratio = 0.75  → drop ~25% of local tiles
  keep_ratio = 0.50  → drop ~50% of local tiles
  keep_ratio = 0.25  → drop ~75% of local tiles

We measure total decode time and ms/token, and record the model's answer.

Results are saved to results/<take>_experiment.csv and a summary
table is printed to stdout.

Usage:
    python experiment.py                          # both takes, all frames
    python experiment.py --takes iiith_cooking_108_5
    python experiment.py --takes iiith_cooking_108_5 nus_cpr_27_3
    python experiment.py --n_frames 8 --prompt "Describe what you see."
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
ALL_TAKES     = ["iiith_cooking_108_5", "nus_cpr_27_3"]
DATA_ROOT     = "data/egoexo_data/takes"
MODEL_ID      = "HuggingFaceTB/SmolVLM-256M-Instruct"

# SmolVLM-256M: 64 tokens per tile (8×8 after pixel shuffle)
TOKENS_PER_TILE = 64

# Aria RGB camera native resolution — gaze 2D coords are in this space
# (extracted frames are 796×448 but gaze is projected onto 1408×1408 sensor)
ARIA_W = 1408
ARIA_H = 1408

KEEP_RATIOS = [1.0, 0.75, 0.50, 0.25]

os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def capture_image_features(model, inputs: dict) -> torch.Tensor:
    """
    Capture connector output via a forward hook.

    We hook into model.model.connector and fire a 1-token generate pass.
    This uses the identical code path as the baseline and works across
    transformers versions (tested on 4.49.0).

    Returns shape: (n_tiles, tokens_per_tile, llm_hidden_dim)  e.g. (13, 64, 576)
    """
    holder = {}

    def _hook(module, inp, out):
        holder["features"] = out.detach().clone()

    handle = model.model.connector.register_forward_hook(_hook)
    with torch.no_grad():
        model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            pixel_attention_mask=inputs["pixel_attention_mask"],
            max_new_tokens=1,
            do_sample=False,
        )
    handle.remove()
    return holder["features"]   # (n_tiles, 64, 576)


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
        # ── Capture image features via hook ──────────────────────────────────
        image_hidden_states = capture_image_features(model, inputs)  # (n_tiles, 64, 576)

        # ── Infer actual tile layout from captured features ──────────────────
        n_actual_tiles = image_hidden_states.shape[0]   # e.g. 13
        n_local        = n_actual_tiles - 1              # subtract global tile
        n_local_side   = max(1, int(n_local ** 0.5))     # e.g. sqrt(12) ≈ 3

        # ── Tile selection by gaze ───────────────────────────────────────────
        pruner = TilePruner(n_local_tiles_side=n_local_side, keep_ratio=keep_ratio)
        pruned_states, new_ids, new_mask, kept = pruner.prune(
            image_hidden_states=image_hidden_states,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_token_id=image_token_id,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
        )
        gen_inputs = dict(
            input_ids=new_ids.to(device),
            attention_mask=new_mask.to(device),
            image_hidden_states=pruned_states.to(device),
        )
        seq_len    = new_ids.shape[-1]
        tiles_kept = len(kept)
        tiles_total = n_actual_tiles
    else:
        gen_inputs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            pixel_attention_mask=inputs["pixel_attention_mask"],
        )
        seq_len    = inputs["input_ids"].shape[-1]
        # infer tile count from sequence length (image tokens ÷ 64)
        n_img_tok  = (inputs["input_ids"][0] == image_token_id).sum().item()
        tiles_total = n_img_tok // TOKENS_PER_TILE
        tiles_kept  = tiles_total

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
        "tiles_kept":    tiles_kept,
        "tiles_total":   tiles_total,
        "vis_tok_kept":  tiles_kept  * TOKENS_PER_TILE,
        "vis_tok_total": tiles_total * TOKENS_PER_TILE,
        "output_tokens": n_out,
        "total_s":       round(elapsed, 4),
        "ms_per_token":  round(elapsed * 1000 / max(n_out, 1), 2),
        "answer":        answer.strip(),
    }


# ---------------------------------------------------------------------------
# Per-take runner
# ---------------------------------------------------------------------------

def run_take(take: str, processor, model, device: str, args) -> str:
    """Run the experiment for a single take. Returns path to CSV."""
    take_dir   = os.path.join(DATA_ROOT, take)
    frames_dir = os.path.join(take_dir, "frames")
    gaze_csv   = os.path.join(take_dir, "eye_gaze", "general_eye_gaze_2d.csv")

    if not os.path.isdir(frames_dir):
        print(f"[SKIP] {take}: no frames directory at {frames_dir}")
        return None

    # ── Load gaze data ───────────────────────────────────────────────────────
    gaze_df = pd.read_csv(gaze_csv).set_index("frame_num")
    print(f"\n{'='*60}")
    print(f"Take: {take}")
    print(f"Gaze CSV: {len(gaze_df)} rows  |  "
          f"x ∈ [{gaze_df['x'].min():.0f}, {gaze_df['x'].max():.0f}]  "
          f"y ∈ [{gaze_df['y'].min():.0f}, {gaze_df['y'].max():.0f}]")
    print(f"Normalising gaze by Aria sensor resolution {ARIA_W}×{ARIA_H}")

    # ── Sample frames ─────────────────────────────────────────────────────────
    all_frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    if args.n_frames == 0 or args.n_frames >= len(all_frames):
        sampled = all_frames
    else:
        step    = max(1, len(all_frames) // args.n_frames)
        sampled = all_frames[::step][: args.n_frames]
    print(f"Using {len(sampled)} / {len(all_frames)} frames\n")

    # ── CSV output ────────────────────────────────────────────────────────────
    results_path = f"results/{take}_experiment.csv"
    fieldnames = [
        "take", "frame", "frame_num", "gaze_x_px", "gaze_y_px",
        "gaze_x_norm", "gaze_y_norm",
        "keep_ratio", "tiles_kept", "tiles_total",
        "vis_tok_kept", "vis_tok_total",
        "output_tokens", "total_s", "ms_per_token", "answer",
    ]
    csvfile = open(results_path, "w", newline="")
    writer  = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    print(f"{'Frame':<20} {'keep':>5} {'tiles':>8} {'vis_tok':>10} "
          f"{'time_s':>7} {'ms/tok':>8}  answer")
    print("-" * 100)

    for fname in sampled:
        frame_idx = int(fname.replace("frame_", "").replace(".jpg", ""))
        # Gaze CSV is at 10 fps; frames were extracted at 1 fps (every 10th gaze frame)
        frame_num = (frame_idx - 1) * 10

        # Gaze lookup — fallback to image center if missing
        if frame_num in gaze_df.index:
            gx_px = gaze_df.loc[frame_num, "x"]
            gy_px = gaze_df.loc[frame_num, "y"]
        else:
            gx_px, gy_px = ARIA_W / 2, ARIA_H / 2

        gx_norm = float(min(max(gx_px / ARIA_W, 0.0), 1.0))
        gy_norm = float(min(max(gy_px / ARIA_H, 0.0), 1.0))

        image = Image.open(os.path.join(frames_dir, fname)).convert("RGB")

        for kr in KEEP_RATIOS:
            res = run_one(
                processor, model, device, image, args.prompt,
                keep_ratio=kr, gaze_x=gx_norm, gaze_y=gy_norm,
                max_new_tokens=args.max_new_tokens,
            )
            print(f"{fname:<20} {kr:>5.2f} {res['tiles_kept']:>4}/{res['tiles_total']:<3}"
                  f" {res['vis_tok_kept']:>5}/{res['vis_tok_total']:<5}"
                  f" {res['total_s']:>7.3f}s {res['ms_per_token']:>7.1f}ms"
                  f"  {res['answer'][:60]}")

            writer.writerow({
                "take":        take,
                "frame":       fname,
                "frame_num":   frame_num,
                "gaze_x_px":   round(gx_px, 1),
                "gaze_y_px":   round(gy_px, 1),
                "gaze_x_norm": round(gx_norm, 4),
                "gaze_y_norm": round(gy_norm, 4),
                "keep_ratio":  kr,
                **res,
            })
            csvfile.flush()

        print()   # blank line between frames

    csvfile.close()

    # ── Per-take summary ──────────────────────────────────────────────────────
    df = pd.read_csv(results_path)
    grp = df.groupby("keep_ratio")
    means = grp[["tiles_kept", "vis_tok_kept", "total_s", "ms_per_token"]].mean()
    stds  = grp[["total_s", "ms_per_token"]].std()

    summary = means.sort_index(ascending=False).rename(columns={
        "tiles_kept":   "avg_tiles",
        "vis_tok_kept": "avg_vis_tokens",
        "total_s":      "avg_time_s",
        "ms_per_token": "avg_ms_per_tok",
    })
    summary["std_time_s"]    = stds["total_s"].sort_index(ascending=False)
    summary["std_ms_per_tok"] = stds["ms_per_token"].sort_index(ascending=False)

    baseline_time = summary.loc[1.0, "avg_time_s"]
    summary["speedup"] = (baseline_time / summary["avg_time_s"]).round(2)

    # Format time as mean ± std for readability
    summary["time (mean±std)"] = (
        summary["avg_time_s"].map(lambda v: f"{v:.3f}")
        + " ± "
        + summary["std_time_s"].map(lambda v: f"{v:.3f}")
    )
    summary["ms/tok (mean±std)"] = (
        summary["avg_ms_per_tok"].map(lambda v: f"{v:.2f}")
        + " ± "
        + summary["std_ms_per_tok"].map(lambda v: f"{v:.2f}")
    )

    print(f"\n===== SUMMARY: {take} ({len(sampled)} frames) =====")
    print(summary[["avg_tiles", "avg_vis_tokens",
                   "time (mean±std)", "ms/tok (mean±std)", "speedup"]].to_string())
    print(f"\nFull results → {results_path}")

    return results_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--takes", nargs="+", default=ALL_TAKES,
        help="Which takes to run (default: all available takes)"
    )
    parser.add_argument(
        "--n_frames", type=int, default=0,
        help="Frames to sample per take; 0 = all frames (default)"
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--prompt", default="What is the person doing in this scene?")
    args = parser.parse_args()

    device = get_device()
    processor, model = load_model(MODEL_ID, device)

    result_paths = []
    for take in args.takes:
        path = run_take(take, processor, model, device, args)
        if path:
            result_paths.append(path)

    # ── Combined summary across all takes ─────────────────────────────────────
    if len(result_paths) > 1:
        combined = pd.concat([pd.read_csv(p) for p in result_paths], ignore_index=True)
        grp   = combined.groupby("keep_ratio")
        means = grp[["vis_tok_kept", "total_s", "ms_per_token"]].mean()
        stds  = grp[["total_s", "ms_per_token"]].std()

        summary = means.sort_index(ascending=False).rename(columns={
            "vis_tok_kept": "avg_vis_tokens",
            "total_s":      "avg_time_s",
            "ms_per_token": "avg_ms_per_tok",
        })
        summary["std_time_s"]     = stds["total_s"].sort_index(ascending=False)
        summary["std_ms_per_tok"] = stds["ms_per_token"].sort_index(ascending=False)

        baseline_time = summary.loc[1.0, "avg_time_s"]
        summary["speedup"] = (baseline_time / summary["avg_time_s"]).round(2)
        summary["time (mean±std)"] = (
            summary["avg_time_s"].map(lambda v: f"{v:.3f}")
            + " ± "
            + summary["std_time_s"].map(lambda v: f"{v:.3f}")
        )
        summary["ms/tok (mean±std)"] = (
            summary["avg_ms_per_tok"].map(lambda v: f"{v:.2f}")
            + " ± "
            + summary["std_ms_per_tok"].map(lambda v: f"{v:.2f}")
        )
        print("\n===== COMBINED SUMMARY (all takes) =====")
        print(summary[["avg_vis_tokens", "time (mean±std)",
                        "ms/tok (mean±std)", "speedup"]].to_string())


if __name__ == "__main__":
    main()
