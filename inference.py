"""
SmolVLM baseline inference with decoder latency measurement.

SmolVLM-256M image encoding:
  - Splits image into 17 tiles: 1 global (low-res full image) + 4x4 local grid
  - Each tile → 64 visual tokens  →  17 × 64 = 1,088 total visual tokens
  - Patch size = 16, image size = 512, scale factor = 4

We measure latency with and without tile-level gaze pruning.

Usage:
    python inference.py                              # unpruned baseline
    python inference.py --keep_ratio 0.5             # keep 50% of local tiles
    python inference.py --keep_ratio 0.5 --gaze_x 0.3 --gaze_y 0.7
    python inference.py --image path/to/img.jpg --prompt "What is she doing?"
"""

import argparse
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

from gaze.pruner import TilePruner

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

# SmolVLM-256M tile layout (fixed by architecture)
N_TILES_TOTAL  = 17   # 1 global + 4x4 local
TOKENS_PER_TILE = 64  # 8x8 after pixel shuffle
TOTAL_VIS_TOKENS = N_TILES_TOTAL * TOKENS_PER_TILE  # 1088


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_id: str = MODEL_ID, device: str | None = None):
    device = device or get_device()
    print(f"Loading {model_id} on {device} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    print(f"Model loaded. Visual tokens per image: {TOTAL_VIS_TOKENS} "
          f"({N_TILES_TOTAL} tiles × {TOKENS_PER_TILE} tokens)")
    return processor, model, device


def build_inputs(processor, image: Image.Image, prompt: str, device: str) -> dict:
    """Tokenize the prompt + image into model inputs."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(text=text, images=[image], return_tensors="pt").to(device)


def run_inference(
    processor,
    model,
    device: str,
    image: Image.Image,
    prompt: str = "What do you see in this image?",
    max_new_tokens: int = 64,
    keep_ratio: float = 1.0,
    gaze_x: float = 0.5,
    gaze_y: float = 0.5,
) -> dict:
    """
    Run SmolVLM on an image + prompt and measure decoder latency.

    If keep_ratio < 1.0, we:
      1. Extract image_hidden_states (the visual token embeddings)
      2. Score each tile by distance to the gaze point
      3. Drop the furthest tiles — removing their tokens from the sequence entirely
      4. Run generate() on the shorter sequence → real speedup

    keep_ratio applies to the 16 LOCAL tiles; tile 0 (global) is always kept.
    """
    inputs = build_inputs(processor, image, prompt, device)
    image_token_id = model.config.image_token_id

    kept_tiles = list(range(N_TILES_TOTAL))  # default: all tiles

    if keep_ratio < 1.0:
        # Step 1: run the vision encoder to get visual token embeddings
        with torch.no_grad():
            image_hidden_states = model.model.get_image_features(
                pixel_values=inputs["pixel_values"].to(torch.float16),
                pixel_attention_mask=inputs["pixel_attention_mask"],
                return_dict=True,
            ).pooler_output
            # shape: (17, 64, 576)

        # Step 2: prune tiles by gaze proximity
        pruner = TilePruner(n_local_tiles_side=4, keep_ratio=keep_ratio)
        pruned_states, new_ids, new_mask, kept_tiles = pruner.prune(
            image_hidden_states=image_hidden_states,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_token_id=image_token_id,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
        )

        n_kept = len(kept_tiles)
        print(f"  Tiles kept: {n_kept}/{N_TILES_TOTAL} "
              f"({n_kept * TOKENS_PER_TILE} visual tokens, "
              f"gaze at x={gaze_x:.2f} y={gaze_y:.2f})")

        # Step 3: pass pruned image_hidden_states + shorter input_ids to generate()
        # The model accepts image_hidden_states directly — it skips re-running
        # the vision encoder and goes straight to merging tokens into the sequence.
        gen_inputs = dict(
            input_ids=new_ids,
            attention_mask=new_mask,
            image_hidden_states=pruned_states.to(device),
        )
        seq_len = new_ids.shape[-1]
    else:
        # No pruning — pass pixel_values normally
        gen_inputs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            pixel_attention_mask=inputs["pixel_attention_mask"],
        )
        seq_len = inputs["input_ids"].shape[-1]

    t_start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    t_total = time.perf_counter() - t_start

    n_output_tokens = outputs.shape[-1] - seq_len
    answer = processor.decode(outputs[0][seq_len:], skip_special_tokens=True)

    return {
        "answer": answer,
        "tiles_kept": len(kept_tiles),
        "tiles_total": N_TILES_TOTAL,
        "visual_tokens_kept": len(kept_tiles) * TOKENS_PER_TILE,
        "visual_tokens_total": TOTAL_VIS_TOKENS,
        "output_tokens": n_output_tokens,
        "total_s": round(t_total, 4),
        "ms_per_token": round(t_total * 1000 / max(n_output_tokens, 1), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--image", default=None, help="Path to image file")
    parser.add_argument("--prompt", default="What do you see in this image?")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "--keep_ratio", type=float, default=1.0,
        help="Fraction of local tiles to keep (1.0 = no pruning)"
    )
    parser.add_argument("--gaze_x", type=float, default=0.5)
    parser.add_argument("--gaze_y", type=float, default=0.5)
    args = parser.parse_args()

    processor, model, device = load_model(args.model)

    if args.image:
        image = Image.open(args.image).convert("RGB")
    else:
        print("No --image provided; using synthetic 512x512 grey image.")
        image = Image.new("RGB", (512, 512), color=(128, 128, 128))

    result = run_inference(
        processor, model, device, image,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        keep_ratio=args.keep_ratio,
        gaze_x=args.gaze_x,
        gaze_y=args.gaze_y,
    )

    label = "Baseline (no pruning)" if args.keep_ratio == 1.0 else f"Gaze pruning (keep {args.keep_ratio:.0%} of tiles)"
    print(f"\n--- {label} ---")
    print(f"Answer        : {result['answer']}")
    print(f"Tiles kept    : {result['tiles_kept']} / {result['tiles_total']}")
    print(f"Visual tokens : {result['visual_tokens_kept']} / {result['visual_tokens_total']}")
    print(f"Output tokens : {result['output_tokens']}")
    print(f"Total time    : {result['total_s']}s")
    print(f"ms / token    : {result['ms_per_token']}ms")


if __name__ == "__main__":
    main()
