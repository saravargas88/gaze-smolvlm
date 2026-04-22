# gaze-smolvlm

**Gaze-Guided Acceleration for On-Device SmolVLM Inference**

Julia Gontijo Lopes · Sara Vargas Martínez — Building LLM Reasoners (Spring 2026)

---

## Overview

We use eye-gaze data from Project Aria glasses to prune visual tiles from [SmolVLM-256M](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) before decoding, reducing sequence length and latency without retraining the model. The central hypothesis is that tiles spatially far from the user's gaze fixation point contribute less to task-relevant responses and can be dropped entirely before the language model ever sees them.

Evaluated on [EgoExo4D](https://ego-exo4d-data.org/) egocentric video with synchronized gaze streams from two takes:
- `iiith_cooking_108_5` — cooking activity, 64 frames
- `nus_cpr_27_3` — CPR activity, 50 frames

---

## How it works

SmolVLM splits each image into tiles (1 global + 4×4 local = 17 tiles for 448×448 frames), encoding each tile into 64 visual tokens. At `keep_ratio=1.0` the language model receives 1,088 visual tokens. Our pipeline:

1. **Gaze alignment** — normalize gaze coordinates from Aria sensor space (1408×1408) to [0,1] and align temporally with extracted frames (gaze at 10fps → frames at 1fps)
2. **Tile scoring** — score each local tile by Gaussian distance of its center from the gaze point (σ=0.3); the global tile is always kept
3. **Feature capture** — register a forward hook on `model.model.connector` and fire a 1-token generate pass to capture `image_hidden_states` of shape `(n_tiles, 64, 576)`
4. **Pruning** — slice the hidden states to kept tiles and remove corresponding `<image>` tokens from `input_ids`, shortening the sequence
5. **Generation** — call `model.generate()` with `image_hidden_states=pruned_states` directly, skipping the vision encoder on the second pass

---

## Results

### Cooking take (64 frames)

| keep_ratio | Tiles kept | Vis. tokens | Time (s) | Speedup |
|---|---|---|---|---|
| 1.00 (baseline) | 17 / 17 | 1088 | 0.466 ± 0.103 | 1.00× |
| 0.75 | 13 / 17 | 832 | 0.302 ± 0.037 | 1.57 ± 0.50× |
| 0.50 | 9 / 17 | 576 | 0.307 ± 0.041 | 1.54 ± 0.33× |
| 0.25 | 5 / 17 | 320 | 0.306 ± 0.046 | 1.55 ± 0.37× |

| keep_ratio | BERTScore F1 vs baseline |
|---|---|
| 0.75 | 0.9666 ± 0.0212 |
| 0.50 | 0.9524 ± 0.0276 |
| 0.25 | 0.9466 ± 0.0246 |

### CPR take (50 frames)

| keep_ratio | Tiles kept | Vis. tokens | Time (s) | Speedup |
|---|---|---|---|---|
| 1.00 (baseline) | 17 / 17 | 1088 | 0.463 ± 0.058 | 1.00× |
| 0.75 | 13 / 17 | 832 | 0.328 ± 0.035 | 1.42 ± 0.21× |
| 0.50 | 9 / 17 | 576 | 0.288 ± 0.048 | 1.64 ± 0.32× |
| 0.25 | 5 / 17 | 320 | 0.281 ± 0.046 | 1.69 ± 0.36× |

| keep_ratio | BERTScore F1 vs baseline |
|---|---|
| 0.75 | 0.9591 ± 0.0253 |
| 0.50 | 0.9405 ± 0.0205 |
| 0.25 | 0.9383 ± 0.0184 |

The speedup plateau between 0.50 and 0.25 is caused by the fixed overhead of the feature-capture hook pass, which dominates when the generation step itself becomes very short.

---

## Repository structure

```
gaze-smolvlm/
├── gaze/
│   └── pruner.py          # TilePruner: Gaussian tile scoring + token removal
├── data/
│   └── egoexo_data/
│       └── takes/
│           ├── iiith_cooking_108_5/
│           │   ├── frames/          # 1fps JPGs extracted from aria01_214-1.mp4
│           │   └── eye_gaze/        # general_eye_gaze_2d.csv
│           └── nus_cpr_27_3/
│               ├── frames/          # 1fps JPGs extracted from aria01_214-1.mp4
│               └── eye_gaze/        # general_eye_gaze_2d.csv
├── results/
│   ├── iiith_cooking_108_5_experiment.csv
│   ├── nus_cpr_27_3_experiment.csv
│   ├── tables/            # LaTeX table snippets from make_tables.py
│   └── viz/               # Tile pruning visualizations per frame
├── experiment.py          # Main experiment: runs SmolVLM at varying keep_ratios
├── make_tables.py         # Computes BERTScore + generates LaTeX tables from CSVs
├── visualize_pruning.py   # Overlays tile grid + gaze point on frames
├── compute_bertscore.py   # Standalone BERTScore computation
├── checkin_report.tex     # Midterm check-in report
└── requirements.txt
```

---

## Setup

**Local (Mac):**
```bash
uv sync
```

**HPC (Singularity/conda):**
```bash
conda create -n gaze-smolvlm python=3.11
conda activate gaze-smolvlm
pip install -r requirements.txt
```

**Extract frames** (only needed once per take — frames must come from `aria01_214-1.mp4`):
```bash
ffmpeg -i data/egoexo_data/takes/<take>/frame_aligned_videos/downscaled/448/aria01_214-1.mp4 \
       -vf "fps=1" -q:v 2 \
       data/egoexo_data/takes/<take>/frames/frame_%04d.jpg
```

---

## Usage

**Run the full experiment (both takes, all frames):**
```bash
python experiment.py
```

**Run with automatic table + BERTScore generation:**
```bash
python experiment.py --tables
```

**Run a quick sanity check (8 frames, one take):**
```bash
python experiment.py --takes iiith_cooking_108_5 --n_frames 8
```

**Generate LaTeX tables from existing CSVs:**
```bash
uv run python make_tables.py
# or specify CSVs explicitly:
uv run python make_tables.py --csv results/iiith_cooking_108_5_experiment.csv results/nus_cpr_27_3_experiment.csv
```

**Visualize which tiles are pruned for a frame:**
```bash
uv run python visualize_pruning.py --take iiith_cooking_108_5 --frame frame_0001.jpg
uv run python visualize_pruning.py   # all frames, both takes
```

---

## Key arguments

| Script | Argument | Default | Description |
|---|---|---|---|
| `experiment.py` | `--takes` | both takes | Which takes to run |
| `experiment.py` | `--n_frames` | 0 (all) | Frames to sample per take |
| `experiment.py` | `--prompt` | "What is the person doing..." | VQA prompt |
| `experiment.py` | `--tables` | off | Run BERTScore + LaTeX tables after experiment |
| `visualize_pruning.py` | `--take` | both | Take to visualize |
| `visualize_pruning.py` | `--frame` | all | Specific frame to visualize |
| `visualize_pruning.py` | `--n_local_tiles` | 12 | Local tile count (16 for 448×448 frames) |

---

## Known limitations

- **Speedup plateau**: the forward hook requires a fixed pre-generation pass whose cost dominates at aggressive pruning ratios, flattening speedup between 0.50 and 0.25
- **Approximate token removal**: pruned tile tokens are removed from the tail of the image token sequence as a proxy — not exact for all interleaving schemes
- **BERTScore vs baseline**: quality is measured relative to the unpruned model output, not against ground-truth annotations
- **No random pruning baseline**: gaze contribution vs. naive spatial heuristics not yet ablated
