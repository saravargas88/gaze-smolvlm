"""
make_tables.py
==============
Reads experiment CSVs and produces two LaTeX tables:
  1. Efficiency table  — tiles, visual tokens, time (mean ± std), speedup (mean ± std)
  2. Quality table     — BERTScore F1 (mean ± std) vs baseline, per keep_ratio

Each table is printed for each take individually, then combined across both takes.

Usage:
    uv run python make_tables.py
    uv run python make_tables.py --csv results/iiith_cooking_108_5_experiment.csv
    uv run python make_tables.py --csv results/iiith_cooking_108_5_experiment.csv results/nus_cpr_27_3_experiment.csv
"""

import argparse
import os

import pandas as pd
from bert_score import score as bert_score

KEEP_RATIOS  = [1.00, 0.75, 0.50, 0.25]
RATIO_LABELS = {1.00: "1.00 (baseline)", 0.75: "0.75", 0.50: "0.50", 0.25: "0.25"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def pm(mean, std, decimals=3):
    """Format mean ± std to given decimal places."""
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} $\\pm$ {fmt.format(std)}"


# ── Efficiency table ──────────────────────────────────────────────────────────

def efficiency_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-keep_ratio efficiency stats with mean and std."""
    grp = df.groupby("keep_ratio")

    means = grp[["tiles_kept", "vis_tok_kept", "total_s", "ms_per_token"]].mean()
    stds  = grp[["total_s", "ms_per_token"]].std().rename(
        columns={"total_s": "std_time", "ms_per_token": "std_ms"}
    )

    stats = means.join(stds).sort_index(ascending=False)

    # Speedup per-frame, then aggregate
    baseline_times = df[df["keep_ratio"] == 1.0].set_index("frame")["total_s"]
    df2 = df.copy()
    df2["speedup"] = df2.apply(
        lambda r: baseline_times.get(r["frame"], float("nan")) / r["total_s"], axis=1
    )
    sp = df2.groupby("keep_ratio")["speedup"].agg(["mean", "std"]).sort_index(ascending=False)
    stats["speedup_mean"] = sp["mean"]
    stats["speedup_std"]  = sp["std"]

    return stats


def latex_efficiency(stats: pd.DataFrame, caption: str, label: str) -> str:
    baseline_tiles = int(stats.loc[1.0, "tiles_kept"])
    baseline_tok   = int(stats.loc[1.0, "vis_tok_kept"])

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{keep\_ratio} & \textbf{Tiles kept} & \textbf{Vis.\ tokens} "
                 r"& \textbf{Time (s)} & \textbf{ms/token} & \textbf{Speedup} \\")
    lines.append(r"\midrule")

    for kr in KEEP_RATIOS:
        if kr not in stats.index:
            continue
        row   = stats.loc[kr]
        tiles = f"{int(row['tiles_kept'])} / {baseline_tiles}"
        toks  = f"{int(row['vis_tok_kept'])} / {baseline_tok}"
        time_ = pm(row["total_s"],    row["std_time"], 3)
        ms_   = pm(row["ms_per_token"], row["std_ms"],  2)

        if kr == 1.0:
            spd = "1.00 (baseline)"
        else:
            spd = pm(row["speedup_mean"], row["speedup_std"], 2) + r"$\times$"

        label_str = RATIO_LABELS[kr]
        lines.append(rf"{label_str} & {tiles} & {toks} & {time_} & {ms_} & {spd} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── BERTScore table ───────────────────────────────────────────────────────────

def bertscore_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-frame BERTScore F1 vs baseline, return mean ± std per keep_ratio."""
    baseline = df[df["keep_ratio"] == 1.0].set_index("frame")["answer"].to_dict()

    rows = []
    for kr, group in df[df["keep_ratio"] < 1.0].groupby("keep_ratio"):
        hyps = [str(r["answer"]) for _, r in group.iterrows()]
        refs = [str(baseline.get(r["frame"], "")) for _, r in group.iterrows()]

        _, _, F1 = bert_score(hyps, refs, lang="en", verbose=False)

        rows.append({
            "keep_ratio": kr,
            "F1_mean":    F1.mean().item(),
            "F1_std":     F1.std().item(),
            "n":          len(hyps),
        })

    return pd.DataFrame(rows).set_index("keep_ratio").sort_index(ascending=False)


def latex_quality(bs_stats: pd.DataFrame, caption: str, label: str) -> str:
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{keep\_ratio} & \textbf{BERTScore F1 (mean $\pm$ std)} & \textbf{$n$ frames} \\")
    lines.append(r"\midrule")
    lines.append(r"1.00 (baseline) & 1.0000 (reference) & --- \\")

    for kr in [0.75, 0.50, 0.25]:
        if kr not in bs_stats.index:
            continue
        row = bs_stats.loc[kr]
        f1  = pm(row["F1_mean"], row["F1_std"], 4)
        lines.append(rf"{kr:.2f} & {f1} & {int(row['n'])} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def process(df: pd.DataFrame, name: str, out_dir: str):
    print(f"\n{'='*60}")
    print(f"Processing: {name}  ({df['frame'].nunique()} frames)")
    print(f"{'='*60}")

    # ── Efficiency ────────────────────────────────────────────────────────────
    eff = efficiency_stats(df)
    print("\n--- Efficiency (mean ± std) ---")
    print(eff[["tiles_kept", "vis_tok_kept", "total_s", "std_time",
               "ms_per_token", "std_ms", "speedup_mean", "speedup_std"]].to_string())

    safe_name = name.replace(" ", "_").replace("/", "_")
    eff_tex = latex_efficiency(
        eff,
        caption=f"Pruning efficiency results ({name}). "
                r"Speedup and time shown as mean $\pm$ std across frames.",
        label=f"tab:efficiency_{safe_name}",
    )
    print(f"\n--- LaTeX: Efficiency ---\n{eff_tex}")

    # ── BERTScore ─────────────────────────────────────────────────────────────
    print("\nComputing BERTScore (this may take a moment)…")
    bs = bertscore_stats(df)
    print("\n--- BERTScore F1 vs baseline (mean ± std) ---")
    print(bs.to_string())

    bs_tex = latex_quality(
        bs,
        caption=f"BERTScore F1 vs.\ baseline ({name}). "
                r"Higher = more semantically similar to the unpruned answer.",
        label=f"tab:bertscore_{safe_name}",
    )
    print(f"\n--- LaTeX: BERTScore ---\n{bs_tex}")

    # ── Save .tex snippets ────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{safe_name}_efficiency.tex"), "w") as f:
        f.write(eff_tex + "\n")
    with open(os.path.join(out_dir, f"{safe_name}_bertscore.tex"), "w") as f:
        f.write(bs_tex + "\n")
    print(f"\nTeX snippets saved → {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", nargs="+",
        default=[
            "results/iiith_cooking_108_5_experiment.csv",
            "results/nus_cpr_27_3_experiment.csv",
        ],
        help="Experiment CSV file(s) to process",
    )
    parser.add_argument("--out", default="results/tables",
                        help="Output directory for .tex snippets")
    args = parser.parse_args()

    all_dfs = []
    for path in args.csv:
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found")
            continue
        df = pd.read_csv(path)
        take = os.path.basename(path).replace("_experiment.csv", "")
        process(df, take, args.out)
        all_dfs.append(df)

    # ── Combined across takes ─────────────────────────────────────────────────
    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        process(combined, "combined_takes", args.out)


if __name__ == "__main__":
    main()
