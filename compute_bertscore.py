"""
compute_bertscore.py
====================
Compute BERTScore for pruned model answers vs. the baseline (keep_ratio=1.0).

For each frame, the baseline answer is used as the reference.
Pruned answers (keep_ratio < 1.0) are scored as hypotheses.
This quantifies quality degradation at each pruning level without
needing human-annotated ground truth.

Usage:
    uv run python compute_bertscore.py --csv results/iiith_cooking_108_5_experiment.csv
    uv run python compute_bertscore.py --csv results/iiith_cooking_108_5_experiment.csv results/nus_cpr_27_3_experiment.csv
"""

import argparse

import pandas as pd
from bert_score import score as bert_score


def evaluate(df: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """
    Compute BERTScore (P, R, F1) for each keep_ratio vs. baseline.

    Returns a DataFrame indexed by keep_ratio with columns:
        BERTScore_P, BERTScore_R, BERTScore_F1  (mean across frames)
    """
    # Build per-frame baseline lookup
    baseline = (
        df[df["keep_ratio"] == 1.0]
        .set_index("frame")["answer"]
        .to_dict()
    )

    rows = []
    for kr, group in df[df["keep_ratio"] < 1.0].groupby("keep_ratio"):
        hypotheses = []
        references = []
        for _, row in group.iterrows():
            ref = baseline.get(row["frame"], "")
            hyp = row["answer"]
            hypotheses.append(str(hyp))
            references.append(str(ref))

        P, R, F1 = bert_score(
            hypotheses, references,
            lang=lang,
            verbose=False,
        )
        rows.append({
            "keep_ratio":       kr,
            "BERTScore_F1":     round(F1.mean().item(), 4),
            "BERTScore_F1_std": round(F1.std().item(),  4),
            "BERTScore_P":      round(P.mean().item(),  4),
            "BERTScore_P_std":  round(P.std().item(),   4),
            "BERTScore_R":      round(R.mean().item(),  4),
            "BERTScore_R_std":  round(R.std().item(),   4),
            "n_frames":         len(hypotheses),
        })

    if not rows:
        raise ValueError("No pruned rows found — check that keep_ratio < 1.0 rows exist in the CSV.")
    return pd.DataFrame(rows).set_index("keep_ratio").sort_index(ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", nargs="+",
        default=["results/iiith_cooking_108_5_experiment.csv"],
        help="Path(s) to experiment CSV(s)",
    )
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()

    dfs = []
    for path in args.csv:
        print(f"Loading {path} …")
        dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)

    print(f"\nScoring {len(df[df['keep_ratio'] < 1.0])} pruned answers "
          f"against baseline across {df['frame'].nunique()} frames …\n")

    results = evaluate(df, lang=args.lang)

    print("===== BERTScore vs. Baseline (keep_ratio=1.0) =====")
    # Pretty-print F1 as mean ± std, P and R as mean ± std
    display = results[["BERTScore_F1", "BERTScore_F1_std",
                        "BERTScore_P",  "BERTScore_P_std",
                        "BERTScore_R",  "BERTScore_R_std",
                        "n_frames"]].copy()
    for metric in ["BERTScore_F1", "BERTScore_P", "BERTScore_R"]:
        display[f"{metric}_fmt"] = (
            display[metric].map(lambda v: f"{v:.4f}")
            + " ± "
            + display[f"{metric}_std"].map(lambda v: f"{v:.4f}")
        )
    print(display[["BERTScore_F1_fmt", "BERTScore_P_fmt",
                   "BERTScore_R_fmt", "n_frames"]]
          .rename(columns={
              "BERTScore_F1_fmt": "F1 (mean ± std)",
              "BERTScore_P_fmt":  "P  (mean ± std)",
              "BERTScore_R_fmt":  "R  (mean ± std)",
          })
          .to_string())
    print()
    print("Note: scores are relative to the baseline answer, not ground truth.")
    print("F1 = 1.00 means identical to baseline; lower = more drift.")

    # Save alongside the input CSV
    out_path = args.csv[0].replace(".csv", "_bertscore.csv").replace(
        "experiment", "bertscore"
    )
    results.to_csv(out_path)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
