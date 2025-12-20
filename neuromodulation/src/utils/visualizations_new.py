import argparse
import glob
import json
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# I/O
# ----------------------------
def load_results(results_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(results_dir, "results_*.json")))
    if not paths:
        raise FileNotFoundError(f"No results_*.json found in: {results_dir}")

    rows: List[Dict] = []
    for p in paths:
        with open(p, "r") as f:
            row = json.load(f)
        row["_file"] = os.path.basename(p)
        rows.append(row)

    df = pd.DataFrame(rows)
    if "config" not in df.columns:
        raise ValueError("Expected a 'config' field in each JSON.")
    df = df.sort_values("config").reset_index(drop=True)
    return df


def ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
def _as_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def compute_summary_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Means for convenience
    if {"MSP_AUROC", "ENG_AUROC"}.issubset(out.columns):
        out["AUROC_mean"] = (_as_float_series(out, "MSP_AUROC") + _as_float_series(out, "ENG_AUROC")) / 2.0

    if {"MSP_FPR95", "ENG_FPR95"}.issubset(out.columns):
        out["FPR95_mean"] = (_as_float_series(out, "MSP_FPR95") + _as_float_series(out, "ENG_FPR95")) / 2.0

    # A simple "higher is better" composite (both in [0,1] typically)
    # If missing, will stay NaN.
    if {"AUROC_mean", "FPR95_mean"}.issubset(out.columns):
        out["OOD_composite"] = (out["AUROC_mean"].astype(float) + (1.0 - out["FPR95_mean"].astype(float))) / 2.0

    return out


def order_df(df: pd.DataFrame, order_by: str, ascending: bool) -> pd.DataFrame:
    if order_by in df.columns:
        return df.sort_values(order_by, ascending=ascending).reset_index(drop=True)
    # fallback
    return df.sort_values("config").reset_index(drop=True)


# ----------------------------
# Plots (dot/lollipop + heatmap)
# ----------------------------
def dotplot_sorted(
    df: pd.DataFrame,
    metric: str,
    out_path: str,
    title: str,
    xlabel: str,
    higher_is_better: bool = True,
):
    d = df.copy()
    d[metric] = _as_float_series(d, metric)

    d = d.sort_values(metric, ascending=not higher_is_better).reset_index(drop=True)

    y = np.arange(len(d))
    x = d[metric].to_numpy()
    labels = d["config"].tolist()

    plt.figure(figsize=(10, max(4, 0.35 * len(d) + 2)))
    plt.scatter(x, y)
    # lollipop stems
    for xi, yi in zip(x, y):
        plt.plot([0, xi], [yi, yi], linewidth=1)

    plt.yticks(y, labels)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def paired_dotplot_sorted(
    df: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    label_a: str,
    label_b: str,
    out_path: str,
    title: str,
    xlabel: str,
    higher_is_better: bool = True,
    sort_by: str = "mean",
):
    d = df.copy()
    d[metric_a] = _as_float_series(d, metric_a)
    d[metric_b] = _as_float_series(d, metric_b)

    if sort_by == metric_a:
        key = d[metric_a]
    elif sort_by == metric_b:
        key = d[metric_b]
    else:
        key = (d[metric_a] + d[metric_b]) / 2.0

    d["_sortkey"] = key
    d = d.sort_values("_sortkey", ascending=not higher_is_better).reset_index(drop=True)

    y = np.arange(len(d))
    xa = d[metric_a].to_numpy()
    xb = d[metric_b].to_numpy()
    labels = d["config"].tolist()

    plt.figure(figsize=(10, max(4, 0.35 * len(d) + 2)))
    plt.scatter(xa, y, label=label_a, marker="o")
    plt.scatter(xb, y, label=label_b, marker="x")

    # Connect pairs per config for easy comparison
    for i in range(len(d)):
        plt.plot([xa[i], xb[i]], [y[i], y[i]], linewidth=1)

    plt.yticks(y, labels)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def scatter_with_labels(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    out_path: str,
    title: str,
):
    x = _as_float_series(df, x_metric).to_numpy()
    y = _as_float_series(df, y_metric).to_numpy()

    plt.figure(figsize=(7, 6))
    plt.scatter(x, y)
    for xi, yi, name in zip(x, y, df["config"].tolist()):
        plt.annotate(name, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fewshot_lineplot(df: pd.DataFrame, out_path: str, title: str) -> None:
    few_cols = [c for c in df.columns if c.startswith("fewshot_")]
    if not few_cols:
        print("No fewshot_* columns found; skipping few-shot plot.")
        return

    shots = []
    for c in few_cols:
        try:
            shots.append(int(c.split("_", 1)[1]))
        except Exception:
            pass
    shots = sorted(set(shots))
    few_cols_sorted = [f"fewshot_{k}" for k in shots if f"fewshot_{k}" in df.columns]
    if not few_cols_sorted:
        print("Few-shot columns present but could not parse shot counts; skipping few-shot plot.")
        return

    plt.figure(figsize=(10, 6))
    xs = np.array(shots, dtype=int)

    # --- NEW: Generate unique colors for each configuration ---
    # nipy_spectral provides a wide spectrum of distinct colors.
    # We sample linearly from 5% to 95% of the map to avoid extreme darks/lights.
    num_configs = len(df)
    colors = plt.cm.nipy_spectral(np.linspace(0.05, 0.95, num_configs))

    for i, (_, row) in enumerate(df.iterrows()):
        ys = np.array([float(row[f"fewshot_{k}"]) for k in shots], dtype=float)
        # --- Pass the specific color here ---
        plt.plot(xs, ys, marker="o", label=row["config"], color=colors[i])

    plt.xticks(xs)
    plt.xlabel("Shots per class (OOD few-shot linear probe)")
    plt.ylabel("Accuracy")
    plt.title(title)
    
    # Adjust legend to fit all items if list is long
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _prepare_heatmap_matrix(
    df: pd.DataFrame,
    metrics: List[Tuple[str, bool]],  # (metric, higher_is_better)
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Returns:
      Z: (n_configs, n_metrics) standardized matrix (higher=better orientation)
      row_labels: config names
      col_labels: metric names
    """
    row_labels = df["config"].tolist()
    col_labels = [m for m, _ in metrics]

    M = []
    for m, higher_is_better in metrics:
        v = _as_float_series(df, m).to_numpy()
        # orient so higher is better
        if not higher_is_better:
            v = -v
        M.append(v)

    X = np.vstack(M).T  # (n, p)

    # z-score by column (ignore NaNs)
    Z = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j].astype(float)
        mu = np.nanmean(col)
        sd = np.nanstd(col)
        if not np.isfinite(sd) or sd < 1e-12:
            Z[:, j] = col - mu
        else:
            Z[:, j] = (col - mu) / sd

    return Z, row_labels, col_labels


def heatmap_configs_by_metrics(
    df: pd.DataFrame,
    metrics: List[Tuple[str, bool]],
    out_path: str,
    title: str,
):
    # Drop metrics missing entirely
    kept = []
    for m, hib in metrics:
        if m in df.columns:
            kept.append((m, hib))
    if not kept:
        print("No requested metrics available for heatmap; skipping.")
        return

    Z, row_labels, col_labels = _prepare_heatmap_matrix(df, kept)

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(col_labels) + 2), max(5, 0.35 * len(row_labels) + 2)))
    im = ax.imshow(Z, aspect="auto")

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("z-score (higher = better)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True, help="Directory containing results_*.json")
    ap.add_argument("--out_dir", type=str, required=True, help="Directory to save plots")
    ap.add_argument(
        "--order_by",
        type=str,
        default="OOD_composite",
        help="Column to order configs by for most plots (default: OOD_composite if available)",
    )
    ap.add_argument(
        "--ascending",
        action="store_true",
        help="If set, order ascending instead of descending",
    )
    args = ap.parse_args()

    ensure_out_dir(args.out_dir)
    df = load_results(args.results_dir)
    df = compute_summary_scores(df)

    # Save table outputs for the report
    csv_path = os.path.join(args.out_dir, "all_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Loaded {len(df)} configs. Wrote: {csv_path}")

    # Order for plotting
    df_plot = order_df(df, args.order_by, ascending=args.ascending)

    # Dot/lollipop plots (sorted)
    if "ID_acc" in df_plot.columns:
        dotplot_sorted(
            df_plot,
            metric="ID_acc",
            out_path=os.path.join(args.out_dir, "01_id_accuracy_dot.png"),
            title="ID Accuracy (Known Classes) — dot/lollipop",
            xlabel="Accuracy",
            higher_is_better=True,
        )

    # OOD AUROC paired dot plots (MSP vs Energy)
    if {"MSP_AUROC", "ENG_AUROC"}.issubset(df_plot.columns):
        paired_dotplot_sorted(
            df_plot,
            metric_a="MSP_AUROC",
            metric_b="ENG_AUROC",
            label_a="MSP_AUROC",
            label_b="ENG_AUROC",
            out_path=os.path.join(args.out_dir, "02_ood_auroc_paired_dot.png"),
            title="OOD Detection AUROC — paired dots (higher is better)",
            xlabel="AUROC",
            higher_is_better=True,
            sort_by="mean",
        )

    # OOD FPR95 paired dot plots (lower is better)
    if {"MSP_FPR95", "ENG_FPR95"}.issubset(df_plot.columns):
        # For sorting with higher_is_better=False, invert logic by sorting descending on negative mean
        # We'll just set higher_is_better=False and sort_by mean.
        paired_dotplot_sorted(
            df_plot,
            metric_a="MSP_FPR95",
            metric_b="ENG_FPR95",
            label_a="MSP_FPR95",
            label_b="ENG_FPR95",
            out_path=os.path.join(args.out_dir, "03_ood_fpr95_paired_dot.png"),
            title="OOD Detection FPR@95TPR — paired dots (lower is better)",
            xlabel="FPR@95TPR",
            higher_is_better=False,
            sort_by="mean",
        )

    # Tradeoffs
    if {"ID_acc", "MSP_AUROC"}.issubset(df_plot.columns):
        scatter_with_labels(
            df_plot,
            "ID_acc",
            "MSP_AUROC",
            os.path.join(args.out_dir, "04_tradeoff_id_vs_msp_auroc.png"),
            "Tradeoff: ID Accuracy vs MSP AUROC",
        )

    if {"ID_acc", "ENG_AUROC"}.issubset(df_plot.columns):
        scatter_with_labels(
            df_plot,
            "ID_acc",
            "ENG_AUROC",
            os.path.join(args.out_dir, "05_tradeoff_id_vs_energy_auroc.png"),
            "Tradeoff: ID Accuracy vs Energy AUROC",
        )

    # Geometry proxy
    if "OOD_sep" in df_plot.columns:
        dotplot_sorted(
            df_plot,
            metric="OOD_sep",
            out_path=os.path.join(args.out_dir, "06_ood_feature_separation_dot.png"),
            title="OOD Feature Separation (d_inter / d_intra) — dot/lollipop (higher is better)",
            xlabel="Separation",
            higher_is_better=True,
        )

    # Covariance frobenius (lower is better)
    if "OOD_frob_cov" in df_plot.columns:
        dotplot_sorted(
            df_plot,
            metric="OOD_frob_cov",
            out_path=os.path.join(args.out_dir, "07_cov_frobenius_dot.png"),
            title="OOD Feature Covariance Distance to Identity (Frobenius) — dot/lollipop (lower is better)",
            xlabel="||Cov - I||_F",
            higher_is_better=False,
        )

    if "OOD_frob_cov_stage0" in df_plot.columns:
        dotplot_sorted(
            df_plot,
            metric="OOD_frob_cov_stage0",
            out_path=os.path.join(args.out_dir, "08_cov_frobenius_stage0_dot.png"),
            title="Stage0 OOD Covariance Distance to Identity (Frobenius) — dot/lollipop (lower is better)",
            xlabel="||Cov - I||_F",
            higher_is_better=False,
        )

    # Few-shot curves (kept)
    fewshot_lineplot(
        df_plot,
        os.path.join(args.out_dir, "09_fewshot_curves.png"),
        title="OOD Few-shot Linear Probe Accuracy Curves",
    )

    # Heatmap: configs x key metrics (z-scored; oriented so higher=better)
    heatmap_metrics = [
        ("ID_acc", True),
        ("MSP_AUROC", True),
        ("ENG_AUROC", True),
        ("MSP_FPR95", False),
        ("ENG_FPR95", False),
        ("OOD_sep", True),
        ("OOD_frob_cov", False),
        ("OOD_frob_cov_stage0", False),
    ]
    # If few-shot columns exist, include a couple common ones if present
    for k in [1, 5, 10, 15]:
        c = f"fewshot_{k}"
        if c in df_plot.columns:
            heatmap_metrics.append((c, True))

    heatmap_configs_by_metrics(
        df_plot,
        metrics=heatmap_metrics,
        out_path=os.path.join(args.out_dir, "10_heatmap_configs_by_metrics.png"),
        title="Configs × Metrics Heatmap (z-scored; higher=better orientation)",
    )

    # Ranking summary CSV (handy for report)
    df_rank = df.copy()
    if "OOD_composite" in df_rank.columns:
        df_rank = df_rank.sort_values("OOD_composite", ascending=False)
        rank_cols = ["config", "ID_acc", "OOD_composite"]
        for c in ["AUROC_mean", "FPR95_mean", "MSP_AUROC", "ENG_AUROC", "MSP_FPR95", "ENG_FPR95", "OOD_sep",
                  "OOD_frob_cov", "OOD_frob_cov_stage0"]:
            if c in df_rank.columns:
                rank_cols.append(c)
        rank_path = os.path.join(args.out_dir, "ranking_summary.csv")
        df_rank[rank_cols].to_csv(rank_path, index=False)
        print(f"Wrote: {rank_path}")

    print("Done. Plots saved to:", args.out_dir)


if __name__ == "__main__":
    main()
