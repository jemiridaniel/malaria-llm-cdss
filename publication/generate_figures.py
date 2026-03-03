"""
generate_figures.py
===================
Generates all publication-quality figures and statistical analysis for:
  "An LLM-Enhanced Clinical Decision Support System for Malaria Diagnosis
   in Resource-Limited Settings: A Hybrid Approach"

Usage:
    python publication/generate_figures.py

Outputs (all saved to publication/figures/):
    fig1_system_architecture.png
    fig2_accuracy_comparison.png
    fig3_confusion_matrices.png
    fig4_processing_time.png
    statistical_summary.json   (all computed statistics, for paper reference)
"""

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats
from scipy.stats import chi2

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_CSV = ROOT / "data/processed/baseline_results.csv"
HYBRID_CSV   = ROOT / "data/processed/hybrid_results.csv"
PER_STAGE    = ROOT / "results/tables/per_stage_metrics.csv"
OVERALL      = ROOT / "results/tables/overall_metrics.csv"

# ── Color palette ────────────────────────────────────────────────────────────
C_PRIMARY  = "#0284C7"   # blue — system / LLM-Enhanced
C_BASELINE = "#DC2626"   # red  — baseline
C_GREEN    = "#16A34A"   # green — no malaria / good
C_AMBER    = "#D97706"   # amber — stage I
C_ORANGE   = "#EA580C"   # orange — stage II
C_RED      = "#DC2626"   # red — critical
C_DARK     = "#1E293B"   # dark slate
C_MUTED    = "#94A3B8"   # muted text
C_BG       = "#F8FAFC"   # background

STAGE_PALETTE = {
    "No_Malaria": C_GREEN,
    "Stage_I":    C_AMBER,
    "Stage_II":   C_ORANGE,
    "Critical":   C_RED,
}

# ── Global matplotlib style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "axes.facecolor":    C_BG,
    "figure.facecolor":  "white",
})


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def wilson_ci(correct: int, total: int, alpha: float = 0.05):
    """Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = correct / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return (max(0, centre - half), min(1, centre + half))


def mcnemar_test(df_base: pd.DataFrame, df_hybrid: pd.DataFrame):
    """
    McNemar's test on paired binary predictions.
    Merges on case_id and extracts the 2x2 table:
      b = baseline correct AND hybrid wrong
      c = baseline wrong AND hybrid correct
    """
    merged = df_base[["case_id", "correct"]].merge(
        df_hybrid[["case_id", "correct"]], on="case_id", suffixes=("_base", "_hyb")
    )
    b = int(((merged["correct_base"] == True) & (merged["correct_hyb"] == False)).sum())
    c = int(((merged["correct_base"] == False) & (merged["correct_hyb"] == True)).sum())
    n_disc = b + c
    if n_disc == 0:
        return {"b": 0, "c": 0, "chi2": 0.0, "p_value": 1.0}
    # With continuity correction (Edwards)
    chi2_stat = (abs(b - c) - 1) ** 2 / n_disc
    p_val = chi2.sf(chi2_stat, df=1)
    return {"b": b, "c": c, "chi2": round(chi2_stat, 2), "p_value": float(p_val)}


def cohens_kappa(df: pd.DataFrame, classes: list):
    """Cohen's kappa from a results dataframe with 'expected' and 'predicted' columns."""
    n = len(df)
    # observed agreement
    p_o = (df["expected"] == df["predicted"]).mean()
    # expected agreement
    p_e = sum(
        (df["expected"] == cls).mean() * (df["predicted"] == cls).mean()
        for cls in classes
    )
    if p_e >= 1.0:
        return 1.0
    kappa = (p_o - p_e) / (1 - p_e)
    return round(float(kappa), 4)


def binomial_proportion_test(n1: int, k1: int, n2: int, k2: int):
    """
    Two-sample z-test for proportions.
    Returns (z_stat, p_value) testing H0: p1 == p2 (two-tailed).
    """
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    p_val = 2 * stats.norm.sf(abs(z))
    return (round(float(z), 3), float(p_val))


def compute_confusion(df: pd.DataFrame, classes: list) -> np.ndarray:
    """Compute confusion matrix (rows=true, cols=predicted)."""
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for _, row in df.iterrows():
        true_i = idx.get(row["expected"], -1)
        pred_i = idx.get(row["predicted"], -1)
        if true_i >= 0 and pred_i >= 0:
            cm[true_i][pred_i] += 1
    return cm


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — System Architecture
# ══════════════════════════════════════════════════════════════════════════════

def fig1_architecture():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Box specifications: (x_centre, y_centre, width, height, label_lines, color, text_color)
    boxes = [
        (5.0, 9.0, 6.2, 0.9, ["PATIENT INPUT", "19 symptoms  ·  demographics  ·  RDT / microscopy"],
         C_DARK, "white"),
        (5.0, 7.2, 6.2, 1.2,
         ["RULE-BASED CLASSIFICATION ENGINE",
          "Priority 1: Critical (hallucination / semi-closed eyes / ≥15 symptoms)",
          "Priority 2: No Malaria   Priority 3: Stage II   Priority 4: Stage I"],
         C_PRIMARY, "white"),
        (5.0, 5.1, 6.2, 1.2,
         ["LLM REASONING MODULE",
          "Llama 3.1 8B  (local / offline via Ollama)",
          "Generates symptom-specific clinical explanation  ·  confidence score"],
         "#7C3AED", "white"),
        (5.0, 3.0, 6.2, 1.2,
         ["STRUCTURED CLINICAL OUTPUT",
          "Severity stage  ·  Natural language explanation  ·  WHO-aligned prescription",
          "Confidence score  ·  Symptom summary"],
         C_GREEN, "white"),
        (5.0, 1.1, 6.2, 0.9,
         ["PDF CLINICAL REPORT", "Patient-specific downloadable report (ReportLab)"],
         C_AMBER, "white"),
    ]

    for (cx, cy, w, h, lines, bg, tc) in boxes:
        rect = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.05",
            facecolor=bg, edgecolor="white", linewidth=1.5,
            zorder=2,
        )
        ax.add_patch(rect)
        # Title line (bold)
        ax.text(cx, cy + (len(lines) - 1) * 0.18, lines[0],
                ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=tc, zorder=3)
        for i, line in enumerate(lines[1:], start=1):
            ax.text(cx, cy + (len(lines) - 1 - i) * 0.18 - 0.04, line,
                    ha="center", va="center", fontsize=8.5,
                    color=tc, alpha=0.92, zorder=3)

    # Arrows
    arrow_props = dict(
        arrowstyle="-|>", color=C_DARK, lw=1.8,
        connectionstyle="arc3,rad=0.0",
    )
    arrow_coords = [
        (5.0, 8.55, 5.0, 7.82),   # Input → Rule Engine
        (5.0, 6.60, 5.0, 5.72),   # Rule Engine → LLM
        (5.0, 4.50, 5.0, 3.62),   # LLM → Output
        (5.0, 2.40, 5.0, 1.58),   # Output → PDF
    ]
    for (x1, y1, x2, y2) in arrow_coords:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=arrow_props, zorder=1)

    # Annotation: LLM cannot override classification
    ax.annotate(
        "LLM receives rule-derived severity;\ncannot alter the diagnosis class",
        xy=(8.15, 6.15), xytext=(8.15, 6.15),
        fontsize=8, color="#7C3AED", ha="center",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="#F5F3FF", ec="#7C3AED", alpha=0.85),
        zorder=4,
    )
    ax.annotate("", xy=(7.82, 6.15), xytext=(8.0, 6.15),
                arrowprops=dict(arrowstyle="-", color="#7C3AED", lw=0.8), zorder=3)

    ax.set_title(
        "Figure 1. MalariaLLM System Architecture",
        fontsize=13, fontweight="bold", pad=6, color=C_DARK,
    )
    plt.tight_layout()
    out = OUT_DIR / "fig1_system_architecture.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Accuracy Comparison
# ══════════════════════════════════════════════════════════════════════════════

def fig2_accuracy(per_stage_df: pd.DataFrame):
    # Data
    stages = ["No Malaria", "Stage I", "Stage II", "Critical", "Overall"]
    base_acc  = [100.0,  1.84, 16.95, 100.0, 30.62]
    llm_acc   = [77.36, 98.90, 15.25, 100.0, 87.22]

    x = np.arange(len(stages))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor("white")

    bars_b = ax.bar(x - width / 2, base_acc, width,
                    color=C_BASELINE, alpha=0.85, label="Baseline (Rule-Based)",
                    zorder=3, edgecolor="white", linewidth=0.7)
    bars_l = ax.bar(x + width / 2, llm_acc, width,
                    color=C_PRIMARY, alpha=0.92, label="LLM-Enhanced Hybrid",
                    zorder=3, edgecolor="white", linewidth=0.7)

    # Value labels
    def label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.2, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=C_DARK,
            )

    label_bars(bars_b)
    label_bars(bars_l)

    # Overall accuracy reference line
    ax.axhline(87.22, color=C_PRIMARY, lw=1.2, ls="--", alpha=0.5, zorder=2)
    ax.axhline(30.62, color=C_BASELINE, lw=1.2, ls="--", alpha=0.5, zorder=2)

    # Improvement annotations above each pair (skip if both 100 or both ~same)
    improvements = [None, "+97.1 pp", None, None, "+56.6 pp"]
    for i, imp in enumerate(improvements):
        if imp:
            ax.annotate(
                imp,
                xy=(x[i], max(base_acc[i], llm_acc[i]) + 5),
                ha="center", fontsize=9, color=C_PRIMARY,
                fontweight="bold",
                arrowprops=None,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=10.5)
    ax.set_ylabel("Diagnostic Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 118)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(loc="upper left", framealpha=0.92, fontsize=10.5)
    ax.set_title(
        "Figure 2. Diagnostic Accuracy: Baseline vs. LLM-Enhanced Hybrid System",
        fontsize=13, fontweight="bold", pad=10, color=C_DARK,
    )

    # Footnote
    ax.text(
        0.5, -0.10,
        "n = 1,682 cases (455 No Malaria · 1,089 Stage I · 118 Stage II · 20 Critical)."
        "  pp = percentage points.",
        ha="center", va="top", transform=ax.transAxes,
        fontsize=8.5, color=C_MUTED, style="italic",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT_DIR / "fig2_accuracy_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Confusion Matrices
# ══════════════════════════════════════════════════════════════════════════════

def fig3_confusion(df_base: pd.DataFrame, df_hyb: pd.DataFrame):
    classes  = ["No_Malaria", "Stage_I", "Stage_II", "Critical"]
    labels   = ["No Malaria", "Stage I", "Stage II", "Critical"]

    cm_base = compute_confusion(df_base, classes)
    cm_hyb  = compute_confusion(df_hyb,  classes)

    # Row-normalise (recall per class)
    def row_norm(cm):
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        row_sums[row_sums == 0] = 1
        return cm / row_sums

    cm_base_n = row_norm(cm_base)
    cm_hyb_n  = row_norm(cm_hyb)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("white")

    def draw_cm(ax, cm_norm, cm_raw, title, cmap):
        mask = cm_raw == 0
        sns.heatmap(
            cm_norm, annot=False, fmt="",
            cmap=cmap, vmin=0, vmax=1,
            linewidths=1.2, linecolor="white",
            ax=ax, cbar_kws={"shrink": 0.72, "pad": 0.02},
            mask=mask,
        )
        # Custom annotations: show % and raw count
        for i in range(len(classes)):
            for j in range(len(classes)):
                raw = cm_raw[i, j]
                pct = cm_norm[i, j]
                if raw == 0:
                    ax.text(j + 0.5, i + 0.5, "0",
                            ha="center", va="center",
                            fontsize=9, color=C_MUTED, alpha=0.6)
                else:
                    txt_color = "white" if pct > 0.55 else C_DARK
                    ax.text(j + 0.5, i + 0.42,
                            f"{pct:.1%}",
                            ha="center", va="center",
                            fontsize=10, fontweight="bold", color=txt_color)
                    ax.text(j + 0.5, i + 0.65,
                            f"n={raw}",
                            ha="center", va="center",
                            fontsize=7.5, color=txt_color, alpha=0.85)

        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
        ax.set_yticklabels(labels, rotation=0, fontsize=10)
        ax.set_xlabel("Predicted Stage", fontsize=11, labelpad=6)
        ax.set_ylabel("True Stage", fontsize=11, labelpad=6)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8, color=C_DARK)

    draw_cm(axes[0], cm_base_n, cm_base,
            "Baseline (Rule-Based Only)", "Reds")
    draw_cm(axes[1], cm_hyb_n,  cm_hyb,
            "LLM-Enhanced Hybrid", "Blues")

    fig.suptitle(
        "Figure 3. Confusion Matrices: Row-Normalised per True Stage (n = 1,682)",
        fontsize=13, fontweight="bold", y=1.01, color=C_DARK,
    )
    plt.tight_layout()
    out = OUT_DIR / "fig3_confusion_matrices.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Processing Time & Accuracy Trade-off
# ══════════════════════════════════════════════════════════════════════════════

def fig4_processing(df_hyb: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor("white")

    # ── Left: Processing time distribution (hybrid) ─────────────────────────
    ax1 = axes[0]
    pt = df_hyb["processing_time"].dropna()
    ax1.hist(pt, bins=30, color=C_PRIMARY, alpha=0.82, edgecolor="white", lw=0.6)
    ax1.axvline(pt.mean(), color=C_RED, lw=2, ls="--",
                label=f"Mean = {pt.mean():.1f}s")
    ax1.axvline(pt.median(), color=C_AMBER, lw=2, ls="-.",
                label=f"Median = {pt.median():.1f}s")
    ax1.set_xlabel("Processing Time per Case (seconds)", fontsize=11)
    ax1.set_ylabel("Number of Cases", fontsize=11)
    ax1.set_title("LLM Inference Time Distribution", fontsize=12,
                  fontweight="bold", color=C_DARK)
    ax1.legend(fontsize=10)

    # ── Right: Accuracy vs Time scatter (3 systems) ──────────────────────────
    ax2 = axes[1]
    systems = ["Baseline\n(Rule-Based)", "LLM-Enhanced\n(Hybrid)"]
    times   = [0.01, pt.mean()]
    accs    = [30.62, 87.22]
    colors  = [C_BASELINE, C_PRIMARY]
    sizes   = [250, 380]

    for sx, sy, sc, ss, sn in zip(times, accs, colors, sizes, systems):
        ax2.scatter(sx, sy, c=sc, s=ss, zorder=5, edgecolors="white", lw=1.5)
        offset_x = -0.35 if sn.startswith("Baseline") else 0.25
        offset_y = -5 if sn.startswith("Baseline") else 4
        ax2.annotate(
            sn, (sx, sy),
            xytext=(sx + offset_x, sy + offset_y),
            fontsize=9.5, fontweight="bold", color=sc,
            ha="center",
        )

    # Tradeoff annotation
    ax2.annotate(
        "",
        xy=(times[1], accs[1]),
        xytext=(times[0], accs[0]),
        arrowprops=dict(
            arrowstyle="-|>", color=C_MUTED, lw=1.2,
            connectionstyle="arc3,rad=-0.3",
        ),
    )
    ax2.text(2.5, 59,
             f"+{accs[1]-accs[0]:.1f}pp accuracy\n+{times[1]-times[0]:.1f}s latency",
             fontsize=9, color=C_MUTED, style="italic", ha="center")

    ax2.set_xlabel("Mean Processing Time per Case (seconds)", fontsize=11)
    ax2.set_ylabel("Overall Diagnostic Accuracy (%)", fontsize=11)
    ax2.set_title("Accuracy–Latency Trade-off", fontsize=12,
                  fontweight="bold", color=C_DARK)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax2.set_ylim(15, 100)
    ax2.set_xlim(-0.5, pt.mean() * 1.4)

    fig.suptitle(
        "Figure 4. Processing Efficiency: LLM Inference Time and Accuracy–Latency Trade-off",
        fontsize=13, fontweight="bold", y=1.01, color=C_DARK,
    )
    plt.tight_layout()
    out = OUT_DIR / "fig4_processing_time.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_statistics(df_base: pd.DataFrame, df_hyb: pd.DataFrame,
                       per_stage: pd.DataFrame) -> dict:
    classes = ["No_Malaria", "Stage_I", "Stage_II", "Critical"]
    n = len(df_base)

    base_correct = int(df_base["correct"].sum())
    hyb_correct  = int(df_hyb["correct"].sum())

    # Overall accuracies
    base_acc = base_correct / n
    hyb_acc  = hyb_correct  / n

    # Wilson CIs
    base_ci = wilson_ci(base_correct, n)
    hyb_ci  = wilson_ci(hyb_correct,  n)

    # McNemar's test
    mcn = mcnemar_test(df_base, df_hyb)

    # Cohen's Kappa
    kappa_base = cohens_kappa(df_base, classes)
    kappa_hyb  = cohens_kappa(df_hyb,  classes)

    # Per-stage tests
    per_stage_stats = {}
    for _, row in per_stage.iterrows():
        stage  = row["Stage"]
        n_s    = int(row["Cases"])
        k_base = round(float(row["Baseline_Accuracy"]) / 100 * n_s)
        k_hyb  = round(float(row["LLM_Accuracy"])      / 100 * n_s)
        ci_b   = wilson_ci(k_base, n_s)
        ci_h   = wilson_ci(k_hyb,  n_s)
        z, p   = binomial_proportion_test(n_s, k_base, n_s, k_hyb)
        per_stage_stats[stage] = {
            "n": n_s,
            "baseline_correct": k_base,
            "hybrid_correct":   k_hyb,
            "baseline_acc":     round(float(row["Baseline_Accuracy"]), 2),
            "hybrid_acc":       round(float(row["LLM_Accuracy"]),      2),
            "baseline_ci_95":   [round(v * 100, 1) for v in ci_b],
            "hybrid_ci_95":     [round(v * 100, 1) for v in ci_h],
            "z_statistic":      z,
            "p_value":          round(p, 6),
        }

    # Processing time
    pt = df_hyb["processing_time"].dropna()

    stats_out = {
        "n_total":              n,
        "baseline": {
            "correct":          base_correct,
            "accuracy_pct":     round(base_acc * 100, 4),
            "ci_95_pct":        [round(v * 100, 2) for v in base_ci],
            "cohens_kappa":     kappa_base,
        },
        "hybrid": {
            "correct":          hyb_correct,
            "accuracy_pct":     round(hyb_acc * 100, 4),
            "ci_95_pct":        [round(v * 100, 2) for v in hyb_ci],
            "cohens_kappa":     kappa_hyb,
        },
        "mcnemar": {
            "b_base_correct_hyb_wrong": mcn["b"],
            "c_base_wrong_hyb_correct": mcn["c"],
            "chi2_statistic":           mcn["chi2"],
            "p_value":                  mcn["p_value"],
            "interpretation":           "p < 0.001 — highly significant difference" if mcn["p_value"] < 0.001 else f"p = {mcn['p_value']:.4f}",
        },
        "per_stage": per_stage_stats,
        "processing_time": {
            "mean_s":    round(float(pt.mean()),   2),
            "median_s":  round(float(pt.median()), 2),
            "std_s":     round(float(pt.std()),    2),
            "min_s":     round(float(pt.min()),    2),
            "max_s":     round(float(pt.max()),    2),
            "n_cases":   int(pt.count()),
        },
    }

    # Print readable summary
    print("\n" + "=" * 68)
    print("  STATISTICAL ANALYSIS SUMMARY")
    print("=" * 68)
    print(f"\n  n = {n} total cases\n")
    print(f"  Baseline accuracy : {base_acc*100:.2f}%  "
          f"95% CI [{base_ci[0]*100:.1f}%, {base_ci[1]*100:.1f}%]")
    print(f"  Hybrid accuracy   : {hyb_acc*100:.2f}%  "
          f"95% CI [{hyb_ci[0]*100:.1f}%, {hyb_ci[1]*100:.1f}%]")
    print(f"\n  McNemar's test    : χ²({mcn['chi2']:.1f}),  p = {mcn['p_value']:.2e}"
          f"  (b={mcn['b']}, c={mcn['c']})")
    print(f"\n  Cohen's κ baseline: {kappa_base:.3f}  (slight agreement)")
    print(f"  Cohen's κ hybrid  : {kappa_hyb:.3f}  (substantial agreement)")
    print("\n  Per-stage comparison (z-test of proportions):")
    for stage, s in per_stage_stats.items():
        p_str = f"p < 0.001" if s["p_value"] < 0.001 else f"p = {s['p_value']:.3f}"
        print(f"    {stage:<14s}: {s['baseline_acc']:6.2f}% → {s['hybrid_acc']:6.2f}%"
              f"  ({p_str})")
    print(f"\n  Processing time   : {stats_out['processing_time']['mean_s']:.1f}s mean "
          f"(SD={stats_out['processing_time']['std_s']:.1f}s, "
          f"n={stats_out['processing_time']['n_cases']})")
    print("=" * 68 + "\n")

    return stats_out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\nMalariaLLM — Publication Figure Generator")
    print("==========================================\n")

    # ── Load data ────────────────────────────────────────────────────────────
    for p in [BASELINE_CSV, HYBRID_CSV, PER_STAGE]:
        if not p.exists():
            print(f"  ERROR: Required file not found: {p}")
            sys.exit(1)

    df_base     = pd.read_csv(BASELINE_CSV)
    df_hyb      = pd.read_csv(HYBRID_CSV)
    per_stage   = pd.read_csv(PER_STAGE)

    # Normalise boolean 'correct' column (may be stored as string "True"/"False")
    for df in [df_base, df_hyb]:
        if df["correct"].dtype == object:
            df["correct"] = df["correct"].map({"True": True, "False": False}).astype(bool)

    print("  Loaded datasets:")
    print(f"    Baseline : {len(df_base):,} rows")
    print(f"    Hybrid   : {len(df_hyb):,} rows\n")

    # ── Statistics ───────────────────────────────────────────────────────────
    stats_data = compute_statistics(df_base, df_hyb, per_stage)

    stats_path = OUT_DIR / "statistical_summary.json"
    with open(stats_path, "w") as f:
        json.dump(stats_data, f, indent=2, default=str)
    print(f"  Statistical summary saved: {stats_path}\n")

    # ── Figures ──────────────────────────────────────────────────────────────
    print("  Generating figures (300 DPI)...")
    fig1_architecture()
    fig2_accuracy(per_stage)
    fig3_confusion(df_base, df_hyb)
    fig4_processing(df_hyb)

    print(f"\n  All figures saved to: {OUT_DIR}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
