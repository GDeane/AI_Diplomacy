"""
Analyze and visualize Theory of Mind probe results.

Loads all _scores.json files from the probe directory structure,
builds a DataFrame, and generates publication-quality figures.

Usage:
    python analyze_tom_results.py
"""

import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Betrayal definitions
# ---------------------------------------------------------------------------
BETRAYAL_DEFINITIONS = {
    ("my_test_run_4", "F1901M", "AUSTRIA", "ITALY"): {
        "has_betrayal": True,
        "betrayer": "ITALY",
        "victim": "AUSTRIA",
        "label": "run4 F1901M\nAustria→Italy",
        "short_label": "R4 AUS→ITA",
        "detailed_label": "R4 AUS→ITA\nEasy-to-spot betrayal",
        "description": "Italy stabs Austria by attacking home SC from Tyrolia",
        "detection_type": "commission",
        # Betrayal detected if ANY of these orders appear in predictions
        "betrayal_orders": ["A TYR - TRI", "A TYR - VIE"],
    },
    ("my_test_run_14", "F1903M", "AUSTRIA", "ITALY"): {
        "has_betrayal": False,
        "betrayer": None,
        "victim": None,
        "label": "run14 F1903M\nAustria→Italy",
        "short_label": "R14 AUS→ITA",
        "detailed_label": "R14 AUS→ITA\nCooperation case",
        "description": "No betrayal (control case)",
        "detection_type": "none",
        # False positive if this order appears
        "false_positive_orders": ["A VEN - TRI"],
    },
    ("my_test_run_14", "S1902M", "ENGLAND", "FRANCE"): {
        "has_betrayal": True,
        "betrayer": "FRANCE",
        "victim": "ENGLAND",
        "label": "run14 S1902M\nEngland→France",
        "short_label": "R14 ENG→FRA",
        "detailed_label": "R14 ENG→FRA\nActive deception",
        "description": "France betrays England by violating Channel DMZ",
        "detection_type": "commission",
        "betrayal_orders": ["F BRE - ENG", "A PIC - BEL"],
    },
    ("my_test_run_14", "S1902M", "TURKEY", "AUSTRIA"): {
        "has_betrayal": True,
        "betrayer": "AUSTRIA",
        "victim": "TURKEY",
        "label": "run14 S1902M\nTurkey→Austria",
        "short_label": "R14 TUR→AUS",
        "detailed_label": "R14 TUR→AUS\nActive deception",
        "description": "Austria betrays Turkey by withholding promised support",
        "detection_type": "omission",
        # Betrayal detected if NONE of these orders appear in predictions
        "expected_support_orders": ["A BUD S A BUL - RUM", "A SER S A BUL - RUM"],
    },
    ("my_test_run_14", "S1904M", "ITALY", "AUSTRIA"): {
        "has_betrayal": True,
        "betrayer": "AUSTRIA",
        "victim": "ITALY",
        "label": "run14 S1904M\nItaly→Austria",
        "short_label": "R14 ITA→AUS",
        "detailed_label": "R14 ITA→AUS\nDeception with warnings",
        "description": "Austria stabs Italy by attacking Venice",
        "detection_type": "commission",
        "betrayal_orders": ["A TRI - VEN"],
    },
}

MODEL_SHORT = {
    "gemini-3.1-pro-preview": "Gemini",
    "openrouter-anthropic/claude-opus-4.6": "Claude",
    "openrouter-openai/gpt-5.4": "GPT",
    "baseline_neutral_random": "Baseline",
}

MODEL_COLORS = {
    "Gemini": "#4285F4",
    "Claude": "#D97706",
    "GPT": "#10B981",
    "Baseline": "#999999",
}

PROBE_HATCHES = {
    "A": "",
    "B": "//",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_scores() -> pd.DataFrame:
    """Load all _scores.json files into a flat DataFrame with one row per prediction."""
    rows = []
    score_files = sorted(glob.glob("results/my_test_run_*/tom_probes/probe_*/*/*_scores.json"))
    for sf in score_files:
        with open(sf) as f:
            data = json.load(f)

        meta = data["metadata"]
        run_dir = os.path.basename(meta["run_dir"])
        phase = meta["phase"]
        predictor = meta["predictor_power"]
        target = meta["target_power"]
        probe_type = meta["probe_type"]

        situation_key = (run_dir, phase, predictor, target)
        betrayal_def = BETRAYAL_DEFINITIONS.get(situation_key, {})

        for pred in data["per_prediction"]:
            if pred.get("error"):
                continue
            scores = pred["scores"]
            model_short = MODEL_SHORT.get(pred["model"], pred["model"])

            row = {
                "run_dir": run_dir,
                "phase": phase,
                "predictor": predictor,
                "target": target,
                "probe_type": probe_type,
                "model": model_short,
                "model_id": pred["model"],
                "run_index": pred["run_index"],
                "situation": betrayal_def.get("short_label", f"{predictor}→{target}"),
                "has_betrayal": betrayal_def.get("has_betrayal", None),
                "rel_accuracy": scores["relationships"]["normalized_accuracy"],
                "rel_bilateral_accuracy": scores.get("relationship_bilateral", {}).get("normalized_accuracy", None),
                "rel_exact_matches": scores["relationships"]["exact_matches"],
                "rel_total": scores["relationships"]["total"],
                "order_exact_jaccard": scores["orders"]["exact_jaccard"],
                "order_dest_accuracy": scores["orders"]["destination_accuracy"],
                "agr_bilateral_jaccard": scores.get("agreements_bilateral", {}).get("jaccard", None),
                "agr_third_party_jaccard": scores.get("agreements_third_party", {}).get("jaccard", None),
                "pred_orders": scores["orders"]["pred_orders"],
                "gt_orders": scores["orders"]["gt_orders"],
            }

            # Betrayal detection
            row["detected_betrayal"] = check_betrayal_detection(
                row["pred_orders"], situation_key
            )

            rows.append(row)

    return pd.DataFrame(rows)


def check_betrayal_detection(pred_orders: list, situation_key: tuple) -> bool | None:
    """Check if a prediction detected the betrayal for a given situation."""
    bdef = BETRAYAL_DEFINITIONS.get(situation_key)
    if not bdef:
        return None

    # Empty/failed predictions should not count as detection
    if not pred_orders:
        return False if bdef["has_betrayal"] else None

    pred_set = set(pred_orders)

    if not bdef["has_betrayal"]:
        # Control case: return False if false positive, True if correctly no betrayal
        fp_orders = bdef.get("false_positive_orders", [])
        return not any(o in pred_set for o in fp_orders)

    if bdef["detection_type"] == "commission":
        return any(o in pred_set for o in bdef["betrayal_orders"])
    elif bdef["detection_type"] == "omission":
        # Betrayal detected if NONE of the expected support orders appear
        return not any(o in pred_set for o in bdef["expected_support_orders"])

    return None


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_model_comparison(df: pd.DataFrame, output_dir: str):
    """Bar chart: mean metrics per model, grouped by probe type. One PNG per metric."""
    for metric, title, fname in [
        ("rel_bilateral_accuracy", "Bilateral Relationship Accuracy", "model_comparison_rel_bilateral"),
        ("order_exact_jaccard", "Order Exact Jaccard", "model_comparison_order_jaccard"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        sub = df.dropna(subset=[metric]) if metric == "rel_bilateral_accuracy" else df
        grouped = sub.groupby(["model", "probe_type"])[metric]
        means = grouped.mean().unstack("probe_type")
        stds = grouped.std().unstack("probe_type")

        x = np.arange(len(means.index))
        width = 0.35

        for i, probe in enumerate(["A", "B"]):
            if probe not in means.columns:
                continue
            ax.bar(
                x + i * width, means[probe], width,
                yerr=stds[probe], capsize=4,
                color=[MODEL_COLORS.get(m, "#999") for m in means.index],
                hatch=PROBE_HATCHES[probe],
                edgecolor="white", linewidth=0.5,
                label=f"Probe {probe}" if i == 0 or probe == "B" else "",
                alpha=0.85 if probe == "A" else 0.6,
            )

        ax.set_ylabel(title)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(means.index)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{title}: Probe A vs Probe B", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{fname}.png"))
        plt.close()


def plot_probe_ab_lift(df: pd.DataFrame, output_dir: str):
    """Paired comparison showing the lift from diary access (Probe A vs B)."""
    for metric, title, fname in [
        ("rel_bilateral_accuracy", "Bilateral Relationship Accuracy", "probe_ab_lift_rel_bilateral"),
        ("order_exact_jaccard", "Order Exact Jaccard", "probe_ab_lift_order_jaccard"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        sub = df.dropna(subset=[metric]) if metric == "rel_bilateral_accuracy" else df
        probe_means = sub.groupby(["model", "probe_type"])[metric].mean().unstack("probe_type")
        probe_stds = sub.groupby(["model", "probe_type"])[metric].std().unstack("probe_type")

        for model in probe_means.index:
            if "A" in probe_means.columns and "B" in probe_means.columns:
                a_val = probe_means.loc[model, "A"]
                b_val = probe_means.loc[model, "B"]
                a_std = probe_stds.loc[model, "A"]
                b_std = probe_stds.loc[model, "B"]
                color = MODEL_COLORS.get(model, "#999")
                ax.errorbar([0, 1], [b_val, a_val], yerr=[b_std, a_std],
                            fmt="o-", color=color, linewidth=2, markersize=8,
                            capsize=4, label=model)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Probe B\n(orders only)", "Probe A\n(diary + orders)"])
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"Effect of Diary Access on {title}", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{fname}.png"))
        plt.close()


def plot_situation_heatmap(df: pd.DataFrame, output_dir: str):
    """Heatmap: model × situation for bilateral relationship accuracy (Probe A only)."""
    probe_a = df[df["probe_type"] == "A"].dropna(subset=["rel_bilateral_accuracy"])
    pivot = probe_a.groupby(["model", "situation"])["rel_bilateral_accuracy"].mean().unstack("situation")

    # Reorder columns by betrayal definition order, using detailed labels
    short_to_detailed = {v["short_label"]: v["detailed_label"] for v in BETRAYAL_DEFINITIONS.values()}
    situation_order = [v["short_label"] for v in BETRAYAL_DEFINITIONS.values()]
    pivot = pivot[[c for c in situation_order if c in pivot.columns]]
    detailed_labels = [short_to_detailed.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(detailed_labels, rotation=30, ha="center", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="black" if val > 0.6 else "white", fontsize=11)

    plt.colorbar(im, ax=ax, label="Bilateral Relationship Accuracy")
    ax.set_title("Bilateral Relationship Accuracy by Model and Situation")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "situation_heatmap.png"))
    plt.close()


def plot_betrayal_detection(df: pd.DataFrame, output_dir: str):
    """Bar chart: betrayal detection rate per model, Probe A vs B."""
    # Separate betrayal cases from control
    betrayal_df = df[df["has_betrayal"] == True]
    control_df = df[df["has_betrayal"] == False]

    # Detection rate by model (Probe A vs B)
    fig, ax = plt.subplots(figsize=(6, 5))
    det = betrayal_df.groupby(["model", "probe_type"])["detected_betrayal"].mean().unstack("probe_type")
    x = np.arange(len(det.index))
    width = 0.35
    for i, probe in enumerate(["A", "B"]):
        if probe not in det.columns:
            continue
        ax.bar(
            x + i * width, det[probe], width,
            color=[MODEL_COLORS.get(m, "#999") for m in det.index],
            hatch=PROBE_HATCHES[probe],
            edgecolor="white", linewidth=0.5,
            alpha=0.85 if probe == "A" else 0.6,
            label=f"Probe {probe}",
        )
    ax.set_ylabel("Detection Rate")
    ax.set_title("Betrayal Detection Rate\n(4 betrayal situations)")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(det.index)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "betrayal_detection_by_model.png"))
    plt.close()

    # Per-situation breakdown (Probe A only)
    fig, ax = plt.subplots(figsize=(10, 5))
    probe_a_betrayal = betrayal_df[(betrayal_df["probe_type"] == "A") & (betrayal_df["model"] != "Baseline")]
    sit_det = probe_a_betrayal.groupby(["situation", "model"])["detected_betrayal"].mean().unstack("model")
    betrayal_short_labels = [v["short_label"] for v in BETRAYAL_DEFINITIONS.values() if v["has_betrayal"]]
    betrayal_detailed_labels = [v["detailed_label"] for v in BETRAYAL_DEFINITIONS.values() if v["has_betrayal"]]
    sit_det = sit_det.reindex(betrayal_short_labels)
    sit_det.plot(kind="bar", ax=ax, color=[MODEL_COLORS.get(m, "#999") for m in sit_det.columns],
                 edgecolor="white", linewidth=0.5, rot=0)
    ax.set_xticklabels(betrayal_detailed_labels, rotation=30, ha="center", fontsize=10)
    ax.set_ylabel("Detection Rate")
    ax.set_title("Betrayal Detection by Situation")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "betrayal_detection_by_situation.png"))
    plt.close()

    # Print summary
    print("\n=== BETRAYAL DETECTION SUMMARY ===")
    print("\nBetrayal cases (Probe A):")
    for sit_key, bdef in BETRAYAL_DEFINITIONS.items():
        if not bdef["has_betrayal"]:
            continue
        sit_df = probe_a_betrayal[probe_a_betrayal["situation"] == bdef["short_label"]]
        for model in ["Gemini", "Claude", "GPT", "Baseline"]:
            m_df = sit_df[sit_df["model"] == model]
            if len(m_df) > 0:
                rate = m_df["detected_betrayal"].mean()
                n = len(m_df)
                print(f"  {bdef['short_label']:15s} {model:8s}: {rate:.0%} ({int(rate*n)}/{n})")

    print("\nControl case (Probe A) — should be ~100% (no false positives):")
    control_a = control_df[control_df["probe_type"] == "A"]
    for model in ["Gemini", "Claude", "GPT", "Baseline"]:
        m_df = control_a[control_a["model"] == model]
        if len(m_df) > 0:
            rate = m_df["detected_betrayal"].mean()
            n = len(m_df)
            print(f"  {model:8s}: {rate:.0%} correct ({int(rate*n)}/{n} avoided false positive)")


def plot_per_situation_orders(df: pd.DataFrame, output_dir: str):
    """Bar chart: order exact Jaccard per situation, by model and probe type."""
    fig, ax = plt.subplots(figsize=(12, 5))

    situation_order = [v["short_label"] for v in BETRAYAL_DEFINITIONS.values()]
    probe_a = df[df["probe_type"] == "A"]
    pivot_mean = probe_a.groupby(["situation", "model"])["order_exact_jaccard"].mean().unstack("model")
    pivot_std = probe_a.groupby(["situation", "model"])["order_exact_jaccard"].std().unstack("model")
    pivot_mean = pivot_mean.reindex([s for s in situation_order if s in pivot_mean.index])
    pivot_std = pivot_std.reindex([s for s in situation_order if s in pivot_std.index])

    models_present = [m for m in pivot_mean.columns]
    x = np.arange(len(pivot_mean.index))
    n_models = len(models_present)
    width = 0.8 / n_models

    for i, model in enumerate(models_present):
        ax.bar(x + i * width, pivot_mean[model], width,
               yerr=pivot_std[model], capsize=3,
               color=MODEL_COLORS.get(model, "#999"),
               edgecolor="white", linewidth=0.5, label=model)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(pivot_mean.index, rotation=30, ha="right")
    ax.set_ylabel("Order Exact Jaccard")
    ax.set_title("Order Prediction Accuracy by Situation (Probe A)")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_situation_orders.png"))
    plt.close()


def plot_agreement_bilateral_vs_thirdparty(df: pd.DataFrame, output_dir: str):
    """Bar chart: bilateral vs third-party agreement Jaccard by model (Probe A)."""
    probe_a = df[df["probe_type"] == "A"]

    fig, ax = plt.subplots(figsize=(8, 5))

    models = ["Gemini", "Claude", "GPT"]
    models_present = [m for m in models if m in probe_a["model"].unique()]
    x = np.arange(len(models_present))
    width = 0.35

    for i, (col, label) in enumerate([
        ("agr_bilateral_jaccard", "Bilateral"),
        ("agr_third_party_jaccard", "Third-Party"),
    ]):
        sub = probe_a.dropna(subset=[col])
        means = sub.groupby("model")[col].mean().reindex(models_present)
        stds = sub.groupby("model")[col].std().reindex(models_present)
        ax.bar(
            x + i * width, means, width,
            yerr=stds, capsize=4,
            color=[MODEL_COLORS.get(m, "#999") for m in models_present],
            hatch="" if i == 0 else "//",
            edgecolor="white", linewidth=0.5,
            alpha=0.85 if i == 0 else 0.6,
            label=label,
        )

    ax.set_ylabel("Agreement Jaccard")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models_present)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Agreement Prediction: Bilateral vs Third-Party (Probe A)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "agreement_bilateral_vs_thirdparty.png"))
    plt.close()


def plot_gemini_probe_ab_detected(df: pd.DataFrame, output_dir: str):
    """Gemini Probe A vs B for the two situations where it detected betrayal."""
    gemini = df[df["model"] == "Gemini"]
    situations = ["R4 AUS→ITA", "R14 ITA→AUS"]
    metrics = [
        ("detected_betrayal", "Betrayal Detection Rate"),
        ("rel_bilateral_accuracy", "Bilateral Relationship Accuracy"),
        ("order_exact_jaccard", "Order Exact Jaccard"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    for ax, (metric, title) in zip(axes, metrics):
        x = np.arange(len(situations))
        width = 0.35

        for i, probe in enumerate(["A", "B"]):
            means, stds = [], []
            for sit in situations:
                sub = gemini[(gemini["situation"] == sit) & (gemini["probe_type"] == probe)]
                vals = sub[metric].dropna()
                means.append(vals.mean() if len(vals) > 0 else 0)
                stds.append(vals.std() if len(vals) > 0 else 0)
            ax.bar(
                x + i * width, means, width,
                yerr=stds, capsize=5,
                color="#4285F4",
                hatch=PROBE_HATCHES[probe],
                edgecolor="white", linewidth=0.5,
                alpha=0.85 if probe == "A" else 0.5,
                label=f"Probe {probe}",
            )

        ax.set_ylabel(title)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(situations)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Gemini: Probe A vs B on Detected Betrayal Situations", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gemini_probe_ab_detected.png"))
    plt.close()


def print_summary_table(df: pd.DataFrame):
    """Print a summary table of all metrics."""
    print("\n=== SUMMARY TABLE ===\n")
    print(f"{'Probe':>5s} {'Model':>8s} {'Rel Bil':>8s} {'Ord Jac':>8s} {'Dest Acc':>9s} {'N':>4s}")
    print("-" * 50)
    for probe in ["A", "B"]:
        for model in ["Gemini", "Claude", "GPT", "Baseline"]:
            sub = df[(df["probe_type"] == probe) & (df["model"] == model)]
            if len(sub) == 0:
                continue
            bil = sub["rel_bilateral_accuracy"].dropna()
            bil_str = f"{bil.mean():>8.3f}" if len(bil) > 0 else "     N/A"
            print(f"{probe:>5s} {model:>8s} "
                  f"{bil_str} "
                  f"{sub['order_exact_jaccard'].mean():>8.3f} "
                  f"{sub['order_dest_accuracy'].mean():>9.3f} "
                  f"{len(sub):>4d}")

    print("\n=== PER-SITUATION BREAKDOWN (Probe A) ===\n")
    probe_a = df[df["probe_type"] == "A"]
    situation_order = [v["short_label"] for v in BETRAYAL_DEFINITIONS.values()]
    for sit in situation_order:
        sit_df = probe_a[probe_a["situation"] == sit]
        if len(sit_df) == 0:
            continue
        bdef = [v for v in BETRAYAL_DEFINITIONS.values() if v["short_label"] == sit][0]
        betrayal_str = f"BETRAYAL ({bdef['description']})" if bdef["has_betrayal"] else "NO BETRAYAL (control)"
        print(f"\n  {sit} — {betrayal_str}")
        print(f"  {'Model':>8s} {'Rel Bil':>8s} {'Ord Jac':>8s} {'Dest Acc':>9s} {'N':>4s}")
        for model in ["Gemini", "Claude", "GPT", "Baseline"]:
            m_df = sit_df[sit_df["model"] == model]
            if len(m_df) == 0:
                continue
            bil = m_df["rel_bilateral_accuracy"].dropna()
            bil_str = f"{bil.mean():>8.3f}" if len(bil) > 0 else "     N/A"
            print(f"  {model:>8s} "
                  f"{bil_str} "
                  f"{m_df['order_exact_jaccard'].mean():>8.3f} "
                  f"{m_df['order_dest_accuracy'].mean():>9.3f} "
                  f"{len(m_df):>4d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    from datetime import datetime

    df = load_all_scores()
    print(f"Loaded {len(df)} predictions from {df['run_dir'].nunique()} game runs")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Probe types: {sorted(df['probe_type'].unique())}")
    print(f"Situations: {sorted(df['situation'].unique())}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"visualization_results/tom_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Print summary
    print_summary_table(df)

    # Generate plots
    plot_model_comparison(df, output_dir)
    plot_probe_ab_lift(df, output_dir)
    plot_situation_heatmap(df, output_dir)
    plot_betrayal_detection(df, output_dir)
    plot_agreement_bilateral_vs_thirdparty(df, output_dir)
    plot_per_situation_orders(df, output_dir)
    plot_gemini_probe_ab_detected(df, output_dir)

    # Save DataFrame for further analysis
    csv_path = os.path.join(output_dir, "all_predictions.csv")
    df.drop(columns=["pred_orders", "gt_orders"]).to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")
    print(f"Figures saved: {output_dir}/")


if __name__ == "__main__":
    main()
