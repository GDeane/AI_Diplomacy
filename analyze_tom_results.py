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
        "description": "Austria stabs Italy by attacking Venice",
        "detection_type": "commission",
        "betrayal_orders": ["A TRI - VEN"],
    },
}

MODEL_SHORT = {
    "gemini-3.1-pro-preview": "Gemini",
    "openrouter-anthropic/claude-opus-4.6": "Claude",
    "openrouter-openai/gpt-5.4": "GPT",
}

MODEL_COLORS = {
    "Gemini": "#4285F4",
    "Claude": "#D97706",
    "GPT": "#10B981",
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
                "rel_exact_matches": scores["relationships"]["exact_matches"],
                "rel_total": scores["relationships"]["total"],
                "order_exact_jaccard": scores["orders"]["exact_jaccard"],
                "order_dest_accuracy": scores["orders"]["destination_accuracy"],
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
    """Bar chart: mean metrics per model, grouped by probe type."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in [
        (axes[0], "rel_accuracy", "Relationship Accuracy"),
        (axes[1], "order_exact_jaccard", "Order Exact Jaccard"),
    ]:
        grouped = df.groupby(["model", "probe_type"])[metric]
        means = grouped.mean().unstack("probe_type")
        sems = grouped.sem().unstack("probe_type")

        x = np.arange(len(means.index))
        width = 0.35

        for i, probe in enumerate(["A", "B"]):
            if probe not in means.columns:
                continue
            bars = ax.bar(
                x + i * width, means[probe], width,
                yerr=sems[probe], capsize=4,
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

    fig.suptitle("Model Comparison: Probe A (diary+orders) vs Probe B (orders only)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    plt.close()


def plot_probe_ab_lift(df: pd.DataFrame, output_dir: str):
    """Paired comparison showing the lift from diary access (Probe A vs B)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in [
        (axes[0], "rel_accuracy", "Relationship Accuracy"),
        (axes[1], "order_exact_jaccard", "Order Exact Jaccard"),
    ]:
        probe_means = df.groupby(["model", "probe_type"])[metric].mean().unstack("probe_type")

        for model in probe_means.index:
            if "A" in probe_means.columns and "B" in probe_means.columns:
                a_val = probe_means.loc[model, "A"]
                b_val = probe_means.loc[model, "B"]
                color = MODEL_COLORS.get(model, "#999")
                ax.plot([0, 1], [b_val, a_val], "o-", color=color, linewidth=2,
                        markersize=8, label=model)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Probe B\n(orders only)", "Probe A\n(diary + orders)"])
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Effect of Diary Access on Prediction Accuracy", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "probe_ab_lift.png"))
    plt.close()


def plot_situation_heatmap(df: pd.DataFrame, output_dir: str):
    """Heatmap: model × situation for relationship accuracy (Probe A only)."""
    probe_a = df[df["probe_type"] == "A"]
    pivot = probe_a.groupby(["model", "situation"])["rel_accuracy"].mean().unstack("situation")

    # Reorder columns by betrayal definition order
    situation_order = [v["short_label"] for v in BETRAYAL_DEFINITIONS.values()]
    pivot = pivot[[c for c in situation_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="black" if val > 0.6 else "white", fontsize=11)

    plt.colorbar(im, ax=ax, label="Relationship Accuracy")
    ax.set_title("Relationship Accuracy by Model × Situation (Probe A)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "situation_heatmap.png"))
    plt.close()


def plot_betrayal_detection(df: pd.DataFrame, output_dir: str):
    """Bar chart: betrayal detection rate per model, Probe A vs B."""
    # Separate betrayal cases from control
    betrayal_df = df[df["has_betrayal"] == True]
    control_df = df[df["has_betrayal"] == False]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: detection rate for betrayal cases
    ax = axes[0]
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

    # Right: per-situation breakdown (Probe A only)
    ax = axes[1]
    probe_a_betrayal = betrayal_df[betrayal_df["probe_type"] == "A"]
    sit_det = probe_a_betrayal.groupby(["situation", "model"])["detected_betrayal"].mean().unstack("model")
    sit_det = sit_det.reindex(
        [v["short_label"] for v in BETRAYAL_DEFINITIONS.values() if v["has_betrayal"]],
    )
    sit_det.plot(kind="bar", ax=ax, color=[MODEL_COLORS.get(m, "#999") for m in sit_det.columns],
                 edgecolor="white", linewidth=0.5, rot=30)
    ax.set_ylabel("Detection Rate")
    ax.set_title("Betrayal Detection by Situation\n(Probe A only)")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "betrayal_detection.png"))
    plt.close()

    # Print summary
    print("\n=== BETRAYAL DETECTION SUMMARY ===")
    print("\nBetrayal cases (Probe A):")
    for sit_key, bdef in BETRAYAL_DEFINITIONS.items():
        if not bdef["has_betrayal"]:
            continue
        sit_df = probe_a_betrayal[probe_a_betrayal["situation"] == bdef["short_label"]]
        for model in ["Gemini", "Claude", "GPT"]:
            m_df = sit_df[sit_df["model"] == model]
            if len(m_df) > 0:
                rate = m_df["detected_betrayal"].mean()
                n = len(m_df)
                print(f"  {bdef['short_label']:15s} {model:8s}: {rate:.0%} ({int(rate*n)}/{n})")

    print("\nControl case (Probe A) — should be ~100% (no false positives):")
    control_a = control_df[control_df["probe_type"] == "A"]
    for model in ["Gemini", "Claude", "GPT"]:
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
    pivot = probe_a.groupby(["situation", "model"])["order_exact_jaccard"].mean().unstack("model")
    pivot = pivot.reindex([s for s in situation_order if s in pivot.index])

    pivot.plot(kind="bar", ax=ax, color=[MODEL_COLORS.get(m, "#999") for m in pivot.columns],
               edgecolor="white", linewidth=0.5, rot=30)
    ax.set_ylabel("Order Exact Jaccard")
    ax.set_title("Order Prediction Accuracy by Situation (Probe A)")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_situation_orders.png"))
    plt.close()


def print_summary_table(df: pd.DataFrame):
    """Print a summary table of all metrics."""
    print("\n=== SUMMARY TABLE ===\n")
    print(f"{'Probe':>5s} {'Model':>8s} {'Rel Acc':>8s} {'Ord Jac':>8s} {'Dest Acc':>9s} {'N':>4s}")
    print("-" * 50)
    for probe in ["A", "B"]:
        for model in ["Gemini", "Claude", "GPT"]:
            sub = df[(df["probe_type"] == probe) & (df["model"] == model)]
            if len(sub) == 0:
                continue
            print(f"{probe:>5s} {model:>8s} "
                  f"{sub['rel_accuracy'].mean():>8.3f} "
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
        print(f"  {'Model':>8s} {'Rel Acc':>8s} {'Ord Jac':>8s} {'Dest Acc':>9s} {'N':>4s}")
        for model in ["Gemini", "Claude", "GPT"]:
            m_df = sit_df[sit_df["model"] == model]
            if len(m_df) == 0:
                continue
            print(f"  {model:>8s} "
                  f"{m_df['rel_accuracy'].mean():>8.3f} "
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
    plot_per_situation_orders(df, output_dir)

    # Save DataFrame for further analysis
    csv_path = os.path.join(output_dir, "all_predictions.csv")
    df.drop(columns=["pred_orders", "gt_orders"]).to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")
    print(f"Figures saved: {output_dir}/")


if __name__ == "__main__":
    main()
