"""Visualisation helpers for the goalkeeper pipeline outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from config import OUTPUTS_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

VISUALS_DIR = OUTPUTS_DIR / "visuals"
VISUALS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_SCORES = [
    "shot_stopping_score",
    "handling_score",
    "distribution_score",
    "sweeping_score",
    "reliability_score",
    "consistency_score",
]

CATEGORY_LABELS = [
    "Shot Stopping",
    "Handling",
    "Distribution",
    "Sweeping",
    "Reliability",
    "Consistency",
]

PALETTE = sns.color_palette("Set2", 8)


_CLOSE_AFTER_SAVE = True


def _save(fig: plt.Figure, name: str) -> Path:
    path = VISUALS_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    if _CLOSE_AFTER_SAVE:
        plt.close(fig)
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# 1.  Label distribution (origin dataset)
# ---------------------------------------------------------------------------

def plot_label_distribution(origin_df: pd.DataFrame) -> plt.Figure:
    """Bar chart of goalkeeper label distribution (PLAYS / BENCH / DROPPED / STAYED)."""
    labels = origin_df["label"].astype(str).str.upper().str.strip()
    order = ["PLAYS", "BENCH", "DROPPED", "STAYED"]
    counts = labels.value_counts().reindex(order, fill_value=0)

    colors = {"PLAYS": "#2ecc71", "BENCH": "#f39c12", "DROPPED": "#e74c3c", "STAYED": "#95a5a6"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index, counts.values, color=[colors[l] for l in counts.index], edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 3, str(v), ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylabel("Number of Goalkeepers")
    ax.set_title("Goalkeeper Label Distribution (Origin Dataset)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "01_label_distribution.png")
    return fig


# ---------------------------------------------------------------------------
# 2.  Category score distributions (current players)
# ---------------------------------------------------------------------------

def plot_score_distributions(current_df: pd.DataFrame) -> plt.Figure:
    """Violin + strip plot for each category score."""
    available = [c for c in CATEGORY_SCORES if c in current_df.columns]
    melted = current_df[available].melt(var_name="Category", value_name="Score")
    melted["Category"] = melted["Category"].str.replace("_score", "").str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=melted, x="Category", y="Score", hue="Category", inner=None, palette="Set2", alpha=0.4, legend=False, ax=ax)
    sns.stripplot(data=melted, x="Category", y="Score", size=2.5, alpha=0.5, color="0.3", jitter=True, ax=ax)
    ax.set_title("Category Score Distributions (Current Players)")
    ax.set_ylabel("Score (0–100)")
    ax.set_xlabel("")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "02_score_distributions.png")
    return fig


# ---------------------------------------------------------------------------
# 3.  Correlation heatmap between category scores
# ---------------------------------------------------------------------------

def plot_score_correlation(current_df: pd.DataFrame) -> plt.Figure:
    """Heatmap of Pearson correlations between category scores."""
    available = [c for c in CATEGORY_SCORES if c in current_df.columns]
    corr = current_df[available].corr()
    labels = [c.replace("_score", "").replace("_", " ").title() for c in available]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1, vmax=1,
                xticklabels=labels, yticklabels=labels, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Between Category Scores")
    fig.tight_layout()
    _save(fig, "03_score_correlation.png")
    return fig


# ---------------------------------------------------------------------------
# 4.  Current Score vs Potential Score scatter
# ---------------------------------------------------------------------------

def plot_current_vs_potential(final_df: pd.DataFrame) -> plt.Figure:
    """Scatter of current score vs potential score, coloured by age."""
    df = final_df.dropna(subset=["current_score", "potential_score"]).copy()
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return fig

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(df["current_score"], df["potential_score"],
                    c=df["age"], cmap="viridis", s=40, alpha=0.75, edgecolors="white", linewidth=0.4)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Age")

    # Highlight top-5 potential
    top5 = df.nsmallest(5, "potential_rank")
    for _, row in top5.iterrows():
        ax.annotate(row["player_name"], (row["current_score"], row["potential_score"]),
                    fontsize=7.5, fontweight="bold",
                    xytext=(6, 6), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

    ax.set_xlabel("Current Score")
    ax.set_ylabel("Potential Score (Success Probability × 100)")
    ax.set_title("Current Score vs Potential Score (Young Keepers)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "04_current_vs_potential.png")
    return fig


# ---------------------------------------------------------------------------
# 5.  Age distribution of young keepers with potential
# ---------------------------------------------------------------------------

def plot_age_vs_potential(potential_df: pd.DataFrame) -> plt.Figure:
    """Box + swarm showing potential score by age bucket."""
    df = potential_df.dropna(subset=["age", "potential_score"]).copy()
    df["age_int"] = df["age"].astype(int)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="age_int", y="potential_score", hue="age_int", palette="coolwarm", width=0.5, legend=False, ax=ax, fliersize=0)
    sns.stripplot(data=df, x="age_int", y="potential_score", color="0.25", size=4, alpha=0.6, jitter=True, ax=ax)
    ax.set_xlabel("Age")
    ax.set_ylabel("Potential Score")
    ax.set_title("Potential Score Distribution by Age")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "05_age_vs_potential.png")
    return fig


# ---------------------------------------------------------------------------
# 6.  Radar chart for a single player vs their best historical match
# ---------------------------------------------------------------------------

def plot_player_radar(player_row: pd.Series, sim_prefix: str = "similar_gk_1") -> plt.Figure:
    """Radar (spider) chart comparing a current player to their most similar historical GK."""
    categories = CATEGORY_LABELS
    player_vals = []
    sim_vals = []
    score_keys = CATEGORY_SCORES
    sim_keys = [
        f"{sim_prefix}_shot_stopping",
        f"{sim_prefix}_handling",
        f"{sim_prefix}_distribution",
        f"{sim_prefix}_sweeping",
        f"{sim_prefix}_reliability",
    ]

    for key in score_keys:
        player_vals.append(float(player_row.get(key, 50)) if pd.notna(player_row.get(key)) else 50)
    for key in sim_keys:
        sim_vals.append(float(player_row.get(key, 50)) if pd.notna(player_row.get(key)) else 50)
    # Pad sim_vals with consistency if available
    sim_consistency_key = f"sim1_consistency"
    sim_vals.append(float(player_row.get(sim_consistency_key, 50)) if pd.notna(player_row.get(sim_consistency_key)) else 50)

    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    player_vals += player_vals[:1]
    sim_vals += sim_vals[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, player_vals, "o-", linewidth=2, label=player_row.get("player_name", "Current"), color="#2980b9")
    ax.fill(angles, player_vals, alpha=0.15, color="#2980b9")
    sim_name = player_row.get(f"{sim_prefix}", "Historical Match")
    ax.plot(angles, sim_vals, "s--", linewidth=2, label=sim_name, color="#e67e22")
    ax.fill(angles, sim_vals, alpha=0.12, color="#e67e22")

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title(f"{player_row.get('player_name', '?')} vs {sim_name}", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.05), fontsize=9)
    fig.tight_layout()
    safe_name = str(player_row.get("player_name", "player")).replace(" ", "_").replace("/", "_")
    _save(fig, f"06_radar_{safe_name}.png")
    return fig


# ---------------------------------------------------------------------------
# 7.  Model comparison bar chart (ROC-AUC, F1, Recall)
# ---------------------------------------------------------------------------

def plot_model_comparison(metrics: dict[str, dict[str, float]]) -> plt.Figure:
    """Grouped bar chart comparing model performance metrics."""
    models = list(metrics.keys())
    metric_names = ["roc_auc", "f1", "recall"]
    friendly = {"roc_auc": "ROC-AUC", "f1": "F1 Score", "recall": "Recall"}

    x = np.arange(len(models))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, m in enumerate(metric_names):
        vals = [metrics[model][m] for model in models]
        bars = ax.bar(x + i * width, vals, width, label=friendly[m], color=PALETTE[i], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "07_model_comparison.png")
    return fig


# ---------------------------------------------------------------------------
# 8.  XGBoost feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(model_artifact: dict) -> plt.Figure:
    """Horizontal bar chart of XGBoost feature importance."""
    model = model_artifact["model"]
    feature_cols = model_artifact["feature_cols"]
    importances = model.feature_importances_

    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_cols[i].replace("_", " ").title() for i in idx], importances[idx], color="#3498db", edgecolor="white")
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("XGBoost — Feature Importance for Potential Prediction")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "08_feature_importance.png")
    return fig


# ---------------------------------------------------------------------------
# 9.  Top-15 current ranking horizontal bar
# ---------------------------------------------------------------------------

def plot_top_current_ranking(final_df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Horizontal stacked bar showing top-N keepers by current score breakdown."""
    df = final_df.sort_values("current_rank").head(top_n).copy()
    df = df.iloc[::-1]  # reverse for horizontal bar (top at top)

    cats = [c for c in CATEGORY_SCORES if c in df.columns and c != "consistency_score"]
    labels = [c.replace("_score", "").replace("_", " ").title() for c in cats]

    fig, ax = plt.subplots(figsize=(10, 6))
    left = np.zeros(len(df))
    for cat, lab, color in zip(cats, labels, PALETTE):
        vals = pd.to_numeric(df[cat], errors="coerce").fillna(0).values
        ax.barh(df["player_name"], vals, left=left, label=lab, color=color, edgecolor="white", linewidth=0.4)
        left += vals

    ax.set_xlabel("Cumulative Category Scores")
    ax.set_title(f"Top {top_n} Goalkeepers — Current Score Breakdown")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "09_top_current_ranking.png")
    return fig


# ---------------------------------------------------------------------------
# 10. Similarity score distribution
# ---------------------------------------------------------------------------

def plot_similarity_distribution(similarity_df: pd.DataFrame) -> plt.Figure:
    """Histogram of top-1 similarity scores."""
    col = "similarity_score_1"
    if col not in similarity_df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No similarity data", transform=ax.transAxes, ha="center")
        return fig

    scores = similarity_df[col].dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(scores, bins=25, color="#1abc9c", edgecolor="white", alpha=0.85)
    ax.axvline(scores.median(), color="#e74c3c", ls="--", lw=1.5, label=f"Median = {scores.median():.3f}")
    ax.set_xlabel("Cosine Similarity (Top-1 Match)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Similarity Scores (Young Keepers to Historical Successful GKs)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "10_similarity_distribution.png")
    return fig


# ---------------------------------------------------------------------------
# 11. Confidence score vs current rank scatter
# ---------------------------------------------------------------------------

def plot_confidence_vs_rank(output_a_df: pd.DataFrame) -> plt.Figure:
    """Scatter showing confidence score vs current rank."""
    df = output_a_df.dropna(subset=["current_rank", "confidence_score"]).copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(df["current_rank"], df["confidence_score"],
                    c=df.get("matches_used", df.get("matches_count", 50)),
                    cmap="plasma", s=35, alpha=0.7, edgecolors="white", linewidth=0.3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Matches Used")
    ax.set_xlabel("Current Rank")
    ax.set_ylabel("Confidence Score")
    ax.set_title("Confidence Score vs Current Rank")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "11_confidence_vs_rank.png")
    return fig


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def generate_all_visuals() -> list[Path]:
    """Load output CSVs + model artifact and generate all plots."""
    tables = OUTPUTS_DIR / "tables"

    # Load data
    final_df = pd.read_csv(tables / "final_goalkeeper_rankings.csv")
    output_a_df = pd.read_csv(tables / "output_a_main_ranking.csv")
    potential_df = pd.read_csv(tables / "potential_ranking.csv")
    similarity_df = pd.read_csv(tables / "young_similarity_output.csv")
    origin_df = pd.read_csv(PROCESSED_DATA_DIR / "origin_player_features.csv")
    current_df = pd.read_csv(PROCESSED_DATA_DIR / "current_player_features.csv")

    model_path = OUTPUTS_DIR / "models" / "potential_model.pkl"
    model_artifact = joblib.load(model_path) if model_path.exists() else None

    paths: list[Path] = []

    print("Generating visualisations...")

    # 1 – Label distribution
    plot_label_distribution(origin_df)

    # 2 – Score distributions
    plot_score_distributions(current_df)

    # 3 – Correlation heatmap
    plot_score_correlation(current_df)

    # 4 – Current vs Potential scatter
    plot_current_vs_potential(output_a_df)

    # 5 – Age vs Potential
    plot_age_vs_potential(potential_df)

    # 6 – Radar charts for top-3 potential players
    top3 = output_a_df.dropna(subset=["potential_rank"]).nsmallest(3, "potential_rank")
    for _, row in top3.iterrows():
        plot_player_radar(row)

    # 7 – Model comparison
    if model_artifact and "metrics" in model_artifact:
        plot_model_comparison(model_artifact["metrics"])

    # 8 – Feature importance
    if model_artifact and "model" in model_artifact and hasattr(model_artifact["model"], "feature_importances_"):
        plot_feature_importance(model_artifact)

    # 9 – Top-15 current ranking
    plot_top_current_ranking(final_df)

    # 10 – Similarity distribution
    plot_similarity_distribution(similarity_df)

    # 11 – Confidence vs rank
    plot_confidence_vs_rank(output_a_df)

    print(f"\nDone — all visuals saved to {VISUALS_DIR}")
    return paths


if __name__ == "__main__":
    generate_all_visuals()
