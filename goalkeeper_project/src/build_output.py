"""Create final tables and artifacts for reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import OUTPUTS_DIR


def _compute_confidence_score(df: pd.DataFrame) -> pd.Series:
    """Compute confidence score from sample size, consistency and similarity density."""
    matches = pd.to_numeric(df.get("matches_count"), errors="coerce").fillna(0)
    sample_size_score = (matches.clip(lower=0, upper=30) / 30.0) * 100.0

    consistency_score = pd.to_numeric(df.get("consistency_score"), errors="coerce").fillna(50.0)

    sim_cols = [c for c in ["similarity_score_1", "similarity_score_2", "similarity_score_3"] if c in df.columns]
    if sim_cols:
        similarity_density_score = (
            df[sim_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True).fillna(0.0) * 100.0
        )
    else:
        similarity_density_score = pd.Series(np.zeros(len(df)), index=df.index)

    confidence = 0.4 * sample_size_score + 0.3 * consistency_score + 0.3 * similarity_density_score
    return confidence.clip(lower=0.0, upper=100.0)


def build_output(
    current_scored_df: pd.DataFrame,
    potential_ranking_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build Output A (main ranking) and Output B (comparison storytelling)."""
    tables_dir = OUTPUTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    merge_cols = [c for c in ["player_id", "success_probability", "potential_score", "potential_rank"] if c in potential_ranking_df.columns]
    potential_small = potential_ranking_df[merge_cols].copy() if merge_cols else pd.DataFrame(columns=["player_id"])

    output_a = current_scored_df.merge(potential_small, on="player_id", how="left")
    if not similarity_df.empty:
        output_a = output_a.merge(similarity_df, on="player_id", how="left", suffixes=("", "_sim"))

    output_a["matches_used"] = output_a.get("matches_count")
    output_a["confidence_score"] = _compute_confidence_score(output_a)

    # Normalize similarity column names for final table contract.
    rename_map = {
        "sim1_young_age": "similar_gk_1_young_age",
        "sim1_shot_stopping": "similar_gk_1_shot_stopping",
        "sim1_handling": "similar_gk_1_handling",
        "sim1_distribution": "similar_gk_1_distribution",
        "sim1_sweeping": "similar_gk_1_sweeping",
        "sim1_reliability": "similar_gk_1_reliability",
        "sim2_young_age": "similar_gk_2_young_age",
        "sim2_shot_stopping": "similar_gk_2_shot_stopping",
        "sim2_handling": "similar_gk_2_handling",
        "sim2_distribution": "similar_gk_2_distribution",
        "sim2_sweeping": "similar_gk_2_sweeping",
        "sim2_reliability": "similar_gk_2_reliability",
        "sim3_young_age": "similar_gk_3_young_age",
        "sim3_shot_stopping": "similar_gk_3_shot_stopping",
        "sim3_handling": "similar_gk_3_handling",
        "sim3_distribution": "similar_gk_3_distribution",
        "sim3_sweeping": "similar_gk_3_sweeping",
        "sim3_reliability": "similar_gk_3_reliability",
    }
    output_a = output_a.rename(columns={k: v for k, v in rename_map.items() if k in output_a.columns})

    output_a = output_a.sort_values(["current_rank", "current_score"], ascending=[True, False]).reset_index(drop=True)
    output_a.to_csv(tables_dir / "output_a_main_ranking.csv", index=False)

    final_cols = [
        "player_name",
        "age",
        "club",
        "matches_used",
        "shot_stopping_score",
        "handling_score",
        "distribution_score",
        "sweeping_score",
        "reliability_score",
        "consistency_score",
        "current_score",
        "current_rank",
        "success_probability",
        "potential_score",
        "potential_rank",
        "confidence_score",
        "similar_gk_1",
        "similarity_score_1",
        "similar_gk_1_young_age",
        "similar_gk_1_shot_stopping",
        "similar_gk_1_handling",
        "similar_gk_1_distribution",
        "similar_gk_1_sweeping",
        "similar_gk_1_reliability",
        "similar_gk_2",
        "similarity_score_2",
        "similar_gk_3",
        "similarity_score_3",
    ]
    final_existing = [c for c in final_cols if c in output_a.columns]
    final_df = output_a[final_existing].copy()
    final_df.to_csv(tables_dir / "final_goalkeeper_rankings.csv", index=False)

    # Output B: side-by-side current vs best historical match (top-1 similarity).
    comparison_cols = [
        "player_name",
        "age",
        "current_score",
        "potential_score",
        "similar_gk_1",
        "similar_gk_1_young_age",
        "shot_stopping_score",
        "similar_gk_1_shot_stopping",
        "handling_score",
        "similar_gk_1_handling",
        "distribution_score",
        "similar_gk_1_distribution",
        "sweeping_score",
        "similar_gk_1_sweeping",
        "reliability_score",
        "similar_gk_1_reliability",
        "consistency_score",
        "sim1_consistency",
        "similarity_score_1",
        "confidence_score",
    ]
    existing_comparison_cols = [c for c in comparison_cols if c in output_a.columns]
    output_b = output_a[existing_comparison_cols].copy()
    output_b = output_b.rename(
        columns={
            "player_name": "Current Player",
            "age": "Age",
            "current_score": "Current Score",
            "potential_score": "Potential Score",
            "similar_gk_1": "Similar Historical GK",
            "similar_gk_1_young_age": "Similar GK Young Age",
            "shot_stopping_score": "Player Shot Stop",
            "similar_gk_1_shot_stopping": "Similar GK Shot Stop",
            "handling_score": "Player Handling",
            "similar_gk_1_handling": "Similar GK Handling",
            "distribution_score": "Player Distribution",
            "similar_gk_1_distribution": "Similar GK Distribution",
            "sweeping_score": "Player Sweeping",
            "similar_gk_1_sweeping": "Similar GK Sweeping",
            "reliability_score": "Player Reliability",
            "similar_gk_1_reliability": "Similar GK Reliability",
            "consistency_score": "Player Consistency",
            "sim1_consistency": "Similar GK Consistency",
            "similarity_score_1": "Similarity Score",
            "confidence_score": "Confidence Score",
        }
    )
    output_b.to_csv(tables_dir / "output_b_comparison.csv", index=False)

    return output_a, output_b
