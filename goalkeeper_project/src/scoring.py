"""Scoring utilities for goalkeeper category ranking."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from config import KPI_GROUPS

CURRENT_SCORE_WEIGHTS = {
    "shot_stopping_score": 0.30,
    "handling_score": 0.20,
    "distribution_score": 0.15,
    "sweeping_score": 0.10,
    "reliability_score": 0.20,
    "consistency_score": 0.05,
}


def z_to_score_0_100(z_value: float | pd.Series) -> float | pd.Series:
    """Convert z-score(s) to clipped 0-100 scale using 50 + 15*z."""
    score = 50 + 15 * z_value
    if isinstance(score, pd.Series):
        return score.clip(lower=0, upper=100)
    return max(0.0, min(100.0, float(score)))


def _existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Return only columns that exist in dataframe."""
    return [col for col in columns if col in df.columns]


def _group_z_columns(group_name: str) -> list[str]:
    """Map KPI group names to expected z-score columns."""
    base = [f"{metric}_z" for metric in KPI_GROUPS.get(group_name, [])]

    # Reliability currently uses `errors_z` in our pipeline output.
    if group_name == "reliability":
        base.append("errors_z")

    return base


def score_players(normalized_match_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-player category scores from normalized match-level KPI data.

    Output scores are in [0, 100]:
    - shot_stopping_score
    - handling_score
    - distribution_score
    - sweeping_score
    - reliability_score
    - consistency_score
    """
    if normalized_match_df.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "player_name",
                "matches_count",
                "shot_stopping_score",
                "handling_score",
                "distribution_score",
                "sweeping_score",
                "reliability_score",
                "consistency_score",
            ]
        )

    df = normalized_match_df.copy()
    if "player_id" not in df.columns:
        raise ValueError("Input dataframe must contain 'player_id'.")

    grouped = df.groupby("player_id", dropna=False)
    result = grouped.size().rename("matches_count").reset_index()

    if "player_name" in df.columns:
        result = grouped["player_name"].first().reset_index().merge(result, on="player_id", how="left")

    score_column_map = {
        "shot_stopping": "shot_stopping_score",
        "handling": "handling_score",
        "distribution": "distribution_score",
        "sweeping": "sweeping_score",
        "reliability": "reliability_score",
    }

    for group_name, score_col in score_column_map.items():
        z_cols = _existing_columns(df, _group_z_columns(group_name))
        if not z_cols:
            result[score_col] = pd.NA
            continue

        # First average per KPI over matches, then average across KPIs per player.
        group_z = grouped[z_cols].mean().mean(axis=1, skipna=True)
        group_scores = z_to_score_0_100(group_z).rename(score_col).reset_index()
        result = result.merge(group_scores, on="player_id", how="left")

    # Consistency score: lower match-to-match variance is better.
    all_group_cols = []
    for grp in ["shot_stopping", "handling", "distribution", "sweeping", "reliability"]:
        all_group_cols.extend(_group_z_columns(grp))
    consistency_source_cols = _existing_columns(df, all_group_cols)

    if consistency_source_cols:
        row_overall_z = df[consistency_source_cols].mean(axis=1, skipna=True)
        consistency_z = -row_overall_z.groupby(df["player_id"]).std()
        consistency_scores = z_to_score_0_100(consistency_z).rename("consistency_score").reset_index()
        result = result.merge(consistency_scores, on="player_id", how="left")
    else:
        result["consistency_score"] = pd.NA

    return result


def add_current_score_and_rank(current_player_df: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted current score and descending rank for current players."""
    if current_player_df.empty:
        out = current_player_df.copy()
        out["current_score"] = pd.NA
        out["current_rank"] = pd.NA
        return out

    out = current_player_df.copy()

    # Ensure all score columns exist and are numeric. Missing values get neutral score 50.
    for col in CURRENT_SCORE_WEIGHTS:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    weighted_sum = 0
    for col, weight in CURRENT_SCORE_WEIGHTS.items():
        weighted_sum += weight * out[col].fillna(50.0)

    out["current_score"] = weighted_sum
    out["current_rank"] = out["current_score"].rank(method="min", ascending=False).astype("Int64")
    out = out.sort_values(["current_rank", "current_score"], ascending=[True, False]).reset_index(drop=True)

    return out
