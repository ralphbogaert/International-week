"""Player similarity calculations and historical-young profile matching."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config import OUTPUTS_DIR

SIM_FEATURES = [
    "shot_stopping_score",
    "handling_score",
    "distribution_score",
    "sweeping_score",
    "reliability_score",
    "consistency_score",
]


def build_successful_gks_subset(
    origin_scored_df: pd.DataFrame,
    current_scored_df: pd.DataFrame,
    min_matches: int = 8,
    min_current_score_quantile: float = 0.5,
) -> pd.DataFrame:
    """Define historical successful goalkeepers based on labels and thresholds."""
    origin = origin_scored_df.copy()
    origin["label"] = origin["label"].astype(str).str.upper().str.strip()
    successful = origin[origin["label"] == "PLAYS"].copy()

    if "matches_count" in successful.columns:
        successful = successful[successful["matches_count"].fillna(0) >= min_matches]

    current_cols = [c for c in ["player_id", "current_score"] if c in current_scored_df.columns]
    if current_cols:
        merged = successful.merge(current_scored_df[current_cols], on="player_id", how="left")
        if "current_score" in merged.columns and merged["current_score"].notna().any():
            threshold = merged["current_score"].quantile(min_current_score_quantile)
            merged = merged[merged["current_score"].fillna(-np.inf) >= threshold]
        successful = merged

    successful = successful.drop_duplicates(subset=["player_id"]).reset_index(drop=True)
    return successful


def build_successful_young_profiles(successful_gks_df: pd.DataFrame) -> pd.DataFrame:
    """Build young-age profile table for successful historical keepers."""
    cols = [
        "player_id",
        "player_name",
        "age",
        "shot_stopping_score",
        "handling_score",
        "distribution_score",
        "sweeping_score",
        "reliability_score",
        "consistency_score",
        "label",
        "matches_count",
    ]
    existing = [c for c in cols if c in successful_gks_df.columns]
    profiles = successful_gks_df[existing].copy()

    if "age" in profiles.columns:
        profiles = profiles.rename(columns={"age": "young_age"})
    if "label" in profiles.columns:
        profiles = profiles.rename(columns={"label": "current_success_label"})

    return profiles.drop_duplicates(subset=["player_id"]).reset_index(drop=True)


def _prepare_similarity_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Build numeric matrix for similarity with safe fill for missing values."""
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    fill_values = x.median(axis=0, skipna=True).fillna(50.0)
    x = x.fillna(fill_values)
    return x.to_numpy(dtype=float)


def find_similar_players(
    current_scored_df: pd.DataFrame,
    successful_profiles_df: pd.DataFrame,
    top_k: int = 3,
    only_young: bool = True,
) -> pd.DataFrame:
    """Find top-k similar historical young keepers for each current young keeper."""
    current = current_scored_df.copy()
    if only_young and "age" in current.columns:
        current = current[current["age"] < 27].copy()

    if current.empty or successful_profiles_df.empty:
        return pd.DataFrame()

    feature_cols = [c for c in SIM_FEATURES if c in current.columns and c in successful_profiles_df.columns]
    if not feature_cols:
        return pd.DataFrame()

    ref = successful_profiles_df.drop_duplicates(subset=["player_id"]).copy()
    x_current = _prepare_similarity_matrix(current, feature_cols)
    x_ref = _prepare_similarity_matrix(ref, feature_cols)

    sim_matrix = cosine_similarity(x_current, x_ref)

    # Avoid matching player to itself when IDs overlap.
    if "player_id" in current.columns and "player_id" in ref.columns:
        ref_id_to_idx = {pid: idx for idx, pid in enumerate(ref["player_id"].tolist())}
        for i, pid in enumerate(current["player_id"].tolist()):
            j = ref_id_to_idx.get(pid)
            if j is not None:
                sim_matrix[i, j] = -1.0

    rows: list[dict[str, Any]] = []
    for i, (_, cur_row) in enumerate(current.iterrows()):
        sim_scores = sim_matrix[i]
        best_idx = np.argsort(sim_scores)[::-1][:top_k]

        out_row: dict[str, Any] = {
            "player_id": cur_row.get("player_id"),
            "player_name": cur_row.get("player_name"),
            "age": cur_row.get("age"),
            "current_score": cur_row.get("current_score"),
            "potential_score": cur_row.get("potential_score"),
            "potential_rank": cur_row.get("potential_rank"),
        }

        for rank, ref_idx in enumerate(best_idx, start=1):
            ref_row = ref.iloc[ref_idx]
            sim_value = float(sim_scores[ref_idx])
            out_row[f"similar_gk_{rank}"] = ref_row.get("player_name")
            out_row[f"similarity_score_{rank}"] = sim_value
            out_row[f"sim{rank}_young_age"] = ref_row.get("young_age")
            out_row[f"sim{rank}_shot_stopping"] = ref_row.get("shot_stopping_score")
            out_row[f"sim{rank}_handling"] = ref_row.get("handling_score")
            out_row[f"sim{rank}_distribution"] = ref_row.get("distribution_score")
            out_row[f"sim{rank}_sweeping"] = ref_row.get("sweeping_score")
            out_row[f"sim{rank}_reliability"] = ref_row.get("reliability_score")
            out_row[f"sim{rank}_consistency"] = ref_row.get("consistency_score")

        rows.append(out_row)

    return pd.DataFrame(rows)


def run_similarity_pipeline(
    origin_scored_df: pd.DataFrame,
    current_scored_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create successful young profiles and similarity matches, then save outputs."""
    successful_gks_df = build_successful_gks_subset(
        origin_scored_df=origin_scored_df,
        current_scored_df=current_scored_df,
    )
    successful_profiles_df = build_successful_young_profiles(successful_gks_df)
    similarity_df = find_similar_players(
        current_scored_df=current_scored_df,
        successful_profiles_df=successful_profiles_df,
        top_k=3,
        only_young=True,
    )

    tables_dir = OUTPUTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    successful_gks_df.to_csv(tables_dir / "successful_gks_df.csv", index=False)
    successful_profiles_df.to_csv(tables_dir / "successful_gks_young_profiles.csv", index=False)
    similarity_df.to_csv(tables_dir / "young_similarity_output.csv", index=False)

    return successful_gks_df, successful_profiles_df, similarity_df
