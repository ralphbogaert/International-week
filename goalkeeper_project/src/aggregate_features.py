"""Clean and aggregate match-level features for downstream modeling."""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import INTERIM_DATA_DIR, KPI_GROUPS, PROCESSED_DATA_DIR


def _expected_kpi_columns() -> list[str]:
    """Flatten configured KPI groups into a deduplicated KPI column list."""
    ordered: list[str] = []
    for group_cols in KPI_GROUPS.values():
        for col in group_cols:
            if col not in ordered:
                ordered.append(col)
    return ordered


def _coerce_numeric(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Convert known numeric columns to float and normalize invalid numbers."""
    out = df.copy()
    for col in numeric_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    return out


def _replace_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible KPI values with missing values."""
    out = df.copy()

    # Percentage-like metrics expected in [0, 1].
    bounded_01 = ["save_pct", "catches", "long_pass_acc", "clean_sheet_rate"]
    for col in bounded_01:
        if col in out.columns:
            mask = (out[col].notna()) & ((out[col] < 0) | (out[col] > 1))
            out.loc[mask, col] = pd.NA

    # Count-like metrics should be non-negative.
    non_negative = [
        "goals_conceded",
        "shots_on_target_faced",
        "parries",
        "cross_claims",
        "sweeper_actions",
        "clearances_outside_box",
        "interceptions",
        "errors",
        "errors_leading_to_shot",
        "errors_leading_to_goal",
    ]
    for col in non_negative:
        if col in out.columns:
            mask = (out[col].notna()) & (out[col] < 0)
            out.loc[mask, col] = pd.NA

    return out


def _drop_empty_rows(df: pd.DataFrame, kpi_columns: list[str]) -> pd.DataFrame:
    """Remove rows with missing identifiers or no KPI signal at all."""
    out = df.copy()
    required = [c for c in ["player_id", "match_id"] if c in out.columns]
    if required:
        out = out.dropna(subset=required)

    existing_kpi_cols = [c for c in kpi_columns if c in out.columns]
    if existing_kpi_cols:
        out = out.dropna(subset=existing_kpi_cols, how="all")

    return out.reset_index(drop=True)


def _build_missing_kpi_log(
    df: pd.DataFrame,
    expected_kpis: list[str],
    scope: str,
) -> list[dict[str, Any]]:
    """Create log records for missing KPI columns and values."""
    logs: list[dict[str, Any]] = []
    total_rows = len(df)

    for kpi in expected_kpis:
        if kpi not in df.columns:
            logs.append(
                {
                    "scope": scope,
                    "kpi": kpi,
                    "status": "missing_column",
                    "missing_rows": total_rows,
                    "missing_ratio": 1.0 if total_rows > 0 else 0.0,
                }
            )
            continue

        missing_rows = int(df[kpi].isna().sum())
        logs.append(
            {
                "scope": scope,
                "kpi": kpi,
                "status": "missing_values",
                "missing_rows": missing_rows,
                "missing_ratio": (missing_rows / total_rows) if total_rows > 0 else 0.0,
            }
        )

    return logs


def clean_match_level_features(df: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean match-level KPI data and return cleaned df + KPI missing-value log."""
    expected_kpis = _expected_kpi_columns()

    numeric_base = ["player_id", "match_id", "age", "is_origin"]
    numeric_columns = numeric_base + expected_kpis + ["errors"]

    cleaned = _coerce_numeric(df, numeric_columns)
    cleaned = _replace_impossible_values(cleaned)
    cleaned = _drop_empty_rows(cleaned, expected_kpis + ["errors"])

    log_rows = _build_missing_kpi_log(cleaned, expected_kpis, scope)
    log_df = pd.DataFrame(log_rows)

    return cleaned, log_df


def save_interim_match_features(
    origin_match_df: pd.DataFrame,
    current_match_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean and save origin/current match feature files to data/interim."""
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_origin, origin_log = clean_match_level_features(origin_match_df, scope="origin")
    cleaned_current, current_log = clean_match_level_features(current_match_df, scope="current")

    cleaned_origin.to_csv(INTERIM_DATA_DIR / "origin_match_features.csv", index=False)
    cleaned_current.to_csv(INTERIM_DATA_DIR / "current_match_features.csv", index=False)

    missing_kpi_log = pd.concat([origin_log, current_log], ignore_index=True)
    missing_kpi_log.to_csv(INTERIM_DATA_DIR / "missing_kpi_log.csv", index=False)

    return cleaned_origin, cleaned_current, missing_kpi_log


def _normalizable_kpi_columns(df: pd.DataFrame) -> list[str]:
    """Return KPI columns present in df that can be competition-normalized."""
    candidates = _expected_kpi_columns() + ["errors"]
    return [col for col in candidates if col in df.columns]


def normalize_competition_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-competition-per-season z-scores and percentiles for KPI columns."""
    if df.empty:
        return df.copy()

    out = df.copy()
    kpi_cols = _normalizable_kpi_columns(out)
    group_cols = [col for col in ["competition", "season"] if col in out.columns]
    if not group_cols or not kpi_cols:
        return out

    grouped = out.groupby(group_cols, dropna=False)
    for col in kpi_cols:
        mean_series = grouped[col].transform("mean")
        std_series = grouped[col].transform("std")

        # Keep std=0 or missing as NA to avoid false certainty in z-scores.
        std_safe = std_series.where(std_series > 0)
        out[f"{col}_z"] = (out[col] - mean_series) / std_safe

        # Percentile rank within competition+season group.
        out[f"{col}_pct"] = grouped[col].rank(method="average", pct=True)

    return out


def save_normalized_match_features(
    cleaned_origin: pd.DataFrame,
    cleaned_current: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize KPI columns by competition+season and save to interim CSV files."""
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    normalized_origin = normalize_competition_kpis(cleaned_origin)
    normalized_current = normalize_competition_kpis(cleaned_current)

    normalized_origin.to_csv(INTERIM_DATA_DIR / "origin_match_features_normalized.csv", index=False)
    normalized_current.to_csv(INTERIM_DATA_DIR / "current_match_features_normalized.csv", index=False)

    return normalized_origin, normalized_current


def aggregate_player_features(match_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate match-level KPI columns to player-level summary features."""
    if match_df.empty:
        return pd.DataFrame(columns=["player_id", "matches_count"])

    out = match_df.copy()

    id_columns = [
        col
        for col in [
            "player_id",
            "player_name",
            "label",
            "age",
            "club",
            "origin_median",
            "current_median",
            "step",
        ]
        if col in out.columns
    ]
    if "player_id" not in id_columns:
        raise ValueError("Input dataframe must contain 'player_id'.")

    # KPI candidates: configured KPI columns + errors, restricted to existing numeric columns.
    kpi_candidates = _normalizable_kpi_columns(out)
    kpi_columns = [
        col
        for col in kpi_candidates
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col])
    ]

    if not kpi_columns:
        base = out[["player_id"]].drop_duplicates().reset_index(drop=True)
        base["matches_count"] = out.groupby("player_id").size().values
        return base

    grouped = out.groupby("player_id", dropna=False)
    agg_df = grouped[kpi_columns].agg(["mean", "median", "std", "min", "max"])
    agg_df.columns = [f"{col}_{stat}" for col, stat in agg_df.columns]
    agg_df = agg_df.reset_index()

    matches_count = grouped.size().rename("matches_count").reset_index()
    player_df = agg_df.merge(matches_count, on="player_id", how="left")

    # Consistency features: lower std/variance means more stable performances.
    shot_cols = [
        col
        for col in KPI_GROUPS.get("shot_stopping", [])
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col])
    ]
    distribution_cols = [
        col
        for col in KPI_GROUPS.get("distribution", [])
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col])
    ]

    if shot_cols:
        shot_consistency = grouped[shot_cols].std().mean(axis=1, skipna=True)
        shot_consistency_df = shot_consistency.rename("shot_stopping_consistency").reset_index()
        player_df = player_df.merge(shot_consistency_df, on="player_id", how="left")
    else:
        player_df["shot_stopping_consistency"] = pd.NA

    if distribution_cols:
        distribution_consistency = grouped[distribution_cols].std().mean(axis=1, skipna=True)
        distribution_consistency_df = distribution_consistency.rename("distribution_consistency").reset_index()
        player_df = player_df.merge(distribution_consistency_df, on="player_id", how="left")
    else:
        player_df["distribution_consistency"] = pd.NA

    if kpi_columns:
        overall_match_score = out[kpi_columns].mean(axis=1, skipna=True)
        overall_variance = overall_match_score.groupby(out["player_id"]).var()
        overall_variance_df = overall_variance.rename("overall_consistency").reset_index()
        player_df = player_df.merge(overall_variance_df, on="player_id", how="left")
        # Alias to make intent explicit in downstream analysis.
        player_df["overall_performance_variance"] = player_df["overall_consistency"]
    else:
        player_df["overall_consistency"] = pd.NA
        player_df["overall_performance_variance"] = pd.NA

    # Add stable player context columns (first observed value).
    context_cols = [col for col in id_columns if col != "player_id"]
    if context_cols:
        context_df = grouped[context_cols].first().reset_index()
        player_df = context_df.merge(player_df, on="player_id", how="right")

    return player_df


def save_player_feature_tables(
    origin_match_df: pd.DataFrame,
    current_match_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate and save player-level feature tables for origin/current matches."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    origin_player_features = aggregate_player_features(origin_match_df)
    current_player_features = aggregate_player_features(current_match_df)

    origin_player_features.to_csv(PROCESSED_DATA_DIR / "origin_player_features.csv", index=False)
    current_player_features.to_csv(PROCESSED_DATA_DIR / "current_player_features.csv", index=False)

    return origin_player_features, current_player_features


def aggregate_features(match_df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for player-level aggregation."""
    return aggregate_player_features(match_df)
