"""Extract per-match goalkeeper features from raw competition files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import RAW_DATA_DIR


def _load_json(file_path: Path) -> dict[str, Any]:
    """Safely load a JSON file and return an empty dict when missing/invalid."""
    if not file_path.exists():
        return {}

    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _build_match_dir_index(competitions_root: Path) -> dict[str, list[Path]]:
    """Map each match directory name to one or more absolute paths."""
    index: dict[str, list[Path]] = {}

    leagues_root = competitions_root / "leagues"
    if not leagues_root.exists():
        return index

    for comp_dir in leagues_root.iterdir():
        if not comp_dir.is_dir():
            continue

        for match_dir in comp_dir.iterdir():
            if not match_dir.is_dir():
                continue
            index.setdefault(match_dir.name, []).append(match_dir)

    return index


def _find_player_block(data: dict[str, Any], player_id: int) -> dict[str, Any] | None:
    """Find a player entry in squadHome/squadAway blocks."""
    payload = data.get("data", {})
    for squad_key in ("squadHome", "squadAway"):
        squad = payload.get(squad_key, {})
        for player in squad.get("players", []):
            if player.get("id") == player_id:
                return player
    return None


def _to_metric_dict(items: list[dict[str, Any]], id_key: str) -> dict[int, float]:
    """Convert list of metric objects into {metric_id: value}."""
    metric_dict: dict[int, float] = {}
    for item in items:
        metric_id = item.get(id_key)
        metric_value = item.get("value")
        if isinstance(metric_id, int) and isinstance(metric_value, (int, float)):
            metric_dict[metric_id] = float(metric_value)
    return metric_dict


def extract_player_matches(
    player_id: int,
    match_dirs: list[str],
    match_scope: str,
    match_index: dict[str, list[Path]],
) -> list[dict[str, Any]]:
    """Extract one row per match for a single player and a list of match dirs."""
    rows: list[dict[str, Any]] = []

    for match_dir_name in match_dirs:
        candidate_paths = match_index.get(match_dir_name, [])
        if not candidate_paths:
            continue

        # Use first match when duplicates exist across competitions.
        match_path = candidate_paths[0]
        competition_key = match_path.parent.name

        scores_json = _load_json(match_path / "player_scores.json")
        kpis_json = _load_json(match_path / "player_kpis.json")
        meta_json = _load_json(match_path / "match_meta.json")

        kpi_player = _find_player_block(kpis_json, player_id)
        if not kpi_player:
            continue

        if kpi_player.get("position") != "GOALKEEPER":
            continue

        score_player = _find_player_block(scores_json, player_id)

        row = {
            "player_id": player_id,
            "match_scope": match_scope,
            "competition_key": competition_key,
            "match_dir": match_dir_name,
            "match_id": meta_json.get("matchId"),
            "match_date": meta_json.get("date"),
            "home_team": meta_json.get("home"),
            "away_team": meta_json.get("away"),
            "match_share": kpi_player.get("matchShare"),
            "play_duration": kpi_player.get("playDuration"),
            "player_scores": _to_metric_dict(
                score_player.get("playerScores", []) if score_player else [],
                "playerScoreId",
            ),
            "player_kpis": _to_metric_dict(kpi_player.get("kpis", []), "kpiId"),
        }
        rows.append(row)

    return rows


def extract_features(
    keepers_df: pd.DataFrame,
    competitions_root: Path | None = None,
) -> pd.DataFrame:
    """Extract per-match rows for all keepers from origin/current directories.

    Required columns in keepers_df:
    - player_id
    - player_name
    - age
    - label
    - current_team
    - origin_match_dirs
    - current_match_dirs
    - origin_median
    - current_median
    - step
    """
    if competitions_root is None:
        competitions_root = RAW_DATA_DIR / "competitions"

    required = [
        "player_id",
        "player_name",
        "age",
        "label",
        "current_team",
        "origin_match_dirs",
        "current_match_dirs",
        "origin_median",
        "current_median",
        "step",
    ]
    missing = [col for col in required if col not in keepers_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in keepers dataframe: {missing}")

    match_index = _build_match_dir_index(competitions_root)

    all_rows: list[dict[str, Any]] = []
    for _, keeper in keepers_df.iterrows():
        player_id = int(keeper["player_id"])

        origin_rows = extract_player_matches(
            player_id=player_id,
            match_dirs=keeper["origin_match_dirs"],
            match_scope="origin",
            match_index=match_index,
        )

        current_rows = extract_player_matches(
            player_id=player_id,
            match_dirs=keeper["current_match_dirs"],
            match_scope="current",
            match_index=match_index,
        )

        base_context = {
            "player_id": player_id,
            "player_name": keeper["player_name"],
            "age": keeper["age"],
            "label": keeper["label"],
            "club": keeper["current_team"],
            "origin_median": keeper["origin_median"],
            "current_median": keeper["current_median"],
            "step": keeper["step"],
        }

        for row in origin_rows + current_rows:
            all_rows.append({**base_context, **row})

    return pd.DataFrame(all_rows)


def _split_competition_key(competition_key: str) -> tuple[str, str]:
    """Split competition key into competition name and season string."""
    if not isinstance(competition_key, str) or not competition_key:
        return "", ""

    if "_" not in competition_key:
        return competition_key, ""

    competition, season = competition_key.rsplit("_", 1)
    return competition, season


def _metric(metrics: dict[int, float], metric_id: int) -> float | None:
    """Get a numeric metric value by ID or return None."""
    value = metrics.get(metric_id)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_match_level_dataframes(
    match_rows_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build match-level tables and split into origin/current dataframes.

    Expected input is the dataframe returned by ``extract_features``.
    Output columns:
    - player_id, player_name, match_id, season, competition, is_origin, age, label
    - save_pct, psxg_delta, catches, long_pass_acc, sweeper_actions, errors
    """
    if match_rows_df.empty:
        empty_columns = [
            "player_id",
            "player_name",
            "match_id",
            "season",
            "competition",
            "is_origin",
            "age",
            "label",
            "club",
            "origin_median",
            "current_median",
            "step",
            "save_pct",
            "psxg_delta",
            "catches",
            "long_pass_acc",
            "sweeper_actions",
            "errors",
        ]
        empty_df = pd.DataFrame(columns=empty_columns)
        return empty_df.copy(), empty_df.copy()

    rows: list[dict[str, Any]] = []
    for _, row in match_rows_df.iterrows():
        competition, season = _split_competition_key(row.get("competition_key", ""))
        player_scores = row.get("player_scores", {})
        player_kpis = row.get("player_kpis", {})

        if not isinstance(player_scores, dict):
            player_scores = {}
        if not isinstance(player_kpis, dict):
            player_kpis = {}

        rows.append(
            {
                "player_id": row.get("player_id"),
                "player_name": row.get("player_name"),
                "match_id": row.get("match_id"),
                "season": season,
                "competition": competition,
                "is_origin": 1 if row.get("match_scope") == "origin" else 0,
                "age": row.get("age"),
                "label": row.get("label"),
                "club": row.get("club"),
                "origin_median": row.get("origin_median"),
                "current_median": row.get("current_median"),
                "step": row.get("step"),
                # Player Score IDs based on project documentation.
                "save_pct": _metric(player_scores, 188),
                "psxg_delta": _metric(player_scores, 164),
                "catches": _metric(player_scores, 190),
                "long_pass_acc": _metric(player_scores, 192),
                "sweeper_actions": _metric(player_scores, 189),
                # KPI 22 is used here as first-match-level proxy for GK errors.
                "errors": _metric(player_kpis, 22),
            }
        )

    match_level_df = pd.DataFrame(rows)
    origin_match_df = match_level_df[match_level_df["is_origin"] == 1].reset_index(drop=True)
    current_match_df = match_level_df[match_level_df["is_origin"] == 0].reset_index(drop=True)

    return origin_match_df, current_match_df
