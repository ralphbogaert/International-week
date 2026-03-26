"""Load and structure goalkeeper label data for phase 1."""

import ast
from typing import Any

import pandas as pd

from config import RAW_DATA_DIR


def parse_match_dirs(value: Any) -> list[str]:
    """Parse match directory values into a clean list of strings.

    Supports Python-like list strings (via ``ast.literal_eval``), pipe-separated
    strings, and already-materialized lists.
    """
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass

        if "|" in value:
            return [part.strip() for part in value.split("|") if part.strip()]

        return [value]

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    return []


def load_goalkeepers() -> pd.DataFrame:
    """Load the goalkeeper dataset and return a cleaned, structured dataframe."""
    df = pd.read_csv(RAW_DATA_DIR / "gk_dataset_final.csv")

    column_map = {
        "playerId": "player_id",
        "name": "player_name",
        "status": "label",
    }
    df = df.rename(columns=column_map)

    selected_columns = [
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

    missing = [col for col in selected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    df = df[selected_columns].copy()
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df["current_team"] = df["current_team"].astype(str).replace("nan", "").str.strip()
    df["origin_match_dirs"] = df["origin_match_dirs"].apply(parse_match_dirs)
    df["current_match_dirs"] = df["current_match_dirs"].apply(parse_match_dirs)

    return df


def load_labels() -> pd.DataFrame:
    """Backward-compatible alias for the main cleaned dataframe loader."""
    return load_goalkeepers()
