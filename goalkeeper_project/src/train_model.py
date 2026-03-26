"""Train potential model and compute potential rankings."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from config import INTERIM_DATA_DIR, OUTPUTS_DIR, PROCESSED_DATA_DIR
from scoring import score_players

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - environment dependent
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None


def _build_origin_training_table() -> pd.DataFrame:
    """Merge origin player features with origin category scores."""
    origin_player_path = PROCESSED_DATA_DIR / "origin_player_features.csv"
    origin_match_norm_path = INTERIM_DATA_DIR / "origin_match_features_normalized.csv"

    origin_player_df = pd.read_csv(origin_player_path)
    origin_match_norm_df = pd.read_csv(origin_match_norm_path)

    origin_category_scores = score_players(origin_match_norm_df)
    score_cols = [
        "player_id",
        "shot_stopping_score",
        "handling_score",
        "distribution_score",
        "sweeping_score",
        "reliability_score",
        "consistency_score",
    ]
    existing_score_cols = [col for col in score_cols if col in origin_category_scores.columns]
    origin_category_scores = origin_category_scores[existing_score_cols].copy()

    origin_df = origin_player_df.merge(
        origin_category_scores,
        on="player_id",
        how="left",
    )
    return origin_df


def _create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary target where PLAYS=1 and others=0."""
    out = df.copy()
    out["label"] = out["label"].astype(str).str.upper().str.strip()
    out["y"] = (out["label"] == "PLAYS").astype(int)
    return out


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Select feature columns for the potential model."""
    base = [
        "shot_stopping_score",
        "handling_score",
        "distribution_score",
        "sweeping_score",
        "reliability_score",
        "consistency_score",
        "age",
        "matches_count",
        "origin_median",
        "current_median",
        "step",
        "shot_stopping_consistency",
        "distribution_consistency",
        "overall_consistency",
    ]
    return [col for col in base if col in df.columns]


def _evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate model with ROC-AUC, F1 and recall."""
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }


def _build_models(random_state: int = 42) -> dict[str, object]:
    """Create model instances for training."""
    if XGBClassifier is None:
        raise ImportError(
            "xgboost is not installed. Install it with 'pip install xgboost'."
        ) from XGBOOST_IMPORT_ERROR

    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def _select_best_model(results: dict[str, dict[str, float]]) -> str:
    """Select best model by ROC-AUC, then F1, then recall."""
    return max(
        results,
        key=lambda name: (
            results[name]["roc_auc"],
            results[name]["f1"],
            results[name]["recall"],
        ),
    )


def train_potential_model(random_state: int = 42) -> tuple[object, dict[str, dict[str, float]], list[str]]:
    """Train candidate models, keep best one, and persist model artifact."""
    df = _create_target(_build_origin_training_table())
    feature_cols = _feature_columns(df)
    if not feature_cols:
        raise ValueError("No feature columns available for model training.")

    x = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df["y"].astype(int)

    # Split on player-level rows (not match rows).
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=random_state,
    )

    models = _build_models(random_state=random_state)
    trained_models: dict[str, object] = {}
    results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        trained_models[name] = model
        results[name] = _evaluate_model(model, x_test, y_test)

    best_name = _select_best_model(results)
    best_model = trained_models[best_name]

    model_dir = OUTPUTS_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "feature_cols": feature_cols,
            "metrics": results,
            "best_model_name": best_name,
        },
        model_dir / "potential_model.pkl",
    )

    return best_model, results, feature_cols


def build_potential_ranking(best_model: object, feature_cols: list[str]) -> pd.DataFrame:
    """Predict success probability for young keepers and build ranking table."""
    current_path = PROCESSED_DATA_DIR / "current_player_features.csv"
    current_df = pd.read_csv(current_path)

    # Young keepers: under 27.
    young_df = current_df[current_df["age"] < 27].copy()
    x_young = young_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    young_df["success_probability"] = best_model.predict_proba(x_young)[:, 1]
    young_df["potential_score"] = young_df["success_probability"] * 100.0
    young_df["potential_rank"] = (
        young_df["potential_score"].rank(method="min", ascending=False).astype("Int64")
    )
    young_df = young_df.sort_values(
        ["potential_rank", "potential_score"],
        ascending=[True, False],
    ).reset_index(drop=True)

    tables_dir = OUTPUTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    young_df.to_csv(tables_dir / "potential_ranking.csv", index=False)
    return young_df


def train_model() -> dict[str, object]:
    """Train potential model, save artifact, and produce potential ranking outputs."""
    best_model, results, feature_cols = train_potential_model()
    potential_ranking_df = build_potential_ranking(best_model, feature_cols)
    return {
        "metrics": results,
        "feature_cols": feature_cols,
        "potential_rows": len(potential_ranking_df),
    }


if __name__ == "__main__":
    summary = train_model()
    print(summary)
