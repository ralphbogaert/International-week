"""Entry point for the goalkeeper project pipeline."""

from pathlib import Path

import pandas as pd

from aggregate_features import (
    save_interim_match_features,
    save_normalized_match_features,
    save_player_feature_tables,
)
from build_output import build_output
from config import OUTPUTS_DIR, PROCESSED_DATA_DIR
from extract_features import build_match_level_dataframes, extract_features
from load_data import load_goalkeepers
from scoring import add_current_score_and_rank, score_players
from similarity import run_similarity_pipeline
from train_model import train_model


def main() -> None:
    """Run phase-1/phase-2 pipeline and persist interim match-level features."""
    keepers_df = load_goalkeepers()
    match_rows_df = extract_features(keepers_df)
    origin_match_df, current_match_df = build_match_level_dataframes(match_rows_df)

    cleaned_origin, cleaned_current, missing_log = save_interim_match_features(
        origin_match_df=origin_match_df,
        current_match_df=current_match_df,
    )
    normalized_origin, normalized_current = save_normalized_match_features(
        cleaned_origin=cleaned_origin,
        cleaned_current=cleaned_current,
    )
    origin_player_features, current_player_features = save_player_feature_tables(
        origin_match_df=cleaned_origin,
        current_match_df=cleaned_current,
    )

    origin_category_scores = score_players(normalized_origin)
    origin_scored = origin_player_features.merge(
        origin_category_scores,
        on="player_id",
        how="left",
        suffixes=("", "_from_matches"),
    )
    origin_scored.to_csv(PROCESSED_DATA_DIR / "origin_player_features.csv", index=False)

    current_category_scores = score_players(normalized_current)
    current_scored = current_player_features.merge(
        current_category_scores,
        on="player_id",
        how="left",
        suffixes=("", "_from_matches"),
    )
    current_scored = add_current_score_and_rank(current_scored)

    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    current_scored.to_csv(PROCESSED_DATA_DIR / "current_player_features.csv", index=False)
    current_scored.to_csv(PROCESSED_DATA_DIR / "current_player_ranking.csv", index=False)

    train_summary = train_model()

    successful_gks_df, successful_profiles_df, similarity_df = run_similarity_pipeline(
        origin_scored_df=origin_scored,
        current_scored_df=current_scored,
    )

    potential_path = OUTPUTS_DIR / "tables" / "potential_ranking.csv"
    potential_ranking_df = pd.read_csv(potential_path) if potential_path.exists() else pd.DataFrame()
    output_a_df, output_b_df = build_output(
        current_scored_df=current_scored,
        potential_ranking_df=potential_ranking_df,
        similarity_df=similarity_df,
    )

    print(
        "Saved interim features:",
        f"origin={len(cleaned_origin)} rows,",
        f"current={len(cleaned_current)} rows,",
        f"missing_log={len(missing_log)} rows,",
        f"normalized_origin={len(normalized_origin)} rows,",
        f"normalized_current={len(normalized_current)} rows,",
        f"origin_players={len(origin_player_features)} rows,",
        f"current_players={len(current_player_features)} rows,",
        f"current_ranking={len(current_scored)} rows,",
        f"successful_gks={len(successful_gks_df)} rows,",
        f"successful_profiles={len(successful_profiles_df)} rows,",
        f"similarity_rows={len(similarity_df)} rows,",
        f"output_a={len(output_a_df)} rows,",
        f"output_b={len(output_b_df)} rows,",
        f"potential_rows={train_summary.get('potential_rows', 0)}",
    )


if __name__ == "__main__":
    main()
