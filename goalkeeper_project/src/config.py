"""Project configuration constants and paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Feature families used across extraction, aggregation, and scoring.
KPI_GROUPS = {
	"shot_stopping": [
		"save_pct",
		"psxg_delta",
		"goals_conceded",
		"shots_on_target_faced",
	],
	"handling": [
		"catches",
		"parries",
		"cross_claims",
	],
	"distribution": [
		"short_pass_acc",
		"medium_pass_acc",
		"long_pass_acc",
	],
	"sweeping": [
		"sweeper_actions",
		"clearances_outside_box",
		"interceptions",
	],
	"reliability": [
		"errors_leading_to_shot",
		"errors_leading_to_goal",
		"clean_sheet_rate",
	],
}

# Central registry that maps model feature names to source file + metric ID.
# NOTE: IDs marked as None still need to be aligned with your exact JSON schema.
METRIC_SOURCE_MAP = {
	# Shot stopping
	"save_pct": {"file": "player_scores.json", "id_type": "playerScoreId", "id": 188},
	"psxg_delta": {"file": "player_scores.json", "id_type": "playerScoreId", "id": 164},
	"goals_conceded": {"file": "player_kpis.json", "id_type": "kpiId", "id": None},
	"shots_on_target_faced": {"file": "player_kpis.json", "id_type": "kpiId", "id": None},
	# Handling
	"catches": {"file": "player_scores.json", "id_type": "playerScoreId", "id": 190},
	"parries": {"file": "player_kpis.json", "id_type": "kpiId", "id": None},
	"cross_claims": {"file": "player_kpis.json", "id_type": "kpiId", "id": None},
	# Distribution
	"short_pass_acc": {"file": "player_kpis.json", "id_type": "kpiId", "id": None},
	"medium_pass_acc": {"file": "player_kpis.json", "id_type": "kpiId", "id": None},
	"long_pass_acc": {"file": "player_scores.json", "id_type": "playerScoreId", "id": 192},
	# Sweeping
	"sweeper_actions": {"file": "player_scores.json", "id_type": "playerScoreId", "id": 189},
	"clearances_outside_box": {"file": "player_kpis.json", "id_type": "kpiId", "id": None},
	"interceptions": {"file": "player_kpis.json", "id_type": "kpiId", "id": None},
	# Reliability
	"errors_leading_to_shot": {"file": "player_kpis.json", "id_type": "kpiId", "id": 22},
	"errors_leading_to_goal": {"file": "player_kpis.json", "id_type": "kpiId", "id": None},
	"clean_sheet_rate": {"file": "derived", "id_type": "computed", "id": None},
}
