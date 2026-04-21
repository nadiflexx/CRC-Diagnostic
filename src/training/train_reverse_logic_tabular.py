"""
Training Script: Reverse Logic Tabular Analysis
Trains models to predict Smoking_History and Alcohol_Consumption from clinical indicators.

Can be run as:
  python train_reverse_logic_tabular.py --target Smoking_History
  python train_reverse_logic_tabular.py --target Alcohol_Consumption
  python train_reverse_logic_tabular.py --both (trains both sequentially)
"""

import argparse
from pathlib import Path
import warnings

import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.constants import (
    REVERSE_ANALYSIS_ALCOHOL_LIGHTGBM_CONFIG,
    REVERSE_ANALYSIS_ALCOHOL_RANDOM_FOREST_CONFIG,
    REVERSE_ANALYSIS_ALCOHOL_XGBOOST_CONFIG,
    REVERSE_ANALYSIS_BIOMARKERS,
    REVERSE_ANALYSIS_CATEGORICAL_FEATURES,
    REVERSE_ANALYSIS_DRINK_FEATURES,
    REVERSE_ANALYSIS_FEATURE_WEIGHTS,
    REVERSE_ANALYSIS_FEATURES,
    REVERSE_ANALYSIS_MODELS,
    REVERSE_ANALYSIS_NUMERIC_FEATURES,
    REVERSE_ANALYSIS_RANDOM_SEED,
    REVERSE_ANALYSIS_SMOKE_FEATURES,
    REVERSE_ANALYSIS_SMOKING_LIGHTGBM_CONFIG,
    REVERSE_ANALYSIS_SMOKING_RANDOM_FOREST_CONFIG,
    REVERSE_ANALYSIS_SMOKING_XGBOOST_CONFIG,
    REVERSE_ANALYSIS_TARGET_ALCOHOL,
    REVERSE_ANALYSIS_TARGET_SMOKING,
    REVERSE_ANALYSIS_TEST_SIZE,
    REVERSE_ANALYSIS_VAL_SIZE,
)
from src.config.paths import paths
from src.evaluation.reverse_logic_analyzer import ReverseLogicAnalyzer
from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

matplotlib.use("Agg")
warnings.filterwarnings("ignore", message="X does not have valid feature names")


class ReverseLogicTrainer:
    """Orchestrates training, evaluation, and analysis for reverse logic models."""

    def __init__(
        self,
        csv_path: Path | None = None,
        model_output_dir: Path | None = None,
        analysis_output_dir: Path | None = None,
        exclude_biomarkers: bool = False,
    ):
        self.csv_path = (
            csv_path or paths.TABULAR_PROCESSED / "smoking_drinking_cleaned.csv"
        )
        self.model_output_dir = model_output_dir or paths.REVERSE_LOGIC_MODEL_DIR
        self.analysis_output_dir = (
            analysis_output_dir or paths.REVERSE_LOGIC_ANALYSIS_DIR
        )
        self.exclude_biomarkers = exclude_biomarkers
        self.random_seed = REVERSE_ANALYSIS_RANDOM_SEED

        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_output_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_binary_target(
        self, series: pd.Series, target_name: str
    ) -> pd.Series:
        """
        Normalizes target columns into stable Yes/No labels without corrupting already-clean data.
        """
        normalized = series.copy()

        if target_name == "SMK_stat_type_cd":
            if pd.api.types.is_numeric_dtype(normalized):
                normalized = normalized.apply(
                    lambda x: "No" if float(x) == 1.0 else "Yes"
                )
            else:
                normalized = normalized.astype(str).str.strip()
                normalized = normalized.replace(
                    {
                        "1": "No",
                        "1.0": "No",
                        "2": "Yes",
                        "2.0": "Yes",
                        "3": "Yes",
                        "3.0": "Yes",
                        "No": "No",
                        "Yes": "Yes",
                    }
                )

        if target_name == "DRK_YN":
            normalized = (
                normalized.astype(str)
                .str.strip()
                .str.upper()
                .replace(
                    {
                        "Y": "Yes",
                        "N": "No",
                        "YES": "Yes",
                        "NO": "No",
                    }
                )
            )

        return normalized

    # ─────────────────────────────────────────────────────────────────────────────
    # DATA LOADING & PREPARATION
    # ─────────────────────────────────────────────────────────────────────────────

    def load_and_prepare_data(self, target: str) -> tuple:
        print(f"\n{'=' * 80}\n  LOADING DATA FOR TARGET: {target}\n{'=' * 80}")
        df = pd.read_csv(self.csv_path)

        if "SMK_stat_type_cd" in df.columns:
            df["SMK_stat_type_cd"] = self._normalize_binary_target(
                df["SMK_stat_type_cd"], "SMK_stat_type_cd"
            )

        if "DRK_YN" in df.columns:
            df["DRK_YN"] = self._normalize_binary_target(df["DRK_YN"], "DRK_YN")

        if target == REVERSE_ANALYSIS_TARGET_SMOKING:
            features_to_use = REVERSE_ANALYSIS_SMOKE_FEATURES.copy()
            print(
                f"[OK] Using SMOKE features ({len(features_to_use)} features): {features_to_use}"
            )
        elif target == REVERSE_ANALYSIS_TARGET_ALCOHOL:
            features_to_use = REVERSE_ANALYSIS_DRINK_FEATURES.copy()
            print(
                f"[OK] Using DRINK features ({len(features_to_use)} features): {features_to_use}"
            )
        else:
            features_to_use = REVERSE_ANALYSIS_FEATURES.copy()
            print(f"[WARN] Using DEFAULT features for unknown target: {target}")

        if self.exclude_biomarkers:
            features_to_use = [
                f for f in features_to_use if f not in REVERSE_ANALYSIS_BIOMARKERS
            ]
            print(f"[WARN] Excluyendo biomarcadores: {REVERSE_ANALYSIS_BIOMARKERS}")

        selected_columns = features_to_use + [target]
        df = df[selected_columns].dropna().copy()

        print(
            f"[OK] Records: {len(df):,}\n  Distribution: {df[target].value_counts().to_dict()}"
        )

        if df[target].nunique() < 2:
            raise ValueError(
                f"Target '{target}' only has one class after normalization: {df[target].unique().tolist()}"
            )

        X = df[features_to_use].copy()
        y = df[target].copy()

        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=(REVERSE_ANALYSIS_VAL_SIZE + REVERSE_ANALYSIS_TEST_SIZE),
            random_state=self.random_seed,
            stratify=y,
        )
        val_ratio = REVERSE_ANALYSIS_VAL_SIZE / (
            REVERSE_ANALYSIS_VAL_SIZE + REVERSE_ANALYSIS_TEST_SIZE
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(1 - val_ratio),
            random_state=self.random_seed,
            stratify=y_temp,
        )
        return df, X_train, X_val, X_test, y_train, y_val, y_test

    # ─────────────────────────────────────────────────────────────────────────────
    # MODEL TRAINING
    # ─────────────────────────────────────────────────────────────────────────────

    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target: str,
        model_types: list | None = None,
    ):
        if model_types is None:
            model_types = REVERSE_ANALYSIS_MODELS

        current_features = X_train.columns.tolist()
        num_feats = [
            f for f in REVERSE_ANALYSIS_NUMERIC_FEATURES if f in current_features
        ]
        cat_feats = [
            f for f in REVERSE_ANALYSIS_CATEGORICAL_FEATURES if f in current_features
        ]

        model = ReverseLogicTabularModel(
            target=target,
            numeric_features=num_feats,
            categorical_features=cat_feats,
            random_seed=self.random_seed,
        )

        if target == REVERSE_ANALYSIS_TARGET_SMOKING:
            configs = {
                "xgboost": {
                    **REVERSE_ANALYSIS_SMOKING_XGBOOST_CONFIG,
                    "feature_weight_overrides": REVERSE_ANALYSIS_FEATURE_WEIGHTS,
                },
                "random_forest": {**REVERSE_ANALYSIS_SMOKING_RANDOM_FOREST_CONFIG},
                "lightgbm": {**REVERSE_ANALYSIS_SMOKING_LIGHTGBM_CONFIG},
            }
        else:
            configs = {
                "xgboost": {
                    **REVERSE_ANALYSIS_ALCOHOL_XGBOOST_CONFIG,
                    "feature_weight_overrides": REVERSE_ANALYSIS_FEATURE_WEIGHTS,
                },
                "random_forest": {**REVERSE_ANALYSIS_ALCOHOL_RANDOM_FOREST_CONFIG},
                "lightgbm": {**REVERSE_ANALYSIS_ALCOHOL_LIGHTGBM_CONFIG},
            }

        for model_type in model_types:
            config = configs.get(model_type, {})
            model.fit(X_train, y_train, model_type=model_type, config=config)
            print(f"    [OK] {model_type} trained successfully")

        return model

    # ─────────────────────────────────────────────────────────────────────────────
    # MODEL EVALUATION
    # ─────────────────────────────────────────────────────────────────────────────

    def evaluate_models(
        self,
        model: ReverseLogicTabularModel,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_val: pd.Series,
        y_test: pd.Series,
        target: str = REVERSE_ANALYSIS_TARGET_SMOKING,
    ) -> dict:
        """
        Evaluates models on validation and test sets.

        Args:
            model: Trained ReverseLogicTabularModel
            X_val: Validation features
            X_test: Test features
            y_val: Validation target
            y_test: Test target
            target: "Smoking_History" or "Alcohol_Consumption"

        Returns:
            Dict of metrics for all models
        """
        print(f"\n{'=' * 80}")
        print("  EVALUATING MODELS")
        print(f"{'=' * 80}")

        all_metrics = {}

        for model_type in model.pipelines:
            print(f"\n  {model_type.upper()}:")

            metrics = model.evaluate(X_test, y_test, model_type=model_type)
            all_metrics[model_type] = metrics

            print(f"    Accuracy:   {metrics['accuracy']:.4f}")
            print(f"    Precision:  {metrics['precision']:.4f}")
            print(f"    Recall:     {metrics['recall']:.4f}")
            print(f"    F1 Score:   {metrics['f1']:.4f}")
            print(f"    ROC-AUC:    {metrics['roc_auc']:.4f}")
            print(f"    Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"    Specificity: {metrics['specificity']:.4f}")

        return all_metrics

    # ─────────────────────────────────────────────────────────────────────────────
    # ANALYSIS GENERATION
    # ─────────────────────────────────────────────────────────────────────────────

    def run_analysis(
        self,
        model: ReverseLogicTabularModel,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        full_df: pd.DataFrame,
        target: str = REVERSE_ANALYSIS_TARGET_SMOKING,
    ):
        """
        Generates comprehensive analysis visualizations.

        Args:
            model: Trained ReverseLogicTabularModel
            X_test: Test features
            y_test: Test target
            full_df: Full dataset (for patterns)
            target: "Smoking_History" or "Alcohol_Consumption"
        """
        print(f"\n{'=' * 80}")
        print("  GENERATING ANALYSIS VISUALIZATIONS")
        print(f"{'=' * 80}")

        analyzer = ReverseLogicAnalyzer(self.analysis_output_dir)
        analyzer.run_full_analysis(
            model=model,
            X_test=X_test,
            y_test=y_test,
            full_df=full_df,
            target_name=target,
            model_types=list(model.pipelines.keys()),
        )

        print(f"\n[OK] Analysis complete. Plots saved to: {self.analysis_output_dir}")

    # ─────────────────────────────────────────────────────────────────────────────
    # SAVE MODELS
    # ─────────────────────────────────────────────────────────────────────────────

    def save_model(
        self,
        model: ReverseLogicTabularModel,
        target: str = REVERSE_ANALYSIS_TARGET_SMOKING,
    ) -> Path:
        """
        Saves trained model to disk.

        Args:
            model: Trained ReverseLogicTabularModel
            target: "Smoking_History" or "Alcohol_Consumption"

        Returns:
            Path to saved model
        """
        safe_target = target.lower().replace("_", "_")
        model_path = self.model_output_dir / f"reverse_logic_{safe_target}.pkl"
        model.save(model_path)
        print(f"\n[OK] Model saved to: {model_path}")
        return model_path

    # ─────────────────────────────────────────────────────────────────────────────
    # MAIN TRAINING PIPELINE
    # ─────────────────────────────────────────────────────────────────────────────

    def train_single_target(
        self,
        target: str = REVERSE_ANALYSIS_TARGET_SMOKING,
        model_types: list | None = None,
    ):
        """
        Trains, evaluates, and analyzes models for a single target.

        Args:
            target: "Smoking_History" or "Alcohol_Consumption"
            model_types: List of models to train (default: all)
        """
        # Load data
        full_df, X_train, X_val, X_test, y_train, y_val, y_test = (
            self.load_and_prepare_data(target)
        )

        # Train
        model = self.train_models(
            X_train, y_train, target=target, model_types=model_types
        )

        # Evaluate
        self.evaluate_models(model, X_val, X_test, y_val, y_test, target=target)

        # Analyze
        self.run_analysis(model, X_test, y_test, full_df, target=target)

        # Save
        self.save_model(model, target=target)

    def train_both_targets(self, model_types: list | None = None):
        """
        Trains models for both Smoking_History and Alcohol_Consumption.

        Args:
            model_types: List of models to train (default: all)
        """
        for target in [
            REVERSE_ANALYSIS_TARGET_SMOKING,
            REVERSE_ANALYSIS_TARGET_ALCOHOL,
        ]:
            self.train_single_target(target=target, model_types=model_types)


def main():
    parser = argparse.ArgumentParser(
        description="Train models on smoking/drinking dataset"
    )
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument(
        "--target",
        type=str,
        choices=["SMK_stat_type_cd", "DRK_YN", "both"],
        default="both",
    )
    parser.add_argument(
        "--exclude-biomarkers",
        action="store_true",
        help="Exclude gamma_GTP, SGOT_AST, SGOT_ALT",
    )
    args = parser.parse_args()

    trainer = ReverseLogicTrainer(
        csv_path=Path(args.csv) if args.csv else None,
        exclude_biomarkers=args.exclude_biomarkers,
    )

    if args.target == "both":
        trainer.train_both_targets()
    else:
        trainer.train_single_target(target=args.target)


if __name__ == "__main__":
    main()
