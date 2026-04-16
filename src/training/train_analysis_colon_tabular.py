"""
Training script for the XGBoost colon tabular analysis model.
"""

import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from src.config.constants import (
    TABULAR_ANALYSIS_POSITIVE_STAGES,
    TABULAR_ANALYSIS_RANDOM_SEED,
    TABULAR_ANALYSIS_RAW_FEATURES,
    TABULAR_ANALYSIS_STAGE_COLUMN,
    TABULAR_ANALYSIS_TARGET,
    TABULAR_ANALYSIS_TEST_SIZE,
)
from src.config.logger import log
from src.config.paths import paths
from src.evaluation.generic_tabular_data_analysis import GenericTabularDataAnalyzer
from src.models.analysis_colon_tabular_model import XGBoostColonAnalysisModel


def load_and_prepare_data():
    """Loads the raw tabular dataset and creates the advanced cancer target."""
    data_path = paths.RAW_TABULAR_CRC_DATASET
    log.info(f"[LOAD] Leyendo dataset tabular: {data_path}")
    df = pd.read_csv(data_path)

    selected_columns = TABULAR_ANALYSIS_RAW_FEATURES + [TABULAR_ANALYSIS_STAGE_COLUMN]
    df = df[selected_columns].dropna().copy()
    df[TABULAR_ANALYSIS_TARGET] = df[TABULAR_ANALYSIS_STAGE_COLUMN].isin(TABULAR_ANALYSIS_POSITIVE_STAGES).astype(int)

    X = df[TABULAR_ANALYSIS_RAW_FEATURES].copy()
    y = df[TABULAR_ANALYSIS_TARGET].copy()

    log.info(f"[LOAD] Registros útiles: {len(df):,}")
    log.info(f"[LOAD] Distribución target: {y.value_counts().to_dict()}")
    return df, X, y


def split_data(X, y):
    """Splits data into train and test partitions."""
    return train_test_split(
        X,
        y,
        test_size=TABULAR_ANALYSIS_TEST_SIZE,
        random_state=TABULAR_ANALYSIS_RANDOM_SEED,
        stratify=y,
    )


def train_model(X_train, y_train, X_test, y_test):
    """Trains and evaluates the XGBoost model."""
    model = XGBoostColonAnalysisModel()
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    importance = model.get_feature_importance()
    model.save()
    return model, metrics, importance


def run_analysis(model, X_test, y_test, full_df, importance):
    """Builds analysis artifacts for the trained model."""
    analyzer = GenericTabularDataAnalyzer()
    analyzer.run_full_analysis(
        model=model,
        X_test=X_test,
        y_test=y_test,
        full_df=full_df,
        feature_importance=importance,
    )


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  XGBOOST ANALYSIS - COLON TABULAR")
    print("=" * 80)

    full_df, X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model, metrics, importance = train_model(X_train, y_train, X_test, y_test)
    run_analysis(model, X_test, y_test, full_df, importance)

    print(f"\nModelo guardado en: {paths.ANALYSIS_TABULAR_XGB_MODEL_PATH}")
    print(f"Gráficas guardadas en: {paths.ANALYSIS_TABULAR_REPORT_DIR}")
    print("=" * 80 + "\n")
