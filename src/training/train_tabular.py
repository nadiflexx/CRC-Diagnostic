"""
Tabular model training pipeline.

Contains two independent pipelines:

  1. `train_tabular_pipeline`  — existing XGBoost/LightGBM pipeline fed by
     SyntheticPatientGenerator (clinical lab features).

  2. `train_csv_mlp_pipeline`  — new MLP pipeline fed by the Kaggle CSV
     (colorectal_cancer_dataset.csv) plus CsvSyntheticGenerator.

Run either pipeline directly:
    python -m src.training.train_tabular            # runs both by default
    python -m src.training.train_tabular --xgb      # only XGBoost
    python -m src.training.train_tabular --mlp      # only CSV MLP
"""

from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.logger import log as logger
from src.config.paths import paths
from src.data.processing.balancer import DataBalancer
from src.data.processing.synthetic_generator import (
    CsvSyntheticGenerator,
    SyntheticPatientGenerator,
)
from src.data.processing.tabular_preprocessor import (
    CsvTabularPreprocessor,
    TabularPreprocessor,
)
from src.evaluation.explainability import TabularExplainer
from src.models.tabular_model import CsvMlpModel, TabularCancerModel


# ═══════════════════════════════════════════════════════════
#  HELPER: data-quality / leakage check
# ═══════════════════════════════════════════════════════════

def validate_data_quality(X, y, feature_names, stage: str = "") -> bool:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    lr = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(lr, X, y, cv=5, scoring="roc_auc")
    mean_auc = scores.mean()
    logger.info(f"[{stage}] LogReg AUC: {mean_auc:.4f} ± {scores.std():.4f}")
    if mean_auc > 0.95:
        logger.warning(f"⚠️ AUC={mean_auc:.3f} → PROBABLE LEAKAGE")
        from sklearn.feature_selection import mutual_info_classif

        mi = mutual_info_classif(X, y, random_state=42)
        top = sorted(
            zip(feature_names, mi, strict=True), key=lambda x: x[1], reverse=True
        )[:10]
        for name, score in top:
            logger.warning(f"  {name}: MI={score:.4f}")
        return False
    return True


# ═══════════════════════════════════════════════════════════
#  PIPELINE 1: XGBoost / LightGBM  (existing)
# ═══════════════════════════════════════════════════════════

def train_tabular_pipeline():
    """
    Existing pipeline: synthetic clinical-lab data → XGBoost → SHAP explainability.
    """
    preprocessor = TabularPreprocessor()
    logger.info("═══ Generating Data ═══")
    generator = SyntheticPatientGenerator(seed=42)
    combined = generator.generate_balanced_dataset(n_per_class=5000)
    logger.info(f"Dataset: {len(combined)} patients")

    logger.info("═══ Preprocessing ═══")
    X, y, feature_names = preprocessor.fit_transform(combined)

    logger.info("═══ Anti-Leakage Check ═══")
    validate_data_quality(X, y, feature_names, stage="Pre-split")

    logger.info("═══ Train/Val/Test Split ═══")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    logger.info("═══ Training ═══")
    class_weights = DataBalancer.compute_class_weights(y_train)
    model = TabularCancerModel(model_type="xgboost")
    model.train(
        X_train, y_train, X_val, y_val,
        feature_names=feature_names,
        class_weights=class_weights,
    )

    logger.info("═══ Evaluation ═══")
    results = model.evaluate(X_test, y_test)
    model.cross_validate(X, y)

    logger.info("═══ Explainability ═══")
    explainer = TabularExplainer(model, feature_names)
    explainer.fit(X_train)
    for i, label in [(0, "first"), (len(X_test) // 2, "middle")]:
        explanation = explainer.explain(X_test[i : i + 1])
        logger.info(f"Sample {label} (real={y_test[i]}):")
        for feat, val in explanation["top_risk_factors"][:3]:
            logger.info(f"  ↑ {feat}: +{val:.4f}")
    explainer.plot_explanation(
        X_test[0:1], save_path=str(paths.ROOT / "shap_explanation.png")
    )

    logger.info("═══ Saving ═══")
    paths.MODELS.mkdir(parents=True, exist_ok=True)
    model.save(paths.TABULAR_MODEL_PATH)
    preprocessor.save(paths.TABULAR_PREPROCESSOR_PATH)
    logger.info("✅ XGBoost pipeline complete")
    return model, preprocessor, results


# ═══════════════════════════════════════════════════════════
#  PIPELINE 2: CSV MLP  (new)
# ═══════════════════════════════════════════════════════════

def train_csv_mlp_pipeline(
    regenerate_data: bool = False,
    save_model: bool = True,
    generate_plots: bool = True,
):
    """
    New pipeline: Kaggle CSV + synthetic healthy patients → MLP model.

    Steps
    -----
    1. Process the raw CSV (real cancer patients, Diagnosis=1)
    2. Generate matching synthetic healthy patients (Diagnosis=0)
    3. Combine into a balanced dataset and optionally cache it
    4. Train CsvMlpModel with recall-optimized threshold
    5. Generate evaluation plots
    6. Save model artefacts
    """
    csv_preprocessor = CsvTabularPreprocessor()

    # ── 1. Real cancer patients ──────────────────────────────────────────────
    logger.info("═══ [CSV MLP] Processing raw CSV ═══")
    df_cancer = csv_preprocessor.process_raw_dataset(
        input_path=paths.CSV_RAW_TABULAR,
        diagnosis_label=1,
    )
    n_cancer = len(df_cancer)
    logger.info(f"Real cancer patients: {n_cancer:,}")

    # ── 2. Synthetic healthy patients ────────────────────────────────────────
    full_path = paths.CSV_FULL_TABULAR
    if regenerate_data or not full_path.exists():
        logger.info(f"═══ [CSV MLP] Generating {n_cancer:,} synthetic healthy patients ═══")
        csv_gen = CsvSyntheticGenerator(seed=42)
        df_healthy = csv_gen.generate_healthy_patients(n=n_cancer)

        df_combined = (
            pd.concat([df_cancer, df_healthy], ignore_index=True)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        df_combined.to_csv(full_path, index=False)
        logger.info(f"Balanced dataset saved → {full_path}")
    else:
        logger.info(f"[CSV MLP] Using cached dataset at {full_path}")
        df_combined = pd.read_csv(full_path)

    logger.info(
        f"Dataset: {len(df_combined):,} rows | "
        f"Diagnosis: {df_combined['Diagnosis'].value_counts().to_dict()}"
    )

    # ── 3. Train MLP ────────────────────────────────────────────────────────
    logger.info("═══ [CSV MLP] Training ═══")
    mlp_model = CsvMlpModel()
    mlp_model.fit(df_combined)

    # ── 4. Evaluation plots ─────────────────────────────────────────────────
    if generate_plots:
        try:
            from src.evaluation.tabular_plots import generate_csv_evaluation_plots
            logger.info("═══ [CSV MLP] Generating evaluation plots ═══")
            generate_csv_evaluation_plots(
                model=mlp_model,
                df=df_combined,
                output_dir=paths.CSV_PLOTS_DIR,
            )
        except ImportError:
            logger.warning("[CSV MLP] tabular_plots not found — skipping plots")

    # ── 5. Save ─────────────────────────────────────────────────────────────
    if save_model:
        mlp_model.save(paths.CSV_MLP_MODEL_DIR)

    logger.info("✅ CSV MLP pipeline complete")
    return mlp_model


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabular training pipelines")
    parser.add_argument("--xgb", action="store_true", help="Run XGBoost pipeline only")
    parser.add_argument("--mlp", action="store_true", help="Run CSV MLP pipeline only")
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Force regeneration of the CSV balanced dataset",
    )
    args = parser.parse_args()

    run_xgb = args.xgb or (not args.xgb and not args.mlp)
    run_mlp = args.mlp or (not args.xgb and not args.mlp)

    if run_xgb:
        train_tabular_pipeline()
    if run_mlp:
        train_csv_mlp_pipeline(regenerate_data=args.regenerate)