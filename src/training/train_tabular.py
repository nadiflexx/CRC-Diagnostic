"""
Tabular model training pipeline.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from src.config.logger import log as logger
from src.config.paths import paths
from src.data.processing.clinical_data_generator import ClinicalDataGenerator
from src.data.processing.tabular_preprocessor import TabularPreprocessor
from src.evaluation.explainability import TabularExplainer
from src.models.tabular_model import TabularCancerModel


def validate_data_quality(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> bool:
    """
    Anti-leakage sanity check: detect features with perfect class separation.

    Checks each feature column for non-overlapping value ranges between
    the healthy (0) and cancer (1) groups. Perfect separation indicates
    data leakage or a trivially separable synthetic dataset.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Binary target array of shape (n_samples,).
        feature_names (list[str]): Feature names aligned with columns of X.

    Returns:
        bool: ``True`` if no perfectly separated features are found,
            ``False`` otherwise.
    """
    perfect_sep = []
    for i, name in enumerate(feature_names):
        col = X[:, i]
        healthy_max = col[y == 0].max()
        healthy_min = col[y == 0].min()
        cancer_max = col[y == 1].max()
        cancer_min = col[y == 1].min()
        if healthy_max < cancer_min or cancer_max < healthy_min:
            perfect_sep.append(name)

    if perfect_sep:
        logger.warning(
            f"⚠️ {len(perfect_sep)} perfectly separated features detected: {perfect_sep}"
        )
        return False

    logger.info(f"✅ Anti-leakage check passed — {len(feature_names)} features OK")
    return True


def train_tabular_pipeline():
    """
    Execute the end-to-end tabular CRC model training pipeline.

    Steps:
        1. Generate the clinical + radiomic dataset via
           ``ClinicalDataGenerator``.
        2. Preprocess features with ``TabularPreprocessor``.
        3. Run anti-leakage quality check on the full dataset.
        4. Split into train / test sets (80 / 20, stratified).
        5. Train an XGBoost model with Optuna hyperparameter search
           and recall-targeted threshold calibration.
        6. Evaluate the model on the held-out test set and
           cross-validate.
        7. Generate SHAP explanations for two representative samples.
        8. Save the trained model and preprocessor to disk.

    Returns:
        tuple[TabularCancerModel, TabularPreprocessor, dict]:
            - Trained ``TabularCancerModel`` instance.
            - Fitted ``TabularPreprocessor`` instance.
            - Evaluation results dictionary as returned by
              ``TabularCancerModel.evaluate``.
    """
    # 1. Generate data
    logger.info("═══ Generating Clinical Dataset ═══")
    generator = ClinicalDataGenerator(seed=42)
    combined = generator.generate_and_validate()
    logger.info(f"Dataset: {len(combined):,} patients")

    # 2. Preprocess
    logger.info("═══ Preprocessing ═══")
    preprocessor = TabularPreprocessor()
    X, y, feature_names = preprocessor.fit_transform(combined)

    # 3. Anti-leakage check
    logger.info("═══ Anti-Leakage Check ═══")
    validate_data_quality(X, y, feature_names)

    # 4. Train / test split (80/20, stratified)
    logger.info("═══ Train/Test Split ═══")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 5. Train — pasar X_val e y_val explícitamente
    logger.info("═══ Training ═══")
    model = TabularCancerModel(model_type="xgboost")
    model.train(
        X_train,
        y_train,
        X_val=X_val,  # ← ahora se usan de verdad
        y_val=y_val,  # ← ahora se usan de verdad
        feature_names=feature_names,
        n_trials=30,
        recall_target=0.80,
    )

    # 6. Evaluate
    logger.info("═══ Evaluation ═══")
    results = model.evaluate(X_test, y_test)
    model.cross_validate(X, y)

    # 7. Explainability
    logger.info("═══ Explainability ═══")
    explainer = TabularExplainer(model, feature_names)
    explainer.fit(X_train)
    for i, label in [(0, "first"), (len(X_test) // 2, "middle")]:
        explanation = explainer.explain(X_test[i : i + 1])
        logger.info(f"Sample {label} (real={y_test[i]}):")
        for feat, val in explanation["top_risk_factors"][:3]:
            logger.info(f"  ↑ {feat}: +{val:.4f}")
    explainer.plot_explanation(
        X_test[0:1],
        save_path=str(paths.MODELS / "shap_explanation.png"),
    )

    # 8. Save
    logger.info("═══ Saving ═══")
    paths.MODELS.mkdir(parents=True, exist_ok=True)
    model.save(paths.TABULAR_MODEL_PATH)
    preprocessor.save(paths.TABULAR_PREPROCESSOR_PATH)
    logger.info("✅ Pipeline complete")
    return model, preprocessor, results


if __name__ == "__main__":
    train_tabular_pipeline()
