"""
Tabular model training pipeline.
"""

from sklearn.model_selection import train_test_split

from src.config.logger import log as logger
from src.config.paths import paths
from src.data.processing.balancer import DataBalancer
from src.data.processing.synthetic_generator import SyntheticPatientGenerator
from src.data.processing.tabular_preprocessor import TabularPreprocessor
from src.evaluation.explainability import TabularExplainer
from src.models.tabular_model import TabularCancerModel


def validate_data_quality(X, y, feature_names, stage=""):
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


def train_tabular_pipeline():
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
        X_train,
        y_train,
        X_val,
        y_val,
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
    logger.info("✅ Pipeline complete")
    return model, preprocessor, results


if __name__ == "__main__":
    train_tabular_pipeline()
