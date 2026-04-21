"""
Tests completos para:
- ReverseLogicAnalyzer (reverse_logic_analyzer.py)
- ReverseLogicTabularModel (reverse_logic_tabular_model.py)
- ReverseLogicTrainer (train_reverse_logic_tabular.py)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES COMPARTIDOS
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir(tmp_path):
    """Directorio temporal para outputs."""
    return tmp_path


@pytest.fixture
def binary_y_numeric():
    """Serie binaria numérica (0/1)."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.integers(0, 2, size=200))


@pytest.fixture
def binary_y_string():
    """Serie binaria de strings (Yes/No)."""
    rng = np.random.default_rng(42)
    return pd.Series(np.where(rng.integers(0, 2, size=200) == 1, "Yes", "No"))


@pytest.fixture
def predictions_proba():
    """Probabilidades de predicción para dos modelos."""
    rng = np.random.default_rng(42)
    return {
        "XGBoost": rng.random(200),
        "Random Forest": rng.random(200),
    }


@pytest.fixture
def predictions_binary(binary_y_numeric):
    """Predicciones binarias para dos modelos."""
    rng = np.random.default_rng(42)
    return {
        "XGBoost": rng.integers(0, 2, size=200),
        "Random Forest": rng.integers(0, 2, size=200),
    }


@pytest.fixture
def sample_df():
    """DataFrame de muestra con columnas básicas."""
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame(
        {
            "Country": rng.choice(["USA", "UK", "Spain", "Japan", "Brazil"], size=n),
            "Age": rng.integers(30, 80, size=n),
            "Gender": rng.choice(["M", "F"], size=n),
            "Smoking_History": rng.choice(["Yes", "No"], size=n),
            "Alcohol_Consumption": rng.choice(["Yes", "No"], size=n),
        }
    )


@pytest.fixture
def importance_dfs():
    """DataFrames de importancia de características."""
    features = [f"feature_{i}" for i in range(20)]
    rng = np.random.default_rng(42)
    return {
        "XGBoost": pd.DataFrame(
            {
                "feature": features,
                "importance": rng.random(20),
            }
        ).sort_values("importance", ascending=False),
        "Random Forest": pd.DataFrame(
            {
                "feature": features,
                "importance": rng.random(20),
            }
        ).sort_values("importance", ascending=False),
    }


@pytest.fixture
def analyzer(tmp_dir):
    """Instancia de ReverseLogicAnalyzer."""
    from src.evaluation.reverse_logic_analyzer import ReverseLogicAnalyzer

    return ReverseLogicAnalyzer(tmp_dir)


@pytest.fixture
def clinical_X():
    """Features clínicas para modelo tabular."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame(
        {
            "age": rng.integers(20, 80, size=n).astype(float),
            "height": rng.integers(155, 185, size=n).astype(float),
            "weight": rng.integers(50, 100, size=n).astype(float),
            "waistline": rng.uniform(65, 110, size=n),
            "BMI": rng.uniform(18, 35, size=n),
            "triglyceride": rng.uniform(60, 300, size=n),
            "HDL_chole": rng.uniform(30, 90, size=n),
            "LDL_chole": rng.uniform(60, 200, size=n),
            "hemoglobin": rng.uniform(10, 18, size=n),
            "sex": rng.choice([0, 1], size=n),
        }
    )


@pytest.fixture
def clinical_y_string():
    """Target de string para modelo tabular."""
    rng = np.random.default_rng(42)
    return pd.Series(np.where(rng.random(500) > 0.45, "Yes", "No"))


@pytest.fixture
def clinical_y_numeric():
    """Target numérico para modelo tabular."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.integers(0, 2, size=500))


@pytest.fixture
def trained_model(clinical_X, clinical_y_string):
    """Modelo entrenado con xgboost y random_forest."""
    from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

    model = ReverseLogicTabularModel(
        target="SMK_stat_type_cd",
        numeric_features=[
            "age",
            "height",
            "weight",
            "waistline",
            "BMI",
            "triglyceride",
            "HDL_chole",
            "LDL_chole",
            "hemoglobin",
        ],
        categorical_features=["sex"],
        random_seed=42,
    )
    model.fit(
        clinical_X,
        clinical_y_string,
        model_type="xgboost",
        config={"n_estimators": 10, "max_depth": 3},
    )
    model.fit(
        clinical_X,
        clinical_y_string,
        model_type="random_forest",
        config={"n_estimators": 10, "max_depth": 3, "random_state": 42},
    )
    return model


# ═════════════════════════════════════════════════════════════════════════════
# TESTS: ReverseLogicAnalyzer
# ═════════════════════════════════════════════════════════════════════════════


class TestReverseLogicAnalyzerInit:
    def test_creates_output_directory(self, tmp_path):
        """El directorio de salida se crea si no existe."""
        from src.evaluation.reverse_logic_analyzer import ReverseLogicAnalyzer

        new_dir = tmp_path / "nested" / "output"
        assert not new_dir.exists()
        ReverseLogicAnalyzer(new_dir)
        assert new_dir.exists()

    def test_output_dir_stored(self, tmp_dir):
        """El atributo output_dir es un Path."""
        from src.evaluation.reverse_logic_analyzer import ReverseLogicAnalyzer

        analyzer = ReverseLogicAnalyzer(tmp_dir)
        assert isinstance(analyzer.output_dir, Path)
        assert analyzer.output_dir == tmp_dir

    def test_accepts_string_path(self, tmp_path):
        """Acepta rutas como string."""
        from src.evaluation.reverse_logic_analyzer import ReverseLogicAnalyzer

        analyzer = ReverseLogicAnalyzer(str(tmp_path / "test_out"))
        assert analyzer.output_dir.exists()


class TestPlotRocCurve:
    def test_creates_file(self, analyzer, binary_y_numeric, predictions_proba):
        """plot_roc_curve genera el archivo PNG."""
        analyzer.plot_roc_curve(
            binary_y_numeric,
            predictions_proba,
            target_name="Smoking_History",
            filename="roc_test.png",
        )
        assert (analyzer.output_dir / "roc_test.png").exists()

    def test_file_is_not_empty(self, analyzer, binary_y_numeric, predictions_proba):
        """El archivo PNG generado no está vacío."""
        analyzer.plot_roc_curve(
            binary_y_numeric, predictions_proba, filename="roc_nonempty.png"
        )
        assert (analyzer.output_dir / "roc_nonempty.png").stat().st_size > 0

    def test_single_model(self, analyzer, binary_y_numeric):
        """Funciona con un solo modelo."""
        rng = np.random.default_rng(0)
        predictions = {"SingleModel": rng.random(200)}
        analyzer.plot_roc_curve(
            binary_y_numeric, predictions, filename="roc_single.png"
        )
        assert (analyzer.output_dir / "roc_single.png").exists()

    def test_multiple_models(self, analyzer, binary_y_numeric, predictions_proba):
        """Funciona con múltiples modelos."""
        analyzer.plot_roc_curve(
            binary_y_numeric,
            predictions_proba,
            target_name="Alcohol_Consumption",
            filename="roc_multi.png",
        )
        assert (analyzer.output_dir / "roc_multi.png").exists()

    def test_custom_target_name_in_filename(
        self, analyzer, binary_y_numeric, predictions_proba
    ):
        """El archivo se guarda con el nombre especificado."""
        filename = "custom_roc.png"
        analyzer.plot_roc_curve(binary_y_numeric, predictions_proba, filename=filename)
        assert (analyzer.output_dir / filename).exists()


class TestPlotConfusionMatrices:
    def test_creates_file_numeric_labels(
        self, analyzer, binary_y_numeric, predictions_binary
    ):
        """Genera archivo con etiquetas numéricas."""
        analyzer.plot_confusion_matrices(
            binary_y_numeric, predictions_binary, filename="cm_numeric.png"
        )
        assert (analyzer.output_dir / "cm_numeric.png").exists()

    def test_creates_file_string_labels(
        self, analyzer, binary_y_string, predictions_binary
    ):
        """Genera archivo con etiquetas de string Yes/No."""
        analyzer.plot_confusion_matrices(
            binary_y_string, predictions_binary, filename="cm_string.png"
        )
        assert (analyzer.output_dir / "cm_string.png").exists()

    def test_file_nonempty(self, analyzer, binary_y_numeric, predictions_binary):
        """Archivo generado no vacío."""
        analyzer.plot_confusion_matrices(
            binary_y_numeric, predictions_binary, filename="cm_nonempty.png"
        )
        assert (analyzer.output_dir / "cm_nonempty.png").stat().st_size > 0

    def test_single_model(self, analyzer, binary_y_numeric):
        """Funciona con un solo modelo."""
        rng = np.random.default_rng(0)
        preds = {"OnlyModel": rng.integers(0, 2, size=200)}
        analyzer.plot_confusion_matrices(
            binary_y_numeric, preds, filename="cm_single.png"
        )
        assert (analyzer.output_dir / "cm_single.png").exists()

    def test_alcohol_target(self, analyzer, binary_y_numeric, predictions_binary):
        """Funciona con target Alcohol_Consumption."""
        analyzer.plot_confusion_matrices(
            binary_y_numeric,
            predictions_binary,
            target_name="Alcohol_Consumption",
            filename="cm_alcohol.png",
        )
        assert (analyzer.output_dir / "cm_alcohol.png").exists()

    def test_all_zeros_prediction(self, analyzer, binary_y_numeric):
        """Maneja predicciones todas cero sin error."""
        preds = {"ZeroModel": np.zeros(200, dtype=int)}
        analyzer.plot_confusion_matrices(
            binary_y_numeric, preds, filename="cm_zeros.png"
        )
        assert (analyzer.output_dir / "cm_zeros.png").exists()

    def test_all_ones_prediction(self, analyzer, binary_y_numeric):
        """Maneja predicciones todas uno sin error."""
        preds = {"OnesModel": np.ones(200, dtype=int)}
        analyzer.plot_confusion_matrices(
            binary_y_numeric, preds, filename="cm_ones.png"
        )
        assert (analyzer.output_dir / "cm_ones.png").exists()


class TestPlotConfusionMatrixDetailed:
    def test_creates_file(self, analyzer, binary_y_numeric, predictions_binary):
        """Genera el archivo de métricas detalladas."""
        analyzer.plot_confusion_matrix_detailed(
            binary_y_numeric, predictions_binary, filename="cmd_test.png"
        )
        assert (analyzer.output_dir / "cmd_test.png").exists()

    def test_creates_file_string_labels(
        self, analyzer, binary_y_string, predictions_binary
    ):
        """Funciona con etiquetas Yes/No."""
        analyzer.plot_confusion_matrix_detailed(
            binary_y_string, predictions_binary, filename="cmd_string.png"
        )
        assert (analyzer.output_dir / "cmd_string.png").exists()

    def test_single_model(self, analyzer, binary_y_numeric):
        """Funciona con un solo modelo."""
        rng = np.random.default_rng(0)
        preds = {"Solo": rng.integers(0, 2, size=200)}
        analyzer.plot_confusion_matrix_detailed(
            binary_y_numeric, preds, filename="cmd_single.png"
        )
        assert (analyzer.output_dir / "cmd_single.png").exists()

    def test_file_nonempty(self, analyzer, binary_y_numeric, predictions_binary):
        """Archivo no vacío."""
        analyzer.plot_confusion_matrix_detailed(
            binary_y_numeric, predictions_binary, filename="cmd_nonempty.png"
        )
        assert (analyzer.output_dir / "cmd_nonempty.png").stat().st_size > 0


class TestPlotFeatureImportance:
    def test_creates_files(self, analyzer, importance_dfs):
        """Genera el archivo de importancia."""
        analyzer.plot_feature_importance(importance_dfs, filename="fi_test.png")
        assert (analyzer.output_dir / "fi_test.png").exists()

    def test_empty_dict_does_nothing(self, analyzer):
        """No falla si el dict está vacío."""
        analyzer.plot_feature_importance({})  # No debe lanzar excepción

    def test_empty_dataframe_skipped(self, analyzer):
        """Skippea DataFrames vacíos."""
        importance_dfs = {"XGBoost": pd.DataFrame()}
        analyzer.plot_feature_importance(importance_dfs, filename="fi_empty.png")
        # No debe lanzar excepción; el archivo puede o no existir

    def test_single_model(self, analyzer, importance_dfs):
        """Funciona con un solo modelo."""
        single = {"XGBoost": importance_dfs["XGBoost"]}
        analyzer.plot_feature_importance(single, filename="fi_single.png")
        assert (analyzer.output_dir / "fi_single.png").exists()

    def test_top_n_respected(self, analyzer, importance_dfs):
        """No falla con top_n=5."""
        analyzer.plot_feature_importance(
            importance_dfs, top_n=5, filename="fi_top5.png"
        )
        assert (analyzer.output_dir / "fi_top5.png").exists()

    def test_alcohol_target(self, analyzer, importance_dfs):
        """Funciona con target Alcohol_Consumption."""
        analyzer.plot_feature_importance(
            importance_dfs, target_name="Alcohol_Consumption", filename="fi_alcohol.png"
        )
        assert (analyzer.output_dir / "fi_alcohol.png").exists()


class TestPlotCountryHabitPatterns:
    def test_creates_file_string_habit(self, analyzer, sample_df):
        """Genera archivo con hábito Yes/No."""
        analyzer.plot_country_habit_patterns(
            sample_df, habit_column="Smoking_History", filename="country_smoke.png"
        )
        assert (analyzer.output_dir / "country_smoke.png").exists()

    def test_creates_file_numeric_habit(self, analyzer, sample_df):
        """Funciona con columna numérica."""
        df = sample_df.copy()
        df["NumericHabit"] = (df["Smoking_History"] == "Yes").astype(int)
        analyzer.plot_country_habit_patterns(
            df, habit_column="NumericHabit", filename="country_num.png"
        )
        assert (analyzer.output_dir / "country_num.png").exists()

    def test_alcohol_column(self, analyzer, sample_df):
        """Funciona con Alcohol_Consumption."""
        analyzer.plot_country_habit_patterns(
            sample_df,
            habit_column="Alcohol_Consumption",
            filename="country_alcohol.png",
        )
        assert (analyzer.output_dir / "country_alcohol.png").exists()

    def test_file_nonempty(self, analyzer, sample_df):
        """Archivo no vacío."""
        analyzer.plot_country_habit_patterns(sample_df, filename="country_nonempty.png")
        assert (analyzer.output_dir / "country_nonempty.png").stat().st_size > 0


class TestPlotPredictionHistograms:
    def test_creates_file_per_model(
        self, analyzer, binary_y_numeric, predictions_proba
    ):
        """Crea un archivo por modelo."""
        analyzer.plot_prediction_histograms(
            binary_y_numeric,
            predictions_proba,
            target_name="Smoking_History",
            filename_prefix="hist_test",
        )
        for model_name in predictions_proba:
            safe = model_name.lower().replace(" ", "_")
            expected = analyzer.output_dir / f"hist_test_{safe}_smoking_history.png"
            assert expected.exists(), f"Archivo no creado: {expected}"

    def test_files_nonempty(self, analyzer, binary_y_numeric, predictions_proba):
        """Archivos no vacíos."""
        analyzer.plot_prediction_histograms(
            binary_y_numeric, predictions_proba, filename_prefix="hist_nonempty"
        )
        for model_name in predictions_proba:
            safe = model_name.lower().replace(" ", "_")
            path = analyzer.output_dir / f"hist_nonempty_{safe}_smoking_history.png"
            assert path.stat().st_size > 0

    def test_string_y(self, analyzer, binary_y_string, predictions_proba):
        """Funciona con etiquetas Yes/No."""
        analyzer.plot_prediction_histograms(
            binary_y_string,
            predictions_proba,
            target_name="Smoking_History",
            filename_prefix="hist_str",
        )
        for model_name in predictions_proba:
            safe = model_name.lower().replace(" ", "_")
            path = analyzer.output_dir / f"hist_str_{safe}_smoking_history.png"
            assert path.exists()

    def test_single_model(self, analyzer, binary_y_numeric):
        """Funciona con un solo modelo."""
        rng = np.random.default_rng(0)
        preds = {"XGBoost": rng.random(200)}
        analyzer.plot_prediction_histograms(
            binary_y_numeric, preds, target_name="DRK_YN", filename_prefix="hist_single"
        )
        assert (analyzer.output_dir / "hist_single_xgboost_drk_yn.png").exists()

    def test_empty_class(self, analyzer):
        """Maneja clase sin muestras sin error."""
        y = pd.Series(np.zeros(100, dtype=int))  # Solo clase 0
        rng = np.random.default_rng(0)
        preds = {"Model": rng.random(100)}
        # No debe lanzar excepción
        analyzer.plot_prediction_histograms(
            y, preds, filename_prefix="hist_empty_class"
        )


class TestRunFullAnalysis:
    def test_run_full_analysis_smoke(
        self, analyzer, trained_model, clinical_X, clinical_y_string, sample_df
    ):
        """run_full_analysis se ejecuta sin excepciones."""
        # Usar un df con Country para evitar skip del country plot
        full_df = sample_df.copy()
        full_df["SMK_stat_type_cd"] = np.where(
            np.random.default_rng(0).random(len(full_df)) > 0.5, "Yes", "No"
        )

        analyzer.run_full_analysis(
            model=trained_model,
            X_test=clinical_X.iloc[:100],
            y_test=clinical_y_string.iloc[:100],
            full_df=full_df,
            target_name="SMK_stat_type_cd",
            model_types=["xgboost", "random_forest"],
        )

    def test_run_full_analysis_without_country(
        self, analyzer, trained_model, clinical_X, clinical_y_string
    ):
        """run_full_analysis sin columna Country no falla."""
        df_no_country = pd.DataFrame(
            {
                "Age": np.random.randint(30, 80, 100),
                "SMK_stat_type_cd": np.where(np.random.random(100) > 0.5, "Yes", "No"),
            }
        )
        analyzer.run_full_analysis(
            model=trained_model,
            X_test=clinical_X.iloc[:100],
            y_test=clinical_y_string.iloc[:100],
            full_df=df_no_country,
            target_name="SMK_stat_type_cd",
            model_types=["xgboost"],
        )

    def test_run_full_analysis_creates_output_files(
        self, analyzer, trained_model, clinical_X, clinical_y_string, sample_df
    ):
        """run_full_analysis genera al menos algunos archivos."""
        full_df = sample_df.copy()
        full_df["SMK_stat_type_cd"] = "No"

        analyzer.run_full_analysis(
            model=trained_model,
            X_test=clinical_X.iloc[:100],
            y_test=clinical_y_string.iloc[:100],
            full_df=full_df,
            target_name="SMK_stat_type_cd",
            model_types=["xgboost"],
        )
        png_files = list(analyzer.output_dir.glob("*.png"))
        assert len(png_files) > 0


# ═════════════════════════════════════════════════════════════════════════════
# TESTS: FeatureWeightedXGBClassifier
# ═════════════════════════════════════════════════════════════════════════════


class TestFeatureWeightedXGBClassifier:
    @pytest.fixture
    def classifier(self):
        from src.models.reverse_logic_tabular_model import FeatureWeightedXGBClassifier

        return FeatureWeightedXGBClassifier(
            numeric_features=["age", "height", "BMI"],
            categorical_features=["sex"],
            random_seed=42,
            model_config={"n_estimators": 10, "max_depth": 3},
        )

    @pytest.fixture
    def simple_X(self):
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "age": rng.uniform(20, 80, 100),
                "height": rng.uniform(150, 190, 100),
                "BMI": rng.uniform(18, 35, 100),
                "sex": rng.choice([0, 1], 100),
            }
        )

    @pytest.fixture
    def simple_y(self):
        rng = np.random.default_rng(42)
        return pd.Series(rng.integers(0, 2, 100))

    def test_fit_returns_self(self, classifier, simple_X, simple_y):
        """fit() devuelve la instancia."""
        result = classifier.fit(simple_X, simple_y)
        assert result is classifier

    def test_predict_shape(self, classifier, simple_X, simple_y):
        """predict() retorna array de misma longitud que X."""
        classifier.fit(simple_X, simple_y)
        preds = classifier.predict(simple_X)
        assert preds.shape == (len(simple_X),)

    def test_predict_binary_values(self, classifier, simple_X, simple_y):
        """predict() retorna solo 0 y 1."""
        classifier.fit(simple_X, simple_y)
        preds = classifier.predict(simple_X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, classifier, simple_X, simple_y):
        """predict_proba() retorna shape (n, 2)."""
        classifier.fit(simple_X, simple_y)
        proba = classifier.predict_proba(simple_X)
        assert proba.shape == (len(simple_X), 2)

    def test_predict_proba_sums_to_one(self, classifier, simple_X, simple_y):
        """Las probabilidades suman 1 por fila."""
        classifier.fit(simple_X, simple_y)
        proba = classifier.predict_proba(simple_X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(simple_X)), atol=1e-5)

    def test_feature_importances_available(self, classifier, simple_X, simple_y):
        """feature_importances_ disponible tras fit."""
        classifier.fit(simple_X, simple_y)
        importances = classifier.feature_importances_
        assert importances is not None
        assert len(importances) > 0

    def test_no_categorical_features(self, simple_X, simple_y):
        """Funciona sin categorical_features."""
        from src.models.reverse_logic_tabular_model import FeatureWeightedXGBClassifier

        clf = FeatureWeightedXGBClassifier(
            numeric_features=["age", "height", "BMI"],
            categorical_features=[],
            model_config={"n_estimators": 5},
        )
        simple_X_num = simple_X[["age", "height", "BMI"]]
        clf.fit(simple_X_num, simple_y)
        preds = clf.predict(simple_X_num)
        assert len(preds) == len(simple_X_num)

    def test_feature_weight_overrides_applied(self, simple_X, simple_y):
        """feature_weight_overrides se aplican sin error."""
        from src.models.reverse_logic_tabular_model import FeatureWeightedXGBClassifier

        clf = FeatureWeightedXGBClassifier(
            numeric_features=["age", "height", "BMI"],
            categorical_features=["sex"],
            model_config={"n_estimators": 5},
            feature_weight_overrides={"age": 2.0, "BMI": 0.5},
        )
        clf.fit(simple_X, simple_y)
        preds = clf.predict(simple_X)
        assert len(preds) == len(simple_X)

    def test_classes_set_after_fit(self, classifier, simple_X, simple_y):
        """classes_ se establece tras fit."""
        classifier.fit(simple_X, simple_y)
        assert hasattr(classifier, "classes_")
        assert set(classifier.classes_).issubset({0, 1})


# ═════════════════════════════════════════════════════════════════════════════
# TESTS: ReverseLogicTabularModel
# ═════════════════════════════════════════════════════════════════════════════


class TestReverseLogicTabularModelInit:
    def test_default_target(self):
        """Target por defecto es Smoking_History."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel()
        assert model.target == "Smoking_History"

    def test_custom_target(self):
        """Target personalizado se almacena."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(target="DRK_YN")
        assert model.target == "DRK_YN"

    def test_default_numeric_features(self):
        """Features numéricas por defecto incluyen Age."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel()
        assert "Age" in model.numeric_features

    def test_empty_categorical_features(self):
        """categorical_features vacío por defecto."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel()
        assert model.categorical_features == []

    def test_custom_features(self):
        """Features personalizadas se almacenan."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(
            numeric_features=["age", "BMI"],
            categorical_features=["sex"],
        )
        assert model.numeric_features == ["age", "BMI"]
        assert model.categorical_features == ["sex"]

    def test_pipelines_empty_initially(self):
        """pipelines vacío al inicializar."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel()
        assert model.pipelines == {}

    def test_metrics_empty_initially(self):
        """metrics vacío al inicializar."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel()
        assert model.metrics == {}


class TestEncodeTarget:
    def test_string_target_encoded_to_binary(self, clinical_X, clinical_y_string):
        """Target de string se codifica a 0/1."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(numeric_features=["age"])
        encoded = model._encode_target(clinical_y_string, "xgboost", fit_encoder=True)
        assert set(encoded.unique()).issubset({0, 1})

    def test_numeric_binary_unchanged(self, clinical_y_numeric):
        """Target numérico 0/1 no se modifica."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(numeric_features=["age"])
        encoded = model._encode_target(clinical_y_numeric, "xgboost", fit_encoder=False)
        assert set(encoded.unique()).issubset({0, 1})

    def test_encoder_stored(self, clinical_y_string):
        """LabelEncoder se almacena en label_encoders."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(numeric_features=["age"])
        model._encode_target(clinical_y_string, "xgboost", fit_encoder=True)
        assert "xgboost" in model.label_encoders

    def test_reuse_encoder(self, clinical_y_string):
        """Reutiliza encoder existente sin fit."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(numeric_features=["age"])
        enc1 = model._encode_target(clinical_y_string, "xgboost", fit_encoder=True)
        enc2 = model._encode_target(clinical_y_string, "xgboost", fit_encoder=False)
        pd.testing.assert_series_equal(enc1, enc2)

    def test_preserves_index(self, clinical_y_string):
        """Preserva el índice original."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(numeric_features=["age"])
        encoded = model._encode_target(clinical_y_string, "xgboost", fit_encoder=True)
        assert list(encoded.index) == list(clinical_y_string.index)


class TestFitMethod:
    def test_fit_xgboost(self, clinical_X, clinical_y_string):
        """fit() con xgboost almacena pipeline."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(
            numeric_features=["age", "height", "BMI"],
            categorical_features=["sex"],
        )
        model.fit(
            clinical_X,
            clinical_y_string,
            model_type="xgboost",
            config={"n_estimators": 5},
        )
        assert "xgboost" in model.pipelines

    def test_fit_random_forest(self, clinical_X, clinical_y_string):
        """fit() con random_forest almacena pipeline."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(
            numeric_features=["age", "height", "BMI"],
            categorical_features=["sex"],
        )
        model.fit(
            clinical_X,
            clinical_y_string,
            model_type="random_forest",
            config={"n_estimators": 5, "random_state": 42},
        )
        assert "random_forest" in model.pipelines

    def test_fit_lightgbm(self, clinical_X, clinical_y_string):
        """fit() con lightgbm almacena pipeline."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(
            numeric_features=["age", "height", "BMI"],
            categorical_features=["sex"],
        )
        model.fit(
            clinical_X,
            clinical_y_string,
            model_type="lightgbm",
            config={"n_estimators": 5, "verbose": -1},
        )
        assert "lightgbm" in model.pipelines

    def test_fit_invalid_model_type(self, clinical_X, clinical_y_string):
        """fit() con model_type inválido lanza ValueError."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(numeric_features=["age"])
        with pytest.raises(ValueError, match="Unknown model_type"):
            model.fit(clinical_X[["age"]], clinical_y_string, model_type="svm")

    def test_fit_returns_self(self, clinical_X, clinical_y_string):
        """fit() devuelve la instancia para chaining."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(numeric_features=["age"])
        result = model.fit(
            clinical_X[["age"]],
            clinical_y_string,
            model_type="xgboost",
            config={"n_estimators": 5},
        )
        assert result is model

    def test_fit_with_string_target(self, clinical_X, clinical_y_string):
        """fit() acepta target de strings Yes/No."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(numeric_features=["age", "BMI"])
        model.fit(
            clinical_X[["age", "BMI"]],
            clinical_y_string,
            model_type="random_forest",
            config={"n_estimators": 5, "random_state": 0},
        )
        assert "random_forest" in model.pipelines

    def test_fit_multiple(self, clinical_X, clinical_y_string):
        """fit_multiple() entrena varios modelos."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(numeric_features=["age", "BMI"])
        X = clinical_X[["age", "BMI"]]
        model.fit_multiple(
            X,
            clinical_y_string,
            model_configs={
                "xgboost": {"n_estimators": 5},
                "random_forest": {"n_estimators": 5, "random_state": 0},
            },
        )
        assert "xgboost" in model.pipelines
        assert "random_forest" in model.pipelines


class TestPredictMethods:
    def test_predict_returns_array(self, trained_model, clinical_X):
        """predict() retorna ndarray."""
        preds = trained_model.predict(clinical_X.iloc[:50], "xgboost")
        assert isinstance(preds, np.ndarray)

    def test_predict_correct_length(self, trained_model, clinical_X):
        """predict() retorna longitud correcta."""
        preds = trained_model.predict(clinical_X.iloc[:50], "xgboost")
        assert len(preds) == 50

    def test_predict_binary_values(self, trained_model, clinical_X):
        """predict() retorna solo 0 y 1."""
        preds = trained_model.predict(clinical_X, "xgboost")
        assert set(preds).issubset({0, 1})

    def test_predict_untrained_raises(self, clinical_X):
        """predict() lanza ValueError para modelo no entrenado."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(numeric_features=["age"])
        with pytest.raises(ValueError, match="not trained"):
            model.predict(clinical_X[["age"]], "xgboost")

    def test_predict_proba_shape(self, trained_model, clinical_X):
        """predict_proba() retorna (n, 2)."""
        proba = trained_model.predict_proba(clinical_X.iloc[:50], "xgboost")
        assert proba.shape == (50, 2)

    def test_predict_proba_sums_to_one(self, trained_model, clinical_X):
        """Las probabilidades suman 1."""
        proba = trained_model.predict_proba(clinical_X, "xgboost")
        np.testing.assert_allclose(
            proba.sum(axis=1), np.ones(len(clinical_X)), atol=1e-5
        )

    def test_predict_risk_range(self, trained_model, clinical_X):
        """predict_risk() retorna valores en [0, 1]."""
        risk = trained_model.predict_risk(clinical_X, "xgboost")
        assert risk.min() >= 0.0
        assert risk.max() <= 1.0

    def test_predict_risk_shape(self, trained_model, clinical_X):
        """predict_risk() retorna (n,)."""
        risk = trained_model.predict_risk(clinical_X, "xgboost")
        assert risk.shape == (len(clinical_X),)

    def test_predict_profile_float(self, trained_model, clinical_X):
        """predict_profile() retorna un float."""
        profile = clinical_X.iloc[0].to_dict()
        result = trained_model.predict_profile(profile, "xgboost")
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_predict_proba_untrained_raises(self, clinical_X):
        """predict_proba() lanza ValueError."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(numeric_features=["age"])
        with pytest.raises(ValueError):
            model.predict_proba(clinical_X[["age"]], "xgboost")


class TestEnsemblePrediction:
    def test_ensemble_mean(self, trained_model, clinical_X):
        """predict_ensemble() con method=mean retorna (n,)."""
        result = trained_model.predict_ensemble(clinical_X, method="mean")
        assert result.shape == (len(clinical_X),)

    def test_ensemble_median(self, trained_model, clinical_X):
        """predict_ensemble() con method=median."""
        result = trained_model.predict_ensemble(clinical_X, method="median")
        assert result.shape == (len(clinical_X),)

    def test_ensemble_max(self, trained_model, clinical_X):
        """predict_ensemble() con method=max."""
        result = trained_model.predict_ensemble(clinical_X, method="max")
        assert result.shape == (len(clinical_X),)

    def test_ensemble_invalid_method(self, trained_model, clinical_X):
        """predict_ensemble() con method inválido lanza ValueError."""
        with pytest.raises(ValueError, match="Unknown ensemble method"):
            trained_model.predict_ensemble(clinical_X, method="vote")

    def test_ensemble_range(self, trained_model, clinical_X):
        """ensemble retorna valores en [0, 1]."""
        result = trained_model.predict_ensemble(clinical_X, method="mean")
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_ensemble_specific_models(self, trained_model, clinical_X):
        """ensemble con modelos específicos."""
        result = trained_model.predict_ensemble(clinical_X, model_types=["xgboost"])
        assert result.shape == (len(clinical_X),)


class TestEvaluateMethod:
    def test_evaluate_returns_dict(self, trained_model, clinical_X, clinical_y_string):
        """evaluate() retorna diccionario."""
        metrics = trained_model.evaluate(
            clinical_X.iloc[:100], clinical_y_string.iloc[:100], "xgboost"
        )
        assert isinstance(metrics, dict)

    def test_evaluate_keys_present(self, trained_model, clinical_X, clinical_y_string):
        """evaluate() contiene las métricas esperadas."""
        metrics = trained_model.evaluate(
            clinical_X.iloc[:100], clinical_y_string.iloc[:100], "xgboost"
        )
        expected_keys = {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "sensitivity",
            "specificity",
            "confusion_matrix",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    def test_evaluate_accuracy_range(
        self, trained_model, clinical_X, clinical_y_string
    ):
        """accuracy está en [0, 1]."""
        metrics = trained_model.evaluate(
            clinical_X.iloc[:100], clinical_y_string.iloc[:100], "xgboost"
        )
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_evaluate_stores_metrics(
        self, trained_model, clinical_X, clinical_y_string
    ):
        """evaluate() almacena métricas en self.metrics."""
        trained_model.evaluate(
            clinical_X.iloc[:100], clinical_y_string.iloc[:100], "xgboost"
        )
        assert "xgboost" in trained_model.metrics

    def test_evaluate_multiple(self, trained_model, clinical_X, clinical_y_string):
        """evaluate_multiple() retorna métricas para todos los modelos."""
        results = trained_model.evaluate_multiple(
            clinical_X.iloc[:100], clinical_y_string.iloc[:100]
        )
        assert "xgboost" in results
        assert "random_forest" in results

    def test_confusion_matrix_in_metrics(
        self, trained_model, clinical_X, clinical_y_string
    ):
        """confusion_matrix está en las métricas."""
        metrics = trained_model.evaluate(
            clinical_X.iloc[:100], clinical_y_string.iloc[:100], "xgboost"
        )
        assert "confusion_matrix" in metrics
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_sensitivity_specificity_range(
        self, trained_model, clinical_X, clinical_y_string
    ):
        """sensitivity y specificity en [0, 1]."""
        metrics = trained_model.evaluate(
            clinical_X.iloc[:100], clinical_y_string.iloc[:100], "xgboost"
        )
        assert 0.0 <= metrics["sensitivity"] <= 1.0
        assert 0.0 <= metrics["specificity"] <= 1.0


class TestGetFeatureImportance:
    def test_returns_dataframe(self, trained_model):
        """get_feature_importance() retorna DataFrame."""
        df = trained_model.get_feature_importance("xgboost")
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_columns(self, trained_model):
        """DataFrame tiene columnas 'feature' e 'importance'."""
        df = trained_model.get_feature_importance("xgboost")
        assert "feature" in df.columns
        assert "importance" in df.columns

    def test_sorted_descending(self, trained_model):
        """DataFrame ordenado por importancia descendente."""
        df = trained_model.get_feature_importance("xgboost")
        if len(df) > 1:
            assert df["importance"].iloc[0] >= df["importance"].iloc[-1]

    def test_untrained_model_returns_empty(self):
        """Modelo no entrenado retorna DataFrame vacío."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(numeric_features=["age"])
        df = model.get_feature_importance("xgboost")
        assert df.empty

    def test_random_forest_importance(self, trained_model):
        """Funciona para random_forest también."""
        df = trained_model.get_feature_importance("random_forest")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_get_coef_returns_empty(self, trained_model):
        """get_coef() retorna DataFrame vacío (no aplicable)."""
        df = trained_model.get_coef("xgboost")
        assert df.empty


class TestCrossValidation:
    def test_cv_returns_dict(self, clinical_X, clinical_y_string):
        """cross_validate_model() retorna diccionario."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(numeric_features=["age", "BMI"])
        X = clinical_X[["age", "BMI"]]
        result = model.cross_validate_model(
            X,
            clinical_y_string,
            model_type="xgboost",
            config={"n_estimators": 5},
            cv_folds=3,
        )
        assert isinstance(result, dict)

    def test_cv_contains_metrics(self, clinical_X, clinical_y_string):
        """CV contiene las métricas esperadas."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(numeric_features=["age", "BMI"])
        X = clinical_X[["age", "BMI"]]
        result = model.cross_validate_model(
            X,
            clinical_y_string,
            model_type="xgboost",
            config={"n_estimators": 5},
            cv_folds=3,
        )
        for metric in ["accuracy_mean", "f1_mean", "roc_auc_mean"]:
            assert metric in result

    def test_cv_stores_in_cv_metrics(self, clinical_X, clinical_y_string):
        """CV almacena resultados en cv_metrics."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(numeric_features=["age", "BMI"])
        X = clinical_X[["age", "BMI"]]
        model.cross_validate_model(
            X,
            clinical_y_string,
            model_type="xgboost",
            config={"n_estimators": 5},
            cv_folds=3,
        )
        assert "xgboost" in model.cv_metrics

    def test_cv_invalid_model_type_raises(self, clinical_X, clinical_y_string):
        """CV con model_type inválido lanza ValueError."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        model = ReverseLogicTabularModel(numeric_features=["age"])
        with pytest.raises(ValueError):
            model.cross_validate_model(
                clinical_X[["age"]], clinical_y_string, model_type="unsupported_model"
            )


class TestSaveLoad:
    def test_save_creates_file(self, trained_model, tmp_dir):
        """save() crea el archivo pickle."""
        path = tmp_dir / "model.pkl"
        trained_model.save(path)
        assert path.exists()

    def test_save_file_nonempty(self, trained_model, tmp_dir):
        """Archivo guardado no vacío."""
        path = tmp_dir / "model.pkl"
        trained_model.save(path)
        assert path.stat().st_size > 0

    def test_load_returns_model(self, trained_model, tmp_dir):
        """load() retorna instancia de ReverseLogicTabularModel."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        path = tmp_dir / "model.pkl"
        trained_model.save(path)
        loaded = ReverseLogicTabularModel.load(path)
        assert isinstance(loaded, ReverseLogicTabularModel)

    def test_load_preserves_target(self, trained_model, tmp_dir):
        """Modelo cargado preserva el target."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        path = tmp_dir / "model.pkl"
        trained_model.save(path)
        loaded = ReverseLogicTabularModel.load(path)
        assert loaded.target == trained_model.target

    def test_load_preserves_pipelines(self, trained_model, tmp_dir):
        """Modelo cargado tiene los mismos pipelines."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        path = tmp_dir / "model.pkl"
        trained_model.save(path)
        loaded = ReverseLogicTabularModel.load(path)
        assert set(loaded.pipelines.keys()) == set(trained_model.pipelines.keys())

    def test_load_and_predict(self, trained_model, clinical_X, tmp_dir):
        """Modelo cargado puede predecir."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        path = tmp_dir / "model.pkl"
        trained_model.save(path)
        loaded = ReverseLogicTabularModel.load(path)
        preds = loaded.predict(clinical_X.iloc[:10], "xgboost")
        assert len(preds) == 10

    def test_save_creates_parent_dirs(self, trained_model, tmp_dir):
        """save() crea directorios padre si no existen."""
        path = tmp_dir / "nested" / "deep" / "model.pkl"
        trained_model.save(path)
        assert path.exists()

    def test_save_returns_path(self, trained_model, tmp_dir):
        """save() retorna el Path del archivo."""
        path = tmp_dir / "model.pkl"
        result = trained_model.save(path)
        assert isinstance(result, Path)
        assert result == path


# ═════════════════════════════════════════════════════════════════════════════
# TESTS: ReverseLogicTrainer
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_csv(tmp_path):
    """CSV de muestra con datos de smoking/drinking."""
    rng = np.random.default_rng(42)
    n = 1000

    df = pd.DataFrame(
        {
            "sex": rng.choice([0, 1], size=n),
            "age": rng.integers(20, 80, size=n).astype(float),
            "height": rng.integers(155, 185, size=n).astype(float),
            "weight": rng.integers(50, 100, size=n).astype(float),
            "waistline": rng.uniform(65, 110, size=n),
            "SBP": rng.uniform(100, 160, size=n),
            "DBP": rng.uniform(60, 100, size=n),
            "BLDS": rng.uniform(70, 140, size=n),
            "tot_chole": rng.uniform(150, 250, size=n),
            "HDL_chole": rng.uniform(30, 90, size=n),
            "LDL_chole": rng.uniform(60, 200, size=n),
            "triglyceride": rng.uniform(60, 300, size=n),
            "hemoglobin": rng.uniform(10, 18, size=n),
            "urine_protein": rng.uniform(0, 3, size=n),
            "serum_creatinine": rng.uniform(0.5, 2.0, size=n),
            "SGOT_AST": rng.uniform(10, 80, size=n),
            "SGOT_ALT": rng.uniform(5, 100, size=n),
            "gamma_GTP": rng.uniform(5, 150, size=n),
            "BMI": rng.uniform(18, 35, size=n),
            "AST_ALT_ratio": rng.uniform(0.5, 3.0, size=n),
            "waist_height_ratio": rng.uniform(0.4, 0.7, size=n),
            "hemoglobin_per_height": rng.uniform(0.05, 0.12, size=n),
            "gamma_GTP_log": rng.uniform(1.6, 5.0, size=n),
            "liver_index": rng.uniform(10, 80, size=n),
            "age_sex_interaction": rng.uniform(20, 80, size=n),
            "bmi_category": rng.integers(0, 4, size=n).astype(float),
            "SMK_stat_type_cd": rng.choice([1.0, 2.0, 3.0], size=n),
            "DRK_YN": rng.choice(["Y", "N"], size=n),
        }
    )
    path = tmp_path / "smoking_drinking_cleaned.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def trainer(sample_csv, tmp_path):
    """Instancia de ReverseLogicTrainer con CSV de muestra."""
    from src.training.train_reverse_logic_tabular import ReverseLogicTrainer

    return ReverseLogicTrainer(
        csv_path=sample_csv,
        model_output_dir=tmp_path / "models",
        analysis_output_dir=tmp_path / "analysis",
    )


class TestReverseLogicTrainerInit:
    def test_creates_model_output_dir(self, sample_csv, tmp_path):
        """Crea el directorio de modelos."""
        from src.training.train_reverse_logic_tabular import ReverseLogicTrainer

        out = tmp_path / "my_models"
        ReverseLogicTrainer(csv_path=sample_csv, model_output_dir=out)
        assert out.exists()

    def test_creates_analysis_output_dir(self, sample_csv, tmp_path):
        """Crea el directorio de análisis."""
        from src.training.train_reverse_logic_tabular import ReverseLogicTrainer

        out = tmp_path / "my_analysis"
        ReverseLogicTrainer(csv_path=sample_csv, analysis_output_dir=out)
        assert out.exists()

    def test_random_seed_stored(self, trainer):
        """Random seed se almacena."""
        from src.config.constants import REVERSE_ANALYSIS_RANDOM_SEED

        assert trainer.random_seed == REVERSE_ANALYSIS_RANDOM_SEED


class TestNormalizeBinaryTarget:
    def test_smk_numeric_one_to_no(self, trainer):
        """SMK_stat_type_cd == 1.0 → No."""
        series = pd.Series([1.0, 2.0, 3.0, 1.0])
        result = trainer._normalize_binary_target(series, "SMK_stat_type_cd")
        assert result.iloc[0] == "No"
        assert result.iloc[3] == "No"

    def test_smk_numeric_two_to_yes(self, trainer):
        """SMK_stat_type_cd == 2.0 → Yes."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = trainer._normalize_binary_target(series, "SMK_stat_type_cd")
        assert result.iloc[1] == "Yes"

    def test_smk_numeric_three_to_yes(self, trainer):
        """SMK_stat_type_cd == 3.0 → Yes."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = trainer._normalize_binary_target(series, "SMK_stat_type_cd")
        assert result.iloc[2] == "Yes"

    def test_drk_yn_y_to_yes(self, trainer):
        """DRK_YN: Y → Yes."""
        series = pd.Series(["Y", "N", "y", "n"])
        result = trainer._normalize_binary_target(series, "DRK_YN")
        assert result.iloc[0] == "Yes"
        assert result.iloc[2] == "Yes"

    def test_drk_yn_n_to_no(self, trainer):
        """DRK_YN: N → No."""
        series = pd.Series(["Y", "N"])
        result = trainer._normalize_binary_target(series, "DRK_YN")
        assert result.iloc[1] == "No"

    def test_drk_yn_already_yes(self, trainer):
        """DRK_YN: YES → Yes."""
        series = pd.Series(["YES", "NO"])
        result = trainer._normalize_binary_target(series, "DRK_YN")
        assert result.iloc[0] == "Yes"
        assert result.iloc[1] == "No"

    def test_unknown_target_unchanged(self, trainer):
        """Target desconocido no se modifica."""
        series = pd.Series(["A", "B", "C"])
        result = trainer._normalize_binary_target(series, "UNKNOWN_TARGET")
        pd.testing.assert_series_equal(result, series)


class TestLoadAndPrepareData:
    def test_returns_tuple_of_seven(self, trainer):
        """load_and_prepare_data() retorna 7 elementos."""
        result = trainer.load_and_prepare_data("SMK_stat_type_cd")
        assert len(result) == 7

    def test_dataframe_first_element(self, trainer):
        """Primer elemento es DataFrame."""
        df, *_ = trainer.load_and_prepare_data("SMK_stat_type_cd")
        assert isinstance(df, pd.DataFrame)

    def test_splits_are_dataframes(self, trainer):
        """X_train, X_val, X_test son DataFrames."""
        _, X_train, X_val, X_test, *_ = trainer.load_and_prepare_data(
            "SMK_stat_type_cd"
        )
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_val, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)

    def test_splits_are_series(self, trainer):
        """y_train, y_val, y_test son Series."""
        *_, y_train, y_val, y_test = trainer.load_and_prepare_data("SMK_stat_type_cd")
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_val, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_no_nan_in_splits(self, trainer):
        """No hay NaN en los splits."""
        _, X_train, X_val, X_test, y_train, y_val, y_test = (
            trainer.load_and_prepare_data("SMK_stat_type_cd")
        )
        assert not X_train.isnull().any().any()
        assert not y_train.isnull().any()

    def test_alcohol_target_loads(self, trainer):
        """Funciona con target DRK_YN."""
        result = trainer.load_and_prepare_data("DRK_YN")
        assert len(result) == 7

    def test_target_binary_after_normalize(self, trainer):
        """Target contiene solo Yes/No después de normalizar."""
        *_, y_train, _, _ = trainer.load_and_prepare_data("SMK_stat_type_cd")
        assert set(y_train.unique()).issubset({"Yes", "No"})


class TestTrainModels:
    def test_returns_model_instance(self, trainer):
        """train_models() retorna ReverseLogicTabularModel."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        _, X_train, _, _, y_train, _, _ = trainer.load_and_prepare_data(
            "SMK_stat_type_cd"
        )
        model = trainer.train_models(X_train, y_train, "SMK_stat_type_cd", ["xgboost"])
        assert isinstance(model, ReverseLogicTabularModel)

    def test_trained_pipelines_exist(self, trainer):
        """Modelo entrenado tiene pipelines."""
        _, X_train, _, _, y_train, _, _ = trainer.load_and_prepare_data(
            "SMK_stat_type_cd"
        )
        model = trainer.train_models(X_train, y_train, "SMK_stat_type_cd", ["xgboost"])
        assert "xgboost" in model.pipelines

    def test_multiple_model_types(self, trainer):
        """Entrena múltiples tipos de modelo."""
        _, X_train, _, _, y_train, _, _ = trainer.load_and_prepare_data(
            "SMK_stat_type_cd"
        )
        model = trainer.train_models(
            X_train, y_train, "SMK_stat_type_cd", ["xgboost", "random_forest"]
        )
        assert "xgboost" in model.pipelines
        assert "random_forest" in model.pipelines


class TestEvaluateModels:
    def test_returns_dict(self, trainer):
        """evaluate_models() retorna diccionario."""
        _, X_train, X_val, X_test, y_train, y_val, y_test = (
            trainer.load_and_prepare_data("SMK_stat_type_cd")
        )
        model = trainer.train_models(X_train, y_train, "SMK_stat_type_cd", ["xgboost"])
        result = trainer.evaluate_models(
            model, X_val, X_test, y_val, y_test, "SMK_stat_type_cd"
        )
        assert isinstance(result, dict)

    def test_contains_model_type(self, trainer):
        """Resultado contiene el tipo de modelo entrenado."""
        _, X_train, X_val, X_test, y_train, y_val, y_test = (
            trainer.load_and_prepare_data("SMK_stat_type_cd")
        )
        model = trainer.train_models(X_train, y_train, "SMK_stat_type_cd", ["xgboost"])
        result = trainer.evaluate_models(
            model, X_val, X_test, y_val, y_test, "SMK_stat_type_cd"
        )
        assert "xgboost" in result

    def test_metrics_have_accuracy(self, trainer):
        """Métricas incluyen accuracy."""
        _, X_train, X_val, X_test, y_train, y_val, y_test = (
            trainer.load_and_prepare_data("SMK_stat_type_cd")
        )
        model = trainer.train_models(X_train, y_train, "SMK_stat_type_cd", ["xgboost"])
        result = trainer.evaluate_models(
            model, X_val, X_test, y_val, y_test, "SMK_stat_type_cd"
        )
        assert "accuracy" in result["xgboost"]


class TestSaveModel:
    def test_save_creates_file(self, trainer):
        """save_model() crea el archivo."""
        _, X_train, _, _, y_train, _, _ = trainer.load_and_prepare_data(
            "SMK_stat_type_cd"
        )
        model = trainer.train_models(X_train, y_train, "SMK_stat_type_cd", ["xgboost"])
        path = trainer.save_model(model, "SMK_stat_type_cd")
        assert path.exists()

    def test_save_returns_path(self, trainer):
        """save_model() retorna Path."""
        _, X_train, _, _, y_train, _, _ = trainer.load_and_prepare_data(
            "SMK_stat_type_cd"
        )
        model = trainer.train_models(X_train, y_train, "SMK_stat_type_cd", ["xgboost"])
        path = trainer.save_model(model, "SMK_stat_type_cd")
        assert isinstance(path, Path)

    def test_saved_file_loadable(self, trainer):
        """Archivo guardado se puede cargar."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        _, X_train, _, _, y_train, _, _ = trainer.load_and_prepare_data(
            "SMK_stat_type_cd"
        )
        model = trainer.train_models(X_train, y_train, "SMK_stat_type_cd", ["xgboost"])
        path = trainer.save_model(model, "SMK_stat_type_cd")
        loaded = ReverseLogicTabularModel.load(path)
        assert isinstance(loaded, ReverseLogicTabularModel)


class TestTrainSingleTarget:
    def test_smoke_runs_without_error(self, trainer):
        """train_single_target() se ejecuta sin excepciones."""
        trainer.train_single_target(target="SMK_stat_type_cd", model_types=["xgboost"])

    def test_creates_model_file(self, trainer):
        """train_single_target() crea el archivo de modelo."""
        trainer.train_single_target(target="SMK_stat_type_cd", model_types=["xgboost"])
        model_files = list(trainer.model_output_dir.glob("*.pkl"))
        assert len(model_files) > 0

    def test_alcohol_target(self, trainer):
        """train_single_target() funciona con DRK_YN."""
        trainer.train_single_target(target="DRK_YN", model_types=["xgboost"])
        model_files = list(trainer.model_output_dir.glob("*.pkl"))
        assert len(model_files) > 0


class TestTrainBothTargets:
    def test_trains_both(self, trainer):
        """train_both_targets() crea dos archivos de modelo."""
        trainer.train_both_targets(model_types=["xgboost"])
        model_files = list(trainer.model_output_dir.glob("*.pkl"))
        assert len(model_files) >= 2


# ═════════════════════════════════════════════════════════════════════════════
# TESTS DE INTEGRACIÓN
# ═════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    def test_full_pipeline_smoke_xgboost(self, sample_csv, tmp_path):
        """Pipeline completo para Smoking con XGBoost."""
        from src.training.train_reverse_logic_tabular import ReverseLogicTrainer

        trainer = ReverseLogicTrainer(
            csv_path=sample_csv,
            model_output_dir=tmp_path / "models",
            analysis_output_dir=tmp_path / "analysis",
        )
        trainer.train_single_target(target="SMK_stat_type_cd", model_types=["xgboost"])

        model_files = list((tmp_path / "models").glob("*.pkl"))
        assert len(model_files) == 1

    def test_full_pipeline_alcohol_random_forest(self, sample_csv, tmp_path):
        """Pipeline completo para Alcohol con Random Forest."""
        from src.training.train_reverse_logic_tabular import ReverseLogicTrainer

        trainer = ReverseLogicTrainer(
            csv_path=sample_csv,
            model_output_dir=tmp_path / "models",
            analysis_output_dir=tmp_path / "analysis",
        )
        trainer.train_single_target(target="DRK_YN", model_types=["random_forest"])

        model_files = list((tmp_path / "models").glob("*.pkl"))
        assert len(model_files) == 1

    def test_save_and_load_predictions_match(self, trained_model, clinical_X, tmp_path):
        """Predicciones antes y después de guardar/cargar son iguales."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        path = tmp_path / "test_model.pkl"
        trained_model.save(path)
        loaded = ReverseLogicTabularModel.load(path)

        original_preds = trained_model.predict(clinical_X.iloc[:20], "xgboost")
        loaded_preds = loaded.predict(clinical_X.iloc[:20], "xgboost")

        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_analyzer_with_trained_model(
        self, analyzer, trained_model, clinical_X, clinical_y_string
    ):
        """ReverseLogicAnalyzer funciona con modelo entrenado real."""
        y_binary = (clinical_y_string == "Yes").astype(int)
        proba_preds = {
            mt: trained_model.predict_risk(clinical_X, mt)
            for mt in trained_model.pipelines
        }
        binary_preds = {
            mt: trained_model.predict(clinical_X, mt) for mt in trained_model.pipelines
        }

        analyzer.plot_roc_curve(y_binary, proba_preds, filename="integration_roc.png")
        analyzer.plot_confusion_matrices(
            y_binary, binary_preds, filename="integration_cm.png"
        )

        assert (analyzer.output_dir / "integration_roc.png").exists()
        assert (analyzer.output_dir / "integration_cm.png").exists()

    def test_pipeline_deterministic(self, clinical_X, clinical_y_string):
        """Mismo seed produce mismas predicciones."""
        from src.models.reverse_logic_tabular_model import ReverseLogicTabularModel

        def build_and_predict():
            model = ReverseLogicTabularModel(
                numeric_features=["age", "BMI"],
                random_seed=42,
            )
            model.fit(
                clinical_X[["age", "BMI"]],
                clinical_y_string,
                model_type="xgboost",
                config={"n_estimators": 10},
            )
            return model.predict_risk(clinical_X[["age", "BMI"]].iloc[:20], "xgboost")

        preds1 = build_and_predict()
        preds2 = build_and_predict()
        np.testing.assert_allclose(preds1, preds2, rtol=1e-5)
