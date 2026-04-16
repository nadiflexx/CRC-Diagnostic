"""
Evaluation utilities for the XGBoost colon tabular analysis workflow.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

from src.config.constants import TABULAR_ANALYSIS_IMAGE_CONFIG
from src.config.paths import paths


class GenericTabularDataAnalyzer:
    """Generates analysis plots focused on country, smoking and alcohol."""

    def __init__(self, output_dir=None):
        self.output_dir = output_dir or paths.ANALYSIS_TABULAR_REPORT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def run_full_analysis(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        full_df: pd.DataFrame,
        feature_importance: pd.DataFrame,
    ):
        """Creates and saves all analysis artifacts."""
        y_prob = model.predict_risk(X_test)
        self._plot_roc_curve(y_test, y_prob)
        self._plot_feature_importance(feature_importance)
        self._plot_country_advanced_rate(full_df)
        self._plot_country_smoking_heatmap(full_df)
        self._plot_country_alcohol_heatmap(full_df)
        self._plot_age_smoking_distribution(full_df)
        self._export_country_summary(full_df)

    def _plot_roc_curve(self, y_test, y_prob):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color="#067a5f", linewidth=2.5, label="XGBoost")
        plt.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Advanced CRC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / TABULAR_ANALYSIS_IMAGE_CONFIG["roc_curve"]["filename"], dpi=150)
        plt.close()

    def _plot_feature_importance(self, feature_importance: pd.DataFrame):
        if feature_importance.empty:
            return
        top_features = feature_importance.head(15).sort_values("importance", ascending=True)
        plt.figure(figsize=(9, 6))
        sns.barplot(data=top_features, x="importance", y="feature", color="#067a5f")
        plt.title("Top XGBoost Features")
        plt.xlabel("Importance")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(self.output_dir / TABULAR_ANALYSIS_IMAGE_CONFIG["feature_importance"]["filename"], dpi=150)
        plt.close()

    def _plot_country_advanced_rate(self, df: pd.DataFrame):
        summary = (
            df.groupby("Country")
            .agg(advanced_rate=("Advanced_Cancer", "mean"), patients=("Advanced_Cancer", "size"))
            .reset_index()
            .sort_values("advanced_rate", ascending=False)
            .head(12)
        )
        plt.figure(figsize=(10, 6))
        sns.barplot(data=summary, x="advanced_rate", y="Country", palette="crest")
        plt.title("Observed Advanced Cancer Rate by Country")
        plt.xlabel("Advanced cancer rate")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(self.output_dir / TABULAR_ANALYSIS_IMAGE_CONFIG["country_advanced_rate"]["filename"], dpi=150)
        plt.close()

    def _plot_country_smoking_heatmap(self, df: pd.DataFrame):
        summary = (
            df.groupby(["Country", "Smoking_History"])["Advanced_Cancer"]
            .mean()
            .reset_index()
            .pivot(index="Country", columns="Smoking_History", values="Advanced_Cancer")
            .sort_index()
        )
        plt.figure(figsize=(8, 8))
        sns.heatmap(summary, cmap="YlGnBu", annot=False)
        plt.title("Advanced Cancer Rate by Country and Smoking")
        plt.tight_layout()
        plt.savefig(self.output_dir / TABULAR_ANALYSIS_IMAGE_CONFIG["country_smoking_heatmap"]["filename"], dpi=150)
        plt.close()

    def _plot_country_alcohol_heatmap(self, df: pd.DataFrame):
        summary = (
            df.groupby(["Country", "Alcohol_Consumption"])["Advanced_Cancer"]
            .mean()
            .reset_index()
            .pivot(index="Country", columns="Alcohol_Consumption", values="Advanced_Cancer")
            .sort_index()
        )
        plt.figure(figsize=(8, 8))
        sns.heatmap(summary, cmap="OrRd", annot=False)
        plt.title("Advanced Cancer Rate by Country and Alcohol")
        plt.tight_layout()
        plt.savefig(self.output_dir / TABULAR_ANALYSIS_IMAGE_CONFIG["country_alcohol_heatmap"]["filename"], dpi=150)
        plt.close()

    def _plot_age_smoking_distribution(self, df: pd.DataFrame):
        plot_df = df.copy()
        plot_df["Stage_Label"] = plot_df["Advanced_Cancer"].map({0: "Localized", 1: "Advanced"})
        plt.figure(figsize=(9, 6))
        sns.boxplot(
            data=plot_df,
            x="Smoking_History",
            y="Age",
            hue="Stage_Label",
            palette=["#9bd3c7", "#ef6c57"],
        )
        plt.title("Age Distribution by Smoking and Cancer Stage")
        plt.tight_layout()
        plt.savefig(self.output_dir / TABULAR_ANALYSIS_IMAGE_CONFIG["age_smoking_distribution"]["filename"], dpi=150)
        plt.close()

    def _export_country_summary(self, df: pd.DataFrame):
        summary = (
            df.groupby("Country")
            .agg(
                patients=("Advanced_Cancer", "size"),
                advanced_rate=("Advanced_Cancer", "mean"),
                smoker_share=("Smoking_History", lambda values: (values == "Yes").mean()),
                alcohol_share=("Alcohol_Consumption", lambda values: (values == "Yes").mean()),
                average_age=("Age", "mean"),
            )
            .reset_index()
            .sort_values(["advanced_rate", "patients"], ascending=[False, False])
        )
        summary.to_csv(paths.ANALYSIS_TABULAR_COUNTRY_SUMMARY_PATH, index=False)
