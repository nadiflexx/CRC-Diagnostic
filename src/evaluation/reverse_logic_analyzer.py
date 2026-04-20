"""
Reverse Logic Tabular Analysis: Evaluation and Visualization
Generates comprehensive analysis for smoking/alcohol prediction models (XGBoost and Random Forest).
Main focus: Confusion Matrices for detailed error analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional
import warnings

# Configure matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from src.config.paths import paths

# Suppress feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)


class ReverseLogicAnalyzer:
    """
    Generates analysis artifacts for reverse logic tabular models.
    Focus on XGBoost and Random Forest with detailed confusion matrix analysis.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize analyzer.
        
        Args:
            output_dir: Directory to save analysis plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="husl")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 11

    # ─────────────────────────────────────────────────────────────────────────────
    # ROC CURVES
    # ─────────────────────────────────────────────────────────────────────────────

    def plot_roc_curve(
        self,
        y_test: pd.Series,
        predictions: Dict[str, np.ndarray],
        target_name: str = "Smoking_History",
        filename: str = "roc_curves.png",
    ):
        """
        Plots ROC curves for XGBoost and Random Forest.
        
        Args:
            y_test: Binary test labels (0/1)
            predictions: Dict of {model_name: probability_array}
            target_name: "Smoking_History" or "Alcohol_Consumption"
            filename: Output filename
        """
        plt.figure(figsize=(10, 7))

        for model_name, y_prob in predictions.items():
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                linewidth=2.5,
                label=f"{model_name} (AUC = {roc_auc:.3f})",
                marker="o",
                markersize=4,
            )

        plt.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=2, label="Random")
        plt.xlabel("False Positive Rate", fontsize=12, fontweight="bold")
        plt.ylabel("True Positive Rate", fontsize=12, fontweight="bold")
        plt.title(f"ROC Curves - {target_name} Prediction", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    # ─────────────────────────────────────────────────────────────────────────────
    # CONFUSION MATRICES - MAIN FOCUS
    # ─────────────────────────────────────────────────────────────────────────────

    def plot_confusion_matrices(
        self,
        y_test: pd.Series,
        predictions: Dict[str, np.ndarray],
        target_name: str = "Smoking_History",
        filename: str = "confusion_matrices.png",
    ):
        """
        Plots detailed confusion matrices for XGBoost and Random Forest side-by-side.
        
        Args:
            y_test: Binary test labels (0/1)
            predictions: Dict of {model_name: binary_predictions}
            target_name: "Smoking_History" or "Alcohol_Consumption"
            filename: Output filename
        """
        n_models = len(predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(16, 6))
        
        if n_models == 1:
            axes = [axes]

        # Convert to binary if needed
        if y_test.dtype == "object":
            y_test_binary = (y_test == "Yes").astype(int)
        else:
            y_test_binary = y_test.values

        for idx, (model_name, y_pred) in enumerate(predictions.items()):
            cm = confusion_matrix(y_test_binary, y_pred, labels=[0, 1])
            
            # Normalize for percentage display
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_percent = np.divide(
                cm.astype(float) * 100,
                row_sums,
                out=np.zeros_like(cm, dtype=float),
                where=row_sums != 0,
            )
            
            # Create heatmap with both counts and percentages
            annotations = np.array([[f"{cm[i,j]}\n({cm_percent[i,j]:.1f}%)" for j in range(2)] for i in range(2)])
            
            sns.heatmap(
                cm,
                annot=annotations,
                fmt="",
                cmap="Blues",
                ax=axes[idx],
                cbar=True,
                xticklabels=["No", "Yes"],
                yticklabels=["No", "Yes"],
                cbar_kws={"label": "Count"}
            )
            
            axes[idx].set_title(f"{model_name}\nConfusion Matrix", fontweight="bold", fontsize=12)
            axes[idx].set_ylabel("True Label", fontweight="bold")
            axes[idx].set_xlabel("Predicted Label", fontweight="bold")
            
            # Add metrics below the matrix
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
            sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            
            metrics_text = f"Accuracy: {accuracy:.1f}%\nSensitivity: {sensitivity:.1f}%\nSpecificity: {specificity:.1f}%\nPrecision: {precision:.1f}%"
            axes[idx].text(0.5, -0.35, metrics_text, transform=axes[idx].transAxes,
                          ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f"Confusion Matrices - {target_name}", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix_detailed(
        self,
        y_test: pd.Series,
        predictions: Dict[str, np.ndarray],
        target_name: str = "Smoking_History",
        filename: str = "confusion_matrix_detailed.png",
    ):
        """
        Plots detailed confusion matrix metrics comparison.
        """
        if y_test.dtype == "object":
            y_test_binary = (y_test == "Yes").astype(int)
        else:
            y_test_binary = y_test.values

        metrics_data = []
        
        for model_name, y_pred in predictions.items():
            cm = confusion_matrix(y_test_binary, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
            sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            metrics_data.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "Sensitivity (TPR)": sensitivity,
                "Specificity (TNR)": specificity,
                "Precision (PPV)": precision,
                "F1 Score": f1,
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics_df["Model"]))
        width = 0.15
        
        metrics_to_plot = ["Accuracy", "Sensitivity (TPR)", "Specificity (TNR)", "Precision (PPV)", "F1 Score"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        
        for i, metric in enumerate(metrics_to_plot):
            ax.bar(x + i*width, metrics_df[metric], width, label=metric, color=colors[i])
        
        ax.set_xlabel("Model", fontweight="bold", fontsize=12)
        ax.set_ylabel("Score (%)", fontweight="bold", fontsize=12)
        ax.set_title(f"Model Performance Metrics - {target_name}", fontweight="bold", fontsize=14)
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(metrics_df["Model"])
        ax.legend(fontsize=10)
        ax.set_ylim([0, 105])
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    # ─────────────────────────────────────────────────────────────────────────────
    # FEATURE IMPORTANCE
    # ─────────────────────────────────────────────────────────────────────────────

    def plot_feature_importance(
        self,
        importance_dfs: Dict[str, pd.DataFrame],
        target_name: str = "Smoking_History",
        filename: str = "feature_importance.png",
        top_n: int = 15,
    ):
        """
        Plots feature importance comparison for XGBoost and Random Forest.
        
        Args:
            importance_dfs: Dict of {model_name: importance_dataframe}
            target_name: "Smoking_History" or "Alcohol_Consumption"
            filename: Output filename
            top_n: Number of top features to show
        """
        if not importance_dfs:
            return

        fig, axes = plt.subplots(1, len(importance_dfs), figsize=(16, 6))
        if len(importance_dfs) == 1:
            axes = [axes]

        for idx, (model_name, importance_df) in enumerate(importance_dfs.items()):
            if importance_df.empty:
                continue
                
            top_features = importance_df.head(top_n).sort_values("importance", ascending=True)
            
            axes[idx].barh(range(len(top_features)), top_features["importance"].values, color="#067a5f")
            axes[idx].set_yticks(range(len(top_features)))
            axes[idx].set_yticklabels(top_features["feature"].values, fontsize=10)
            axes[idx].set_xlabel("Importance", fontweight="bold")
            axes[idx].set_title(f"{model_name}\nTop {top_n} Features", fontweight="bold", fontsize=12)
            axes[idx].grid(axis="x", alpha=0.3)

        plt.suptitle(f"Feature Importance - {target_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    # ─────────────────────────────────────────────────────────────────────────────
    # COUNTRY PATTERNS
    # ─────────────────────────────────────────────────────────────────────────────

    def plot_country_habit_patterns(
        self,
        df: pd.DataFrame,
        habit_column: str = "Smoking_History",
        filename: str = "country_patterns.png",
    ):
        """
        Plots habit prevalence by country.
        """
        # Convert Yes/No to binary
        if df[habit_column].dtype == "object":
            habit_binary = (df[habit_column] == "Yes").astype(int)
        else:
            habit_binary = df[habit_column]

        summary = (
            pd.DataFrame({
                "Country": df["Country"],
                "Habit": habit_binary,
            })
            .groupby("Country")["Habit"]
            .agg(["mean", "size"])
            .reset_index()
            .rename(columns={"mean": "prevalence", "size": "patients"})
            .sort_values("prevalence", ascending=False)
            .head(15)
        )

        plt.figure(figsize=(11, 7))
        sns.barplot(data=summary, x="prevalence", y="Country", palette="crest")
        plt.xlabel("Prevalence", fontsize=12, fontweight="bold")
        plt.title(f"{habit_column} by Country", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    # ─────────────────────────────────────────────────────────────────────────────
    # PERFORMANCE HISTOGRAMS
    # ─────────────────────────────────────────────────────────────────────────────

    def plot_prediction_histograms(
        self,
        y_test: pd.Series,
        predictions: Dict[str, np.ndarray],
        target_name: str = "Smoking_History",
        filename_prefix: str = "prediction_histogram",
    ) -> None:
        """
        Generates one histogram per trained model using predicted probabilities.
        Each histogram compares the score distribution for true No vs true Yes cases.
        """
        if y_test.dtype == "object":
            y_test_binary = (y_test == "Yes").astype(int).values
        else:
            y_test_binary = y_test.values

        for model_name, y_prob in predictions.items():
            y_prob = np.asarray(y_prob)
            probs_no = y_prob[y_test_binary == 0]
            probs_yes = y_prob[y_test_binary == 1]

            fig = plt.figure(figsize=(12, 7))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.05)
            ax_top = fig.add_subplot(gs[0])
            ax_hist = fig.add_subplot(gs[1], sharex=ax_top)

            # Summary distribution band on top
            box = ax_top.boxplot(
                [probs_no, probs_yes],
                vert=False,
                widths=[0.5, 0.5],
                patch_artist=True,
                labels=["True No", "True Yes"],
                medianprops={"color": "#111827", "linewidth": 1.8},
                whiskerprops={"linewidth": 1.4},
                capprops={"linewidth": 1.4},
                flierprops={
                    "marker": "o",
                    "markersize": 2.5,
                    "markerfacecolor": "#6b7280",
                    "markeredgecolor": "#6b7280",
                    "alpha": 0.55,
                },
            )
            box_colors = ["#1f77b4", "#d62728"]
            for patch, color in zip(box["boxes"], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.45)
                patch.set_edgecolor(color)
                patch.set_linewidth(1.5)

            ax_top.set_ylabel("Real label", fontweight="bold")
            ax_top.set_title(
                f"{model_name} - Probability Distribution Summary ({target_name})",
                fontsize=14,
                fontweight="bold",
                pad=10,
            )
            ax_top.grid(axis="x", alpha=0.25)
            ax_top.grid(axis="y", alpha=0.0)
            ax_top.spines["top"].set_visible(False)
            ax_top.spines["right"].set_visible(False)
            ax_top.spines["left"].set_visible(False)
            ax_top.tick_params(axis="x", labelbottom=False)

            # Histogram section
            ax_hist.hist(
                probs_no,
                bins=25,
                alpha=0.45,
                label="True No",
                color="#94a3b8",
                edgecolor="white",
                linewidth=0.6,
            )
            ax_hist.hist(
                probs_yes,
                bins=25,
                alpha=0.65,
                label="True Yes",
                color="#3b82f6",
                edgecolor="white",
                linewidth=0.6,
            )

            # Reference lines and annotations for medians
            median_no = float(np.median(probs_no)) if len(probs_no) else 0.0
            median_yes = float(np.median(probs_yes)) if len(probs_yes) else 0.0
            ax_hist.axvline(median_no, color="#475569", linestyle="--", linewidth=1.5, alpha=0.9)
            ax_hist.axvline(median_yes, color="#1d4ed8", linestyle="--", linewidth=1.5, alpha=0.9)

            ymax = ax_hist.get_ylim()[1]
            ax_hist.text(
                median_no,
                ymax * 0.95,
                f"Median No: {median_no:.3f}",
                rotation=35,
                ha="left",
                va="top",
                fontsize=9,
                color="#475569",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "#cbd5e1"},
            )
            ax_hist.text(
                median_yes,
                ymax * 0.78,
                f"Median Yes: {median_yes:.3f}",
                rotation=35,
                ha="left",
                va="top",
                fontsize=9,
                color="#1d4ed8",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "#bfdbfe"},
            )

            separation = abs(median_yes - median_no)
            ax_hist.text(
                0.99,
                0.96,
                f"Median gap: {separation:.3f}\nNo samples: {len(probs_no):,}\nYes samples: {len(probs_yes):,}",
                transform=ax_hist.transAxes,
                ha="right",
                va="top",
                fontsize=9.5,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "alpha": 0.95, "edgecolor": "#cbd5e1"},
            )

            ax_hist.set_xlabel("Predicted probability", fontsize=12, fontweight="bold")
            ax_hist.set_ylabel("Patients", fontsize=12, fontweight="bold")
            ax_hist.legend(fontsize=11)
            ax_hist.grid(axis="y", alpha=0.3)
            ax_hist.spines["top"].set_visible(False)
            ax_hist.spines["right"].set_visible(False)
            fig.tight_layout()

            safe_model_name = model_name.lower().replace(" ", "_")
            output_path = self.output_dir / f"{filename_prefix}_{safe_model_name}_{target_name.lower()}.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────────
    # COMPREHENSIVE ANALYSIS RUNNER
    # ─────────────────────────────────────────────────────────────────────────────

    def run_full_analysis(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        full_df: pd.DataFrame,
        target_name: str = "Smoking_History",
        model_types: Optional[List[str]] = None,
    ):
        """
        Runs all analysis visualizations with focus on confusion matrices.
        """
        if model_types is None:
            model_types = list(model.pipelines.keys())

        # Convert y_test to binary if needed
        if isinstance(y_test.iloc[0], str):
            y_test_binary = (y_test == "Yes").astype(int)
        else:
            y_test_binary = y_test

        # Collect predictions from all models
        predictions_proba = {
            mt: model.predict_risk(X_test, mt) for mt in model_types
        }
        predictions_binary = {
            mt: model.predict(X_test, mt) for mt in model_types
        }

        # Collect feature importance
        importance_dfs = {}
        for mt in model_types:
            try:
                imp_df = model.get_feature_importance(mt)
                if not imp_df.empty:
                    importance_dfs[mt] = imp_df
            except (AttributeError, ValueError):
                pass

        # Generate plots
        self.plot_roc_curve(
            y_test_binary,
            predictions_proba,
            target_name=target_name,
            filename=f"roc_curves_{target_name.lower()}.png",
        )

        # MAIN FOCUS: Confusion Matrices
        self.plot_confusion_matrices(
            y_test_binary,
            predictions_binary,
            target_name=target_name,
            filename=f"confusion_matrices_{target_name.lower()}.png",
        )

        self.plot_confusion_matrix_detailed(
            y_test_binary,
            predictions_binary,
            target_name=target_name,
            filename=f"confusion_matrix_metrics_{target_name.lower()}.png",
        )

        if importance_dfs:
            self.plot_feature_importance(
                importance_dfs,
                target_name=target_name,
                filename=f"feature_importance_{target_name.lower()}.png",
            )

        if "Country" in full_df.columns:
            self.plot_country_habit_patterns(
                full_df,
                habit_column=target_name,
                filename=f"country_{target_name.lower()}.png",
            )
        else:
            print(f"Skipping country plot: 'Country' not in dataset.")
        
        self.plot_prediction_histograms(
            y_test,
            predictions_proba,
            target_name=target_name,
            filename_prefix="prediction_histogram",
        )
        
