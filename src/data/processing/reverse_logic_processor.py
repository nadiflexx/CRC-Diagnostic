"""
Reverse Logic Data Processor & Analyzer
Generates graphical analyses of the processed dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config.paths import paths


class ReverseLogicDataAnalyzer:
    """
    Analyzer of the processed dataset for predicting habits (Smoking/Alcohol).
    Generates key visualizations about the distribution and quality of the data.
    """

    def __init__(self, output_dir: Path | None = None):
        """
        Initializes the analyzer.
        """
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else paths.REVERSE_LOGIC_ANALYSIS_DIR
        )

        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["font.size"] = 10

    def generate_dataset_report(self, df: pd.DataFrame):
        """
        Executes the entire visualization pipeline on the provided DataFrame.
        Args:
            df (pd.DataFrame): The processed dataset to analyze.
        """
        print(f"📊 Generating analysis plots for the dataset in: {self.output_dir}")

        self.plot_target_distributions(df)
        self.plot_correlation_matrix(df)
        self.plot_demographics(df)

        print("✅ Analysis of the dataset completed.")

    def plot_target_distributions(self, df: pd.DataFrame):
        """Plots the distribution of the target variables (Smoking and Alcohol).
        Args:
            df (pd.DataFrame): The processed dataset.
        """
        targets = [
            col
            for col in ["Smoking_History", "Alcohol_Consumption"]
            if col in df.columns
        ]

        if not targets:
            return

        fig, axes = plt.subplots(1, len(targets), figsize=(12, 5))
        if len(targets) == 1:
            axes = [axes]

        for ax, target in zip(axes, targets, strict=True):
            sns.countplot(data=df, x=target, ax=ax, hue=target, legend=False)
            ax.set_title(f"Distribution of {target}")
            ax.set_ylabel("Number of Patients")

            total = len(df)
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(
                    f"{height / total:.1%}",
                    xy=(p.get_x() + p.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.savefig(self.output_dir / "target_distributions.png", dpi=300)
        plt.close()

    def plot_correlation_matrix(self, df: pd.DataFrame):
        """Plots a heatmap with the correlations of the numeric variables.
        Args:
            df (pd.DataFrame): The processed dataset.
        """
        numeric_df = df.select_dtypes(include=["int64", "float64"])

        if numeric_df.empty:
            return

        plt.figure(figsize=(14, 10))
        corr = numeric_df.corr()

        import numpy as np

        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr,
            mask=mask,
            annot=False,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )

        plt.title("Correlation Matrix of Clinical Variables")
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_matrix.png", dpi=300)
        plt.close()

    def plot_demographics(self, df: pd.DataFrame):
        """Plots the relationship between Age, Gender (if exists) and habits.
        Args:
            df (pd.DataFrame): The processed dataset.
        """
        if "Age" not in df.columns:
            return

        targets = [
            col
            for col in ["Smoking_History", "Alcohol_Consumption"]
            if col in df.columns
        ]

        for target in targets:
            plt.figure(figsize=(10, 6))

            hue_col = (
                "Gender"
                if "Gender" in df.columns
                else "sex"
                if "sex" in df.columns
                else None
            )

            sns.boxplot(data=df, x=target, y="Age", hue=hue_col)
            plt.title(f"Distribution of Age by {target}")

            plt.tight_layout()
            plt.savefig(self.output_dir / f"age_vs_{target.lower()}.png", dpi=300)
            plt.close()
