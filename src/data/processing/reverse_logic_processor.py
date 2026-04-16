"""
Reverse Logic Data Processor & Analyzer
Genera gráficos y análisis exploratorio (EDA) del dataset tabular procesado
antes de la fase de entrenamiento.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.config.paths import paths

class ReverseLogicDataAnalyzer:
    """
    Analizador del dataset procesado para la predicción de hábitos (Smoking/Alcohol).
    Genera visualizaciones clave sobre la distribución y calidad de los datos.
    """

    def __init__(self, output_dir: Path = paths.REVERSE_LOGIC_ANALYSIS_DIR):
        """
        Inicializa el analizador.
        """
        
        if output_dir is None:
            output_dir = paths.REVERSE_LOGIC_ANALYSIS_DIR
            self.output_dir = output_dir
        else:
            self.output_dir = output_dir
        
        # Configuración visual global
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["font.size"] = 10

    def generate_dataset_report(self, df: pd.DataFrame):
        """
        Ejecuta todo el pipeline de visualización sobre el DataFrame proporcionado.
        """
        print(f"📊 Generando gráficos de análisis del dataset en: {self.output_dir}")
        
        self.plot_target_distributions(df)
        self.plot_correlation_matrix(df)
        self.plot_demographics(df)
        
        print("✅ Análisis del dataset completado.")

    def plot_target_distributions(self, df: pd.DataFrame):
        """Grafica el balance de las variables objetivo (Smoking y Alcohol)."""
        targets = [col for col in ["Smoking_History", "Alcohol_Consumption"] if col in df.columns]
        
        if not targets:
            return

        fig, axes = plt.subplots(1, len(targets), figsize=(12, 5))
        if len(targets) == 1:
            axes = [axes]

        for ax, target in zip(axes, targets):
            sns.countplot(data=df, x=target, ax=ax, hue=target, legend=False)
            ax.set_title(f'Distribución de {target}')
            ax.set_ylabel('Cantidad de Pacientes')
            
            # Añadir porcentajes sobre las barras
            total = len(df)
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height/total:.1%}', 
                            xy=(p.get_x() + p.get_width() / 2, height),
                            xytext=(0, 3),  # 3 puntos de offset vertical
                            textcoords="offset points",
                            ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / "target_distributions.png", dpi=300)
        plt.close()

    def plot_correlation_matrix(self, df: pd.DataFrame):
        """Grafica un mapa de calor con las correlaciones de las variables numéricas."""
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        
        if numeric_df.empty:
            return

        plt.figure(figsize=(14, 10))
        corr = numeric_df.corr()
        
        # Máscara para la mitad superior del triángulo (opcional, para mayor limpieza)
        import numpy as np
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0, 
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        
        plt.title('Matriz de Correlación de Variables Clínicas')
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_matrix.png", dpi=300)
        plt.close()

    def plot_demographics(self, df: pd.DataFrame):
        """Grafica la relación entre Edad, Género (si existe) y los hábitos."""
        if "Age" not in df.columns:
            return

        targets = [col for col in ["Smoking_History", "Alcohol_Consumption"] if col in df.columns]
        
        for target in targets:
            plt.figure(figsize=(10, 6))
            
            # Si existe la columna de género (Gender o sex), la usamos para separar
            hue_col = "Gender" if "Gender" in df.columns else "sex" if "sex" in df.columns else None
            
            sns.boxplot(data=df, x=target, y="Age", hue=hue_col)
            plt.title(f'Distribución de Edad por {target}')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"age_vs_{target.lower()}.png", dpi=300)
            plt.close()