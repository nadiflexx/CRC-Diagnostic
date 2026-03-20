"""
xgb_clinical_model.py

Entrena un clasificador XGBoost para detectar cáncer colorrectal a partir
del dataset clínico sintético generado con clinical_data_generator.py.

Flujo completo:
  1. cargar_datos()            -- Lee el CSV y separa X / y
  2. buscar_hiperparametros()  -- Optuna con 30 trials y CV 5-fold
  3. entrenar_modelo_final()   -- XGBoost con los mejores parámetros
  4. encontrar_umbral_optimo() -- Busca el umbral que garantiza recall >= objetivo
  5. evaluar_modelo()          -- Métricas + panel gráfico 2x2
  6. grafico_shap()            -- Beeswarm + barplot de importancias

"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")           # renderizado sin pantalla (servidor / CI)
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier


optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)


def cargar_datos(csv_path: str):
    """
    Lee el CSV, quita las columnas no predictivas y devuelve X, y y los nombres de features.

    Eliminamos Patient_ID y Diagnosis de X para no hacer trampas (data leakage).
    Nos quedamos solo con columnas numéricas por si hubiera alguna categórica inesperada.
    """
    df = pd.read_csv(csv_path)

    # Quitamos el identificador de paciente y la variable objetivo
    cols_excluir = ["Patient_ID", "Diagnosis"]
    X = df.drop(columns=[c for c in cols_excluir if c in df.columns])
    y = df["Diagnosis"]

    # nos aseguramos de quedarnos solo con numéricas 
    X = X.select_dtypes(include=[np.number])

    print(f"  Datos cargados: {X.shape[0]} muestras x {X.shape[1]} features")
    print(f"  Distribucion de clases -- 0: {(y==0).sum()}  1: {(y==1).sum()}")
    return X, y, list(X.columns)


def objetivo_optuna(trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Define el espacio de búsqueda de hiperparámetros y evalúa cada combinación
    con validación cruzada de 5 folds. Devuelve el ROC-AUC medio (a maximizar).

    Optuna usa TPE (Tree-structured Parzen Estimator): aprende qué regiones
    del espacio dan mejores resultados y concentra los trials donde hay más
    probabilidad de mejora.
    """
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
        "max_depth":         trial.suggest_int("max_depth", 3, 9),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        "subsample":         trial.suggest_float("subsample", 0.60, 1.00),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.60, 1.00),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        # scale_pos_weight compensa el desbalanceo: si hay 2x más sanos que cánceres, vale 2.0
        "scale_pos_weight":  (y == 0).sum() / max((y == 1).sum(), 1),
        "eval_metric":       "auc",
        "use_label_encoder": False,
        "random_state":      42,
        "n_jobs":            -1,
    }

    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs  = []
    for fold_train, fold_val in cv.split(X, y):
        Xtr, Xval = X.iloc[fold_train], X.iloc[fold_val]
        ytr, yval = y.iloc[fold_train], y.iloc[fold_val]
        modelo = XGBClassifier(**params)
        modelo.fit(Xtr, ytr)
        prob = modelo.predict_proba(Xval)[:, 1]
        aucs.append(roc_auc_score(yval, prob))

    return float(np.mean(aucs))


def buscar_hiperparametros(X_train: pd.DataFrame, y_train: pd.Series,
                           n_trials: int = 30) -> dict:
    """
    Lanza la búsqueda de hiperparámetros con Optuna.

    Cada trial está guiado por los resultados anteriores, por eso converge más rápido que un grid search clásico.

    Devuelve un diccionario con los mejores hiperparámetros encontrados.
    """
    estudio = optuna.create_study(
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=42),
        pruner     = optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    estudio.optimize(
        lambda trial: objetivo_optuna(trial, X_train, y_train),
        n_trials   = n_trials,
        show_progress_bar = True,
    )

    print(f"  Mejor ROC-AUC CV: {estudio.best_value:.4f}")
    print(f"  Mejores params: {estudio.best_params}")
    return estudio.best_params


def entrenar_modelo_final(X_train: pd.DataFrame, y_train: pd.Series,
                          mejores_params: dict) -> XGBClassifier:
    """
    Entrena el modelo XGBoost definitivo sobre todo el conjunto de entrenamiento
    usando los hiperparámetros que encontró Optuna.

    Usamos todo el training set (sin CV) para darle al modelo el máximo de datos posible.
    """
    params_finales = {
        **mejores_params,
        "scale_pos_weight":  (y_train == 0).sum() / max((y_train == 1).sum(), 1),
        "eval_metric":       "auc",
        "use_label_encoder": False,
        "random_state":      42,
        "n_jobs":            -1,
    }
    modelo = XGBClassifier(**params_finales)
    modelo.fit(X_train, y_train)
    print("  Modelo final entrenado.")
    return modelo


def encontrar_umbral_optimo(modelo: XGBClassifier,
                            X_train: pd.DataFrame, y_train: pd.Series,
                            recall_objetivo: float = 1.0) -> float:
    """
    Busca el umbral de decisión más alto que garantiza recall >= recall_objetivo
    en el conjunto de entrenamiento.

    Calibramos sobre train, no sobre test, para evitar data leakage: si usamos
    el test para elegir el umbral estaríamos usando información que el modelo no
    debería conocer en producción.

    En contexto médico, un falso negativo (decir "sano" a un paciente con cáncer)
    puede suponer un retraso diagnóstico grave, por eso forzamos un recall alto.
    """
    probs_train = modelo.predict_proba(X_train)[:, 1]
    umbral_optimo = 0.50       # empezamos con el umbral por defecto

    for t in np.arange(0.01, 0.51, 0.01):
        preds_t = (probs_train >= t).astype(int)
        rec_t   = recall_score(y_train, preds_t, zero_division=0)
        if rec_t >= recall_objetivo:
            umbral_optimo = round(float(t), 2)   # nos quedamos con el mayor umbral que cumple

    print(f"  Umbral optimo encontrado: {umbral_optimo:.2f}  "
          f"(recall_objetivo={recall_objetivo:.2f})")
    return umbral_optimo


def evaluar_modelo(modelo: XGBClassifier,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   umbral: float, plots_dir: str) -> None:
    """
    Calcula las métricas de clasificación con el umbral elegido y genera
    un panel 2x2 con:
      - Matriz de confusión
      - Curva ROC
      - Curva Precisión-Recall
      - Distribución de probabilidades por clase

    El gráfico se guarda en plots_dir/evaluacion_modelo.png
    """
    probs = modelo.predict_proba(X_test)[:, 1]
    preds = (probs >= umbral).astype(int)

    auc  = roc_auc_score(y_test, probs)
    rec  = recall_score(y_test, preds, zero_division=0)
    f1   = f1_score(y_test, preds, zero_division=0)
    cm   = confusion_matrix(y_test, preds)

    print(f"\n  METRICAS (umbral={umbral:.2f}):")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Recall  : {rec:.4f}  (FN={cm[1,0]})")
    print(f"  F1-Score: {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["Sano", "Cancer"]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Evaluacion del Modelo XGBoost  (umbral={umbral:.2f})", fontsize=14)

    # Matriz de confusión
    ax = axes[0, 0]
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Matriz de Confusion")
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Sano", "Cancer"]); ax.set_yticklabels(["Sano", "Cancer"])
    fig.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, probs)
    ax = axes[0, 1]
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="navy", lw=1)
    ax.set_title("Curva ROC")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Curva Precisión-Recall
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, probs)
    ap = float(np.trapezoid(prec_arr[::-1], rec_arr[::-1]))
    ax = axes[1, 0]
    ax.plot(rec_arr, prec_arr, color="green", lw=2, label=f"AP = {ap:.3f}")
    ax.set_title("Curva Precision-Recall")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Distribución de probabilidades: queremos ver que los dos grupos estén bien separados
    ax = axes[1, 1]
    ax.hist(probs[y_test == 0], bins=30, alpha=0.6, color="steelblue",  label="Sano")
    ax.hist(probs[y_test == 1], bins=30, alpha=0.6, color="tomato",     label="Cancer")
    ax.axvline(umbral, color="black", ls="--", lw=2, label=f"Umbral={umbral:.2f}")
    ax.set_title("Distribucion de Probabilidades")
    ax.set_xlabel("P(Cancer)"); ax.set_ylabel("Frecuencia")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, "evaluacion_modelo.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Panel grafico guardado en: {out_path}")


def grafico_shap(modelo: XGBClassifier,
                 X_test: pd.DataFrame,
                 nombres_features: list,
                 plots_dir: str) -> None:
    """
    Genera dos gráficos SHAP para explicar qué features impulsan las predicciones:
      1. Beeswarm: cada punto es un paciente, el eje X muestra el impacto en log-odds.
      2. Bar plot: importancia media |SHAP| por feature.

    Usamos una submuestra de 3000 pacientes para que el cálculo sea razonablemente rápido.
    """
    # Submuestra para no mas velocidad (SHAP tarda con muchas muestras)
    n_shap  = min(3000, len(X_test))
    idx_sub = np.random.choice(len(X_test), size=n_shap, replace=False)
    X_sub   = X_test.iloc[idx_sub].copy()
    X_sub.columns = nombres_features

    explainer   = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_sub)
    # Para clasificación binaria, XGBoost devuelve shap_values de shape (n, f) directamente
    sv = shap_values if not isinstance(shap_values, list) else shap_values[1]

    os.makedirs(plots_dir, exist_ok=True)

    # Beeswarm: muestra la distribución de impactos por feature
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_sub, feature_names=nombres_features, show=False)
    plt.title("SHAP -- Impacto de cada feature (Beeswarm)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot: ranking de features por importancia media absoluta
    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, X_sub, feature_names=nombres_features,
                      plot_type="bar", show=False)
    plt.title("SHAP -- Importancia media por feature")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Graficos SHAP guardados en: {plots_dir}")



if __name__ == "__main__":
   
    
        directorio_actual = os.path.dirname(os.path.abspath(__file__))

        # Configuración del modelo (dataset clínico sintético)
        cfg = {
            "nombre": "clinical",
            "csv": os.path.join(directorio_actual, "..", "Data", "processed", "dataset_clinico_tumoral.csv"),
            "dir_artefactos": os.path.join(directorio_actual, "artifacts"),
            "dir_plots": os.path.join(directorio_actual, "..", "Data", "processed", "plots"),
            "recall_objetivo": 0.90
        }
        print(f"  EXPERIMENTO {cfg['nombre']}  --  {cfg['csv']}")
        
        # 1. Cargar datos
        X, y, nombres = cargar_datos(cfg["csv"])
        # 2. Split 80/20 estratificado para mantener la proporción de clases en ambos conjuntos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

        # 3. Búsqueda de hiperparámetros con Optuna
        print("\n[Optuna] Buscando hiperparametros...")
        mejores_params = buscar_hiperparametros(X_train, y_train, n_trials=30)

        # 4. Entrenamiento final con los mejores hiperparámetros encontrados
        print("\n[Entrenamiento] Ajustando modelo final...")
        modelo = entrenar_modelo_final(X_train, y_train, mejores_params)

        # 5. Umbral: buscamos el mayor umbral que garantiza recall >= recall_objetivo
        print("\n[Umbral] Buscando umbral optimo en train...")
        umbral = encontrar_umbral_optimo(modelo, X_train, y_train,
                                         recall_objetivo=cfg["recall_objetivo"])

        # 6. Evaluación final en test (datos que el modelo nunca ha visto)
        print("\n[Evaluacion] Metricas en test set:")
        evaluar_modelo(modelo, X_test, y_test, umbral, cfg["dir_plots"])

        # 7. Gráficos SHAP para interpretar qué variables explican las predicciones
        print("\n[SHAP] Generando graficos de explicabilidad...")
        grafico_shap(modelo, X_test, nombres, cfg["dir_plots"])

        # 8. Guardamos el modelo en JSON (formato nativo XGBoost, ligero y portátil)
        os.makedirs(cfg["dir_artefactos"], exist_ok=True)
        model_path = os.path.join(cfg["dir_artefactos"], "xgb_clinical_model.json")
        modelo.save_model(model_path)
        print(f"  Modelo guardado en: {model_path}")

        # 9. Paquete de inferencia completo en PKL para usar desde Streamlit
     
        paquete_inferencia = {
            "model": modelo,
            "threshold": umbral,
            "feature_names": nombres
        }
        pkl_path = os.path.join(cfg["dir_artefactos"], "xgb_inference_package.pkl")
        joblib.dump(paquete_inferencia, pkl_path)
        print(f"  Paquete PKL guardado en: {pkl_path}")

        print("\n Entrenamiento del modelo finalizado.")



       