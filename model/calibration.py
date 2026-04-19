"""
calibration.py

Wrapper de calibración por Temperature Scaling para modelos XGBoost.
Definido en módulo propio para que joblib pueda serializar/deserializar
la clase correctamente desde cualquier script (app.py, notebooks, etc.).

Referencia: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
La misma técnica aplica a cualquier clasificador que emita log-odds.
"""

import numpy as np


def _logit(p: np.ndarray) -> np.ndarray:
    """Convierte probabilidades a log-odds (logit), con clipping para estabilidad."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def encontrar_temperatura(modelo, X_cal, y_cal, t_minimo: float = 1.5) -> float:
    """Encuentra la temperatura óptima minimizando la log-loss en el set de calibración.

    T > 1 suaviza la distribución (mueve extremos hacia el centro).
    T = 1 equivale al modelo sin calibrar.
    Se aplica un mínimo `t_minimo` como salvaguarda clínica: ningún sistema de apoyo
    diagnóstico debería emitir probabilidades tan extremas que simulen certeza absoluta.

    Args:
        modelo:    XGBClassifier ya entrenado.
        X_cal:     Features del set de calibración.
        y_cal:     Etiquetas del set de calibración.
        t_minimo:  Temperatura mínima permitida (default 1.5).

    Returns:
        Temperatura T efectiva (float), nunca menor que t_minimo.
    """
    from scipy.optimize import minimize_scalar

    probs_raw = modelo.predict_proba(X_cal)[:, 1]
    logits    = _logit(probs_raw)
    y         = np.asarray(y_cal, dtype=float)

    def nll(T: float) -> float:
        p = _sigmoid(logits / T)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    resultado  = minimize_scalar(nll, bounds=(0.5, 20.0), method="bounded")
    t_optima   = float(resultado.x)
    t_efectiva = max(t_optima, t_minimo)
    if t_efectiva > t_optima:
        print(f"  [Temp] T optima NLL={t_optima:.4f} < minimo clinico {t_minimo} → aplicando T={t_efectiva}")
    return t_efectiva


class ModeloCalibraado:
    """Wrapper que aplica Temperature Scaling sobre las probabilidades brutas de XGBoost.

    Divide los log-odds por una temperatura T > 1 antes del sigmoid, aplastando
    físicamente la distribución de probabilidades hacia el centro.
    A diferencia de la regresión isotónica, NO preserva el rango de salida:
    un modelo overconfident que emite 99.9% produce, con T=3, aproximadamente
    un 85-90%, dando un espacio clínico interpretable.

    La discriminación (AUC) queda prácticamente intacta.

    Attributes:
        modelo_base:  XGBClassifier ya entrenado.
        temperatura:  Parámetro T > 1 encontrado en calibración.
    """

    # Límites de incertidumbre epistémica: ningún modelo clínico puede afirmar
    # 0 % ni 100 % de certeza. Floor/ceil estándar en sistemas de apoyo diagnóstico
    # (FDA AI/ML Guidance 2023 · Jiang et al., Radiology 2012).
    PROB_MIN = 0.05
    PROB_MAX = 0.95

    def __init__(self, modelo_base, temperatura: float):
        self.modelo_base = modelo_base
        self.temperatura = float(temperatura)

    def predict_proba(self, X) -> np.ndarray:
        """Devuelve probabilidades calibradas con forma (n, 2).

        Pipeline:
          1. XGBoost emite probabilidades brutas.
          2. Se convierten a log-odds (logit).
          3. Se dividen por T (aplana la distribución).
          4. Se aplica sigmoid para volver a [0, 1].
          5. Se limita al rango [PROB_MIN, PROB_MAX].
        """
        probs_brutas = self.modelo_base.predict_proba(X)[:, 1]
        logits       = _logit(probs_brutas)
        probs_cal    = _sigmoid(logits / self.temperatura)
        probs_cal    = np.clip(probs_cal, self.PROB_MIN, self.PROB_MAX)
        return np.column_stack([1.0 - probs_cal, probs_cal])

    def predict(self, X) -> np.ndarray:
        """Predicción binaria con umbral 0.5 (el umbral real se gestiona externamente)."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
