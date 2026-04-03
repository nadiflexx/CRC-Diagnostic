"""
Modelo de Predicción de Cáncer Colorrectal — Red Neuronal Multicapa
====================================================================
Módulo que define la arquitectura y utilidades del modelo MLP para
predicción de cáncer colorrectal basado en características tabulares.

Uso:
    from src.models.generic_tabular_model import GenericTabularMLPClassifier
    
    model = GenericTabularMLPClassifier(seed=42)
    model.fit(X_train, y_train, X_val, y_val)
    predictions = model.predict(X_test)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.neural_network import MLPClassifier


class GenericTabularMLPClassifier:
    """
    Clasificador MLP para predicción de cáncer colorrectal.
    
    Arquitectura:
        Input (N features) → 256 → 128 → 64 → Output (1)
    
    Parámetros:
        seed (int): Semilla para reproducibilidad
        hidden_layers (tuple): Tamaños de capas ocultas
        alpha (float): Factor de regularización L2
        learning_rate_init (float): Tasa de aprendizaje inicial
        max_iter (int): Número máximo de iteraciones
    """
    
    def __init__(self, seed=42, hidden_layers=(256, 128, 64), 
                 alpha=1e-4, learning_rate_init=1e-3, max_iter=200):
        """Inicializa el clasificador con parámetros configurables."""
        self.seed = seed
        self.hidden_layers = hidden_layers
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.model = None
        self.is_fitted = False
        self.n_features_ = None
    
    def _create_model(self, validation_fraction):
        """Crea la instancia del MLPClassifier."""
        return MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            activation='relu',
            solver='adam',
            alpha=self.alpha,
            learning_rate='adaptive',
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=validation_fraction,
            n_iter_no_change=20,
            random_state=self.seed,
            verbose=False,
        )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, validation_fraction=0.125):
        """
        Entrena el modelo.
        
        Los datos de train y validación se combinan internamente, y se usa
        validation_fraction para reservar una porción para early stopping.
        """
        self.model = self._create_model(validation_fraction)
        
        # Combinar train y validación si se proporcionan
        if X_val is not None and y_val is not None:
            X_full = np.vstack([X_train, X_val])
            y_full = np.concatenate([y_train, y_val])
        else:
            X_full = X_train
            y_full = y_train
        
        self.model.fit(X_full, y_full)
        self.n_features_ = X_full.shape[1]
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """Predice las etiquetas del conjunto de entrada."""
        if not self.is_fitted:
            raise ValueError("El modelo debe entrenarse primero con .fit()")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Retorna las probabilidades predichas."""
        if not self.is_fitted:
            raise ValueError("El modelo debe entrenarse primero con .fit()")
        return self.model.predict_proba(X)
    
    def get_model_info(self):
        """Retorna información del modelo entrenado."""
        if not self.is_fitted:
            raise ValueError("El modelo no ha sido entrenado aún")
        
        return {
            'n_features': self.n_features_,
            'hidden_layers': self.hidden_layers,
            'n_iter': self.model.n_iter_,
            'loss': self.model.loss_,
            'alpha': self.alpha,
            'seed': self.seed,
        }


# Función auxiliar para crear un modelo estándar
def create_default_model(seed=42):
    """Crea un modelo con los parámetros por defecto."""
    return GenericTabularMLPClassifier(seed=seed)


# ============================================================================
# CÓDIGO ANTIGUO REMOVIDO
#
# Todo el código de entrenamiento, evaluación y gráficas ha sido movido a:
#    src/training/train_generic_tabular.py
#
# Para entrenar el modelo, ejecuta:
#    python src/training/train_generic_tabular.py
# ============================================================================
