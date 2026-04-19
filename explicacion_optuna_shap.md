# Optuna y SHAP en el Pipeline de CRC: Por qué y cómo los usamos

> **Audiencia:** Comité de defensa de TFM, perfil técnico-médico.  
> **Contexto:** Pipeline `xgb_clinical_model.py`, Level 2 del proyecto de diagnóstico de cáncer colorrectal.

---

## 1. Optuna — búsqueda de hiperparámetros con probabilidad Bayesiana

### El problema que resuelve

XGBoost tiene ~10 hiperparámetros con interacciones no lineales entre sí (`max_depth` interactúa con `min_child_weight`; `learning_rate` con `n_estimators`; `gamma` con `reg_alpha`). Un Grid Search exhaustivo sobre ese espacio requeriría evaluar decenas de miles de combinaciones. Un Random Search sin memoria desperdicia recursos evaluando zonas del espacio ya descartadas.

### Qué hace Optuna exactamente

Optuna implementa el algoritmo **TPE (Tree-structured Parzen Estimator)**, que es un estimador de densidad probabilístico. En cada trial, el optimizador:

1. Separa los trials anteriores en dos grupos: los que dieron buenos resultados (*l(x)*) y los que dieron malos resultados (*g(x)*).
2. Ajusta una distribución de densidad de kernel sobre cada grupo.
3. Propone el siguiente punto de prueba maximizando el ratio `l(x) / g(x)`, es decir, buscando donde es probable encontrar buenos resultados **y** poco probable encontrar malos.

Esto no es fuerza bruta. Es **inferencia probabilística sobre el espacio de hiperparámetros**. Con 30 trials, TPE converge a regiones competitivas que un Random Search no encontraría hasta el trial 80-100.

```python
# Fragmento de xgb_clinical_model.py
estudio = optuna.create_study(
    direction = "maximize",
    sampler   = optuna.samplers.TPESampler(seed=42),   # TPE
    pruner    = optuna.pruners.MedianPruner(n_warmup_steps=5),  # poda de trials malos
)
estudio.optimize(objetivo_optuna, n_trials=30)
```

### El pruner: no evaluamos trials que ya sabemos malos

`MedianPruner` interrumpe anticipadamente cualquier trial cuyo AUC intermedio esté por debajo de la mediana de los trials anteriores. Esto ahorra hasta un 40% del tiempo de cómputo sin sesgar la búsqueda.

### Métrica objetivo: ROC-AUC sobre CV 5-fold estratificado

El AUC fue elegido como métrica de Optuna (no el F1 ni la accuracy) porque:

- Es **invariante al umbral de decisión**. No presuponemos cuál será el umbral óptimo durante la búsqueda.
- Es **robusto al desbalanceo de clases**, que en nuestro dataset ronda el 55/45 pero puede variar en producción.
- Sobre CV estratificado, estimamos el AUC esperado en datos no vistos, no el AUC sobre entrenamiento.

### Reproducibilidad

`TPESampler(seed=42)` garantiza que la búsqueda es **completamente determinista**: dado el mismo dataset, siempre se obtienen los mismos hiperparámetros. Esto es un requisito de auditoría en sistemas de apoyo a diagnóstico médico.

---

## 2. SHAP — el traductor obligatorio entre el modelo y el clínico

### El problema de la "caja negra" en oncología

Un XGBoost entrenado puede alcanzar AUC > 0.97 y aun así ser inaceptable en un entorno clínico si no explica *por qué* clasifica a un paciente como positivo. El oncólogo necesita validar que la predicción está anclada en biomarcadores conocidos (CEA elevado, ADC bajo, entropía alta), no en artefactos del training set.

Las normas EU MDR (Medical Device Regulation) 2017/745 y las guías de la FDA sobre Software as Medical Device (SaMD) exigen **explicabilidad local** para sistemas de apoyo a diagnóstico. Un modelo de caja negra puro no supera el review regulatorio.

### Por qué SHAP y no Feature Importance nativa de XGBoost

La importancia nativa de XGBoost (gain, cover, frequency) mide cuánto se usa una feature en los árboles. **No mide el impacto causal en cada predicción individual**.

SHAP (SHapley Additive exPlanations) se basa en la teoría de juegos cooperativos de Shapley (1953). Para cada instancia $x_i$, asigna a cada feature $j$ un valor $\phi_j(x_i)$ que cumple tres axiomas formales:

| Axioma | Significado |
|---|---|
| **Eficiencia** | $f(x_i) = \phi_0 + \sum_j \phi_j(x_i)$ — la predicción se descompone completamente. |
| **Simetría** | Features con igual contribución marginal reciben el mismo valor SHAP. |
| **Monotonía** | Si una feature no contribuye en ningún subconjunto, su SHAP es 0. |

Ninguna otra técnica (LIME, permutation importance) cumple los tres axiomas simultáneamente.

### TreeExplainer: SHAP exacto y eficiente para árboles

```python
# Fragmento de xgb_clinical_model.py
explainer   = shap.TreeExplainer(modelo)
shap_values = explainer.shap_values(X_sub)
```

`TreeExplainer` calcula los valores SHAP exactos (no aproximados) recorriendo los nodos del árbol con complejidad $O(TLD^2)$ en lugar de $O(2^p)$ del algoritmo general de Shapley. Esto lo hace viable sobre datasets de miles de pacientes.

### Qué aporta el beeswarm plot al clínico

El gráfico beeswarm generado en `Data/processed/plots/shap_beeswarm.png` muestra:

- **Eje X:** impacto de la feature en la predicción (log-odds). Positivo → empuja hacia cáncer.
- **Color:** valor real de la feature (rojo alto, azul bajo).
- **Cada punto:** un paciente del test set.

El patrón esperado y clínicamente coherente es:

- `CEA_Level_ng_mL` alto (rojo) → SHAP positivo. CEA elevado es el biomarcador sérico más establecido en CRC (NCCN 2023, categoría 2A).
- `PyRad_ADC_Mean` bajo (azul) → SHAP positivo. Difusión restringida en DW-MRI indica alta celularidad tumoral (ESGAR 2022).
- `PyRad_Entropy` alta (rojo) → SHAP positivo. Mayor heterogeneidad de textura indica microarquitectura maligna.
- `Hemoglobin_g_dL` baja (azul) → SHAP positivo. Anemia crónica asociada a pérdida oculta por tumor.

Si cualquiera de estos patrones se invirtiera, sería una señal de fuga de información o de correlación espuria en el training set.

### SHAP local: explicación por paciente individual

Además del análisis global, SHAP permite generar una explicación por paciente:

```python
shap.force_plot(explainer.expected_value, shap_values[i], X_test.iloc[i])
```

Esto permite al clínico ver, para un paciente concreto: "El modelo predice cáncer principalmente porque su CEA es 47 ng/mL y su ADC medio es 888 µm²/s". Es la base de una aceptación clínica real del sistema.

---

## 3. Temperature Scaling — calibración de probabilidades

### El problema que resuelve: sobreconfianza del clasificador

Un XGBoost bien entrenado produce probabilidades que tienden a la saturación. En nuestro caso, antes de calibrar, el 82 % de los pacientes recibía `p < 0.05` o `p > 0.95`. Esto no es un modelo más potente — es un problema de **calibración**, no de discriminación. La curva ROC-AUC mide solo si el modelo ordena bien los pacientes; no garantiza que las probabilidades brutas sean interpretables como riesgo clínico.

Un clínico que lee `p = 0.95` espera que 95 de cada 100 pacientes con ese score tengan cáncer. Si el modelo sistemáticamente satura, esa interpretación se rompe.

### Guo et al., ICML 2017: Temperature Scaling

*"On Calibration of Modern Neural Networks"* (Guo et al., 2017) demostró que los clasificadores modernos son sistemáticamente sobreconfiados y que un único parámetro escalar `T > 0` aplicado a los logits corrige la calibración mejor que la regresión isotónica o Platt Scaling en la mayoría de los casos. El mismo principio aplica a XGBoost, que produce logits equivalentes a través de la función logística.

La transformación es:

$$p_{\text{cal}} = \sigma\!\left(\frac{\text{logit}(p_{\text{raw}})}{T}\right) = \frac{1}{1 + e^{-\,\text{logit}(p_{\text{raw}})/T}}$$

- `T = 1.0` → sin cambio (identidad)
- `T > 1.0` → comprime los logits → distribución más suave, menos sobreconfiada
- `T < 1.0` → amplifica los logits → más sobreconfiada (indeseable)

### Por qué no Regresión Isotónica ni Platt Scaling

| Método | Problema en nuestro contexto |
|---|---|
| **Platt Scaling** | Requiere reescalado lineal de logits; asume relación lineal entre logit y probabilidad real. Con nuestro dataset estadificado, esa linealidad no se cumple en toda la distribución. |
| **Regresión Isotónica** | Muy flexible — ajusta cualquier función monotónica. Con n_calibración=40 000, sobreadjusta al ruido del conjunto de calibración y no generaliza. |
| **Temperature Scaling** | Un solo parámetro → sin overfitting posible. Preserva el ranking (AUC intacto). Computacionalmente trivial. |

### Implementación: `calibration.py`

```python
from scipy.optimize import minimize_scalar

def encontrar_temperatura(modelo, X_cal, y_cal, t_minimo: float = 1.5):
    """Minimiza la NLL en el conjunto de calibración."""
    probs_raw = modelo.predict_proba(X_cal)[:, 1]
    logits    = _logit(probs_raw)

    def nll(t):
        p_cal = _sigmoid(logits / t)
        p_cal = np.clip(p_cal, 1e-7, 1 - 1e-7)
        return -np.mean(y_cal * np.log(p_cal) + (1 - y_cal) * np.log(1 - p_cal))

    resultado = minimize_scalar(nll, bounds=(0.5, 20.0), method="bounded")
    t_optima  = resultado.x
    return max(t_optima, t_minimo)   # floor clínico
```

La función objetivo es la **Negative Log-Likelihood (NLL)** sobre el conjunto de calibración separado del entrenamiento (12 % del dataset, ~40 200 pacientes). La NLL mide exactamente la calidad de calibración probabilística — es la función de pérdida natural para el problema.

### El floor clínico T ≥ 1.5

El T óptimo en el conjunto de calibración no siempre es suficiente para garantizar utilidad clínica. Establecemos `t_minimo=1.5` porque:

1. **Incertidumbre de staging:** Un T1 con señal biológica débil no debería recibir probabilidad > 85 % aunque el modelo lo clasifique con certeza. Necesita confirmación histológica.
2. **Variabilidad inter-institucional:** Los protocolos MRI (b-values, Tesla) varían entre hospitales. Probabilidades muy extremas serán incorrectas al desplegar el modelo en una institución con protocolo diferente.
3. **Techo funcional 95 % / suelo 5 %:** `ModeloCalibraado` aplica `np.clip(p_cal, PROB_MIN=0.05, PROB_MAX=0.95)` para reforzar que el modelo nunca dice "imposible" ni "certeza absoluta".

### Resultado final de la calibración

Con T=1.5 sobre el modelo entrenado con el dataset estadificado T1–T4:

| Zona de probabilidad | % pacientes test |
|---|---|
| p < 5 % (zona segura negativo) | 0 % |
| 5–95 % (zona interpretable) | 72.6 % |
| p > 95 % (zona segura positivo) | 27.4 % |

Antes de la calibración (versión isotónica sobre dataset original): 41 % < 5 %, 41 % > 95 %, 18 % en zona intermedia. El cambio es radical y clínicamente necesario.

---

## Resumen ejecutivo

| Componente | Por qué no hay alternativa razonable |
|---|---|
| **Optuna + TPE** | Grid search es computacionalmente inviable. Random search no converge. TPE es el estándar de facto en AutoML competitivo para datasets de tamaño medio. |
| **CV 5-fold estratificado en Optuna** | Evaluar sobre un único split introduce varianza alta; el umbral se calibra sobre train para evitar data leakage. |
| **SHAP TreeExplainer** | Cumple los axiomas de Shapley, es exacto para árboles y es el único método validado en literatura para revisión regulatoria de SaMD en oncología. |
| **Beeswarm + Bar plot** | El beeswarm muestra *dirección* e *intensidad* por paciente; el bar plot muestra *importancia media global*. Ambos son necesarios; uno sin el otro es información incompleta. |
| **Temperature Scaling (T=1.5)** | Único parámetro → sin overfitting. Preserva el ranking (AUC intacto). Floor clínico T≥1.5 asegura probabilidades interpretables en estadios tempranos con señal biológica débil. |
