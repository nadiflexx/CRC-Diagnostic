"""
Shared utility for consistent index calculation across training and frontend.
Ensures Risk_Score, Prevention_Index, and Access_Score are calculated the same way.
"""


def calculate_risk_indices(data_dict):
    """
    Calcula índices de riesgo para un paciente (raw, antes de normalizar).
    
    NOTA: Screening_History y Early_Detection fueron removidos por data leakage.
    
    Retorna:
    - risk_score: 0-150 (suma de factores de riesgo)
    - prevention_index: 0-100 (suma de factores protectores, simplificado)
    - access_score: 50-70 (basado en ubicación)
    """
    # Risk_Score: basado en factores de riesgo (sin Obesity_BMI)
    risk_score = 0.0
    if data_dict.get('Family_History') == 'Yes':
        risk_score += 20
    if data_dict.get('Smoking_History') == 'Yes':
        risk_score += 15
    if data_dict.get('Alcohol_Consumption') == 'Yes':
        risk_score += 10
    if data_dict.get('Diabetes') == 'Yes':
        risk_score += 15
    if data_dict.get('Inflammatory_Bowel_Disease') == 'Yes':
        risk_score += 25
    if data_dict.get('Genetic_Mutation') == 'Yes':
        risk_score += 30
    if data_dict.get('Diet_Risk') == 'Yes':
        risk_score += 10
    
    # Prevention_Index: basado en factores protectores (simplificado sin Screening_History/Early_Detection)
    # NOTA: Sin Screening_History ni Early_Detection, solo se basa en Physical_Activity
    # Escalamos a 0-100 para mantener consistencia con escala del dataset
    prevention_index = 0.0
    if data_dict.get('Physical_Activity') == 'Yes':
        prevention_index += 100
    prevention_index = min(prevention_index, 100)
    
    # Access_Score: basado en acceso a recursos
    access_score = 70.0 if data_dict.get('Urban_or_Rural') == 'Urban' else 50.0
    
    return risk_score, prevention_index, access_score


def normalize_risk_indices(risk_score, prevention_index, access_score):
    """
    Normaliza los índices crudos a [0, 1] para uso en el modelo.
    
    Entradas:
    - risk_score: 0-150 (raw)
    - prevention_index: 0-100 (raw)
    - access_score: 50-70 (raw)
    
    Salidas:
    - risk_score_norm: 0-1
    - prevention_index_norm: 0-1
    - access_score_norm: 0-1
    """
    risk_score_norm = risk_score / 150.0
    prevention_index_norm = prevention_index / 100.0
    access_score_norm = (access_score - 50.0) / 20.0  # maps [50, 70] -> [0, 1]
    
    return risk_score_norm, prevention_index_norm, access_score_norm


def denormalize_risk_indices(risk_score_norm, prevention_index_norm, access_score_norm):
    """
    Desnormaliza los índices normalizados [0, 1] de vuelta a sus valores crudos.
    Inversa de normalize_risk_indices.
    """
    risk_score = risk_score_norm * 150.0
    prevention_index = prevention_index_norm * 100.0
    access_score = (access_score_norm * 20.0) + 50.0
    
    return risk_score, prevention_index, access_score
