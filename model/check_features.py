"""
check_features.py

Auditoría anti-data leakage sobre el dataset clínico-tumoral de CRC.

Detecta si alguna feature por sí sola separa perfectamente a los pacientes
sanos de los enfermos (rangos de valores sin ningún solapamiento). La presencia
de esa separación perfecta indicaría una fuga de información o un dataset
artificialmente trivial.

Uso:
    python model/check_features.py
"""

import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "..", "Data", "processed", "dataset_clinico_tumoral.csv")


def detectar_separacion_perfecta(df: pd.DataFrame) -> tuple[list, list]:
    """Identifica features que separan perfectamente las clases o son idénticas al target.

    Una feature está perfectamente separada cuando el rango de valores del
    grupo sano (Diagnosis=0) y el del grupo cáncer (Diagnosis=1) no se solapan.
    Eso indicaría data leakage o una variable derivada del propio diagnóstico.

    Args:
        df: DataFrame que contiene al menos la columna ``Diagnosis`` y las
            features predictivas.

    Returns:
        Tupla (matches_label, perfect_sep) donde:
            - matches_label: lista de nombres de features idénticas a ``Diagnosis``.
            - perfect_sep: lista de tuplas (feature, min_sano, max_sano,
              min_cancer, max_cancer) para cada feature sin solapamiento.
    """
    cols = [c for c in df.columns if c not in ("Patient_ID", "Diagnosis")]
    perfect_sep = []
    matches_label = []

    for c in cols:
        group_sano   = df[df["Diagnosis"] == 0][c]
        group_cancer = df[df["Diagnosis"] == 1][c]

        if df[c].equals(df["Diagnosis"]):
            matches_label.append(c)

        if group_sano.max() < group_cancer.min() or group_cancer.max() < group_sano.min():
            perfect_sep.append((
                c,
                float(group_sano.min()),
                float(group_sano.max()),
                float(group_cancer.min()),
                float(group_cancer.max()),
            ))

    return matches_label, perfect_sep


def imprimir_informe(
    matches_label: list,
    perfect_sep: list,
    n_features: int,
) -> None:
    """Imprime el informe de auditoría en stdout.

    Args:
        matches_label: Features idénticas al vector de etiquetas.
        perfect_sep: Features con separación perfecta entre clases.
        n_features: Número total de features evaluadas.

    Returns:
        None
    """
    print(f"Total de features evaluadas : {n_features}")
    print(f"Features idénticas al target: {matches_label}")
    print(f"Features perfectamente separadas (mostrando ≤10):")
    for item in perfect_sep[:10]:
        print(f"  {item}")
    print(f"Total perfectamente separadas: {len(perfect_sep)}")


if __name__ == "__main__":
    print(f"CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    cols_predictivas = [c for c in df.columns if c not in ("Patient_ID", "Diagnosis")]
    matches, separadas = detectar_separacion_perfecta(df)

    imprimir_informe(matches, separadas, n_features=len(cols_predictivas))

