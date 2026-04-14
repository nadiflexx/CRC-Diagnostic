"""
src/utils/export_onnx.py

Exporta modelos .pth y .pkl a ONNX / ONNX-ML.
Genera automáticamente un JSON de metadata junto a cada .onnx.

Uso:
  uv run src/utils/export_onnx.py --all
  uv run src/utils/export_onnx.py --pth models/saved/best_classifier.pth
  uv run src/utils/export_onnx.py --pth models/saved/best_tissue_classifier.pth
  uv run src/utils/export_onnx.py --pth models/saved/best_segmenter.pth
  uv run src/utils/export_onnx.py --pkl models/saved/tabular_model.pkl
"""

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import onnx
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as XGBFloat
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import torch
import torch.nn as nn

from src.config.paths import paths

ONNX_DIR = paths.MODELS / "onnx"
IMAGE_SIZE = 384
OPSET = 16

MODEL_REGISTRY: dict[str, type] = {}


# ═══════════════════════════════════════════════════════════════════════════════
#  REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


def _build_registry():
    """Registra todas las clases de modelos del proyecto."""
    try:
        from src.models.image_classifier import ColonCancerClassifier

        MODEL_REGISTRY["ColonCancerClassifier"] = ColonCancerClassifier
        MODEL_REGISTRY["classifier"] = ColonCancerClassifier
    except ImportError:
        pass

    try:
        from src.models.tissue_classifier import TissueOnlyClassifier

        MODEL_REGISTRY["TissueOnlyClassifier"] = TissueOnlyClassifier
        MODEL_REGISTRY["tissue_classifier"] = TissueOnlyClassifier
        MODEL_REGISTRY["tissue"] = TissueOnlyClassifier
    except ImportError:
        pass

    try:
        from src.models.polyp_segmenter import ColonPolypSegmenter

        MODEL_REGISTRY["ColonPolypSegmenter"] = ColonPolypSegmenter
        MODEL_REGISTRY["segmenter"] = ColonPolypSegmenter
        MODEL_REGISTRY["polyp"] = ColonPolypSegmenter
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  TEMPERATURE WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════


class _TemperatureWrapper(nn.Module):
    """
    Incorpora temperatura en el grafo ONNX.
    Solo se aplica a clasificadores (salida logits 2D).
    """

    def __init__(self, model: nn.Module, temperature: float):
        super().__init__()
        self.model = model
        self.register_buffer(
            "temperature",
            torch.tensor(temperature, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) / self.temperature


# ═══════════════════════════════════════════════════════════════════════════════
#  INSTANCIACIÓN ROBUSTA
# ═══════════════════════════════════════════════════════════════════════════════


def _instantiate(cls: type, model_name: str, num_classes: int) -> nn.Module | None:
    """
    Prueba firmas de constructor en orden hasta que una funcione.

      1. (model_name, pretrained=False, num_classes)  ← clasificadores timm
      2. (pretrained=None)                             ← segmentadores smp
      3. (pretrained=False)                            ← fallback
      4. ()                                            ← sin argumentos
    """
    attempts = [
        {
            "kwargs": {
                "model_name": model_name,
                "pretrained": False,
                "num_classes": num_classes,
            },
            "label": "model_name + pretrained=False + num_classes",
        },
        {
            # smp: pretrained=None → arquitectura sin pesos, sin descarga
            # pretrained=False falla porque smp busca pesos llamados "False"
            "kwargs": {"pretrained": None},
            "label": "pretrained=None  (smp / segmentadores)",
        },
        {
            "kwargs": {"pretrained": False},
            "label": "pretrained=False",
        },
        {
            "kwargs": {},
            "label": "sin argumentos",
        },
    ]

    for attempt in attempts:
        try:
            model = cls(**attempt["kwargs"])  # type: ignore
            print(f"  💡 Constructor: {attempt['label']}")
            return model
        except TypeError:
            continue
        except Exception as e:
            print(f"  ⚠️  '{attempt['label']}' falló: {e}")
            continue

    print(
        f"  ❌ No se pudo instanciar {cls.__name__}\n"
        f"     Instancia manualmente y pasa model= a export_pth()"
    )
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  METADATA JSON  (novedad: se guarda para TODOS los .pth)
# ═══════════════════════════════════════════════════════════════════════════════


def _save_pth_metadata(output_onnx: Path, ckpt: dict, pth_path: Path):
    """
    Guarda metadata del checkpoint junto al .onnx.

    El engine la usa para recuperar class_names, num_classes, etc.
    sin necesidad de cargar PyTorch.
    """
    meta = {
        "source": pth_path.name,
        "model_name": ckpt.get("model_name"),
        "model_type": ckpt.get("model_type"),
        "num_classes": ckpt.get("num_classes"),
        "temperature": ckpt.get("temperature"),
        "class_mapping": ckpt.get("class_mapping", {}),
        "best_f1": ckpt.get("best_f1"),
        "best_dice": ckpt.get("best_dice"),
        "best_iou": ckpt.get("best_iou"),
        "n_crops": ckpt.get("n_crops"),
        "crop_strategy": ckpt.get("crop_strategy"),
    }
    # Eliminar claves None para JSON limpio
    meta = {k: v for k, v in meta.items() if v is not None}

    json_path = output_onnx.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  💾 Metadata → {json_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
#  INFERIR MODELO DESDE CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════════════


def _load_model_from_ckpt(pth_path: Path) -> tuple[nn.Module | None, dict]:
    """
    Carga e instancia automáticamente el modelo desde un .pth.

    Returns:
        (modelo_con_pesos, checkpoint_dict)
        modelo es None si no se reconoce.
    """
    _build_registry()

    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        print(f"  ❌ Checkpoint inesperado (no es dict): {type(ckpt)}")
        return None, {}

    print(f"  Keys: {list(ckpt.keys())}")

    num_classes = ckpt.get("num_classes", 3)
    model_name = ckpt.get("model_name", "tf_efficientnetv2_s.in21k")
    model_type = ckpt.get("model_type", "")
    temperature = ckpt.get("temperature", 1.0)

    print(f"  model_type:  {model_type or '(no definido)'}")
    print(f"  model_name:  {model_name}")
    print(f"  num_classes: {num_classes}")
    print(f"  temperature: {temperature}")

    # 1. Por model_type del checkpoint
    cls = MODEL_REGISTRY.get(model_type)

    # 2. Por nombre del archivo
    if cls is None:
        stem = pth_path.stem.lower()
        for alias, klass in MODEL_REGISTRY.items():
            if alias.lower() in stem:
                cls = klass
                print(f"  💡 Inferido por nombre de archivo: {klass.__name__}")
                break

    if cls is None:
        print(
            f"  ❌ No se reconoce el modelo.\n"
            f"     model_type='{model_type}', archivo='{pth_path.name}'\n"
            f"     Aliases registrados: {list(MODEL_REGISTRY.keys())}\n"
            f"     Añade tu clase en _build_registry()"
        )
        return None, ckpt

    # 3. Instanciar
    model = _instantiate(cls, model_name=model_name, num_classes=num_classes)
    if model is None:
        return None, ckpt

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  ✅ {cls.__name__} cargado")

    # Temperatura solo para clasificadores (no segmentadores)
    is_segmenter = (
        "segmenter" in pth_path.stem.lower() or "polyp" in pth_path.stem.lower()
    )
    if temperature != 1.0 and not is_segmenter:
        model = _TemperatureWrapper(model, temperature)
        print(f"  🌡️  Temperatura {temperature:.3f} incorporada al grafo")

    return model, ckpt


# ═══════════════════════════════════════════════════════════════════════════════
#  .pth → ONNX
# ═══════════════════════════════════════════════════════════════════════════════


def export_pth(
    pth_path: Path,
    model: nn.Module | None = None,
    image_size: int = IMAGE_SIZE,
) -> Path | None:
    """
    Exporta un .pth a ONNX + JSON de metadata.

    Args:
        pth_path:   ruta del .pth original
        model:      modelo ya instanciado con pesos (opcional).
                    Si es None se infiere automáticamente del checkpoint.
        image_size: tamaño H=W de la imagen de entrada

    Returns:
        Path del .onnx generado o None si falla.
    """
    pth_path = Path(pth_path)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    output = ONNX_DIR / pth_path.with_suffix(".onnx").name

    print(f"\n{'─' * 50}")
    print(f"  PTH → ONNX: {pth_path.name}")
    print(f"{'─' * 50}")

    ckpt: dict = {}
    if model is None:
        model, ckpt = _load_model_from_ckpt(pth_path)
        if model is None:
            return None
    else:
        # Si se pasa el modelo manualmente, cargar ckpt solo para metadata
        try:
            ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
        except Exception:
            ckpt = {}

    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)

    with torch.no_grad():
        try:
            ref_out = model(dummy).numpy()
        except Exception as e:
            print(f"  ❌ Forward pass falló: {e}")
            return None

    print(f"  Input:  {tuple(dummy.shape)}")
    print(f"  Output: {tuple(ref_out.shape)}")

    # Exportar
    try:
        torch.onnx.export(
            model,
            dummy,
            str(output),
            opset_version=OPSET,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            do_constant_folding=True,
            verbose=False,
        )
    except Exception as e:
        print(f"  ❌ Export falló: {e}")
        return None

    # Validar grafo
    try:
        onnx.checker.check_model(onnx.load(str(output)))
        print("  ✅ Grafo ONNX válido")
    except Exception as e:
        print(f"  ❌ Grafo inválido: {e}")
        return None

    # Verificar outputs PyTorch vs ONNX Runtime
    try:
        sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input": dummy.numpy()})[0]
        diff = float(np.max(np.abs(ref_out - ort_out)))
        status = "✅" if diff < 1e-3 else "⚠️ "
        print(f"  {status} PyTorch vs ONNX: max_diff={diff:.2e}")
    except Exception as e:
        print(f"  ⚠️  Verificación: {e}")

    # Guardar metadata JSON (siempre, para todos los .pth)
    if ckpt:
        _save_pth_metadata(output, ckpt, pth_path)

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  📦 {output.name}  {size_mb:.1f} MB")

    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  .pkl → ONNX-ML
# ═══════════════════════════════════════════════════════════════════════════════


def export_pkl(
    pkl_path: Path,
    n_features: int | None = None,
    feature_names: list[str] | None = None,
) -> Path | None:
    """
    Exporta un .pkl (sklearn o xgboost) a ONNX-ML + JSON de metadata.

    El .pkl puede ser:
      - Estimador sklearn/xgboost directamente
      - Dict con clave 'model' (el resto va al JSON de metadata)

    Args:
        pkl_path:      ruta del .pkl
        n_features:    features de entrada (se infiere si es None)
        feature_names: nombres de features (van al JSON)

    Returns:
        Path del .onnx generado o None si falla.
    """
    pkl_path = Path(pkl_path)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    output = ONNX_DIR / pkl_path.with_suffix(".onnx").name

    print(f"\n{'─' * 50}")
    print(f"  PKL → ONNX-ML: {pkl_path.name}")
    print(f"{'─' * 50}")

    # Cargar
    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"  ❌ Corrupto o ilegible: {e}")
        return None

    # Extraer modelo del dict envolvente
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        extra_meta = {k: v for k, v in obj.items() if k != "model"}
        if feature_names is None and "feature_names" in obj:
            feature_names = obj["feature_names"]
    else:
        model = obj
        extra_meta = {}

    module = type(model).__module__
    model_name = type(model).__name__

    print(f"  Tipo:   {model_name}")
    print(f"  Módulo: {module}")

    # Inferir n_features
    if n_features is None:
        if feature_names:
            n_features = len(feature_names)
        elif hasattr(model, "n_features_in_"):
            n_features = int(model.n_features_in_)
        else:
            print("  ❌ No se puede determinar n_features. Pasa --n-features N")
            return None

    print(f"  n_features: {n_features}")

    # XGBoost → ONNX-ML
    if "xgboost" in module:
        try:
            onnx_model = convert_xgboost(
                model,
                initial_types=[("input", XGBFloat([None, n_features]))],
            )
            onnx.save(onnx_model, str(output))
        except Exception as e:
            print(f"  ❌ XGBoost export falló: {e}")
            return None

    # sklearn → ONNX-ML
    elif "sklearn" in module:
        try:
            onnx_model = convert_sklearn(
                model,
                initial_types=[("input", FloatTensorType([None, n_features]))],
                target_opset=OPSET,
                options={type(model): {"zipmap": False}},
            )
            onnx.save(onnx_model, str(output))
        except Exception as e:
            print(f"  ❌ sklearn export falló: {e}")
            return None

    # No soportado
    else:
        print(f"  ⚠️  {model_name} ({module}) no exportable a ONNX-ML")
        if extra_meta:
            json_out = ONNX_DIR / pkl_path.with_suffix(".json").name
            with open(json_out, "w") as f:
                json.dump(extra_meta, f, indent=2, default=str)
            print(f"  💾 Metadata → {json_out.name}")
        return None

    # Validar grafo
    try:
        onnx.checker.check_model(onnx.load(str(output)))
        print("  ✅ Grafo ONNX-ML válido")
    except Exception as e:
        print(f"  ❌ Grafo inválido: {e}")
        return None

    # Verificar outputs
    try:
        X = np.random.randn(3, n_features).astype(np.float32)
        ref = model.predict(X)
        sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        ort_pred = sess.run(None, {sess.get_inputs()[0].name: X})[0]
        matches = int(np.sum(ref == ort_pred))
        print(f"  ✅ sklearn/xgb vs ONNX-ML: {matches}/3 coinciden")
    except Exception as e:
        print(f"  ⚠️  Verificación: {e}")

    # Metadata JSON
    meta = {
        "source": pkl_path.name,
        "n_features": n_features,
        "feature_names": feature_names or [],
        **extra_meta,
    }
    with open(output.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  💾 Metadata → {output.with_suffix('.json').name}")

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  📦 {output.name}  {size_mb:.3f} MB")

    return output


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORTAR TODO
# ═══════════════════════════════════════════════════════════════════════════════


def export_all():
    """
    Exporta todos los .pth y .pkl en models/saved.
    Archivos corruptos o no reconocidos se omiten con aviso.
    """
    print(f"\n{'═' * 50}")
    print("  EXPORTANDO TODOS LOS MODELOS")
    print(f"{'═' * 50}")

    results: dict[str, Path] = {}
    skipped: list[str] = []

    for pth in sorted(paths.MODELS.glob("*.pth")):
        out = export_pth(pth)
        if out:
            results[pth.stem] = out
        else:
            skipped.append(pth.name)

    for pkl in sorted(paths.MODELS.glob("*.pkl")):
        out = export_pkl(pkl)
        if out:
            results[pkl.stem] = out
        else:
            skipped.append(pkl.name)

    print(f"\n{'═' * 50}")
    print(f"  Exportados: {len(results)}")
    for name, p in results.items():
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  ✅ {name:30s} {size_mb:.2f} MB")

    if skipped:
        print(f"\n  Omitidos: {len(skipped)}")
        for name in skipped:
            print(f"  ⏭️  {name}")

    print(f"\n  Directorio: {ONNX_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Exporta modelos a ONNX / ONNX-ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  uv run src/utils/export_onnx.py --all
  uv run src/utils/export_onnx.py --pth models/saved/best_classifier.pth
  uv run src/utils/export_onnx.py --pth models/saved/best_tissue_classifier.pth
  uv run src/utils/export_onnx.py --pth models/saved/best_segmenter.pth
  uv run src/utils/export_onnx.py --pkl models/saved/tabular_model.pkl
  uv run src/utils/export_onnx.py --pkl models/saved/tabular_model.pkl --n-features 30
        """,
    )
    parser.add_argument("--pth", type=str, help="Ruta a un .pth")
    parser.add_argument("--pkl", type=str, help="Ruta a un .pkl")
    parser.add_argument("--n-features", type=int, help="Nº features para .pkl")
    parser.add_argument("--all", action="store_true", help="Exporta todo")
    args = parser.parse_args()

    if args.all:
        export_all()
    elif args.pth:
        export_pth(Path(args.pth))
    elif args.pkl:
        export_pkl(Path(args.pkl), n_features=args.n_features)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
