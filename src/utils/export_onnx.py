"""
src/utils/export_onnx.py

Exports models from .pth y .pkl a ONNX / ONNX-ML.
Generates automatically a JSON of metadata of each.onnx.

Usage:
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
    """Register all model classes from the project."""
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
    Wraps a model to apply temperature scaling to its outputs.
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


def _instantiate(cls: type, model_name: str, num_classes: int) -> nn.Module | None:
    """
    Test constructor signatures in order until one works.

      1. (model_name, pretrained=False, num_classes)  ← clasification timm
      2. (pretrained=None)                             ← segmentation smp
      3. (pretrained=False)                            ← fallback
      4. ()                                            ← sin argumentos
    Args:
        cls (type): The model class to instantiate.
        model_name (str): The name of the model.
        num_classes (int): The number of classes for the model.
    Returns:
        nn.Module | None: The instantiated model or None if instantiation fails.
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
        f"  ❌ Cannot instantiate {cls.__name__}\n"
        f"     Instantiate manually and pass model= to export_pth()"
    )
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  METADATA JSON
# ═══════════════════════════════════════════════════════════════════════════════


def _save_pth_metadata(output_onnx: Path, ckpt: dict, pth_path: Path):
    """
    Save metadata of the checkpoint with each .onnx file.
    Args:
        output_onnx (Path): Path to the output ONNX file.
        ckpt (dict): Checkpoint dictionary.
        pth_path (Path): Path to the input PyTorch model file.
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
    meta = {k: v for k, v in meta.items() if v is not None}

    json_path = output_onnx.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  💾 Metadata → {json_path.name}")


def _load_model_from_ckpt(pth_path: Path) -> tuple[nn.Module | None, dict]:
    """
    Load and instantiate the model automatically from a .pth file.
    Args:
        pth_path (Path): Path to the input PyTorch model file.
    Returns:
        tuple[nn.Module | None, dict]: The instantiated model and the checkpoint dictionary.
    """
    _build_registry()

    ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        print(f"  ❌ Checkpoint not expected (it's not a dict): {type(ckpt)}")
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

    cls = MODEL_REGISTRY.get(model_type)

    if cls is None:
        stem = pth_path.stem.lower()
        for alias, klass in MODEL_REGISTRY.items():
            if alias.lower() in stem:
                cls = klass
                print(f"  💡 Inferred by file name: {klass.__name__}")
                break

    if cls is None:
        print(
            f"  ❌ Model not recognized.\n"
            f"     model_type='{model_type}', file='{pth_path.name}'\n"
            f"     Registered aliases: {list(MODEL_REGISTRY.keys())}\n"
            f"     Add your class to _build_registry()"
        )
        return None, ckpt

    model = _instantiate(cls, model_name=model_name, num_classes=num_classes)
    if model is None:
        return None, ckpt

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  ✅ {cls.__name__} loaded")

    is_segmenter = (
        "segmenter" in pth_path.stem.lower() or "polyp" in pth_path.stem.lower()
    )
    if temperature != 1.0 and not is_segmenter:
        model = _TemperatureWrapper(model, temperature)
        print(f"  🌡️  Temperature {temperature:.3f} incorporated into the graph")

    return model, ckpt


def export_pth(
    pth_path: Path,
    model: nn.Module | None = None,
    image_size: int = IMAGE_SIZE,
) -> Path | None:
    """
    Export a .pth file to ONNX + JSON metadata.

    Args:
        pth_path:   path to the original .pth file
        model:      instantiated model with weights (optional).
                    If None, it will be inferred automatically from the checkpoint.
        image_size: height and width of the input image

    Returns:
        Path to the generated .onnx file or None if it fails.
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
            print(f"  ❌ Forward pass failed: {e}")
            return None

    print(f"  Input:  {tuple(dummy.shape)}")
    print(f"  Output: {tuple(ref_out.shape)}")

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
        print(f"  ❌ Export failed: {e}")
        return None

    try:
        onnx.checker.check_model(onnx.load(str(output)))
        print("  ✅ ONNX Graph valid")
    except Exception as e:
        print(f"  ❌ Invalid ONNX Graph: {e}")
        return None

    try:
        sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input": dummy.numpy()})[0]
        diff = float(np.max(np.abs(ref_out - ort_out)))
        status = "✅" if diff < 1e-3 else "⚠️ "
        print(f"  {status} PyTorch vs ONNX: max_diff={diff:.2e}")
    except Exception as e:
        print(f"  ⚠️  Verification failed: {e}")

    if ckpt:
        _save_pth_metadata(output, ckpt, pth_path)

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  📦 {output.name}  {size_mb:.1f} MB")

    return output


def export_pkl(
    pkl_path: Path,
    n_features: int | None = None,
    feature_names: list[str] | None = None,
) -> Path | None:
    """
    Export a .pkl file to ONNX-ML + JSON metadata.

    Supported formats:
      - ReverseLogicTabularModel artifact  (dict with key 'pipelines')
      - Generic project format      (dict with key 'model')
      - Direct sklearn/xgboost estimator

    Args:
        pkl_path:      path to the .pkl file
        n_features:    input features (inferred if None)
        feature_names: names of transformed features (go to JSON)

    Returns:
        Path to the generated .onnx file or None if it fails.
    """
    pkl_path = Path(pkl_path)
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    output = ONNX_DIR / pkl_path.with_suffix(".onnx").name

    print(f"\n{'─' * 50}")
    print(f"  PKL → ONNX-ML: {pkl_path.name}")
    print(f"{'─' * 50}")

    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        print(f"  ❌ Corrupt or ilegible: {e}")
        return None

    extra_meta: dict = {}
    transformed_feature_names: list[str] = []

    if isinstance(obj, dict):
        if _is_reverse_logic_artifact(obj):
            model, extra_meta, transformed_feature_names, inferred_n = (
                _extract_reverse_logic_pipeline(obj)
            )
            if model is None:
                return None
            if n_features is None:
                n_features = inferred_n
            if feature_names is None and transformed_feature_names:
                feature_names = transformed_feature_names

        elif "model" in obj:
            model = obj["model"]
            extra_meta = {k: v for k, v in obj.items() if k != "model"}
            if feature_names is None and "feature_names" in obj:
                feature_names = obj["feature_names"]

        else:
            print(
                f"  ❌ Dict with unrecognized keys.\n"
                f"     Found keys : {list(obj.keys())}\n"
                f"     Expected keys   : 'pipelines'  or  'model'"
            )
            return None

    else:
        model = obj
        extra_meta = {}

    module = type(model).__module__
    model_name = type(model).__name__
    print(f"  Type:   {model_name}")
    print(f"  MModule: {module}")

    if n_features is None:
        if feature_names:
            n_features = len(feature_names)
        else:
            _preprocessor = None
            if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
                _preprocessor = model.named_steps["preprocessor"]
            elif hasattr(model, "preprocessor_"):
                _preprocessor = model.preprocessor_

            if _preprocessor is not None and hasattr(_preprocessor, "n_features_in_"):
                n_features = int(_preprocessor.n_features_in_)
            elif hasattr(model, "n_features_in_"):
                n_features = int(model.n_features_in_)

        if n_features is None or not isinstance(n_features, int):
            print(
                "  ❌ Cannot determine n_features.\n"
                "     Pass --n-features N with the number of input columns"
            )
            return None

    print(f"  n_features: {n_features}")

    n_transformed: int = n_features if n_features is not None else 0

    if transformed_feature_names:
        n_transformed = len(transformed_feature_names)
    else:
        try:
            _preprocessor = None
            if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
                _preprocessor = model.named_steps["preprocessor"]
            elif hasattr(model, "preprocessor_"):
                _preprocessor = model.preprocessor_

            if _preprocessor is not None:
                n_transformed = len(_preprocessor.get_feature_names_out())
        except Exception:
            pass

    if "xgboost" in module:
        try:
            onnx_model = convert_xgboost(
                model,
                initial_types=[("input", XGBFloat([None, n_transformed]))],
            )
            onnx.save(onnx_model, str(output))
        except Exception as e:
            print(f"  ❌ XGBoost export falló: {e}")
            return None

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

    elif "lightgbm" in module:
        try:
            onnx_model = convert_sklearn(
                model,
                initial_types=[("input", FloatTensorType([None, n_features]))],
                target_opset=OPSET,
            )
            onnx.save(onnx_model, str(output))
        except Exception as e:
            print(f"  ❌ LightGBM export failed: {e}")
            print(
                "     Install: pip install onnxmltools lightgbm\n"
                "     Or use --pipeline random_forest to export that pipeline"
            )
            return None

    elif hasattr(model, "model_") and hasattr(model, "preprocessor_"):
        print(
            "  💡 FeatureWeightedXGBClassifier detected → "
            "exporting internal XGBClassifier"
        )
        try:
            onnx_model = convert_xgboost(
                model.model_,
                initial_types=[("input", XGBFloat([None, n_transformed]))],
            )
            onnx.save(onnx_model, str(output))
            extra_meta["note"] = (
                "The preprocessing (StandardScaler + OHE) is NOT included in the graph. "
                "Apply the ColumnTransformer manually before inference."
            )
        except Exception as e:
            print(f"  ❌ FeatureWeightedXGBClassifier export failed: {e}")
            return None

    else:
        print(f"  ⚠️  {model_name} ({module}) not exportable to ONNX-ML")
        if extra_meta:
            json_out = ONNX_DIR / pkl_path.with_suffix(".json").name
            with open(json_out, "w") as f:
                json.dump(extra_meta, f, indent=2, default=str)
            print(f"  💾 Metadata → {json_out.name}")
        return None

    try:
        onnx.checker.check_model(onnx.load(str(output)))
        print("  ✅ Grafo ONNX-ML válido")
    except Exception as e:
        print(f"  ❌ Grafo inválido: {e}")
        return None

    try:
        _n_verify = (
            n_transformed
            if (hasattr(model, "model_") and hasattr(model, "preprocessor_"))
            else n_features
        )
        X_verify = np.random.randn(3, _n_verify).astype(np.float32)
        ref = model.predict(X_verify)  # type: ignore[union-attr]
        sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        ort_pred = sess.run(None, {sess.get_inputs()[0].name: X_verify})[0]
        matches = int(np.sum(ref == ort_pred))
        print(f"  ✅ sklearn/xgb vs ONNX-ML: {matches}/3 coinciden")
    except Exception as e:
        print(f"  ⚠️  Verification: {e}")

    meta = {
        "source": pkl_path.name,
        "n_features_input": n_features,
        "n_features_transformed": n_transformed,
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
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _is_reverse_logic_artifact(obj: object) -> bool:
    """
    Detects if the loaded object from the .pkl is an artifact of ReverseLogicTabularModel.
    Minimum criterion: dict with key 'pipelines' that is also a dict.
    """
    return (
        isinstance(obj, dict)
        and "pipelines" in obj
        and isinstance(obj["pipelines"], dict)
    )


def _extract_reverse_logic_pipeline(
    artifact: dict,
) -> tuple[object | None, dict, list[str], int | None]:
    """
    Extracts the best exportable pipeline from a ReverseLogicTabularModel artifact.

    Selection strategy (in order of priority):
      1. Pipeline with highest F1 score in saved metrics
      2. First available pipeline as fallback

    Order of preference by ONNX exportability:
      random_forest > xgboost > lightgbm
      (lightgbm needs additional converter)

    Returns:
        (pipeline, extra_meta, feature_names_list, n_features)
        pipeline is None if there is no exportable one.
    """
    pipelines: dict = artifact["pipelines"]
    metrics: dict = artifact.get("metrics", {})
    feature_names_by_type: dict = artifact.get("feature_names", {})
    numeric_features: list = artifact.get("numeric_features", [])
    categorical_features: list = artifact.get("categorical_features", [])
    target: str = artifact.get("target", "unknown")

    if not pipelines:
        print("  ❌ The artifact does not contain any trained pipelines")
        return None, {}, [], None

    best_name: str | None = None
    best_f1 = -1.0

    for name in pipelines:
        f1 = metrics.get(name, {}).get("f1", 0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    if best_name is None:
        best_name = next(iter(pipelines))

    best_pipeline = pipelines[best_name]

    scaler_params: dict = {}
    try:
        if hasattr(best_pipeline, "preprocessor_"):
            preprocessor = best_pipeline.preprocessor_
        elif hasattr(best_pipeline, "named_steps"):
            preprocessor = best_pipeline.named_steps.get("preprocessor")
        else:
            preprocessor = None

        if preprocessor is not None:
            for t_name, transformer, cols in preprocessor.transformers_:
                if t_name == "num" and hasattr(transformer, "mean_"):
                    scaler_params = {
                        "scaler_mean": transformer.mean_.tolist(),
                        "scaler_scale": transformer.scale_.tolist(),
                        "scaler_feature_names": list(cols),
                    }
                    print(
                        f"  📊 StandardScaler params extracted ({len(cols)} features)"
                    )
                    break
    except Exception as e:
        print(f"  ⚠️  Could not extract scaler params: {e}")

    feature_names_list: list[str] = feature_names_by_type.get(best_name, [])

    if not feature_names_list:
        try:
            if hasattr(best_pipeline, "named_steps"):
                preprocessor = best_pipeline.named_steps["preprocessor"]
                feature_names_list = preprocessor.get_feature_names_out().tolist()
            elif hasattr(best_pipeline, "preprocessor_"):
                feature_names_list = (
                    best_pipeline.preprocessor_.get_feature_names_out().tolist()
                )
        except Exception as e:
            print(f"  ⚠️  Could not retrieve feature_names: {e}")

    n_features_input = len(numeric_features) + len(categorical_features)
    if n_features_input == 0:
        try:
            if hasattr(best_pipeline, "named_steps"):
                preprocessor = best_pipeline.named_steps["preprocessor"]
                n_features_input = int(preprocessor.n_features_in_)
            elif hasattr(best_pipeline, "preprocessor_"):
                n_features_input = int(best_pipeline.preprocessor_.n_features_in_)
        except (Exception, TypeError):
            n_features_input = 0

    extra_meta = {
        "model_framework": "ReverseLogicTabularModel",
        "exported_pipeline": best_name,
        "target": target,
        "all_pipelines": list(pipelines.keys()),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "metrics": {
            name: {k: v for k, v in m.items() if k != "confusion_matrix"}
            for name, m in metrics.items()
        },
        "label_encoders": {
            name: list(le.classes_)
            for name, le in artifact.get("label_encoders", {}).items()
        },
        **scaler_params,
    }
    if best_f1 >= 0:
        extra_meta["exported_f1"] = round(best_f1, 6)

    print("  💡 ReverseLogicTabularModel detected")
    print(f"     target       : {target}")
    print(f"     pipelines    : {list(pipelines.keys())}")
    print(f"     exporting   : '{best_name}'  (F1={best_f1:.4f})")
    print(f"     n_features   : {n_features_input}")
    print(
        f"     feature_names: {feature_names_list[:5]}{'...' if len(feature_names_list) > 5 else ''}"
    )

    return best_pipeline, extra_meta, feature_names_list, n_features_input


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORTAR TODO
# ═══════════════════════════════════════════════════════════════════════════════


def export_all():
    """
    Exports all .pth and .pkl files in models/saved.
    Corrupted or unrecognized files are skipped with a warning.
    """
    print(f"\n{'═' * 50}")
    print("  EXPORTING ALL MODELS")
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
    print(f"  Exported: {len(results)}")
    for name, p in results.items():
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  ✅ {name:30s} {size_mb:.2f} MB")

    if skipped:
        print(f"\n  Skipped: {len(skipped)}")
        for name in skipped:
            print(f"  ⏭️  {name}")

    print(f"\n  Directory: {ONNX_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Exports models to ONNX / ONNX-ML",
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
    parser.add_argument("--n-features", type=int, help="Nº features for .pkl")
    parser.add_argument("--all", action="store_true", help="Exports all models")
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
