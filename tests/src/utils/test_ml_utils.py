# tests/src/utils/test_ml_utils.py
"""
Tests for src/utils/ml_utils.py
"""

import numpy as np

from src.utils.ml_utils import (
    _convert_feature,
    _to_bool_float,
    _to_categorical_float,
    _to_numeric_float,
    encode_patient_row,
    numpy_to_base64,
    softmax_np,
)

# ─────────────────────────────────────────────────────────────
#  _to_bool_float
# ─────────────────────────────────────────────────────────────


class TestToBoolFloat:
    def test_true_bool(self):
        assert _to_bool_float(True) == 1.0

    def test_false_bool(self):
        assert _to_bool_float(False) == 0.0

    def test_yes_string(self):
        assert _to_bool_float("yes") == 1.0

    def test_no_string(self):
        assert _to_bool_float("no") == 0.0

    def test_sí_string(self):
        assert _to_bool_float("sí") == 1.0

    def test_true_string(self):
        assert _to_bool_float("true") == 1.0

    def test_1_string(self):
        assert _to_bool_float("1") == 1.0

    def test_0_string(self):
        assert _to_bool_float("0") == 0.0

    def test_integer_1(self):
        assert _to_bool_float(1) == 1.0

    def test_integer_0(self):
        assert _to_bool_float(0) == 0.0

    def test_none_returns_zero(self):
        assert _to_bool_float(None) == 0.0

    def test_uppercase_yes(self):
        assert _to_bool_float("YES") == 1.0

    def test_unknown_string(self):
        assert _to_bool_float("maybe") == 0.0


# ─────────────────────────────────────────────────────────────
#  _to_categorical_float
# ─────────────────────────────────────────────────────────────


class TestToCategoricalFloat:
    def test_gender_male(self):
        assert _to_categorical_float("gender", "male") == 0.0

    def test_gender_female(self):
        assert _to_categorical_float("gender", "female") == 1.0

    def test_gender_m_short(self):
        assert _to_categorical_float("gender", "m") == 0.0

    def test_smoking_never(self):
        assert _to_categorical_float("smoking_status", "never") == 0.0

    def test_smoking_current(self):
        assert _to_categorical_float("smoking_status", "current") == 2.0

    def test_alcohol_moderate(self):
        assert _to_categorical_float("alcohol_consumption", "moderate") == 1.0

    def test_physical_activity_high(self):
        assert _to_categorical_float("physical_activity", "high") == 3.0

    def test_diet_western(self):
        assert _to_categorical_float("diet_type", "western") == 0.0

    def test_unknown_value_returns_zero(self):
        assert _to_categorical_float("gender", "unknown_value") == 0.0

    def test_numeric_passthrough(self):
        assert _to_categorical_float("gender", 1) == 1.0

    def test_none_returns_zero(self):
        assert _to_categorical_float("gender", None) == 0.0

    def test_case_insensitive(self):
        assert _to_categorical_float("gender", "MALE") == 0.0


# ─────────────────────────────────────────────────────────────
#  _to_numeric_float
# ─────────────────────────────────────────────────────────────


class TestToNumericFloat:
    def test_integer(self):
        assert _to_numeric_float("age", 45) == 45.0

    def test_float(self):
        assert _to_numeric_float("bmi", 24.5) == 24.5

    def test_string_number(self):
        assert _to_numeric_float("age", "30") == 30.0

    def test_none_returns_zero(self):
        assert _to_numeric_float("age", None) == 0.0

    def test_invalid_string_returns_zero(self):
        assert _to_numeric_float("age", "not_a_number") == 0.0

    def test_negative_number(self):
        assert _to_numeric_float("score", -5.5) == -5.5


# ─────────────────────────────────────────────────────────────
#  _convert_feature
# ─────────────────────────────────────────────────────────────


class TestConvertFeature:
    def test_boolean_feature(self):
        result = _convert_feature("family_history_ccr", "yes")
        assert result == 1.0

    def test_categorical_feature(self):
        result = _convert_feature("gender", "female")
        assert result == 1.0

    def test_numeric_feature(self):
        result = _convert_feature("age", 55)
        assert result == 55.0


# ─────────────────────────────────────────────────────────────
#  encode_patient_row
# ─────────────────────────────────────────────────────────────


class TestEncodePatientRow:
    def test_output_shape(self):
        data = {"age": 55, "gender": "male"}
        result = encode_patient_row(data, ["age", "gender"])
        assert result.shape == (1, 2)

    def test_output_dtype_float32(self):
        data = {"age": 55}
        result = encode_patient_row(data, ["age"])
        assert result.dtype == np.float32

    def test_missing_feature_uses_default(self):
        result = encode_patient_row({}, ["age"])
        assert result.shape == (1, 1)
        assert np.isfinite(result).all()

    def test_none_feature_uses_default(self):
        result = encode_patient_row({"age": None}, ["age"])
        assert result.shape == (1, 1)

    def test_correct_encoding(self):
        data = {"gender": "female", "age": 40}
        result = encode_patient_row(data, ["gender", "age"])
        assert result[0, 0] == 1.0  # female
        assert result[0, 1] == 40.0

    def test_boolean_feature_encoded(self):
        data = {"family_history_ccr": "yes"}
        result = encode_patient_row(data, ["family_history_ccr"])
        assert result[0, 0] == 1.0

    def test_multiple_features_order_preserved(self):
        data = {"age": 50, "gender": "male", "has_ibd": "no"}
        features = ["age", "gender", "has_ibd"]
        result = encode_patient_row(data, features)
        assert result[0, 0] == 50.0
        assert result[0, 1] == 0.0  # male
        assert result[0, 2] == 0.0  # no

    def test_empty_feature_list(self):
        result = encode_patient_row({"age": 30}, [])
        assert result.shape == (1, 0)


# ─────────────────────────────────────────────────────────────
#  numpy_to_base64
# ─────────────────────────────────────────────────────────────


class TestNumpyToBase64:
    def test_returns_string(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = numpy_to_base64(img)
        assert isinstance(result, str)

    def test_starts_with_data_uri(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = numpy_to_base64(img)
        assert result.startswith("data:image/png;base64,")

    def test_non_empty_base64(self):
        img = np.ones((32, 32, 3), dtype=np.uint8) * 128
        result = numpy_to_base64(img)
        b64_part = result.split(",", 1)[1]
        assert len(b64_part) > 0

    def test_different_images_give_different_output(self):
        img1 = np.zeros((32, 32, 3), dtype=np.uint8)
        img2 = np.ones((32, 32, 3), dtype=np.uint8) * 255
        r1 = numpy_to_base64(img1)
        r2 = numpy_to_base64(img2)
        assert r1 != r2

    def test_base64_decodable(self):
        import base64

        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = numpy_to_base64(img)
        b64_part = result.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert len(decoded) > 0


# ─────────────────────────────────────────────────────────────
#  softmax_np
# ─────────────────────────────────────────────────────────────


class TestSoftmaxNp:
    def test_output_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax_np(x)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_max_input_has_max_output(self):
        x = np.array([1.0, 5.0, 2.0])
        result = softmax_np(x)
        assert result.argmax() == 1

    def test_all_values_in_0_1(self):
        x = np.random.randn(10)
        result = softmax_np(x)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_uniform_input_uniform_output(self):
        x = np.zeros(4)
        result = softmax_np(x)
        np.testing.assert_allclose(result, [0.25] * 4, atol=1e-6)

    def test_large_input_numerically_stable(self):
        x = np.array([1000.0, 1000.0, 1000.0])
        result = softmax_np(x)
        assert np.all(np.isfinite(result))

    def test_negative_inputs(self):
        x = np.array([-1.0, -2.0, -3.0])
        result = softmax_np(x)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_single_element(self):
        result = softmax_np(np.array([5.0]))
        np.testing.assert_allclose(result, [1.0], atol=1e-6)
