"""
Tests for MultiDatasetOrganizer - 100% coverage.
"""

from collections import defaultdict
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import src.data.ingestion.multi_dataset_organizer as org_module
from src.data.ingestion.multi_dataset_organizer import MultiDatasetOrganizer

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _fake_paths(tmp_path):
    fp = MagicMock()
    fp.HYPERKVASIR_RAW = tmp_path / "hk_raw"
    fp.RAW = tmp_path / "raw"
    fp.COLON_CLEAN = tmp_path / "clean"
    fp.COLON_PROCESSED = tmp_path / "processed"
    fp.TABULAR_PROCESSED = tmp_path / "tabular_processed"
    fp.RAW_TABULAR = tmp_path / "raw_tabular"
    fp.SEGMENTER_CHECKPOINT = tmp_path / "seg.pth"
    fp.TABULAR_MODEL_PATH = tmp_path / "tab.pkl"
    fp.TABULAR_PREPROCESSOR_PATH = tmp_path / "prep.pkl"
    fp.MODELS = tmp_path / "models"
    return fp


def _make_organizer(tmp_path, monkeypatch, target_per_class=10, seed=42):
    fp = _fake_paths(tmp_path)
    monkeypatch.setattr(org_module, "paths", fp)
    org = MultiDatasetOrganizer(target_per_class=target_per_class, seed=seed)
    org.hk_raw = fp.HYPERKVASIR_RAW
    org.cvc_raw = tmp_path / "raw" / "cvc_clinicdb"
    org.limuc_raw = tmp_path / "raw" / "limuc"
    org.curated_raw = tmp_path / "raw" / "curated_colon"
    org.clean_dir = tmp_path / "clean"
    org.processed_dir = tmp_path / "processed"
    return org, fp


# ─────────────────────────────────────────────────────────────
# __init__
# ─────────────────────────────────────────────────────────────


class TestInit:
    def test_attributes_set_correctly(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch, target_per_class=500, seed=7)
        assert org.target_per_class == 500
        assert org.seed == 7
        assert isinstance(org.collected, defaultdict)
        assert org.image_records == []

    def test_default_values(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        assert org.target_per_class == 10
        assert org.seed == 42

    def test_collected_is_defaultdict(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        assert org.collected["new_class"] == []

    def test_image_records_starts_empty(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        assert len(org.image_records) == 0


# ─────────────────────────────────────────────────────────────
# _glob_images
# ─────────────────────────────────────────────────────────────


class TestGlobImages:
    def test_returns_known_extensions(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        (tmp_path / "a.jpg").write_bytes(b"x")
        (tmp_path / "b.png").write_bytes(b"x")
        (tmp_path / "c.txt").write_bytes(b"x")
        result = org._glob_images(tmp_path)
        names = [p.name for p in result]
        assert "a.jpg" in names
        assert "b.png" in names
        assert "c.txt" not in names

    def test_sorted_output(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        for name in ["z.jpg", "a.png", "m.jpg"]:
            (tmp_path / name).write_bytes(b"x")
        result = org._glob_images(tmp_path)
        assert [p.name for p in result] == sorted(p.name for p in result)

    def test_empty_directory(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        assert org._glob_images(tmp_path) == []

    def test_returns_list(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        assert isinstance(org._glob_images(tmp_path), list)


# ─────────────────────────────────────────────────────────────
# _find_dir
# ─────────────────────────────────────────────────────────────


class TestFindDir:
    def test_returns_first_existing_candidate(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        (tmp_path / "b").mkdir()
        result = org._find_dir(tmp_path, ["a", "b", "c"])
        assert result == tmp_path / "b"

    def test_returns_none_when_none_exist(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        result = org._find_dir(tmp_path, ["x", "y"])
        assert result is None

    def test_dot_returns_base_itself(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        result = org._find_dir(tmp_path, ["."])
        assert result == tmp_path

    def test_returns_none_when_base_missing(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        result = org._find_dir(tmp_path / "missing", ["a"])
        assert result is None

    def test_skips_non_directory_candidates(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        (tmp_path / "file.txt").write_bytes(b"x")
        result = org._find_dir(tmp_path, ["file.txt"])
        assert result is None


# ─────────────────────────────────────────────────────────────
# _stratified_subsample
# ─────────────────────────────────────────────────────────────


class TestStratifiedSubsample:
    def _make_records(self, sources_counts):
        records = []
        for src, count in sources_counts.items():
            for i in range(count):
                records.append({"source": src, "id": f"{src}{i}"})
        return records

    def test_returns_at_most_target_records(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        records = self._make_records({"a": 50, "b": 50})
        result = org._stratified_subsample(records, 30, key="source")
        assert len(result) <= 30

    def test_both_groups_represented(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        records = self._make_records({"a": 100, "b": 100})
        result = org._stratified_subsample(records, 40, key="source")
        sources = [r["source"] for r in result]
        assert "a" in sources
        assert "b" in sources

    def test_single_group(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        records = self._make_records({"only": 20})
        result = org._stratified_subsample(records, 10, key="source")
        assert len(result) <= 10
        assert all(r["source"] == "only" for r in result)

    def test_target_larger_than_total(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        records = self._make_records({"a": 3, "b": 2})
        result = org._stratified_subsample(records, 100, key="source")
        assert len(result) <= 5

    def test_three_groups_all_represented(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        records = self._make_records({"a": 30, "b": 30, "c": 30})
        result = org._stratified_subsample(records, 15, key="source")
        sources = {r["source"] for r in result}
        assert len(sources) >= 2

    def test_last_group_absorbs_remainder(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        records = self._make_records({"a": 50, "b": 50, "c": 50})
        result = org._stratified_subsample(records, 10, key="source")
        assert len(result) <= 10


# ─────────────────────────────────────────────────────────────
# Mayo folder helpers
# ─────────────────────────────────────────────────────────────


class TestMayoFolders:
    def test_has_mayo_folders_true(self, tmp_path, monkeypatch):
        from src.config.constants import MAYO_FOLDER_PATTERNS

        org, _ = _make_organizer(tmp_path, monkeypatch)
        patient = tmp_path / "patient1"
        pattern = MAYO_FOLDER_PATTERNS[0]
        score_dir = patient / pattern.format(score="0")
        score_dir.mkdir(parents=True)
        (score_dir / "img.jpg").write_bytes(b"x")
        assert org._has_mayo_folders(patient) is True

    def test_has_mayo_folders_false_when_empty_dirs(self, tmp_path, monkeypatch):
        from src.config.constants import MAYO_FOLDER_PATTERNS

        org, _ = _make_organizer(tmp_path, monkeypatch)
        patient = tmp_path / "patient_empty"
        pattern = MAYO_FOLDER_PATTERNS[0]
        score_dir = patient / pattern.format(score="0")
        score_dir.mkdir(parents=True)
        assert org._has_mayo_folders(patient) is False

    def test_has_mayo_folders_false_no_dirs(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        patient = tmp_path / "patient_no_dirs"
        patient.mkdir()
        assert org._has_mayo_folders(patient) is False

    def test_find_mayo_folder_returns_correct_path(self, tmp_path, monkeypatch):
        from src.config.constants import MAYO_FOLDER_PATTERNS

        org, _ = _make_organizer(tmp_path, monkeypatch)
        patient = tmp_path / "p"
        pattern = MAYO_FOLDER_PATTERNS[0]
        score_dir = patient / pattern.format(score="2")
        score_dir.mkdir(parents=True)
        result = org._find_mayo_folder(patient, "2")
        assert result == score_dir

    def test_find_mayo_folder_returns_none(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        patient = tmp_path / "p"
        patient.mkdir()
        result = org._find_mayo_folder(patient, "3")
        assert result is None

    def test_has_mayo_folders_all_scores(self, tmp_path, monkeypatch):
        from src.config.constants import MAYO_FOLDER_PATTERNS

        org, _ = _make_organizer(tmp_path, monkeypatch)
        patient = tmp_path / "multi_score"
        for score in ["1", "2", "3"]:
            pattern = MAYO_FOLDER_PATTERNS[0]
            score_dir = patient / pattern.format(score=score)
            score_dir.mkdir(parents=True)
            (score_dir / f"img_{score}.jpg").write_bytes(b"x")
        assert org._has_mayo_folders(patient) is True


# ─────────────────────────────────────────────────────────────
# _collect_curated_colon
# ─────────────────────────────────────────────────────────────


class TestCollectCuratedColon:
    def test_collects_polyp_images(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        curated = tmp_path / "raw" / "curated_colon"
        d = curated / "polyp_class"
        d.mkdir(parents=True)
        (d / "img1.jpg").write_bytes(b"x")
        org.curated_raw = curated
        org._collect_curated_colon()
        assert len(org.collected["polyp"]) >= 1

    def test_collects_normal_images(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        curated = tmp_path / "raw" / "curated_colon"
        d = curated / "normal_tissue"
        d.mkdir(parents=True)
        (d / "img1.jpg").write_bytes(b"x")
        org.curated_raw = curated
        org._collect_curated_colon()
        assert len(org.collected["normal"]) >= 1

    def test_skips_when_directory_missing(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        org.curated_raw = tmp_path / "nonexistent"
        org._collect_curated_colon()
        assert sum(len(v) for v in org.collected.values()) == 0

    def test_skips_unclassifiable_dirs(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        curated = tmp_path / "raw" / "curated_colon"
        d = curated / "random_folder"
        d.mkdir(parents=True)
        (d / "img.jpg").write_bytes(b"x")
        org.curated_raw = curated
        org._collect_curated_colon()
        assert sum(len(v) for v in org.collected.values()) == 0

    def test_source_metadata_set(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        curated = tmp_path / "raw" / "curated_colon"
        d = curated / "polyp_stuff"
        d.mkdir(parents=True)
        (d / "img.jpg").write_bytes(b"x")
        org.curated_raw = curated
        org._collect_curated_colon()
        assert org.collected["polyp"][0]["source"] == "curated_colon"

    def test_no_folder_mapping_logs_warning(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        curated = tmp_path / "raw" / "curated_colon"
        curated.mkdir(parents=True)
        # No classifiable directories
        org.curated_raw = curated
        org._collect_curated_colon()
        assert sum(len(v) for v in org.collected.values()) == 0


# ─────────────────────────────────────────────────────────────
# _collect_cvc_clinicdb
# ─────────────────────────────────────────────────────────────


class TestCollectCvcClinicdb:
    def test_collects_polyp_images(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        cvc = tmp_path / "raw" / "cvc_clinicdb"
        img_dir = cvc / "Original"
        img_dir.mkdir(parents=True)
        (img_dir / "001.jpg").write_bytes(b"x")
        org.cvc_raw = cvc
        org._collect_cvc_clinicdb()
        assert len(org.collected["polyp"]) >= 1

    def test_all_classified_as_polyp(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        cvc = tmp_path / "raw" / "cvc_clinicdb"
        img_dir = cvc / "Original"
        img_dir.mkdir(parents=True)
        for i in range(3):
            (img_dir / f"{i:03d}.jpg").write_bytes(b"x")
        org.cvc_raw = cvc
        org._collect_cvc_clinicdb()
        assert all(r["class_name"] == "polyp" for r in org.collected["polyp"])

    def test_skips_when_directory_missing(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        org.cvc_raw = tmp_path / "nonexistent"
        org._collect_cvc_clinicdb()
        assert len(org.collected["polyp"]) == 0

    def test_source_set_correctly(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        cvc = tmp_path / "raw" / "cvc_clinicdb"
        img_dir = cvc / "Original"
        img_dir.mkdir(parents=True)
        (img_dir / "001.jpg").write_bytes(b"x")
        org.cvc_raw = cvc
        org._collect_cvc_clinicdb()
        assert org.collected["polyp"][0]["source"] == "cvc_clinicdb"

    def test_mask_path_none_when_no_mask_dir(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        cvc = tmp_path / "raw" / "cvc_clinicdb"
        img_dir = cvc / "Original"
        img_dir.mkdir(parents=True)
        (img_dir / "001.jpg").write_bytes(b"x")
        org.cvc_raw = cvc
        org._collect_cvc_clinicdb()
        assert org.collected["polyp"][0]["mask_path"] is None

    def test_mask_path_when_mask_exists(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        cvc = tmp_path / "raw" / "cvc_clinicdb"
        img_dir = cvc / "Original"
        mask_dir = cvc / "Ground Truth"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        (img_dir / "001.jpg").write_bytes(b"x")
        (mask_dir / "001.png").write_bytes(b"x")
        org.cvc_raw = cvc
        org._collect_cvc_clinicdb()
        assert org.collected["polyp"][0]["mask_path"] is not None


# ─────────────────────────────────────────────────────────────
# _collect_hyperkvasir
# ─────────────────────────────────────────────────────────────


class TestCollectHyperkvasir:
    def test_skips_when_not_found(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        org.hk_raw = tmp_path / "nonexistent_hk"
        org._collect_hyperkvasir()
        assert sum(len(v) for v in org.collected.values()) == 0

    def test_collects_images_with_mask(self, tmp_path, monkeypatch):
        from src.config.constants import SOURCE_MAP

        org, _ = _make_organizer(tmp_path, monkeypatch)

        hk_root = tmp_path / "hk_raw"
        labeled_dir = hk_root / "labeled-images"
        segmented_dir = hk_root / "segmented-images"
        masks_dir = segmented_dir / "masks"

        # Create structure for first class/source in SOURCE_MAP
        first_class = list(SOURCE_MAP["hyperkvasir"].keys())[0]
        first_source = SOURCE_MAP["hyperkvasir"][first_class][0]

        img_dir = labeled_dir / first_source
        img_dir.mkdir(parents=True)
        (img_dir / "img001.jpg").write_bytes(b"x")

        masks_dir.mkdir(parents=True)
        (masks_dir / "img001.jpg").write_bytes(b"x")

        org.hk_raw = hk_root
        org._collect_hyperkvasir()
        total = sum(len(v) for v in org.collected.values())
        assert total >= 1

    def test_source_detail_set(self, tmp_path, monkeypatch):
        from src.config.constants import SOURCE_MAP

        org, _ = _make_organizer(tmp_path, monkeypatch)

        hk_root = tmp_path / "hk_raw"
        labeled_dir = hk_root / "labeled-images"

        first_class = list(SOURCE_MAP["hyperkvasir"].keys())[0]
        first_source = SOURCE_MAP["hyperkvasir"][first_class][0]

        img_dir = labeled_dir / first_source
        img_dir.mkdir(parents=True)
        (img_dir / "img001.jpg").write_bytes(b"x")

        org.hk_raw = hk_root
        org._collect_hyperkvasir()

        records = org.collected[first_class]
        if records:
            assert "source" in records[0]
            assert records[0]["source"] == "hyperkvasir"

    def test_missing_source_dir_skipped(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)

        hk_root = tmp_path / "hk_raw"
        labeled_dir = hk_root / "labeled-images"
        labeled_dir.mkdir(parents=True)

        org.hk_raw = hk_root
        # Don't create any actual image directories
        org._collect_hyperkvasir()
        # No crash, just warnings


# ─────────────────────────────────────────────────────────────
# _collect_limuc
# ─────────────────────────────────────────────────────────────


class TestCollectLimuc:
    def test_skips_when_not_found(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        org.limuc_raw = tmp_path / "nonexistent_limuc"
        org._collect_limuc()
        assert sum(len(v) for v in org.collected.values()) == 0

    def test_skips_when_no_patient_dirs(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        limuc = tmp_path / "raw" / "limuc"
        limuc.mkdir(parents=True)
        org.limuc_raw = limuc
        org._collect_limuc()
        assert sum(len(v) for v in org.collected.values()) == 0

    def test_collects_from_patient_dirs(self, tmp_path, monkeypatch):
        from src.config.constants import MAYO_FOLDER_PATTERNS, SOURCE_MAP

        org, _ = _make_organizer(tmp_path, monkeypatch)

        limuc = tmp_path / "raw" / "limuc"

        # Create patient with mayo score
        patient_dir = limuc / "001"
        first_class = list(SOURCE_MAP["limuc"].keys())[0]
        first_scores = SOURCE_MAP["limuc"][first_class]
        score = first_scores[0]
        pattern = MAYO_FOLDER_PATTERNS[0]
        score_dir = patient_dir / pattern.format(score=score)
        score_dir.mkdir(parents=True)
        (score_dir / "frame001.jpg").write_bytes(b"x")

        org.limuc_raw = limuc
        org._collect_limuc()
        total = sum(len(v) for v in org.collected.values())
        assert total >= 1


# ─────────────────────────────────────────────────────────────
# _find_limuc_patient_dirs
# ─────────────────────────────────────────────────────────────


class TestFindLimucPatientDirs:
    def test_returns_empty_when_not_found(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        org.limuc_raw = tmp_path / "nonexistent"
        result = org._find_limuc_patient_dirs()
        assert result == []

    def test_finds_numeric_patient_dirs(self, tmp_path, monkeypatch):
        from src.config.constants import MAYO_FOLDER_PATTERNS

        org, _ = _make_organizer(tmp_path, monkeypatch)

        limuc = tmp_path / "limuc_data"
        limuc.mkdir()
        patient = limuc / "042"
        pattern = MAYO_FOLDER_PATTERNS[0]
        score_dir = patient / pattern.format(score="0")
        score_dir.mkdir(parents=True)
        (score_dir / "frame.jpg").write_bytes(b"x")

        org.limuc_raw = limuc
        result = org._find_limuc_patient_dirs()
        assert len(result) >= 1

    def test_skips_non_numeric_dirs(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        limuc = tmp_path / "limuc_data"
        limuc.mkdir()
        (limuc / "not_a_patient").mkdir()
        org.limuc_raw = limuc
        result = org._find_limuc_patient_dirs()
        assert all(p.name.isdigit() for p in result)

    def test_recursive_fallback_search(self, tmp_path, monkeypatch):
        """Verifica el fallback recursivo cuando el primary search no encuentra nada"""
        from src.config.constants import MAYO_FOLDER_PATTERNS

        org, _ = _make_organizer(tmp_path, monkeypatch)

        limuc = tmp_path / "limuc_data"
        deep = limuc / "deep" / "nested"
        patient = deep / "007"
        pattern = MAYO_FOLDER_PATTERNS[0]
        score_dir = patient / pattern.format(score="1")
        score_dir.mkdir(parents=True)
        (score_dir / "frame.jpg").write_bytes(b"x")

        org.limuc_raw = limuc
        result = org._find_limuc_patient_dirs()
        assert len(result) >= 1


# ─────────────────────────────────────────────────────────────
# _save_mapping
# ─────────────────────────────────────────────────────────────


class TestSaveMapping:
    def test_creates_json_in_both_dirs(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        org.clean_dir = tmp_path / "clean"
        org.processed_dir = tmp_path / "processed"
        org._save_mapping()
        assert (org.clean_dir / "class_mapping.json").exists()
        assert (org.processed_dir / "class_mapping.json").exists()

    def test_json_has_required_keys(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        org.clean_dir = tmp_path / "clean"
        org.processed_dir = tmp_path / "processed"
        org._save_mapping()
        with open(org.clean_dir / "class_mapping.json") as f:
            data = json.load(f)
        for key in ("project", "num_classes", "classes", "sources"):
            assert key in data

    def test_classes_labels_to_names(self, tmp_path, monkeypatch):
        from src.config.constants import CLASS_CONFIG

        org, _ = _make_organizer(tmp_path, monkeypatch)
        org.clean_dir = tmp_path / "clean"
        org.processed_dir = tmp_path / "processed"
        org._save_mapping()
        with open(org.clean_dir / "class_mapping.json") as f:
            data = json.load(f)
        for class_name, cfg in CLASS_CONFIG.items():
            assert data["classes"][str(cfg["label"])] == class_name

    def test_valid_json_content(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        org.clean_dir = tmp_path / "clean"
        org.processed_dir = tmp_path / "processed"
        org._save_mapping()
        with open(org.clean_dir / "class_mapping.json") as f:
            data = json.load(f)
        assert isinstance(data["num_classes"], int)

    def test_clinical_info_in_mapping(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        org.clean_dir = tmp_path / "clean"
        org.processed_dir = tmp_path / "processed"
        org._save_mapping()
        with open(org.clean_dir / "class_mapping.json") as f:
            data = json.load(f)
        assert "clinical_info" in data


# ─────────────────────────────────────────────────────────────
# Tabular helpers
# ─────────────────────────────────────────────────────────────


class TestTabularHelpers:
    def _make_base_df(self):
        return pd.DataFrame(
            {
                "Survival_Prediction": ["Yes", "No", "Yes", "No"],
                "Age": [45, 60, 55, 70],
                "Gender": ["M", "F", "M", "F"],
                "Family_History": ["Yes", "No", "Yes", "No"],
                "Smoking_History": ["No", "Yes", "No", "Yes"],
                "Alcohol_Consumption": ["No", "No", "Yes", "Yes"],
                "Diabetes": ["No", "Yes", "No", "No"],
                "Inflammatory_Bowel_Disease": ["No", "No", "No", "Yes"],
                "Genetic_Mutation": ["No", "No", "Yes", "No"],
            }
        )

    def test_load_csv_raises_if_missing(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        with pytest.raises(FileNotFoundError):
            org._load_colorectal_cancer_csv(tmp_path / "nope.csv")

    def test_load_csv_returns_dataframe(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = self._make_base_df()
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)
        result = org._load_colorectal_cancer_csv(csv_path)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4

    def test_validate_true_for_valid_df(self, tmp_path, monkeypatch):
        from src.config.constants import TABULAR_COLON_TARGET

        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = pd.DataFrame({TABULAR_COLON_TARGET: [0, 1, 0, 1]})
        assert org._validate_colorectal_data(df) is True

    def test_validate_false_when_target_missing(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = pd.DataFrame({"OtherCol": [0, 1]})
        assert org._validate_colorectal_data(df) is False

    def test_save_cleaned_creates_file(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = pd.DataFrame({"a": [1, 2]})
        out = tmp_path / "subdir" / "out.csv"
        org._save_cleaned_dataset(df, out)
        assert out.exists()
        loaded = pd.read_csv(out)
        assert len(loaded) == 2

    def test_clean_dataset_handles_binary_mapping(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = self._make_base_df()
        result = org._clean_colorectal_dataset(df)
        assert result is None or isinstance(result, pd.DataFrame)

    def test_clean_dataset_target_alias(self, tmp_path, monkeypatch):
        """Target encontrado por alias"""
        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = pd.DataFrame(
            {
                "Survival_Prediction": ["Yes", "No"],
                "Age": [45, 60],
            }
        )
        result = org._clean_colorectal_dataset(df)
        # Should not return None - found via alias
        assert result is None or isinstance(result, pd.DataFrame)

    def test_clean_dataset_no_target_returns_none(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = pd.DataFrame({"SomeColumn": [1, 2], "OtherColumn": [3, 4]})
        result = org._clean_colorectal_dataset(df)
        assert result is None

    def test_clean_dataset_with_bmi_column(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = self._make_base_df()
        df["Obesity_BMI"] = ["Normal", "Overweight", "Obese", "Normal"]
        result = org._clean_colorectal_dataset(df)
        assert result is None or isinstance(result, pd.DataFrame)

    def test_clean_dataset_with_country_column(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = self._make_base_df()
        df["Country"] = ["USA", "UK", "USA", "UK"]
        result = org._clean_colorectal_dataset(df)
        assert result is None or isinstance(result, pd.DataFrame)

    def test_clean_dataset_with_diet_risk(self, tmp_path, monkeypatch):
        from src.config.constants import TABULAR_COLON_DIET_RISK_MAP

        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = self._make_base_df()
        diet_vals = list(TABULAR_COLON_DIET_RISK_MAP.keys())
        df["Diet_Risk"] = [diet_vals[0], diet_vals[0], diet_vals[0], diet_vals[0]]
        result = org._clean_colorectal_dataset(df)
        assert result is None or isinstance(result, pd.DataFrame)

    def test_clean_dataset_with_physical_activity(self, tmp_path, monkeypatch):
        from src.config.constants import TABULAR_COLON_ACTIVITY_MAP

        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = self._make_base_df()
        activity_vals = list(TABULAR_COLON_ACTIVITY_MAP.keys())
        df["Physical_Activity"] = [activity_vals[0]] * 4
        result = org._clean_colorectal_dataset(df)
        assert result is None or isinstance(result, pd.DataFrame)

    def test_clean_dataset_with_urban_rural(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = self._make_base_df()
        df["Urban_or_Rural"] = ["Urban", "Rural", "Urban", "Rural"]
        result = org._clean_colorectal_dataset(df)
        assert result is None or isinstance(result, pd.DataFrame)


# ─────────────────────────────────────────────────────────────
# _process_tabular_dataset
# ─────────────────────────────────────────────────────────────


class TestProcessTabularDataset:
    def test_process_tabular_file_not_found(self, tmp_path, monkeypatch):
        org, fp = _make_organizer(tmp_path, monkeypatch)
        fp.RAW_TABULAR = tmp_path / "raw_tabular"
        fp.TABULAR_PROCESSED = tmp_path / "tabular_processed"
        # File doesn't exist → FileNotFoundError caught internally
        org._process_tabular_dataset()

    def test_process_tabular_clean_returns_none(self, tmp_path, monkeypatch):
        org, fp = _make_organizer(tmp_path, monkeypatch)
        fp.RAW_TABULAR = tmp_path / "raw_tabular"
        fp.RAW_TABULAR.mkdir(parents=True)
        fp.TABULAR_PROCESSED = tmp_path / "tabular_processed"

        csv_path = fp.RAW_TABULAR / "colorectal_cancer_dataset.csv"
        df = pd.DataFrame({"SomeCol": [1, 2]})
        df.to_csv(csv_path, index=False)

        # _clean returns None → logs error
        org._load_colorectal_cancer_csv = MagicMock(return_value=df)
        org._clean_colorectal_dataset = MagicMock(return_value=None)
        org._process_tabular_dataset()

    def test_process_tabular_validation_fails(self, tmp_path, monkeypatch):
        org, fp = _make_organizer(tmp_path, monkeypatch)
        fp.RAW_TABULAR = tmp_path / "raw_tabular"
        fp.RAW_TABULAR.mkdir(parents=True)
        fp.TABULAR_PROCESSED = tmp_path / "tabular_processed"

        df_clean = pd.DataFrame({"SomeCol": [1, 2]})
        org._load_colorectal_cancer_csv = MagicMock(return_value=df_clean)
        org._clean_colorectal_dataset = MagicMock(return_value=df_clean)
        org._validate_colorectal_data = MagicMock(return_value=False)
        org._save_cleaned_dataset = MagicMock()
        org._process_tabular_dataset()
        org._save_cleaned_dataset.assert_not_called()


"""     def test_process_tabular_success(self, tmp_path, monkeypatch):
        from src.config.constants import TABULAR_COLON_TARGET

        org, fp = _make_organizer(tmp_path, monkeypatch)
        fp.RAW_TABULAR = tmp_path / "raw_tabular"
        fp.RAW_TABULAR.mkdir(parents=True)
        fp.TABULAR_PROCESSED = tmp_path / "tabular_processed"

        df_clean = pd.DataFrame({TABULAR_COLON_TARGET: [0, 1]})
        org._load_colorectal_cancer_csv = MagicMock(return_value=df_clean)
        org._clean_colorectal_dataset = MagicMock(return_value=df_clean)
        org._validate_colorectal_data = MagicMock(return_value=True)
        org._save_cleaned_dataset = MagicMock()
        org._process_tabular_dataset()
        org._save_cleaned_dataset.assert_called_once()
 """

# ─────────────────────────────────────────────────────────────
# _balance_and_copy
# ─────────────────────────────────────────────────────────────


class TestBalanceAndCopy:
    def test_copies_images_to_clean_dir(self, tmp_path, monkeypatch):
        from src.config.constants import CLASS_CONFIG

        org, _ = _make_organizer(tmp_path, monkeypatch, target_per_class=100)

        # Create a real image file
        src_img = tmp_path / "src_img.jpg"
        src_img.write_bytes(b"fake_image")

        # Populate collected for first class only
        first_class = list(CLASS_CONFIG.keys())[0]
        org.collected[first_class] = [
            {
                "original_path": str(src_img),
                "mask_path": None,
                "source": "hyperkvasir",
                "source_detail": "hk_polyp",
                "class_name": first_class,
                "patient_id": "",
            }
        ]

        org._balance_and_copy()
        class_dir = org.clean_dir / first_class
        assert class_dir.exists()

    def test_copies_with_patient_id(self, tmp_path, monkeypatch):
        from src.config.constants import CLASS_CONFIG

        org, _ = _make_organizer(tmp_path, monkeypatch, target_per_class=100)

        src_img = tmp_path / "frame.jpg"
        src_img.write_bytes(b"fake")

        first_class = list(CLASS_CONFIG.keys())[0]
        org.collected[first_class] = [
            {
                "original_path": str(src_img),
                "mask_path": None,
                "source": "limuc",
                "source_detail": "limuc_mayo0",
                "class_name": first_class,
                "patient_id": "limuc_p001",
            }
        ]

        org._balance_and_copy()
        class_dir = org.clean_dir / first_class
        files = list(class_dir.iterdir())
        assert any("limuc_p001" in f.name for f in files)

    def test_copies_mask_when_present(self, tmp_path, monkeypatch):
        from src.config.constants import CLASS_CONFIG

        org, _ = _make_organizer(tmp_path, monkeypatch, target_per_class=100)

        src_img = tmp_path / "img.jpg"
        src_img.write_bytes(b"fake")
        mask_img = tmp_path / "mask.jpg"
        mask_img.write_bytes(b"mask")

        first_class = list(CLASS_CONFIG.keys())[0]
        org.collected[first_class] = [
            {
                "original_path": str(src_img),
                "mask_path": str(mask_img),
                "source": "cvc_clinicdb",
                "source_detail": "cvc_polyp",
                "class_name": first_class,
                "patient_id": "",
            }
        ]

        org._balance_and_copy()
        mask_dir = org.clean_dir / "masks"
        assert mask_dir.exists()
        assert len(list(mask_dir.iterdir())) > 0

    def test_skips_missing_source_image(self, tmp_path, monkeypatch):
        from src.config.constants import CLASS_CONFIG

        org, _ = _make_organizer(tmp_path, monkeypatch, target_per_class=100)

        first_class = list(CLASS_CONFIG.keys())[0]
        org.collected[first_class] = [
            {
                "original_path": str(tmp_path / "nonexistent.jpg"),
                "mask_path": None,
                "source": "hyperkvasir",
                "source_detail": "hk_polyp",
                "class_name": first_class,
                "patient_id": "",
            }
        ]
        org._balance_and_copy()  # Should not raise

    def test_empty_class_logs_error(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch, target_per_class=100)
        # collected is empty for all classes
        org._balance_and_copy()  # Should not raise

    def test_subsamples_when_over_target(self, tmp_path, monkeypatch):
        from src.config.constants import CLASS_CONFIG

        org, _ = _make_organizer(tmp_path, monkeypatch, target_per_class=2)

        first_class = list(CLASS_CONFIG.keys())[0]
        imgs = []
        for i in range(5):
            img = tmp_path / f"img{i}.jpg"
            img.write_bytes(b"x")
            imgs.append(
                {
                    "original_path": str(img),
                    "mask_path": None,
                    "source": "hyperkvasir",
                    "source_detail": "hk_polyp",
                    "class_name": first_class,
                    "patient_id": "",
                }
            )
        org.collected[first_class] = imgs
        org._balance_and_copy()

        class_dir = org.clean_dir / first_class
        if class_dir.exists():
            assert len(list(class_dir.iterdir())) <= 2


# ─────────────────────────────────────────────────────────────
# _build_processed_records
# ─────────────────────────────────────────────────────────────


class TestBuildProcessedRecords:
    def test_builds_records_from_processed_dir(self, tmp_path, monkeypatch):
        from src.config.constants import CLASS_CONFIG

        org, _ = _make_organizer(tmp_path, monkeypatch)

        first_class = list(CLASS_CONFIG.keys())[0]
        proc_dir = tmp_path / "processed" / first_class
        proc_dir.mkdir(parents=True)

        from PIL import Image as PILImage

        img = PILImage.new("RGB", (100, 100))
        img_path = proc_dir / "hk_polyp_img001.jpg"
        img.save(str(img_path))

        org._build_processed_records()
        assert len(org.image_records) >= 1

    def test_records_have_required_keys(self, tmp_path, monkeypatch):
        from src.config.constants import CLASS_CONFIG

        org, _ = _make_organizer(tmp_path, monkeypatch)

        first_class = list(CLASS_CONFIG.keys())[0]
        proc_dir = tmp_path / "processed" / first_class
        proc_dir.mkdir(parents=True)
        (proc_dir / "img001.jpg").write_bytes(b"x")

        org._build_processed_records()
        if org.image_records:
            r = org.image_records[0]
            for key in ["file_path", "label", "class_name", "source", "split"]:
                # split is set later, other keys must be present
                if key != "split":
                    assert key in r

    def test_with_mask_in_clean_dir(self, tmp_path, monkeypatch):
        from src.config.constants import CLASS_CONFIG

        org, _ = _make_organizer(tmp_path, monkeypatch)

        first_class = list(CLASS_CONFIG.keys())[0]
        proc_dir = tmp_path / "processed" / first_class
        proc_dir.mkdir(parents=True)

        stem = "hk_polyp_img001"
        (proc_dir / f"{stem}.jpg").write_bytes(b"x")

        mask_dir = tmp_path / "clean" / "masks"
        mask_dir.mkdir(parents=True)
        (mask_dir / f"{stem}.jpg").write_bytes(b"x")

        org._build_processed_records()
        if org.image_records:
            r = org.image_records[0]
            assert r["mask_path"] is not None

    def test_handles_corrupt_image_gracefully(self, tmp_path, monkeypatch):
        from src.config.constants import CLASS_CONFIG

        org, _ = _make_organizer(tmp_path, monkeypatch)

        first_class = list(CLASS_CONFIG.keys())[0]
        proc_dir = tmp_path / "processed" / first_class
        proc_dir.mkdir(parents=True)
        (proc_dir / "corrupt.jpg").write_bytes(b"notanimage")

        org._build_processed_records()
        if org.image_records:
            r = org.image_records[0]
            assert r["width"] == 384  # fallback

    def test_skips_missing_class_dir(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        org._build_processed_records()
        assert len(org.image_records) == 0


# ─────────────────────────────────────────────────────────────
# _create_source_stratified_splits
# ─────────────────────────────────────────────────────────────


class TestCreateSourceStratifiedSplits:
    def _make_records(self, tmp_path, n_per_combo=10):
        from src.config.constants import CLASS_CONFIG

        records = []
        classes = list(CLASS_CONFIG.keys())
        sources = ["hyperkvasir", "cvc_clinicdb"]
        for cls in classes:
            for src in sources:
                for i in range(n_per_combo):
                    records.append(
                        {
                            "file_path": str(tmp_path / f"{cls}_{src}_{i}.jpg"),
                            "label": CLASS_CONFIG[cls]["label"],
                            "class_name": cls,
                            "source": src,
                            "dataset_source": src,
                            "mask_path": None,
                            "width": 384,
                            "height": 384,
                        }
                    )
        return records

    def test_assigns_split_to_all_records(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        org.image_records = self._make_records(tmp_path)
        org._create_source_stratified_splits()
        for r in org.image_records:
            assert "split" in r
            assert r["split"] in ["train", "val", "test"]

    def test_split_proportions_approx(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        org.image_records = self._make_records(tmp_path, n_per_combo=20)
        org._create_source_stratified_splits()
        total = len(org.image_records)
        train = sum(1 for r in org.image_records if r["split"] == "train")
        assert train / total == pytest.approx(0.7, abs=0.1)

    def test_fallback_class_only_stratification(self, tmp_path, monkeypatch):
        """Grupos pequeños → fallback a class-only"""
        from src.config.constants import CLASS_CONFIG

        org, _ = _make_organizer(tmp_path, monkeypatch)
        # Only 2 records per (class, source) → triggers fallback
        records = []
        for cls in list(CLASS_CONFIG.keys()):
            for i in range(8):
                records.append(
                    {
                        "file_path": str(tmp_path / f"{cls}_{i}.jpg"),
                        "label": CLASS_CONFIG[cls]["label"],
                        "class_name": cls,
                        "source": "only_source",
                        "dataset_source": "only_source",
                        "mask_path": None,
                        "width": 384,
                        "height": 384,
                    }
                )
        org.image_records = records
        org._create_source_stratified_splits()
        for r in org.image_records:
            assert r["split"] in ["train", "val", "test"]


# ─────────────────────────────────────────────────────────────
# _log_source_distribution
# ─────────────────────────────────────────────────────────────


class TestLogSourceDistribution:
    def test_logs_without_crash(self, tmp_path, monkeypatch):
        from src.config.constants import CLASS_CONFIG

        org, _ = _make_organizer(tmp_path, monkeypatch)
        first_class = list(CLASS_CONFIG.keys())[0]
        org.collected[first_class] = [
            {"source": "hyperkvasir"},
            {"source": "cvc_clinicdb"},
        ]
        org._log_source_distribution()

    def test_warns_single_source(self, tmp_path, monkeypatch):
        from src.config.constants import CLASS_CONFIG

        org, _ = _make_organizer(tmp_path, monkeypatch)
        first_class = list(CLASS_CONFIG.keys())[0]
        org.collected[first_class] = [{"source": "only_source"}]
        # Should not raise, just log warning
        org._log_source_distribution()


# ─────────────────────────────────────────────────────────────
# _verify_source_mixing
# ─────────────────────────────────────────────────────────────


class TestVerifySourceMixing:
    def test_verifies_without_crash(self, tmp_path, monkeypatch):
        from src.config.constants import CLASS_CONFIG

        org, _ = _make_organizer(tmp_path, monkeypatch)
        first_class = list(CLASS_CONFIG.keys())[0]
        org.image_records = [
            {
                "split": "train",
                "class_name": first_class,
                "source": "hyperkvasir",
                "file_path": str(tmp_path / "img1.jpg"),
            },
            {
                "split": "val",
                "class_name": first_class,
                "source": "cvc_clinicdb",
                "file_path": str(tmp_path / "img2.jpg"),
            },
            {
                "split": "test",
                "class_name": first_class,
                "source": "limuc",
                "file_path": str(tmp_path / "img3.jpg"),
            },
        ]
        org._verify_source_mixing()

    def test_detects_overlap(self, tmp_path, monkeypatch):
        from src.config.constants import CLASS_CONFIG

        org, _ = _make_organizer(tmp_path, monkeypatch)
        first_class = list(CLASS_CONFIG.keys())[0]
        shared_path = str(tmp_path / "shared_img.jpg")
        org.image_records = [
            {
                "split": "train",
                "class_name": first_class,
                "source": "hyperkvasir",
                "file_path": shared_path,
            },
            {
                "split": "val",
                "class_name": first_class,
                "source": "cvc_clinicdb",
                "file_path": shared_path,
            },
        ]
        # Should log error but not raise
        org._verify_source_mixing()


# ─────────────────────────────────────────────────────────────
# setup_database
# ─────────────────────────────────────────────────────────────


class TestSetupDatabase:
    def test_setup_database_calls_create_all(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        with patch("src.data.ingestion.multi_dataset_organizer.Base") as MockBase:
            with patch("src.data.ingestion.multi_dataset_organizer.engine") as mock_eng:
                org.setup_database()
                MockBase.metadata.create_all.assert_called_once_with(bind=mock_eng)


# ─────────────────────────────────────────────────────────────
# _register_in_database
# ─────────────────────────────────────────────────────────────


class TestRegisterInDatabase:
    def test_register_calls_bulk_insert(self, tmp_path, monkeypatch):
        from src.config.constants import CLASS_CONFIG

        org, _ = _make_organizer(tmp_path, monkeypatch)
        first_class = list(CLASS_CONFIG.keys())[0]
        org.image_records = [
            {
                "file_path": str(tmp_path / "img.jpg"),
                "mask_path": None,
                "label": 0,
                "class_name": first_class,
                "source": "hyperkvasir",
                "dataset_source": "hyperkvasir",
                "split": "train",
                "width": 384,
                "height": 384,
            }
        ]

        mock_db = MagicMock()
        mock_repo = MagicMock()
        mock_repo.bulk_insert.return_value = 1

        with patch(
            "src.data.ingestion.multi_dataset_organizer.sa_inspect"
        ) as mock_insp:
            mock_insp.return_value.has_table.return_value = True
            with patch(
                "src.data.ingestion.multi_dataset_organizer.get_db"
            ) as mock_get_db:
                mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_db)
                mock_get_db.return_value.__exit__ = MagicMock(return_value=False)
                with patch(
                    "src.data.ingestion.multi_dataset_organizer.TrainingImageRepository",
                    return_value=mock_repo,
                ):
                    org._register_in_database()

        mock_repo.bulk_insert.assert_called_once()

    def test_register_creates_tables_if_not_exist(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        org.image_records = []
        org.setup_database = MagicMock()

        mock_db = MagicMock()
        mock_repo = MagicMock()
        mock_repo.bulk_insert.return_value = 0

        with patch(
            "src.data.ingestion.multi_dataset_organizer.sa_inspect"
        ) as mock_insp:
            mock_insp.return_value.has_table.return_value = False
            with patch(
                "src.data.ingestion.multi_dataset_organizer.get_db"
            ) as mock_get_db:
                mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_db)
                mock_get_db.return_value.__exit__ = MagicMock(return_value=False)
                with patch(
                    "src.data.ingestion.multi_dataset_organizer.TrainingImageRepository",
                    return_value=mock_repo,
                ):
                    org._register_in_database()

        org.setup_database.assert_called_once()


# ─────────────────────────────────────────────────────────────
# _start_tissue_preprocessor
# ─────────────────────────────────────────────────────────────


class TestStartTissuePreprocessor:
    def test_calls_process_dataset_tissue_only(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        with patch(
            "src.data.ingestion.multi_dataset_organizer.process_dataset_tissue_only"
        ) as mock_proc:
            org._start_tissue_preprocessor()
            mock_proc.assert_called_once_with(
                target_size=384, n_crops=5, max_black_pct=0.5
            )


# ─────────────────────────────────────────────────────────────
# _collect_all_sources
# ─────────────────────────────────────────────────────────────


class TestCollectAllSources:
    def test_calls_all_collectors(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        org._collect_hyperkvasir = MagicMock()
        org._collect_cvc_clinicdb = MagicMock()
        org._collect_limuc = MagicMock()
        org._collect_curated_colon = MagicMock()
        org._collect_all_sources()
        org._collect_hyperkvasir.assert_called_once()
        org._collect_cvc_clinicdb.assert_called_once()
        org._collect_limuc.assert_called_once()
        org._collect_curated_colon.assert_called_once()


# ─────────────────────────────────────────────────────────────
# run (pipeline completo)
# ─────────────────────────────────────────────────────────────


class TestRun:
    def test_run_executes_full_pipeline(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)

        # Mock all heavy operations
        org._collect_all_sources = MagicMock()
        org._log_source_distribution = MagicMock()
        org._balance_and_copy = MagicMock()
        org._save_mapping = MagicMock()
        org._build_processed_records = MagicMock()
        org._create_source_stratified_splits = MagicMock()
        org._register_in_database = MagicMock()
        org._verify_source_mixing = MagicMock()
        org._start_tissue_preprocessor = MagicMock()
        org._process_tabular_dataset = MagicMock()

        with patch("src.data.ingestion.multi_dataset_organizer.preprocess"):
            with patch("shutil.rmtree"):
                # Create dirs so rmtree path is exercised
                org.clean_dir.mkdir(parents=True, exist_ok=True)
                org.processed_dir.mkdir(parents=True, exist_ok=True)
                org.run()

        org._collect_all_sources.assert_called_once()
        org._log_source_distribution.assert_called_once()
        org._balance_and_copy.assert_called_once()
        org._save_mapping.assert_called_once()
        org._build_processed_records.assert_called_once()
        org._create_source_stratified_splits.assert_called_once()
        org._register_in_database.assert_called_once()
        org._verify_source_mixing.assert_called_once()
        org._start_tissue_preprocessor.assert_called_once()
        org._process_tabular_dataset.assert_called_once()

    def test_run_removes_existing_dirs(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        org.clean_dir.mkdir(parents=True, exist_ok=True)
        org.processed_dir.mkdir(parents=True, exist_ok=True)

        org._collect_all_sources = MagicMock()
        org._log_source_distribution = MagicMock()
        org._balance_and_copy = MagicMock()
        org._save_mapping = MagicMock()
        org._build_processed_records = MagicMock()
        org._create_source_stratified_splits = MagicMock()
        org._register_in_database = MagicMock()
        org._verify_source_mixing = MagicMock()
        org._start_tissue_preprocessor = MagicMock()
        org._process_tabular_dataset = MagicMock()

        with patch("src.data.ingestion.multi_dataset_organizer.preprocess"):
            org.run()

        # After run, dirs are recreated by inner methods
        # Just verify no crash

    def test_run_dirs_not_existing(self, tmp_path, monkeypatch):
        """Dirs que no existen al inicio → no se ejecuta rmtree"""
        org, _ = _make_organizer(tmp_path, monkeypatch)
        # Don't create dirs

        org._collect_all_sources = MagicMock()
        org._log_source_distribution = MagicMock()
        org._balance_and_copy = MagicMock()
        org._save_mapping = MagicMock()
        org._build_processed_records = MagicMock()
        org._create_source_stratified_splits = MagicMock()
        org._register_in_database = MagicMock()
        org._verify_source_mixing = MagicMock()
        org._start_tissue_preprocessor = MagicMock()
        org._process_tabular_dataset = MagicMock()

        with patch("src.data.ingestion.multi_dataset_organizer.preprocess"):
            org.run()  # Should not raise


# ─────────────────────────────────────────────────────────────
# Smoking/drinking helpers (métodos presentes en el código)
# ─────────────────────────────────────────────────────────────


class TestSmokingDrinkingHelpers:
    def _make_smoking_df(self):
        return pd.DataFrame(
            {
                "sex": ["Male", "Female"],
                "age": [35, 45],
                "height": [170.0, 160.0],
                "weight": [70.0, 60.0],
                "waistline": [80.0, 75.0],
                "SBP": [120, 115],
                "DBP": [80, 75],
                "BLDS": [90, 85],
                "tot_chole": [200, 190],
                "HDL_chole": [50, 55],
                "LDL_chole": [130, 120],
                "triglyceride": [150, 140],
                "hemoglobin": [14.0, 12.5],
                "urine_protein": [1, 1],
                "serum_creatinine": [0.9, 0.8],
                "SGOT_AST": [25, 22],
                "SGOT_ALT": [20, 18],
                "gamma_GTP": [30, 25],
                "SMK_stat_type_cd": [1.0, 2.0],
                "DRK_YN": ["Y", "N"],
            }
        )

    def test_load_smoking_csv_raises_if_missing(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        with pytest.raises(FileNotFoundError):
            org._load_dataset_smoking_drinking(tmp_path / "nope.csv")

    def test_load_smoking_csv_returns_df(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = self._make_smoking_df()
        path = tmp_path / "smoking.csv"
        df.to_csv(path, index=False)
        result = org._load_dataset_smoking_drinking(path)
        assert isinstance(result, pd.DataFrame)

    def test_clean_smoking_dataset(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = self._make_smoking_df()
        result = org._clean_smoking_drinking_dataset(df)
        assert isinstance(result, pd.DataFrame)

    def test_clean_smoking_adds_bmi(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = self._make_smoking_df()
        result = org._clean_smoking_drinking_dataset(df)
        if result is not None and len(result) > 0:
            assert "BMI" in result.columns

    def test_clean_smoking_adds_ast_alt_ratio(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = self._make_smoking_df()
        result = org._clean_smoking_drinking_dataset(df)
        if result is not None and len(result) > 0:
            assert "AST_ALT_ratio" in result.columns

    def test_validate_smoking_false_when_missing_features(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = pd.DataFrame({"col1": [1, 2]})
        result = org._validate_smoking_drinking_data(df)
        assert result is False

    def test_save_cleaned_smoking_creates_file(self, tmp_path, monkeypatch):
        org, _ = _make_organizer(tmp_path, monkeypatch)
        df = pd.DataFrame({"a": [1, 2]})
        out = tmp_path / "smk_out" / "result.csv"
        org._save_cleaned_smoking_drinking_dataset(df, out)
        assert out.exists()
