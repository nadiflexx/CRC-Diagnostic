# 🌿 ENDO-AID: AI Clinical Decision Support System for Colorectal Cancer

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)
![PyTorch](https://img.shields.io/badge/DL-PyTorch-EE4C2C)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-blue)
![LightGBM](https://img.shields.io/badge/ML-LightGBM-green)
![PostgreSQL](https://img.shields.io/badge/DB-PostgreSQL-336791)
![Docker](https://img.shields.io/badge/Infra-Docker-2496ED)

---

## 🧠 Overview

**Endo-AID** is a multimodal clinical decision support system (CDSS) for colorectal cancer detection.

It combines:

- 🖼️ Computer Vision → Colonoscopy image classification & segmentation
- 🧪 Clinical Data Analysis → Tabular risk prediction models

The system acts as a diagnostic simulator designed to support clinicians in decision-making.

---

## 🎯 Key Features

### 🔬 Image Analysis

- 3-class classification: Normal | Polyp | Inflammation
- Dual-model system:
  - Model A → Context-aware
  - Model B → Tissue-only
- Adaptive Ensemble (attention-based)
- Grad-CAM explainability

### 📍 Segmentation

- U-Net + EfficientNet-B4
- Dice ~0.92
- Automatic activation when polyp detected

### 🧪 Tabular Models

- Tumoral risk model (XGBoost)
- Smoking/Drinking reverse model (LightGBM)
- SHAP explainability

### 🔗 Multimodal Diagnosis

- Risk scoring with confidence
- Patient-level analysis

---

## 🏗️ Architecture

- Backend: FastAPI
- Frontend: Streamlit
- Database: PostgreSQL
- ML/DL: PyTorch + XGBoost, LightGBM
- Tracking: MLflow

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/nadiflexx/CRC-Diagnostic.git
cd crc-diagnostic
```

### 2. Install

```bash
uv sync
```

### 3. Environment

Create `.env`:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=colon_diagnosis
DB_USER=admin
DB_PASSWORD=admin123

KAGGLE_USERNAME="your-kaggle-username-here"
KAGGLE_KEY="your-kaggle-key-here"
```

### 4. Start DB

```bash
docker-compose -f docker/docker-compose.yml up -d
```

### 5. Run App

Backend:

```bash
uv run uvicorn src.api.app:app --reload --port 8000
```

Frontend:

```bash
streamlit run frontend/main.py
```

---

## 📊 Performance (Summary)

- Classification F1 ≈ 0.95
- Segmentation Dice ≈ 0.92
- Tabular AUC ≈ 0.97

---

## 📁 Project Structure

```
crc-diagnostic/
├── src/
├── frontend/
├── models/
├── data/
├── docker/
└── pyproject.toml
```

---

## 👥 Authors

- Nadeem Rashid
- David Siles
- Nil Arroyo

---

## 📄 License

Research & educational purposes only.  
Not intended for clinical use.
