# 🌿 ENDO-AID: Clinical Decision Support System for Colorectal Cancer

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)
![PyTorch](https://img.shields.io/badge/DL-PyTorch-EE4C2C)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-blue)
![PostgreSQL](https://img.shields.io/badge/DB-PostgreSQL_16-336791)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2)
![Docker](https://img.shields.io/badge/Infra-Docker-2496ED)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

Endo-AID is an end-to-end AI-powered diagnostic simulator for colorectal cancer (CRC) detection, combining deep learning image classification, U-Net polyp segmentation, and tabular risk scoring into a unified multimodal diagnosis engine.

The system processes colonoscopy images from 4 medical datasets, applies an Adaptive Ensemble with Grad-CAM attention gating, and provides explainable results through an interactive clinical dashboard.

## 🏗️ Architecture Overview

The project follows a Domain-Driven Modular Architecture with centralized configuration and strict separation of concerns.

```
CRC-Diagnostic/
├── 📂 src/
│ ├── 📂 config/ # Centralized configuration (SSOT)
│ │ ├── constants.py # All maps, features, thresholds
│ │ ├── paths.py # All directory paths (auto-created)
│ │ ├── settings.py # Environment-based settings (Pydantic)
│ │ └── logger.py # Loguru structured logging
│ │
│ ├── 📂 core/ # Domain interfaces & shared types
│ │ ├── interfaces.py # ABCs: BaseTrainer, BaseClassifier, etc.
│ │ └── types.py # TypedDicts, dataclasses
│ │
│ ├── 📂 data/ # Data layer
│ │ ├── 📂 ingestion/ # External data acquisition
│ │ │ ├── kaggle_loader.py # Kaggle dataset download
│ │ │ ├── gdc_client.py # GDC API (TCGA-COAD)
│ │ │ └── multi_dataset_organizer.py # 4-source mixing pipeline
│ │ │
│ │ └── 📂 processing/ # Data preprocessing
│ │ ├── image_preprocessor.py # Augmentation & Dataset
│ │ ├── standardizer.py # Multi-source FOV standardization
│ │ ├── tissue_only_preprocessor.py # Zero-padding tissue crops
│ │ ├── tabular_preprocessor.py # Clinical feature engineering
│ │ ├── synthetic_generator.py # Realistic patient generation
│ │ └── balancer.py # SMOTE/ADASYN balancing
│ │
│ ├── 📂 database/ # PostgreSQL ORM layer
│ │ ├── connection.py # SQLAlchemy engine & sessions
│ │ ├── models.py # Patient, Visit, TrainingImage
│ │ └── repositories.py # CRUD repositories
│ │
│ ├── 📂 models/ # ML/DL model definitions
│ │ ├── image_classifier.py # EfficientNetV2-S (3-class)
│ │ ├── tissue_classifier.py # Same backbone, tissue-only input
│ │ ├── polyp_segmenter.py # U-Net + EfficientNet-B4
│ │ ├── tabular_model.py # XGBoost cancer risk scorer
│ │ └── ensemble_predictor.py # Adaptive attention-gated ensemble
│ │
│ ├── 📂 training/ # Training pipelines
│ │ ├── tracking.py # MLflow wrapper (graceful degradation)
│ │ ├── train_image_classifier.py # Model A trainer
│ │ ├── train_tissue_classifier.py # Model B trainer
│ │ ├── train_segmenter.py # U-Net trainer
│ │ └── train_tabular.py # XGBoost trainer
│ │
│ ├── 📂 evaluation/ # Model evaluation & XAI
│ │ ├── gradcam.py # Grad-CAM multi-source analysis
│ │ ├── attention.py # Attention ratio metrics
│ │ └── explainability.py # SHAP + image overlay reports
│ │
│ ├── 📂 diagnosis/ # Inference engine
│ │ └── engine.py # Multimodal fusion + adaptive ensemble
│ │
│ └── 📂 api/ # REST API
│ ├── app.py # FastAPI entry point
│ ├── schemas.py # Pydantic request/response models
│ └── routes/
│ ├── patients.py # Patient CRUD
│ ├── uploads.py # Colonoscopy image upload
│ └── diagnosis.py # AI diagnosis endpoint
│
├── 📂 frontend/ # Streamlit clinical dashboard
│ ├── main.py
│ ├── assets/styles.css
│ ├── components/
│ │ ├── sidebar.py
│ │ ├── cards.py
│ │ ├── banners.py
│ │ ├── charts.py
│ │ └── loading.py
│ ├── pages/
│ │ ├── 01_pacientes.py
│ │ ├── 02_cribado.py
│ │ └── 03_endoscopia.py
│ └── utils/
│ ├── api_client.py
│ └── helpers.py
│
├── 📂 docker/
│ └── docker-compose.yml # PostgreSQL + pgAdmin
│
├── .env # Environment variables
├── pyproject.toml
```

##

### 🔬 Image Analysis (Deep Learning)

🔬 Image Analysis (Deep Learning)

3-Class Classification: Normal | Polyp | Inflammation (Ulcerative Colitis)
Adaptive Ensemble: Model A (context-aware) + Model B (tissue-only) with per-image attention gating
Grad-CAM Explainability: Visual heatmaps showing where the model focuses

### Anti-shortcut Design: Multi-source mixing from 4 datasets breaks frame/border artifact leakage

📍 Polyp Segmentation
U-Net + EfficientNet-B4 encoder for precise polyp localization
TTA (Test-Time Augmentation) reduces over-segmentation
Instance separation via connected components post-processing
Boundary-aware loss (Dice + BCE + boundary weighting)

###

🧪 Clinical Screening (Tabular)
XGBoost risk classifier with 14 blood markers + 10 binary risk factors
Synthetic patient generation with clinically verified overlap distributions
SHAP explainability for per-patient risk factor analysis
Youden's J threshold optimization for recall-first medical context

###

🏥 Multimodal Fusion
Image + Tabular scores combined with clinical priority rules
Polyp detection overrides tabular score (clinical urgency)
Full patient history tracking with longitudinal risk evolution

### 🧠 Model Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ DATA INGESTION │
│ Kaggle (4 datasets) → Multi-Source Organizer → Stratified Split│
│ HyperKvasir + CVC-ClinicDB + LIMUC + Curated Colon │
└──────────────────────────┬──────────────────────────────────────┘
│
┌────────────┼────────────┐
▼ ▼ ▼
┌────────────┐ ┌────────────┐ ┌──────────┐
│ Model A │ │ Model B │ │ U-Net │
│ Context │ │ Tissue-Only│ │ Segmenter│
│ EfficientV2│ │ EfficientV2│ │ EffNet-B4│
│ (inscribed │ │ (slide+ │ │ (masks) │
│ crop) │ │ shrink) │ │ │
└─────┬──────┘ └─────┬──────┘ └─────┬────┘
│ │ │
▼ ▼ │
┌─────────────────────────┐ │
│ Adaptive Ensemble │ │
│ Grad-CAM → α,β │ │
│ Attention Gating │ │
└───────────┬─────────────┘ │
│ │
▼ ▼
┌─────────────────────────────────────┐
│ Diagnosis Engine │
│ Image + Tabular + Segmentation │
│ → Final Risk Score + Report │
└─────────────────────────────────────┘
```

## 🛠️ Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Kaggle API configured (~/.kaggle/kaggle.json)
- (Optional) NVIDIA GPU with CUDA for training
- (Optional) MLflow server for experiment tracking

### 1️⃣ Clone & Install

```bash
git clone https://github.com/your-repo/CRC-Diagnostic.git
cd CRC-Diagnostic
uv venv
source .venv/bin/activate # Linux/Mac

# .venv\Scripts\activate # Windows

uv sync
```

### 2️⃣ Environment Configuration

Create a .env file in the project root:

```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=colon_diagnosis
DB_USER=admin
DB_PASSWORD=admin123

# Kaggle (ensure ~/.kaggle/kaggle.json exists)
# KAGGLE_USERNAME=your_username
# KAGGLE_KEY=your_key
```

### 3️⃣ Start Infrastructure

```bash
# Start PostgreSQL + pgAdmin
docker compose -f docker/docker-compose.yml up -d

# Verify database is running
docker ps # Should show colon_cancer_db healthy
pgAdmin available at http://localhost:5050 (admin@colon.com / admin123)
```

### 4️⃣ Data Pipeline

Run in order — each step depends on the previous:

```bash
# Step 1: Download datasets from Kaggle (requires kaggle.json)
uv run src.data.ingestion.kaggle_loader

# Step 2: Organize, preprocess, split, register in DB + tissue preprocessing
uv run src.data.ingestion.multi_dataset_organizer
```

⚠️ Step 2 is the main pipeline. It executes:

- Multi-source collection (4 datasets)
- Source-balanced subsampling
- FOV standardization + mask alignment
- Stratified train/val/test split (by class + source)
- Database registration
- Tissue-only preprocessing (zero-padding crops)

### 5️⃣ Model Training

Each model trains independently. Order doesn't matter, but all 4 are needed for full functionality:

```bash
# Image Classifier (Model A) — ~25 epochs, GPU recommended
uv run src.training.train_image_classifier

# Tissue-Only Classifier (Model B) — ~25 epochs, GPU recommended
uv run src.training.train_tissue_classifier

# Polyp Segmenter (U-Net) — ~50-100 epochs, GPU recommended
uv run src.training.train_segmenter

# Tabular Risk Model (XGBoost) — ~2 min, CPU only
uv run  src.training.train_tabular
```

MLflow Tracking (optional): Start `mlflow server --port 5000` before training to track experiments at http://localhost:5000.

### 6️⃣ Launch Application

Start both the API backend and the Streamlit frontend:

```bash
# Terminal 1: API Backend
uvicorn src.api.app:app --reload --port 8000

# Terminal 2: Streamlit Frontend
streamlit run frontend/main.py
```

- API: http://localhost:8000 (Swagger docs at /docs)
- Frontend: http://localhost:8501

## 📊 Model Performance

### Image Classification (Test Set)

| Class        | Precision | Recall | F1-Score |
| ------------ | --------- | ------ | -------- |
| Normal       | 0.93      | 0.95   | 0.94     |
| Polyp        | 0.91      | 0.89   | 0.90     |
| Inflammation | 0.88      | 0.87   | 0.87     |

Macro Avg 0.91 0.90 0.90

### Polyp Segmentation

| Metric         | Score |
| -------------- | ----- |
| Dice           | 0.87  |
| IoU (Jaccard)  | 0.79  |
| Pixel Accuracy | 0.96  |

### Tabular Risk Model

| Metric    | Score |
| --------- | ----- |
| AUC-ROC   | 0.86  |
| Recall    | 0.91  |
| Precision | 0.78  |
| F1-Score  | 0.84  |

Metrics are from the held-out test set with multi-source stratification. Actual values may vary depending on training run.

## 🗂️ Data Sources

| Dataset        | Source              | Content                               | Role                                     |
| -------------- | ------------------- | ------------------------------------- | ---------------------------------------- |
| HyperKvasir    | Simula Research     | Labeled GI images + polyp masks       | Normal, Polyp, Inflammation              |
| CVC-ClinicDB   | CVC Barcelona       | Polyp images + ground truth masks     | Polyp (different endoscope)              |
| LIMUC          | University Hospital | UC images by Mayo score (0-3)         | Normal (Mayo 0), Inflammation (Mayo 1-3) |
| Curated Colon  | Kaggle              | Classified colon images               | Polyp diversity                          |
| TCGA-COAD/READ | GDC API             | Clinical data (cancer patients)       | Tabular reference                        |
| Synthetic      | Generated           | 10,000 patients with clinical overlap | Tabular training (balanced)              |

## 🔧 Configuration

All configuration is centralized in `src/config/`:

| File         | Purpose                                                                             |
| ------------ | ----------------------------------------------------------------------------------- |
| constants.py | Class mappings, feature lists, source maps, clinical defaults, physiological ranges |
| paths.py     | All directory paths (auto-created on import)                                        |
| settings.py  | Environment-based settings via Pydantic (DB, model hyperparams, API, thresholds)    |
| logger.py    | Loguru console + rotating file logging                                              |

## 🐳 Docker Services

| Service       | Port | Credentials                |
| ------------- | ---- | -------------------------- |
| PostgreSQL 16 | 5432 | admin / admin123           |
| pgAdmin 4     | 5050 | admin@colon.com / admin123 |

```bash
# Start
docker compose -f docker/docker-compose.yml up -d

# Stop
docker compose -f docker/docker-compose.yml down

# Reset database (delete volume)
docker compose -f docker/docker-compose.yml down -v
```

## 📁 Key Design Decisions

| Decision                                 | Rationale                                                                            |
| ---------------------------------------- | ------------------------------------------------------------------------------------ |
| 4 datasets mixed per class               | Breaks frame/border artifact shortcuts — model can't use endoscope-specific patterns |
| Same backbone for Model A & B            | Diversity from preprocessing, not architecture (Mehrtash et al. 2020)                |
| Adaptive attention gating                | Per-image α/β weights based on where Model A's Grad-CAM focuses                      |
| Tissue-only preprocessing (zero padding) | Slide→Shrink→Resize removes all border artifacts                                     |
| Synthetic patients with clinical overlap | Realistic difficulty: LogReg AUC ≤0.90, prevents trivial separation                  |
| MLflow behind wrapper                    | Training works even without MLflow installed                                         |
| Centralized constants                    | Single source of truth eliminates magic values scattered across files                |

## 👥 Authors

- Nadeem Rashid
- Nil Arroyo
- David Siles

## 📄 License

This project is for research and educational purposes only.

Not intended for clinical use without proper medical validation.

See the LICENSE file for details.
