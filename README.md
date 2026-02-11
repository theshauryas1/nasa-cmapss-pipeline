# NASA C-MAPSS Scientific Data Processing Pipeline

A reproducible data pipeline for analyzing NASA turbofan engine degradation data, with emphasis on statistical validation, scalability, and interpretability.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Abstract

This project implements a comprehensive analysis pipeline for the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) turbofan engine degradation dataset. The pipeline demonstrates scientific computing best practices by:

1. **Statistical analysis before machine learning** to understand sensor-degradation relationships
2. **Multiple modeling approaches** (Linear Regression, Neural Network, LSTM) for Remaining Useful Life prediction
3. **Explainability analysis** using SHAP values to validate model interpretations
4. **Reproducible methodology** with clear documentation and modular code

---

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Citation](#citation)

---

## Dataset

### NASA C-MAPSS FD001

| Property | Value |
|----------|-------|
| Source | NASA Prognostics Center of Excellence |
| Operating Conditions | 1 (sea level) |
| Fault Modes | 1 (HPC degradation) |
| Training Engines | 100 |
| Test Engines | 100 |
| Features | 21 sensors + 3 operational settings |
| Target | Remaining Useful Life (RUL) |

**Download**: See [data/DOWNLOAD_INSTRUCTIONS.md](data/DOWNLOAD_INSTRUCTIONS.md)

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/nasa-cmapss-pipeline.git
cd nasa-cmapss-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **Scientific Computing**: NumPy, Pandas, SciPy
- **Machine Learning**: scikit-learn, PyTorch
- **Visualization**: Matplotlib, Seaborn
- **Explainability**: SHAP
- **Notebooks**: Jupyter

---

## Project Structure

```
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── data/
│   ├── raw/                     # Original C-MAPSS files
│   │   ├── train_FD001.txt
│   │   ├── test_FD001.txt
│   │   └── RUL_FD001.txt
│   ├── processed/               # Cleaned & engineered features
│   └── DOWNLOAD_INSTRUCTIONS.md # Download guide
│
├── notebooks/
│   └── analysis_pipeline.ipynb  # Main reproducible notebook
│
├── src/
│   ├── data/
│   │   ├── ingestion.py         # Data loading & validation
│   │   └── preprocessing.py     # Cleaning & feature engineering
│   │
│   ├── analysis/
│   │   ├── statistical.py       # Statistical analysis functions
│   │   └── correlation.py       # Correlation & trend analysis
│   │
│   ├── models/
│   │   ├── baseline.py          # Linear Regression baseline
│   │   ├── ensemble.py          # Random Forest (for SHAP)
│   │   ├── neural_network.py    # Neural Network & LSTM
│   │   └── evaluation.py        # Metrics & evaluation
│   │
│   ├── explainability/
│   │   └── shap_analysis.py     # SHAP value computation
│   │
│   └── visualization/
│       └── plots.py             # Scientific plotting functions
│
├── reports/
│   ├── figures/                 # Generated plots
│   └── technical_report.md      # Summary report
│
└── tests/
    └── test_pipeline.py         # Unit tests
```

---

## Usage

### Quick Start

1. **Download the dataset** following [data/DOWNLOAD_INSTRUCTIONS.md](data/DOWNLOAD_INSTRUCTIONS.md)

2. **Run the analysis notebook**:
   ```bash
   cd notebooks
   jupyter notebook analysis_pipeline.ipynb
   ```

3. **Or use the modules directly**:

```python
from src.data.ingestion import CMAPSSDataLoader, compute_training_rul
from src.data.preprocessing import DataPreprocessor
from src.models.neural_network import LSTMModel

# Load data
loader = CMAPSSDataLoader('data/raw')
train_df, test_df, rul_df = loader.load_dataset('FD001')
train_df = compute_training_rul(train_df)

# Preprocess
preprocessor = DataPreprocessor()
train_processed = preprocessor.fit_transform(train_df)

# Train LSTM
lstm = LSTMModel(input_dim=14, hidden_dim=64, num_layers=2)
X_seq, y_seq = LSTMModel.prepare_sequences(train_processed, feature_cols, sequence_length=30)
lstm.fit(X_seq, y_seq, epochs=50)

# Predict
predictions = lstm.predict(X_test_seq)
```

---

## Methodology

### 1. Data Ingestion & Validation
- Load multi-file dataset with schema validation
- Handle missing/corrupted records
- Compute piecewise-linear RUL with cap at 125 cycles

### 2. Preprocessing & Feature Engineering
- Min-Max normalization
- Rolling window statistics (5, 10 cycle windows)
- Degradation indicators (rate of change, EWM)
- Drop constant sensors (zero variance)

### 3. Statistical Analysis
- **Sensor-RUL Correlation**: Identify predictive sensors
- **Monotonicity Analysis**: Find consistent degradation trends
- **Distribution Shifts**: Compare healthy vs. degraded states
- **PCA**: Understand dimensionality structure

### 4. Machine Learning Models

| Model | Description | Key Strength |
|-------|-------------|--------------|
| Linear Regression | Ridge-regularized baseline | Interpretable coefficients |
| Neural Network | 3-layer feedforward | Non-linear patterns |
| LSTM | 2-layer recurrent | Temporal dependencies |

### 5. Explainability (SHAP)
- Global feature importance
- Local prediction explanations
- Feature importance evolution over lifecycle
- Overfitting diagnostics

### 6. Evaluation Metrics
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **NASA Score**: Asymmetric penalty (penalizes late predictions)

---

## Results

### Model Performance (FD001 Validation Set)

| Model | RMSE (cycles) | MAE (cycles) | R² |
|-------|---------------|--------------|-----|
| Linear Regression | ~25 | ~20 | ~0.65 |
| Neural Network | ~22 | ~17 | ~0.72 |
| LSTM | ~20 | ~15 | ~0.78 |

*Note: Exact values depend on random seed and hyperparameters.*

### Key Findings

1. **Sensors 4, 11, 12, 15** show strongest correlation with RUL
2. **Sensors 1, 5, 6, 10, 16, 18, 19** are constant and can be dropped
3. **LSTM** performs best by leveraging temporal patterns
4. **SHAP analysis** confirms model uses physically meaningful features

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@software{cmapss_pipeline,
  author = {Your Name},
  title = {NASA C-MAPSS Scientific Data Processing Pipeline},
  year = {2026},
  url = {https://github.com/yourusername/nasa-cmapss-pipeline}
}
```

### Original Dataset Citation

```bibtex
@inproceedings{saxena2008damage,
  title={Damage propagation modeling for aircraft engine run-to-failure simulation},
  author={Saxena, Abhinav and Goebel, Kai and Simon, Don and Eklund, Neil},
  booktitle={2008 International Conference on Prognostics and Health Management},
  pages={1--9},
  year={2008},
  organization={IEEE}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- NASA Prognostics Center of Excellence for the C-MAPSS dataset
- SHAP library authors for explainability tools
- scikit-learn and PyTorch communities
