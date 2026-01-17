# NASA C-MAPSS Dataset Download Instructions

## Overview

The **C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset is a benchmark dataset for predictive maintenance and prognostics research. It simulates turbofan engine run-to-failure trajectories.

---

## Download Options

### Option 1: Kaggle (Recommended)

The dataset is available on Kaggle at:
**https://www.kaggle.com/datasets/behrad3d/nasa-cmaps**

#### Steps:

1. **Create a Kaggle account** (if you don't have one):
   - Go to https://www.kaggle.com
   - Sign up for a free account

2. **Download the dataset**:
   - Visit: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
   - Click the **"Download"** button (top right)
   - This will download `archive.zip`

3. **Extract and move files**:
   - Extract `archive.zip`
   - Copy these files to `data/raw/`:
     ```
     train_FD001.txt
     test_FD001.txt
     RUL_FD001.txt
     ```

---

### Option 2: Alternative Kaggle Links

If the above link doesn't work, try these alternatives:

- https://www.kaggle.com/datasets/danieeeld/nasa-cmapss-dataset
- https://www.kaggle.com/datasets/suraj520/aerospace-turbofan-engine-degradation-dataset

---

### Option 3: Direct Download (if available)

NASA's Prognostics Data Repository:
- https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

> Note: Direct download from NASA may not always be available.

---

## Expected File Structure

After downloading, your `data/raw/` folder should contain:

```
data/raw/
├── train_FD001.txt    (~2.5 MB)
├── test_FD001.txt     (~1.2 MB)
└── RUL_FD001.txt      (~1 KB)
```

---

## File Descriptions

| File | Description | Rows | Engines |
|------|-------------|------|---------|
| `train_FD001.txt` | Complete run-to-failure trajectories | ~20,631 | 100 |
| `test_FD001.txt` | Partial trajectories (stopped before failure) | ~13,096 | 100 |
| `RUL_FD001.txt` | Ground truth RUL for test engines | 100 | 100 |

---

## Data Format

Each row in the data files contains 26 columns (space-separated, no header):

| Columns | Description |
|---------|-------------|
| 1 | Engine unit number (1-100) |
| 2 | Time cycle (1 to max lifecycle) |
| 3-5 | Operational settings (altitude, Mach, throttle) |
| 6-26 | Sensor measurements (21 sensors) |

---

## FD001 Dataset Characteristics

| Property | Value |
|----------|-------|
| Operating Conditions | 1 (sea level) |
| Fault Modes | 1 (HPC degradation) |
| Training Engines | 100 |
| Test Engines | 100 |

---

## Verification

After placing files in `data/raw/`, run this to verify:

```python
from src.data.ingestion import CMAPSSDataLoader

loader = CMAPSSDataLoader('data/raw')
train, test, rul = loader.load_dataset('FD001')
loader.print_summary(train, test, rul)
```

Expected output:
```
==================================================
NASA C-MAPSS Dataset Summary
==================================================

📊 TRAINING DATA
  • Number of engines: 100
  • Total samples: 20,631
  • Lifecycle range: 128 - 362 cycles
  • Average lifecycle: 206.3 cycles

🧪 TEST DATA
  • Number of engines: 100
  • Total samples: 13,096
  ...
```

---

## Troubleshooting

### "FileNotFoundError"
- Ensure files are in `data/raw/` directory
- Check file names match exactly (case-sensitive on Linux/Mac)

### Encoding Issues
- Files should be plain text, space-separated
- If downloaded as CSV, rename to .txt

### Missing Columns
- Verify files have 26 columns per row
- Some unofficial sources may have modified data

---

## Citation

If using this dataset in research, please cite:

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
