# Urdu Handwritten Alphabets Recognition

Handwritten Urdu alphabet recognition using DNN and CNN on student-collected images.

## Overview
This repository contains code, dataset samples, and reports for a course project that builds models to recognize handwritten Urdu alphabets captured with a mobile phone camera.

## Features
- DNN baseline (2 dense hidden layers)
- CNN baseline (2 convolutional layers)
- Preprocessing and augmentation (rotation, translation, zoom, contrast, gaussian noise)
- Experiments with multiple input sizes (28×28, 64×64, 128×128)
- Evaluation: accuracy, precision/recall/F1, confusion matrices

## Technologies
- Python 3.11
- TensorFlow / Keras
- Pillow (PIL), scikit-learn, matplotlib, seaborn
- FPDF for report generation

## Quick Start
1. Create a Python virtual environment and install dependencies (example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

2. Prepare dataset (if not present): place unaugmented submission images in `2023cs607(A)/alphabets_128x128` or use `dataset/` structure shown in the repo.

3. Train models (example, 128×128):

```powershell
python run_all.py --dataset dataset --img-size 128 --batch-size 32 --epochs 30
```

4. Generated outputs (models, reports) are saved under `outputs*` directories.

## Dataset
- The repository includes a small student-collected dataset and a submission folder `2023cs607(A)` with single-sample images (pre-augmentation).
- Image sizes included: 128×128 (also stored: 64×64, 28×28 outputs).
- Note: Teacher requested 34–35 submission images; this project contains 40 images in `2023cs607(A)/alphabets_128x128`. Confirm with your instructor if needed.

## Results
- See `report.pdf` and `report.md` for evaluation details and comparative analysis (best reported model: CNN at 128×128 in this project).

## Files of interest
- `run_all.py` — training and evaluation script
- `source_code_bundle.txt` — all source code in single text file (submission requirement)
- `report.pdf` — 3–4 page project report (course requirement)
- `2023cs607(A)/alphabets_128x128` — submission images (pre-augmentation)

## License
This project is released under the MIT License — see `LICENSE`.

## Contact
Submitted by: Muhammad Rateeb (2023-CS-607)
Email: (add your contact email here)
# ML CCP - Handwritten Urdu Alphabet Recognition

## What this project does
- Trains two models (DNN + CNN) on 40-class Urdu alphabet dataset
- Applies preprocessing and 5 data augmentations
- Saves metrics, confusion matrices, and trained models
- Generates a short report

## Run training
```
.venv311\\Scripts\\python.exe run_all.py --img-size 128 --epochs 30
```

Outputs are written to the `outputs/` folder.

## Generate report
```
.venv311\\Scripts\\python.exe generate_report.py
```

This creates `report.md` and `report.pdf`.
