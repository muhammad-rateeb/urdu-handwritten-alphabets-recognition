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
