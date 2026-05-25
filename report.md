# Handwritten Urdu Alphabets Recognition System

## 1. Problem Statement and Scope
Build a system to recognize handwritten Urdu alphabets from mobile-camera images collected by the student. The task is a 40-class image classification problem with uniform class labels and a fixed image size input. The scope is limited to single-character recognition with no word-level context, and success is measured on a held-out test set using standard classification metrics.

## 2. Dataset
- Total classes: 40
- Images per class: 72
- Image size: 64x64 (also evaluated at 28x28 and 128x128 for comparison)
- Data split: stratified train/validation/test split (70/15/15)

## 3. Data Preparation and Preprocessing
- Grayscale conversion to reduce color noise and focus on strokes
- Normalization to 0-1 for stable optimization
- Augmentation for robustness to capture variations in handwriting and camera conditions:
	- Rotation and translation for pose variance
	- Zoom for scale variance
	- Contrast adjustment for lighting variance
	- Gaussian noise for sensor noise and blur

The split is stratified to preserve class balance across sets, reducing leakage and ensuring a fair evaluation.

## 4. Model Selection and Implementation
Two baseline families were selected to compare a simple dense classifier with a spatial model:
- DNN: 2 hidden layers (512, 256) with dropout for regularization
- CNN: 2 convolution layers (32, 64) with pooling and dropout to learn local stroke patterns

Rationale: CNNs are expected to perform better on images because they preserve spatial structure, while DNNs provide a baseline to quantify how much spatial modeling helps.

## 5. Evaluation and Metrics
Metrics include test accuracy and loss, with per-class precision/recall/F1 and confusion matrices for detailed error analysis.

Results by input size:
- 28x28: DNN accuracy 0.0255, CNN accuracy 0.0278
- 64x64: DNN accuracy 0.0255, CNN accuracy 0.0579
- 128x128: DNN accuracy 0.0255, CNN accuracy 0.1134

Best overall model: CNN at 128x128 (test accuracy 0.1134).

Detailed metrics (128x128):
- CNN: macro avg precision 0.1353, recall 0.1116, F1 0.0757; weighted avg precision 0.1371, recall 0.1134, F1 0.0768
- DNN: macro avg precision 0.0006, recall 0.0250, F1 0.0012; weighted avg precision 0.0006, recall 0.0255, F1 0.0013

Per-class behavior highlights (128x128 CNN):
- Higher recall appears for a few classes such as class 37 (recall 1.0000), class 32 (recall 0.7273), and class 13 (recall 0.5455), but precision remains low, indicating confusion with visually similar classes.
- Many classes still show zero precision/recall, showing the model is not yet robust across all characters.

Compact metrics summary:

| Input size | Model | Accuracy | Macro F1 |
|-----------:|:------|---------:|---------:|
| 28x28 | DNN | 0.0255 | 0.0012 |
| 28x28 | CNN | 0.0278 | 0.0035 |
| 64x64 | DNN | 0.0255 | 0.0012 |
| 64x64 | CNN | 0.0579 | 0.0492 |
| 128x128 | DNN | 0.0255 | 0.0012 |
| 128x128 | CNN | 0.1134 | 0.0757 |

## 6. Comparative Analysis
CNN consistently outperforms DNN across all image sizes, and performance improves as resolution increases. This supports the hypothesis that spatial feature learning is critical for distinguishing similar Urdu characters. The DNN stays near chance level and tends to collapse into predicting a small subset of labels, which is reflected in near-zero macro F1.

Trade-offs:
- Accuracy vs. complexity: CNN yields higher accuracy but is more compute-intensive than DNN.
- Resolution vs. speed: 128x128 improves accuracy but increases training time and inference cost.
- Consistency vs. coverage: CNN improves for some classes but still misses many, indicating the need for more data or stronger architectures.

## 7. Limitations and Next Steps
- Accuracy is still low relative to a production target, suggesting a need for more data, better class balance checks, and stronger architectures (e.g., deeper CNNs or transfer learning).
- Incorporate per-class error analysis from the confusion matrix to identify specific confusing pairs and targeted augmentation.

## 8. Conclusion
The CNN consistently outperforms the DNN across all image sizes, with the best performance at 128x128. The macro F1 values show that class coverage is still limited, so future work should focus on expanding data diversity, refining augmentation, and testing stronger CNN architectures to improve per-class recognition and overall robustness.