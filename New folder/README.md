# White Wine Quality Classification

This project develops and evaluates machine-learning classifiers that categorize Portuguese white wines as either **lower quality** (quality scores 3–5) or **acceptable/higher quality** (quality scores 6–9) from 11 physicochemical measurements.

## Objective

The primary objective was to exceed 70% test accuracy while also examining how well each model detects lower-quality wines. This class-specific review matters because approximately 66% of the cleaned observations belong to the acceptable/higher-quality class.

## Dataset

- Source: [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
- Original white-wine observations: 4,898
- Duplicate rows removed: 937
- Cleaned observations: 3,961
- Predictor variables: 11 numeric physicochemical measurements
- Missing values: none

The dataset is not duplicated in this repository unless permitted by its source terms. Download `winequality-white.csv` from UCI and place it beside the notebook or update the loading path.

## Workflow

1. Data inspection and duplicate removal
2. Binary target engineering
3. Outlier, skewness and correlation analysis
4. Stratified 80/20 train-test split
5. Majority-class baseline
6. KNN before and after Min–Max normalization
7. Five-fold stratified cross-validation and Grid Search
8. Random Forest and Gradient Boosting ensembles
9. Class-specific evaluation and decision-threshold tuning
10. Bootstrap confidence intervals
11. Surrogate Decision Tree for interpretability

All learned preprocessing is fitted only on training data to prevent data leakage.

## Key Results

| Model | Test accuracy | Lower-quality recall | Lower-quality F1 |
|---|---:|---:|---:|
| Majority-class baseline | 65.95% | 0.00% | 0.00% |
| Unscaled KNN, k=10 | 67.84% | — | — |
| Normalized KNN, k=5 | 73.27% | 54.00% | 58.00% |
| Tuned KNN | 75.16% | 54.00% | 60.00% |
| Initial Random Forest | 76.29% | 60.00% | 63.00% |
| Tuned Random Forest | 74.91% | 65.00% | 64.00% |
| Tuned Gradient Boosting, default threshold | **76.67%** | 60.74% | 63.94% |
| Tuned Gradient Boosting, threshold 0.68 | 71.88% | **76.67%** | **64.99%** |

The tuned Gradient Boosting classifier was selected as the general-purpose final model. Its five-fold cross-validation accuracy was 77.43%, and its 95% bootstrap confidence interval for test accuracy was 73.52%–79.57%.

For stricter quality control, increasing the decision threshold to 0.68 substantially improved lower-quality recall, although overall accuracy decreased.

## Interpretability

Alcohol was the most influential feature in the tree-based analysis, followed by density and acidity- or sulfur-related measurements. A depth-three surrogate Decision Tree reproduced 87.14% of the final Gradient Boosting model's test predictions. The surrogate is an explanatory approximation, not a replacement for the final model.

## Repository Structure

```text
wine-quality-classification/
├── notebooks/
│   └── white_wine_quality_classification.ipynb
├── src/
│   └── wine_modeling.py
├── requirements.txt
└── README.md
```

## Running the Project

```bash
pip install -r requirements.txt
jupyter lab
```

Open the notebook and run its cells from top to bottom after placing `winequality-white.csv` in the notebook's working directory.

## Limitations

- Binary target engineering removes differences within the original quality scores.
- The dataset represents one wine type and region, limiting external generalization.
- Model comparisons reused the same held-out test set; independent external validation would strengthen the evidence.
- Bootstrap intervals quantify sampling uncertainty for this evaluation but do not guarantee performance in every production setting.
- Feature importance and surrogate rules describe predictive behaviour, not causation.

## Future Work

Future development could include external validation, probability calibration, cost-based threshold selection, additional boosting algorithms, multiclass prediction and monitoring after deployment.
