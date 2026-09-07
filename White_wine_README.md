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

## Citation
This dataset is public available for research. The details are described in [Cortez et al., 2009]. 
  Please include this citation if you plan to use this database:

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
                [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
                [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib

1. Title: Wine Quality 

2. Sources
   Created by: Paulo Cortez (Univ. Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos and Jose Reis (CVRVV) @ 2009
   
3. Past Usage:

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  In the above reference, two datasets were created, using red and white wine samples.
  The inputs include objective tests (e.g. PH values) and the output is based on sensory data
  (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality 
  between 0 (very bad) and 10 (very excellent). Several data mining methods were applied to model
  these datasets under a regression approach. The support vector machine model achieved the
  best results. Several metrics were computed: MAD, confusion matrix for a fixed error tolerance (T),
  etc. Also, we plot the relative importances of the input variables (as measured by a sensitivity
  analysis procedure).
 
4. Relevant Information:

   The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.
   For more details, consult: http://www.vinhoverde.pt/en/ or the reference [Cortez et al., 2009].
   Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables 
   are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

   These datasets can be viewed as classification or regression tasks.
   The classes are ordered and not balanced (e.g. there are munch more normal wines than
   excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent
   or poor wines. Also, we are not sure if all input variables are relevant. So
   it could be interesting to test feature selection methods. 

5. Number of Instances: red wine - 1599; white wine - 4898. 

6. Number of Attributes: 11 + output attribute
  
   Note: several of the attributes may be correlated, thus it makes sense to apply some sort of
   feature selection.

7. Attribute information:

   For more information, read [Cortez et al., 2009].

   Input variables (based on physicochemical tests):
   1 - fixed acidity
   2 - volatile acidity
   3 - citric acid
   4 - residual sugar
   5 - chlorides
   6 - free sulfur dioxide
   7 - total sulfur dioxide
   8 - density
   9 - pH
   10 - sulphates
   11 - alcohol
   Output variable (based on sensory data): 
   12 - quality (score between 0 and 10)

8. Missing Attribute Values: None

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
Min-project-2-wine/
├── notebooks/
│   └── Analysis_completed_Paul_Angelica_reviewed_white_wine_quality_classification.ipynb
├── src/
│   └── wine_modeling.py
├── data/
│   └── winequality.txt
    └── winequality-white.csv
├── slices/
│   └── White_Wine_Quality_Classification_final.pptx
├── requirements.txt
└── White_wine_README.md
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
