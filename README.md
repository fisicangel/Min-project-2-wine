# Min-project-2-wine
Predicting High-Quality White Wine Using Machine Learning

Business Objective
The objective is to develop a classification model that predicts whether a white wine will be classified as high quality based on its physicochemical properties.

A high-quality wine is defined as one receiving a quality score of 6 or higher. Such a model could provide an initial data-driven quality assessment to support quality-control and product-screening processes in the wine and beverage industry.

Research Questions
Primary Research Question
Can a machine-learning classification model predict whether a white wine will receive a high-quality score based on its physicochemical properties?

Secondary Research Questions
Which physicochemical properties have the strongest relationships with high-quality wine classification?

Which model—Logistic Regression or a tuned K-Nearest Neighbours Classifier—performs better at identifying high-quality white wines?

How well does the best model perform in terms of accuracy, precision, recall, F1-score, and ROC-AUC?

1. INITIAL DATA SET INFORMATION
Number of rows: 4898
Number of columns: 12

RangeIndex: 4898 entries, 0 to 4897
Data columns (total 12 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   fixed acidity         4898 non-null   float64
 1   volatile acidity      4898 non-null   float64
 2   citric acid           4898 non-null   float64
 3   residual sugar        4898 non-null   float64
 4   chlorides             4898 non-null   float64
 5   free sulfur dioxide   4898 non-null   float64
 6   total sulfur dioxide  4898 non-null   float64
 7   density               4898 non-null   float64
 8   pH                    4898 non-null   float64
 9   sulphates             4898 non-null   float64
 10  alcohol               4898 non-null   float64
 11  quality               4898 non-null   int64  
dtypes: float64(11), int64(1)
memory usage: 459.3 KB

2. DUPLICATES
The dataset contains 4,898 white-wine observations and 12 columns.

All columns contain 4,898 non-null values, confirming that there are no missing values in the dataset. Therefore, no missing-value treatment, such as removing rows or filling values, is required.

The dataset will also be checked for duplicated rows before proceeding with the analysis and machine-learning model.

Duplicate Treatment
The dataset contained 937 duplicated rows. These rows were removed to prevent identical observations from receiving excessive influence during model training and potentially appearing in both the training and test sets.

After removing duplicates, the cleaned dataset contains 3,961 unique observations.

3. TARGET DISTRIBUTION ANALYSIS

The original target variable, quality, is an integer score representing the sensory quality assigned to each wine.

Because this project uses classification, the numerical quality scores will later be converted into categorical classes. However, the classification threshold should not be selected arbitrarily. We must first examine:

The number of observations in each quality category.
The percentage represented by each category.
Whether the proposed classes would be balanced.
Whether a majority-class prediction could already produce high accuracy.
For a quality score (q), a possible binary target can be expressed as: y = 0, if q < t y = 1, if q ≥ t

where:

(q) is the original wine-quality score,
(t) is the classification threshold,
(y=0) represents lower-quality wine,
(y=1) represents higher-quality wine.
The threshold (t) will be selected after examining the target distribution.

4. Target Engineering for Binary Classification
The original target, 'quality', contains integer scores from 3 to 9. For this classification task, it is converted into a binary target called 'quality-class'.

The classification rule is: yi​={0,1,​if qi​≤5if qi​≥6​)

where:

(q_i) is the original quality score for wine (i),
(y_i) is the newly engineered binary target,
0 represents lower-quality wine,
1 represents acceptable or higher-quality wine.
The original 'quality' column is retained for interpretation, while 'quality_class' becomes the target used for classification.

This is target engineering, not one-hot encoding. One-hot encoding is unnecessary for the predictor variables because all original wine features are numerical.


5. Quantitative Assessment of Potential Outliers
The IQR analysis identified potential outliers in 10 of the 11 predictor variables. Citric acid had the highest proportion at 5.63%, followed by chlorides at 4.49% and volatile acidity at 3.36%. Alcohol had no observations outside its calculated IQR boundaries.

The IQR method identifies observations using:

IQR = Q3 − Q1

[ ]

[ \text{Upper bound}=Q_3+1.5(IQR) ]

These boundaries are statistical rules rather than scientific limits. For example, the lower IQR boundary for citric acid is 0.09. Consequently, chemically possible values between 0 and 0.09 may be labelled as potential outliers even though they are not necessarily errors.

Similarly, the calculated lower boundaries for residual sugar and free sulfur dioxide are negative. This does not indicate that negative observations exist; it only means that the mathematical boundary falls below the physically meaningful minimum of zero.

The potential outlier percentages for individual features range from 0% to 5.63%. These observations will be retained because there is currently no evidence that they represent measurement or data-entry errors.

Rather than deleting valid but unusual wines, the project will investigate skewness and compare preprocessing methods. This is particularly important for KNN because its distance calculation is sensitive to feature scale and extreme values.

The analysis therefore distinguishes between:

a statistically unusual observation,
a scientifically impossible observation,
and an erroneous observation.
Only observations supported by evidence as erroneous or impossible should be removed.

5.1 Initial Interpretation of the Feature Boxplots
The boxplots indicate potential outliers in several physicochemical features. The most noticeable upper-tail observations appear in volatile acidity, citric acid, residual sugar, chlorides and free sulfur dioxide.

Some features, particularly residual sugar and chlorides, also appear positively skewed. A positively skewed distribution has a longer right tail because most observations are concentrated at lower values while a smaller number have much higher values.

However, the points outside the boxplot whiskers are not automatically data errors. Wine composition can naturally contain unusual but valid measurements. Removing all observations identified by the IQR rule could discard genuine wines, reduce the sample size and create selection bias.

Potential outliers may influence distance-based models such as KNN because the model calculates distances between observations. They may also affect StandardScaler, which uses the mean and standard deviation.

Therefore, no observations will be removed solely because they appear outside the boxplot whiskers. The potential outliers will first be quantified and checked for physical plausibility. Model performance can later be compared using standardization, normalization and, if justified, a robust preprocessing method.

6. Analysis of Feature Skewness
Skewness measures the asymmetry of a distribution.A simplified mathematical expression is: Skewness = (1/n) Σ ((xi − x̄) / s)³

where:

(n) is the number of observations,
(x_i) is an individual feature value,
(\bar{x}) is the feature mean,
(s) is the standard deviation.
The sign indicates the direction:

Skewness > 0 = longer right tail
Skewness < 0 = longer left tail
Skewness ≈ 0 = approximately symmetrical distribution

The following practical thresholds will guide interpretation:

Absolute skewness interpretation:

Below 0.5 = approximately symmetrical
0.5 to below 1.0 = moderately skewed
1.0 or above = strongly skewed

Skewness matters because extreme asymmetric values may affect distance-based models such as KNN and preprocessing methods based on the mean and standard deviation.

However, skewness does not automatically require transformation. Any transformation must be justified and later evaluated using model performance.

Interpretation of Feature Skewness
The skewness analysis shows that six predictor variables are strongly positively skewed:

chlorides,
volatile acidity,
free sulfur dioxide,
residual sugar,
citric acid,
and density.
Chlorides has the strongest skewness:

Skewness(chlorides) = 4.97

This indicates that most wines have relatively low chloride values, while a small number of wines have much higher values.

Two variables, sulphates and fixed acidity, are moderately positively skewed. Total sulfur dioxide, pH and alcohol are approximately symmetrical according to the selected threshold:

|Skewness| < 0.5

All calculated skewness values are positive. Therefore, the asymmetric distributions have longer right tails.

Strong skewness may influence K-Nearest Neighbors because extreme values can affect the calculated distances between observations. It may also influence standardization because the mean and standard deviation are sensitive to extreme values.

However, scaling and transformation are different operations:

Transformation changes the shape of a distribution.
Scaling changes the numerical range or scale.
Standardization does not necessarily remove skewness.
Min–max normalization does not remove skewness.
Standardization is calculated as:

z = (x − μtrain) / σtrain

where:

x = original feature value
μ_train = training-set mean
σ_train = training-set standard deviation
z = standardized value

Min–max normalization is calculated as:

x′ = (x − xmin_train) / (xmax_train − xmin_train)

Both scaling methods change the numerical scale but generally preserve the original distribution's skewness.

A power transformation, such as Yeo–Johnson, can be used to reduce skewness. Because it estimates transformation parameters from the data, it must be fitted only on the training set.

The project will therefore compare classification models using:

original unscaled features,
standardized features,
normalized features,
and transformed plus standardized features.
The train–test split will be performed before fitting any scaler or learned transformation. This prevents information from the test set from influencing preprocessing and avoids data leakage.

7. Defining the Features and Target
In supervised machine learning, the dataset is split into:

X = predictor variables/features
y = target variable to be predicted

For this project:

X = physicochemical wine features, such as fixed acidity, volatile acidity, alcohol, and others

y = quality_class

The original quality column is not included in X because quality_class was created from it. Including it would cause target leakage, meaning the model would already have access to the answer.

The model relationship is:

X → f(X) → ŷ

where:

X = wine features
f = classification model
ŷ = predicted quality class

The feature matrix is:

X ∈ R^(3961 × 11)

This means there are 3,961 wine samples and 11 predictor variables.

The target vector is:

y ∈ {0, 1}^3961

where:

y = 0 = lower-quality wine
y = 1 = acceptable or higher-quality wine


8. Stratified Train–Test Split
The dataset is divided into a training set and a test set:

D = D_train ∪ D_test

and:

D_train ∩ D_test = ∅

This means the training and test sets do not overlap.

An 80/20 split is used:

80% = training data
20% = test data

The training set is used to fit preprocessing methods and train the model. The test set is kept unseen and used to evaluate model performance on new data.

Stratification is applied because the target classes are not perfectly balanced:

P_train(y = k) ≈ P_test(y = k) ≈ P_complete(y = k), where k ∈ {0, 1}

This keeps the class proportions similar in the full dataset, training set, and test set.

random_state = 42 makes the split reproducible, so the same train-test split is produced each time.

P_train(y = k) ≈ P_test(y = k) ≈ P_complete(y = k), where k ∈ {0, 1}

P = proportion
P_train = proportion in the training set
P_test = proportion in the test set
P_complete = proportion in the full dataset
y = target variable
k = class 0 or class 1
y = k = observations in class k
≈ = approximately equal


