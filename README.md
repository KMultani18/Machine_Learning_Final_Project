# Predicting Chip Defects: Optimizing Semiconductor Yield

## Project Overview
This project applies exploratory data analysis (EDA), unsupervised learning, and supervised machine learning techniques to semiconductor manufacturing sensor data in order to understand and predict chip defects. By analyzing process-level sensor measurements, the goal is to identify operating conditions associated with higher defect risk and to evaluate the effectiveness of predictive models for yield optimization.

The dataset is loaded programmatically from Kaggle to ensure reproducibility and compliance with course requirements.

---

## Dataset
**Source:** Kaggle  
**Dataset:** Semiconductor Sensor Data for Predictive Quality  
**Rows:** 4,219  
**Columns:** 16  

The dataset includes:
- Numeric sensor measurements (temperature, pressure, power, vibration, particle count, etc.)
- Categorical process metadata
- A binary target label (`Defect`) indicating whether a chip is defective

---

## Tools & Libraries
- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy
- kagglehub

All dependencies are listed in `requirements.txt`.

---

## Exploratory Data Analysis (EDA)

### Summary Statistics
- Dataset shape verified: **4,219 × 16**
- All numeric sensor features exhibit meaningful variability
- No dead sensors (no constant-valued features)
- Overall defect rate is approximately **14.6%**, indicating class imbalance

### Data Quality Checks
- No missing values
- No duplicate rows
- Target label (`Defect`) is clean and binary (`0` / `1`)

### Visualizations
The following EDA visualizations were generated:
1. Bar chart showing the distribution of defective vs non-defective chips  
2. Histogram illustrating the distribution of chamber temperature  
3. Correlation heatmap showing relationships among sensor features and defects  

EDA results show weak linear correlations between individual sensors and defects, suggesting that failures are driven by multivariate interactions rather than single-variable thresholds.

---

## Unsupervised Learning (Clustering)

Unsupervised learning was used to identify hidden operating regimes based solely on numeric sensor data (excluding the defect label).

### Preprocessing
- Selected numeric sensor features
- Standardized features using `StandardScaler`
- Applied PCA (2 components) for visualization

---

### K-Means Clustering
The optimal number of clusters was selected as **k = 2**, based on silhouette score analysis, which peaked at **0.077**. The elbow plot did not show a strong inflection point at higher values of k, supporting the use of a smaller number of clusters.

Cluster-level defect analysis showed:
- Cluster 0 defect rate: **15.19%**
- Cluster 1 defect rate: **14.07%**

These results indicate two overlapping operating regimes with modest differences in defect risk.

---

### Agglomerative Clustering
Agglomerative clustering using Ward linkage was also applied with **k = 2** clusters for comparison.

Results:
- Silhouette score: **0.034**
- Cluster 0 defect rate: **15.07%**
- Cluster 1 defect rate: **14.03%**

The similarity in defect rates suggests that hierarchical clustering does not reveal sharply isolated failure regimes at this level of granularity.

---

## Supervised Learning

Three supervised classification models were trained to predict chip defects:
- Logistic Regression
- Random Forest
- SVM (RBF kernel)

### Evaluation Metrics
Because `Defect` is a binary target, models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Baseline models achieved high accuracy (~85%) but **zero recall**, indicating that they defaulted to predicting the majority (non-defective) class due to class imbalance.

---

## Hyperparameter Tuning

Hyperparameter tuning was performed using **GridSearchCV with 5-fold cross-validation**, optimizing for **recall** to prioritize defect detection.

### Tuned Model Results

**Random Forest (Tuned):**
- Best parameters:  
  `n_estimators = 100`, `max_depth = 10`, `min_samples_split = 5`, `min_samples_leaf = 2`
- Recall: **1.6%**
- F1-score: **0.027**
- ROC-AUC: **0.437**

**SVM (RBF, Tuned):**
- Best parameters:  
  `C = 10`, `gamma = 0.01`
- Recall: **51.2%**
- F1-score: **0.201**
- ROC-AUC: **0.450**

The tuned SVM demonstrated the strongest defect detection capability, trading off accuracy for significantly improved recall.

---

## Feature Importance & Interpretation

Feature importance analysis was conducted using the tuned Random Forest model. The most influential features were:

- **Rotation Speed (0.107)**
- **Etch Depth (0.104)**
- **Vacuum Pressure (0.102)**
- **UV Exposure Intensity (0.101)**
- **RF Power (0.100)**

Additional contributing features included **Particle Count (0.097)**, **Vibration Level (0.096)**, **Chamber Temperature (0.093)**, and **Gas Flow Rate (0.091)**. The relatively uniform importance values indicate that defects arise from complex interactions among multiple process variables rather than a single dominant factor. The low importance of cluster-based features (**Agglo_Cluster = 0.010**) confirms that unsupervised regimes provide limited predictive value for defect classification.

---

## Key Insights
- Chip defects are driven by multivariate sensor interactions
- Unsupervised clustering reveals overlapping operating regimes with limited predictive power
- Baseline supervised models fail under class imbalance
- Hyperparameter tuning is essential for meaningful defect detection
- Tuned SVM provides the best trade-off between recall and overall performance

---

## Project Status
✔ Data loading and validation  
✔ Exploratory data analysis  
✔ K-Means and Agglomerative clustering  
✔ Supervised learning models  
✔ Hyperparameter tuning  
✔ Feature importance analysis  

---
