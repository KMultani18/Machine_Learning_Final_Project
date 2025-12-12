# Predicting Chip Defects: Optimizing Semiconductor Yield

## Project Overview
This project applies exploratory data analysis and unsupervised machine learning techniques to semiconductor manufacturing sensor data with the goal of understanding and predicting chip defects. By analyzing process-level sensor measurements, the project identifies operating regimes associated with higher defect rates and provides insights that support yield optimization in semiconductor fabrication.

The dataset is loaded programmatically from Kaggle to ensure reproducibility and to meet course requirements.

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
- Verified dataset shape: **4,219 × 16**
- All numeric sensor features exhibit meaningful variability
- No dead sensors (no constant-valued features)
- Overall defect rate is approximately **14.6%**, indicating a moderately imbalanced dataset

### Data Quality Checks
- No missing values across all columns
- No duplicate rows
- Target label (`Defect`) is clean and binary (`0` / `1`)

### Visualizations
The following EDA visualizations were generated:
1. Bar chart showing the distribution of defective vs non-defective chips  
2. Histogram illustrating the distribution of chamber temperature  
3. Correlation heatmap showing relationships among sensor features and defects  

EDA results indicate weak linear correlations between individual sensors and the defect label, suggesting that chip defects arise from complex, multivariate process interactions rather than single-variable thresholds.

---

## Unsupervised Learning (Clustering)

Unsupervised learning was applied to discover hidden operating regimes using only numeric sensor data (excluding the defect label).

### Preprocessing
- Selected numeric sensor features only
- Standardized features using `StandardScaler`
- Applied PCA with two components for visualization

---

### K-Means Clustering
The optimal number of clusters was determined using:
- Elbow method (SSE / inertia)
- Silhouette score analysis

**Final choice:** **k = 5**

**Results:**
- Final silhouette score: **0.059**
- Defect rates across clusters ranged from approximately **12.18%** to **16.08%**

These results indicate overlapping but distinct operating regimes with differing defect risks.

---

### Agglomerative Clustering
Agglomerative clustering was performed using Ward linkage with **k = 5** clusters for comparison.

**Results:**
- Silhouette score: **0.033**
- One cluster exhibited a defect rate of **100%**
- All remaining clusters had a defect rate of **0%**

This suggests the presence of a narrowly defined process regime strongly associated with chip defects.

---

## Key Insights
- Chip defects are not driven by a single sensor value
- Manufacturing process regimes overlap significantly
- Certain operating conditions consistently show elevated defect risk
- Hierarchical clustering can isolate rare but critical failure states
- Results motivate the use of supervised learning for generalizable defect prediction

---

## Project Status
✔ Data loading and validation  
✔ Exploratory data analysis  
✔ K-Means clustering  
✔ Agglomerative clustering  

**Next Step:**  
Supervised learning models (Logistic Regression, Random Forest, SVM) to predict defects and evaluate performance using classification metrics.

---
