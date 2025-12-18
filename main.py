#for data loading
import kagglehub
from kagglehub import KaggleDatasetAdapter

#for data manipulation
import pandas as pd
import numpy as np # Added for data math

#for plots
import matplotlib.pyplot as plt
import seaborn as sns # for pretty plots 
import os # Added to manage your 'figures' folder

# Create a folder for your outputs if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')


##1) SETUP
# Set the path to the file you'd like to load
file_path = "semiconductor_quality_control.csv"

# Load the latest version
# Using dataset_load to ensure the file path is recognized correctly
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "programmer3/semiconductor-sensor-data-for-predictive-quality",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

#print(df.shape)
#print(df.columns)
#print(df.head())



##2) MAKING GRAPHS 
'''''
Exploratory Data Analysis (EDA)
- Summary statistics
- Address missing and messy data
- At least 3 clear visualizations
- Correlation or relationship insights
'''''

# --- CLEAN DATA HEALTH SUMMARY ---
# (We updated this part to look better in Jupyter while keeping your logic)

health_stats = {
    "Metric": ["Total Rows", "Total Columns", "Missing Values", "Duplicate Rows"],
    "Value": [df.shape[0], df.shape[1], df.isna().sum().sum(), df.duplicated().sum()]
}
print("========== DATA HEALTH SUMMARY ==========")
display(pd.DataFrame(health_stats))

#check for "dead" Sensors where std = 0 OR only 1 unique value
numeric_df = df.select_dtypes(include=["int64", "float64"]).drop(columns=["Defect"], errors='ignore')
dead_sensors = numeric_df.columns[numeric_df.nunique() <= 1].tolist()
print("\n========== SENSOR SIGNAL CHECK ==========")
print("Dead sensors:", dead_sensors)

#descriptive statistics (The Vertical View for better eyes)
print("\n========== STATISTICAL SUMMARY (VERTICAL VIEW) ==========")
display(df.describe().T.round(2))

#Check for duplicate rows: (Already shown in the table above)

sns.set(style="whitegrid")#white background for plots also add gridlines

#bar chart:
## Defect Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="Defect", data=df)
plt.title("Distribution of Defective vs Non-Defective Chips")
plt.xlabel("Defect (0 = No, 1 = Yes)")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("figures/defect_distribution_bar.png")
plt.show()


#histogram
## Sensor Distribution
plt.figure(figsize=(6, 4))
plt.hist(df["Chamber_Temperature"], bins=30)
plt.title("Distribution of Chamber Temperature")
plt.xlabel("Chamber Temperature")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("figures/chamber_temperature_hist.png")
plt.show()

#correlation heatmap
'''
Relationships
Sensor–sensor relationships
Sensor–defect relationships
'''
numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(10, 8))
corr = numeric_df.corr()

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5
)

plt.title("Correlation Heatmap of Sensor Features")
plt.tight_layout()
plt.savefig("figures/correlation_heatmap.png")
plt.show()


##3) Unsupervised Learning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
#KMEANS
#AGGLOMERATIVE CLUSTERING
'''''
Evaluate:
- Silhouette score
- Visualizations using PCA or t-SNE projections
- Compare clusters with meaningful real-world groups if possible
'''''
#prepare data for clustering
cluster_data = df.select_dtypes(include=["int64", "float64"]).drop(columns=["Defect"])
#scale the data: 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_data)
#PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("Explained variance ratio by PCA components:", pca.explained_variance_ratio_)
print("Total explained variance by first 2 components:", sum(pca.explained_variance_ratio_))


#determine best k using elbow method: 
inertias = []
k_value = range(2, 11)

for k in k_value:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
#plot to find elbow curve
plt.figure(figsize=(6, 4))
plt.plot(k_value, inertias, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (SSE)")
plt.title("Elbow Method for K-Means")
plt.tight_layout()
plt.savefig("figures/kmeans_elbow.png")
plt.show()


#silhouette score analysis
silhouette_scores = []

for k in k_value:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
# plot silhouette scores
plt.figure(figsize=(6, 4))
plt.plot(k_value, silhouette_scores, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs Number of Clusters")
plt.tight_layout()
plt.savefig("figures/kmeans_silhouette.png")
plt.show()

##KMeans Clustering
best_k = 2 # choose based on previous analysis of SS and elbow method graph. 

kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

#final silhouette score
final_kmeans_ss = silhouette_score(X_scaled, kmeans_labels)
print("Final K-Means Silhouetter Score:", final_kmeans_ss)
#visualize clusters in PCA space
plt.figure(figsize=(6, 5))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=kmeans_labels,
    palette="tab10"
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clusters (PCA Projection)")
plt.tight_layout()
plt.savefig("figures/kmeans_pca.png")
plt.show()

#compare clusters to real world group
df["KMeans_Cluster"] = kmeans_labels

print("Defect rate per K-Means cluster:")
print(df.groupby("KMeans_Cluster")["Defect"].mean())


##Agglomerative Clustering
#Goal: discover hierarchical process regimes using the same number of clusters (k = 5) for fair comparison with K-Means.
agglo = AgglomerativeClustering(n_clusters=2, linkage="ward")
agglo_labels = agglo.fit_predict(X_scaled)

#Silhouette Score
agglo_silhouette = silhouette_score(X_scaled, agglo_labels)
print("Agglomerative Silhouette Score:", agglo_silhouette)

#PCA Visualization
plt.figure(figsize=(6, 5))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=agglo_labels,
    palette="tab10"
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Agglomerative Clusters (PCA Projection)")
plt.tight_layout()
plt.savefig("figures/agglo_pca.png")
plt.show()

#compare to real world groups
df["Agglo_Cluster"] = agglo_labels

print("Defect rate per Agglomerative cluster:")
print(df.groupby("Agglo_Cluster")["Defect"].mean())


## 4) Supervised Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

# --- DATA PREPARATION ---
# Selecting numeric features and defining target
X = df.select_dtypes(include=["int64", "float64"]).drop(columns=["Defect", "KMeans_Cluster", "Agglo_Cluster"], errors='ignore')
y = df["Defect"]

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling (Required for Logistic Regression and SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- MODEL TRAINING ---

# 1. Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=1000).fit(X_train_scaled, y_train)

# 2. Random Forest (Balanced)
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced").fit(X_train, y_train)

# 3. SVM with RBF Kernel (Added balanced weight to fix the 0.00 recall issue)
svm = SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced").fit(X_train_scaled, y_train)

# --- CLEAN EVALUATION VIEW ---

def evaluate_and_display(name, y_true, y_pred, y_prob):
    print(f"{f' {name} PERFORMANCE ':=^40}")
    stats = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        "Score": [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred),
            f1_score(y_true, y_pred),
            roc_auc_score(y_true, y_prob)
        ]
    }
    display(pd.DataFrame(stats).round(4))
    print("="*40 + "\n")

# Run evaluations
evaluate_and_display("Logistic Regression", y_test, log_reg.predict(X_test_scaled), log_reg.predict_proba(X_test_scaled)[:, 1])
evaluate_and_display("Random Forest", y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1])
evaluate_and_display("SVM (RBF)", y_test, svm.predict(X_test_scaled), svm.predict_proba(X_test_scaled)[:, 1])



##5) Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Grid Search for Random Forest ---
rf_param_grid = {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42, class_weight="balanced"),
                       param_grid=rf_param_grid, scoring="recall", cv=5, n_jobs=-1)
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

# Styled Best Params Table for RF
rf_params_df = pd.DataFrame(rf_grid.best_params_.items(), columns=['Parameter', 'Optimal Value'])
rf_params_styled = rf_params_df.style.set_caption("Random Forest: Best Hyperparameters")\
    .set_table_styles([{'selector': 'caption', 'props': [('font-size', '16pt'), ('font-weight', 'bold')]}])\
    .bar(subset=['Optimal Value'], color='#d65f5f')  # Highlight values
display(rf_params_styled)

# Add CV Recall Score
print(f"Best Cross-Validation Recall (Macro for Minority Class Focus): {rf_grid.best_score_:.4f}")

evaluate_model("Random Forest (Tuned)", y_test, best_rf.predict(X_test), best_rf.predict_proba(X_test)[:, 1])

print("\n" + "="*60 + "\n")

# --- 2. Grid Search for SVM ---
svm_param_grid = {"C": [0.1, 1, 10], "gamma": ["scale", 0.01]}
svm_grid = GridSearchCV(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
                        param_grid=svm_param_grid, scoring="recall", cv=5, n_jobs=-1)
svm_grid.fit(X_train_scaled, y_train)

best_svm = svm_grid.best_estimator_

# Styled Best Params Table for SVM
svm_params_df = pd.DataFrame(svm_grid.best_params_.items(), columns=['Parameter', 'Optimal Value'])
svm_params_styled = svm_params_df.style.set_caption("SVM: Best Hyperparameters")\
    .set_table_styles([{'selector': 'caption', 'props': [('font-size', '16pt'), ('font-weight', 'bold')]}])\
    .bar(subset=['Optimal Value'], color='#5fba7d')
display(svm_params_styled)

print(f"Best Cross-Validation Recall: {svm_grid.best_score_:.4f}")

evaluate_model("SVM (Tuned)", y_test, best_svm.predict(X_test_scaled), best_svm.predict_proba(X_test_scaled)[:, 1])

##6) Feature Importance & Interpretation
#Random Forest Feature Importance
'''''
Explain what features matter and why — interpret in context.
'''''

# extract feature importances from tuned random forest
import pandas as pd

feature_importances = best_rf.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

print(importance_df)

import matplotlib.pyplot as plt
import seaborn as sns

#visualize feature importances
plt.figure(figsize=(8, 6))
sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature",
    hue="Feature",
    palette="viridis",
    legend=False
)
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Sensor Feature")
plt.tight_layout()
plt.savefig("figures/random_forest_feature_importance.png")
plt.show()
