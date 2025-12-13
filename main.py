#for data loading
import kagglehub
from kagglehub import KaggleDatasetAdapter

#for data manipulation
import pandas as pd

#for plots
import matplotlib.pyplot as plt
import seaborn as sns # for pretty plots 


##1) SETUP
# Set the path to the file you'd like to load
file_path = "semiconductor_quality_control.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "programmer3/semiconductor-sensor-data-for-predictive-quality",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

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

#summary statistics
print(df.shape) #for rows and columns 

#check for "dead" Sensors where std = 0 OR only 1 unique value
numeric_df = df.select_dtypes(include=["int64", "float64"]).drop(columns=["Defect"]) # removes the defect column not the rows 
dead_sensors = numeric_df.columns[numeric_df.nunique() <= 1]
print("Dead sensors:", dead_sensors.tolist())

#descriptive statistics
print(df.describe())

#Missing and Messy Data
missing_data = df.isna().sum()
print(missing_data) # 0 missing values. 

#Check for duplicate rows: 
print("Duplicate rows:", df.duplicated().sum()) # 0 duplicate rows

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



##4) Supervised Learning
#Logistic Regression
#Random Forest
#SVM(RBF Kernel)
'''''
classification problem: 
- accuracy, precision, recall, F1-score, ROC-AUC
'''''
#define feature (X) and target (y)
X = df.select_dtypes(include=["int64", "float64"]).drop(columns=["Defect"])
y = df["Defect"]

#train-test split 80/20
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42, stratify=y
)

#feature scaling for LR and SVM: 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
#initialize model
log_reg = LogisticRegression(random_state=42, max_iter=1000) # max_iter = 1000 ensures convergence
log_reg.fit(X_train_scaled, y_train)

#predictions
y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

#Random Forest
from sklearn.ensemble import RandomForestClassifier
#initialize model
rf = RandomForestClassifier(n_estimators = 200, random_state = 42, class_weight="balanced") # n_estimators=200 for more robust results, class_weight="balanced" to handle class imbalance
rf.fit(X_train, y_train)

#predictions
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

#SVM with RBF Kernel
from sklearn.svm import SVC
#initialize SVM with RBF kernel
svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
#predictions
y_pred_svm = svm.predict(X_test_scaled)
y_proba_svm = svm.predict_proba(X_test_scaled)[:, 1]

#Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\n{name} Performance:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_prob))

#Evaluate all models:
evaluate_model("Logistic Regression", y_test, y_pred_lr, y_proba_lr)
evaluate_model("Random Forest", y_test, y_pred_rf, y_proba_rf)
evaluate_model("SVM (RBF)", y_test, y_pred_svm, y_proba_svm)



##5) Hyperparameter Tuning
'''''
Use GridSearchCV or RandomizedSearchCV with cross-validation.
Discuss:
- Best parameters found
- How performance changed vs baseline
'''''

'''''
Random Forest-> sensitive to tree depth and number of trees.
SVm(RBF)-> sensitive to C and gamma.
Defaults are not optimal.
'''''

#Grid Search for Random Forest
from sklearn.model_selection import GridSearchCV

#define parameter grid
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

#run grid search
rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=42, 
        class_weight="balanced"
        ),
        param_grid=rf_param_grid,
        scoring="recall", #focus on detecting defects
        cv=5,
        n_jobs=-1
)

rf_grid.fit(X_train, y_train)

#best parameters
print("\n\nBest Random Forest Parameters:") 
print(rf_grid.best_params_)

#evaluate tuned RF
best_rf = rf_grid.best_estimator_

y_pred_rf_tuned = best_rf.predict(X_test)
y_proba_rf_tuned = best_rf.predict_proba(X_test)[:, 1]

evaluate_model(
    "Random Forest (Tuned)",
    y_test,
    y_pred_rf_tuned,
    y_proba_rf_tuned
)


#GridSearchCV for SVM (RBF)
#SVM is expensive to train, so we limit the grid size
svm_param_grid = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.01, 0.1],
}

#run grid search
svm_grid = GridSearchCV(
    estimator=SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    ),
    param_grid=svm_param_grid,
    scoring="recall",
    cv=5,
    n_jobs=-1
)

svm_grid.fit(X_train_scaled, y_train)

#best parameters
print("\n\nBest SVM Parameters:")
print(svm_grid.best_params_)

#evaluate tuned SVM
best_svm = svm_grid.best_estimator_

y_pred_svm_tuned = best_svm.predict(X_test_scaled)
y_proba_svm_tuned = best_svm.predict_proba(X_test_scaled)[:, 1]

evaluate_model(
    "SVM (RBF, Tuned)",
    y_test,
    y_pred_svm_tuned,
    y_proba_svm_tuned
)



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

