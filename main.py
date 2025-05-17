import os
import time 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, silhouette_score
import matplotlib.pyplot as plt
from k_means import *


DATA_DIR = "data/Base.csv"


## Download latest version
## path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

## print("Path to dataset files:", path)

## 1. Load the dataset. Adjust the file path if necessary.
## For example, the common Kaggle Credit Card dataset.
data = pd.read_csv(DATA_DIR)
print("First 5 records:", data.head())


## 2. Separate features and target variable.
## Assume the target column is named 'Class' where 1 indicates fraud and 0 indicates normal
X = data.drop('fraud_bool', axis=1)
print(X.shape)
y = data['fraud_bool']



X = X.select_dtypes(include=[np.number]).fillna(0)
print(X.shape)
## 3. Split the data into training and testing sets.
## Stratify to maintain the imbalance ratio between classes in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

## 4. Feature Scaling: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## No Baseline
## calculate the latency 
import time
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- 0. Prep: scale your data once up front ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)




n_features = X_train.shape[1]
times = []
component_counts = []


n_pcs = 4
pca = PCA(n_components=n_pcs, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)

# 2. Grab the loadings matrix and explained‐variance ratios
#    pca.components_ is shape (n_pcs, n_features)
loadings = pca.components_         # rows = PCs, cols = features
var_ratio = pca.explained_variance_ratio_  # length = n_pcs

# 3. Compute aggregated importance per feature:
#    importance_j = sum_i |loading[i, j]| * var_ratio[i]
abs_loadings = np.abs(loadings)
weighted = abs_loadings * var_ratio[:, np.newaxis]
feature_scores = weighted.sum(axis=0)  # length = n_features

# 4. Put into a Series and sort
feat_importance = pd.Series(feature_scores, index=X.columns)
feat_importance = feat_importance.sort_values(ascending=False)

# 5. View the top features
top_k = 10
print(f"Top {top_k} features by aggregated PCA importance:")
print(feat_importance.head(top_k))
# Show the top 20 rows as a nicely formatted table
# 1. Take the top 20 from your Series and turn it into a DataFrame
tbl = feat_importance.head(top_k).reset_index()
tbl.columns = ["feature", "aggregated_importance"]

# 2. Create a matplotlib table
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')  # no axes

table = ax.table(
    cellText=tbl.values,
    colLabels=tbl.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)  # increase row height

# 3. Save to PNG
plt.tight_layout()
plt.savefig(f"{n_pcs}_components_feature_importance_top_{top_k}.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Wrote PNG to feature_importance_{top_k}.png")

# (Optional) Visualize

# --- 1. Loop over number of PCA components ---

repeats = 3
all_times = {k: [] for k in range(1, n_features+1)}

# for k in tqdm(range(1, n_features + 1)):
#     for _ in range(repeats):
#         pca = PCA(n_components=k, random_state=42)
#         Xk = pca.fit_transform(X_train_scaled)
#         clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
#         t0 = time.process_time()
#         clf.fit(Xk, y_train)
#         t1 = time.process_time()
#         all_times[k].append(t1 - t0)
loadings2 = pd.DataFrame(
    pca.components_.T,            # transpose so rows=features
    index=X.columns,              # feature names
    columns=[f"PC{i+1}" for i in range(pca.n_components_)]
)

n_clusters = X_train_pca.shape[1]
clusterings = cluster_features_pca(loadings2, 5)
# # Compute statistics
component_counts = []
mean_times  = []
std_times   = []
for k, ts in all_times.items():
    component_counts.append(k)
    mean_times.append(np.mean(ts))        
    # std_times.append(np.std(ts, ddof=1))    #
plt.plot(component_counts, mean_times, marker = "o")
plt.xlabel("Number of PCA Components")
plt.ylabel("Training Latency (sec)")
plt.title("Latency vs #Components")
plt.grid(True)
plt.show()


# # --- 6. Plot latency vs number of components ---
# plt.figure()
# plt.plot(component_counts, times, marker="o")
# plt.xlabel("Number of PCA Components")
# plt.ylabel("Training Latency (seconds)")
# plt.title("Complexity by PCA Components on Baseline Logistic Regression")
# plt.grid(True)
# plt.show()



## Optional: Plot the explained variance ratio
# plt.figure(figsize=(8, 5))
# plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, marker='o')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance (%)')
# plt.title('Explained Variance by PCA Components')
# plt.grid(True)
# plt.show()

## 6. Train a Logistic Regression classifier on the PCA-transformed data
## For imbalanced data, it can be useful to set class_weight parameter to "balanced"





    
# clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
# ## Baseline Training Latency
# baseline_start = time.perf_counter()
# clf.fit(X_train_scaled, y_train)
# baseline_end   = time.perf_counter()
# baseline_time = baseline_end - baseline_start




# print(f"Elapsed baseline training time: {baseline_end - baseline_start:.6f} seconds")
# print(f"Elapsed PCA training time: {pca_end - pca_start:.6f} seconds")
# print(f"Ratio Comparison PCA/Baseline: {pca_time/baseline_time}")



## 7. Evaluate the model
# y_pred = clf.predict(X_test_scaled)
# y_proba = clf.predict_proba(X_test_scaled)[:, 1]
# fpr_o, tpr_o, _ = roc_curve(y_test, y_proba)
# auc_o = roc_auc_score(y_test, y_proba)

# y_pred_pca = clf_pca.predict(X_test_pca)
# y_proba_pca = clf_pca.predict_proba(X_test_pca)[:, 1]
# fpr, tpr, _ = roc_curve(y_test, y_proba_pca)
# auc = roc_auc_score(y_test, y_proba_pca)




# plt.figure()
# plt.plot(fpr_o, tpr_o, color="green", label = "Original")
# plt.plot(fpr, tpr, color = "red", label = "PCA")
# plt.plot([0, 1], [0, 1], color = "blue", linestyle='--', label = "Random Classifier")  # chance line
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title(f'ROC Curve (AUC = {auc_o:.3f})')
# plt.grid(True)
# plt.legend()
# plt.show()

## 2. Confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(values_format='d')  # integer format
# plt.title('Confusion Matrix')
# plt.show()


# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))
# print("\nROC AUC Score (Original):", auc_o)
# print("\nROC AUC Score (PCA):", auc)
