import os
import time 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
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

## 5. Apply PCA
## You can choose a threshold for explained variance. In this case, we'll try to retain 95% variance.
pca = PCA(n_components=.5, random_state=42)  # alternatively, specify an integer number of components.
X_train_pca = pca.fit_transform(X_train_scaled)


loadings = pd.DataFrame(
    pca.components_.T,            # transpose so rows=features
    index=X.columns,              # feature names
    columns=[f"PC{i+1}" for i in range(pca.n_components_)]
)
## 3. See the raw loadings for PC1 (signed contributions)
print("PC1 loadings (largest positive → negative):")
print(loadings["PC1"].sort_values(ascending=False).head(10))

## 4. If you want the magnitude of contribution (absolute values):
abs_contrib = loadings["PC5"].abs()
print("\nPC1 absolute contributions (top 10):")
print((abs_contrib / abs_contrib.sum()).sort_values(ascending=False).head(10))

## 5. Optionally, view all components at once
print("\nAll feature loadings:")
print(loadings)

n_clusters = X_train_pca.shape[1]
clusterings = cluster_features_pca(loadings, 2)


X_test_pca = pca.transform(X_test_scaled)

print("Original number of features:", X_train.shape[1])
print("Number of PCA components:", X_train_pca.shape[1])

## Optional: Plot the explained variance ratio
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.show()

## 6. Train a Logistic Regression classifier on the PCA-transformed data
## For imbalanced data, it can be useful to set class_weight parameter to "balanced"

clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

## Baseline Training Latency
baseline_start = time.perf_counter()
clf.fit(X_train_scaled, y_train)
baseline_end   = time.perf_counter()
baseline_time = baseline_end - baseline_start

clf_pca = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
## PCA Training Latency
pca_start = time.perf_counter()
clf_pca.fit(X_train_pca, y_train)
pca_end   = time.perf_counter()
pca_time = pca_end - pca_start



print(f"Elapsed baseline training time: {baseline_end - baseline_start:.6f} seconds")
print(f"Elapsed PCA training time: {pca_end - pca_start:.6f} seconds")
print(f"Ratio Comparison PCA/Baseline: {pca_time/baseline_time}")



## 7. Evaluate the model
y_pred = clf.predict(X_test_scaled)
y_proba = clf.predict_proba(X_test_scaled)[:, 1]
fpr_o, tpr_o, _ = roc_curve(y_test, y_proba)
auc_o = roc_auc_score(y_test, y_proba)

y_pred_pca = clf_pca.predict(X_test_pca)
y_proba_pca = clf_pca.predict_proba(X_test_pca)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba_pca)
auc = roc_auc_score(y_test, y_proba_pca)




plt.figure()
plt.plot(fpr_o, tpr_o, color="green", label = "Original")
plt.plot(fpr, tpr, color = "red", label = "PCA")
plt.plot([0, 1], [0, 1], color = "blue", linestyle='--', label = "Random Classifier")  # chance line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (AUC = {auc_o:.3f})')
plt.grid(True)
plt.legend()
plt.show()

## 2. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(values_format='d')  # integer format
plt.title('Confusion Matrix')
plt.show()


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nROC AUC Score (Original):", auc_o)
print("\nROC AUC Score (PCA):", auc)
