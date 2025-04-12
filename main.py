import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from baseline_logistic import *
from autoencoder import *

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

DATA_DIR = "data/Base.csv"


# Download latest version
# path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

# print("Path to dataset files:", path)

# 1. Load the dataset. Adjust the file path if necessary.
# For example, the common Kaggle Credit Card dataset.
data = pd.read_csv(DATA_DIR)
print("First 5 records:", data.head())


# 2. Separate features and target variable.
# Assume the target column is named 'Class' where 1 indicates fraud and 0 indicates normal
X = data.drop('fraud_bool', axis=1)
print(X.shape)
y = data['fraud_bool']
X = X.select_dtypes(include=[np.number]).fillna(0)
print(X.shape)
# 3. Split the data into training and testing sets.
# Stratify to maintain the imbalance ratio between classes in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)



# 4. Feature Scaling: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Apply PCA
# You can choose a threshold for explained variance. In this case, we'll try to retain 95% variance.
pca = PCA(n_components=1, random_state=42)  # alternatively, specify an integer number of components.
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Original number of features:", X_train.shape[1])
print("Number of PCA components:", X_train_pca.shape[1])

# Optional: Plot the explained variance ratio
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_) * 100, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.show()

fpr_pca_log, tpr_pca_log, y_pred_pca_log, auc_pca_log, clf_pca_log = baseline_logistic(X_train_pca, X_test_pca, y_train, y_test) #PCA Logisitc Classifier

fpr_log, tpr_log, y_pred_log, auc_log, clf_log = baseline_logistic(X_train_scaled, X_test_scaled, y_train, y_test)

fpr_autoencoder_pca, tpr_autoencoder_pca, y_pred_autoencoder_pca, auc_autoencoder_pca, clf_autoencoder_pca = autoencoder_classifier(X_train_pca, X_test_pca, y_train, y_test)

fpr_autoencoder, tpr_autoencoder, y_pred_autoencoder, auc_autoencoder, clf_autoencoder = autoencoder_classifier(X_train_scaled, X_test_scaled, y_train, y_test)







explained_variance = pca.explained_variance_ratio_

# Scree Plot
# plt.figure(figsize=(8, 5))
# plt.plot(np.arange(1, len(explained_variance) + 1), explained_variance, 'o-', linewidth=2)
# plt.title('Scree Plot')
# plt.xlabel('Principal Component')
# plt.ylabel('Explained Variance Ratio')
# plt.grid(True)
# plt.show()






# plt.figure()
# plt.plot(fpr_pca_log, tpr_pca_log)
# plt.plot([0, 1], [0, 1], linestyle='--')  # chance line
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title(f'ROC Curve (AUC = {auc_pca_log:.3f})')
# plt.grid(True)
# plt.show()

# # 2. Confusion matrix
# cm = confusion_matrix(y_test, y_pred_pca_log)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# plt.figure()
# disp.plot(values_format='d')  # integer format
# plt.title('Confusion Matrix')
# plt.show()

# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_pca_log))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_pca_log))
# print("\nROC AUC Score:", roc_auc_score(y_test, clf_pca_log.predict_proba(X_test_pca)[:, 1]))







# plt.figure()
# plt.plot(fpr_log, tpr_log)
# plt.plot([0, 1], [0, 1], linestyle='--')  # chance line
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title(f'ROC Curve (AUC = {auc_log:.3f})')
# plt.grid(True)
# plt.show()

# # 2. Confusion matrix
# cm_log = confusion_matrix(y_test, y_pred_log)
# disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log)
# plt.figure()
# disp_log.plot(values_format='d')  # integer format
# plt.title('Confusion Matrix')
# plt.show()

# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_log))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_log))
# print("\nROC AUC Score:", roc_auc_score(y_test, clf_log.predict_proba(X_test_scaled)[:, 1]))





# plt.figure()
# plt.plot(fpr_autoencoder_pca, tpr_autoencoder_pca)
# plt.plot([0, 1], [0, 1], linestyle='--')  # chance line
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title(f'ROC Curve (AUC = {auc_autoencoder_pca:.3f})')
# plt.grid(True)
# plt.show()

# # 2. Confusion matrix
# cm_autoencoder_pca = confusion_matrix(y_test, y_pred_autoencoder_pca)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm_autoencoder_pca)
# plt.figure()
# disp.plot(values_format='d')  # integer format
# plt.title('Confusion Matrix')
# plt.show()

# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_autoencoder_pca))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_autoencoder_pca))
# print("\nROC AUC Score:", roc_auc_score(y_test, clf_autoencoder_pca.predict_proba(X_test_pca)[:, 1]))







# plt.figure()
# plt.plot(fpr_autoencoder, tpr_autoencoder)
# plt.plot([0, 1], [0, 1], linestyle='--')  # chance line
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title(f'ROC Curve (AUC = {auc_autoencoder:.3f})')
# plt.grid(True)
# plt.show()

# # 2. Confusion matrix
# cm_autoencoder = confusion_matrix(y_test, y_pred_autoencoder)
# disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_autoencoder)
# plt.figure()
# disp_log.plot(values_format='d')  # integer format
# plt.title('Confusion Matrix')
# plt.show()

# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred_autoencoder))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_autoencoder))
# print("\nROC AUC Score:", roc_auc_score(y_test, clf_autoencoder.predict_proba(X_test_scaled)[:, 1]))
