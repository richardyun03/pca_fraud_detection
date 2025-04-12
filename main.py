import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
pca = PCA(n_components=0.95, random_state=42)  # alternatively, specify an integer number of components.
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

# 6. Train a Logistic Regression classifier on the PCA-transformed data
# For imbalanced data, it can be useful to set class_weight parameter to "balanced"
clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
clf.fit(X_train_pca, y_train)

# 7. Evaluate the model
y_pred = clf.predict(X_test_pca)

y_proba = clf.predict_proba(X_test_pca)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)




plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')  # chance line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (AUC = {auc:.3f})')
plt.grid(True)
plt.show()

# 2. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.figure()
disp.plot(values_format='d')  # integer format
plt.title('Confusion Matrix')
plt.show()

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, clf.predict_proba(X_test_pca)[:, 1]))
