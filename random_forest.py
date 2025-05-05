import kagglehub
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


def randomforest(X_train, X_test, Y_train, Y_test):



# Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, Y_train)

    # Predict probabilities
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(Y_test, y_pred_proba)
    auc_score = roc_auc_score(Y_test, y_pred_proba)

    # Threshold probabilities to get class labels
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Print classification report and confusion matrix
    print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
    print("\nClassification Report:\n", classification_report(Y_test, y_pred))
    print("\nROC AUC Score:", auc_score)

    return fpr, tpr, y_pred_proba, auc_score, clf




