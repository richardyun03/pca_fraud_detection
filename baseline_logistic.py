from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


def baseline_logistic(X_train, X_test, Y_train, Y_test):


    # 6. Train a Logistic Regression classifier on the PCA-transformed data
    # For imbalanced data, it can be useful to set class_weight parameter to "balanced"
    clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    clf.fit(X_train, Y_train)

    # 7. Evaluate the model
    y_pred = clf.predict(X_test)

    y_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, y_proba)
    auc = roc_auc_score(Y_test, y_proba)

    return fpr, tpr, y_pred, auc, clf