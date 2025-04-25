import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, ConfusionMatrixDisplay


def kmeans_analysis(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: pd.Series,
    k_values: list[int] = [2,3,4,5],
    random_state: int = 42,
    plot: bool = True
):
    """
    Performs KMeans clustering on PCA-transformed data, selects best k by silhouette score,
    computes cluster sizes and fraud rates, and optionally plots clusters & confusion matrix.

    Returns a dict with:
      - 'best_k', 'silhouette_scores', 'kmeans',
        'train_labels', 'test_labels', 'cluster_sizes', 'fraud_rates'
    """
    # 1. Silhouette analysis to pick k
    sil_scores = {}
    for k in tqdm(k_values, desc="Silhouette analysis", unit="k"):
        # km = KMeans(n_clusters=k, random_state=random_state)
        # labels = km.fit_predict(X_train)
    #     sil_scores[k] = silhouette_score(X_train, labels)
    # best_k = max(sil_scores, key=sil_scores.get)

    # 2. Fit KMeans with best k
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        train_labels = kmeans.fit_predict(X_train)
        test_labels  = kmeans.predict(X_test)
        # 3. Compute cluster sizes & fraud rates on test set
        dfc = pd.DataFrame({
            'cluster': test_labels,
            'isFraud': y_test.reset_index(drop=True)
        })
        cluster_sizes = dfc['cluster'].value_counts().sort_index()
        fraud_rates   = dfc.groupby('cluster')['isFraud'].mean()

        # 4. Optional plotting
        if plot:
            # a) scatter clusters on PC1–PC2
            plt.figure(figsize=(6,5))
            sc = plt.scatter(
                X_test[:,0], X_test[:,1],
                c=test_labels, cmap='tab10', alpha=0.6
            )
            plt.xlabel('PC1'); plt.ylabel('PC2')
            plt.title(f'KMeans (k={k}) in PCA space')
            plt.legend(*sc.legend_elements(), title='cluster')
            plt.grid(True)
            plt.show()

            # b) fraud vs legit overlay
            plt.figure(figsize=(6,5))
            mask = y_test.reset_index(drop=True)==1
            plt.scatter(
                X_test[~mask,0], X_test[~mask,1],
                label='Legit', alpha=0.3
            )
            plt.scatter(
                X_test[ mask,0], X_test[ mask,1],
                label='Fraud', alpha=0.6
            )
            plt.xlabel('PC1'); plt.ylabel('PC2')
            plt.title('Fraud vs Legit on PC1–PC2')
            plt.legend(); plt.grid(True)
            plt.show()

            # c) Confusion matrix of cluster→fraud classification (optional heuristic)
            # if we treat each cluster as a “predict fraud if in cluster X”
            # here we just show the raw matrix of true vs cluster label
            cm = confusion_matrix(y_test, test_labels, labels=sorted(cluster_sizes.index))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                        display_labels=[f"C{k}" for k in sorted(cluster_sizes.index)])
            plt.figure(figsize=(5,4))
            disp.plot(values_format='d')
            plt.title('Confusion Matrix: true fraud vs cluster')
            plt.show()

    return {
        # 'best_k': best_k,
        # 'silhouette_scores': sil_scores,
        'kmeans': kmeans,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'cluster_sizes': cluster_sizes,
        'fraud_rates': fraud_rates
    }


def cluster_features_pca(
    loadings: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
    plot: bool = True
) -> pd.DataFrame:
    """
    Cluster features based on their PCA loadings and optionally plot PC1 vs PC2.

    Parameters
    ----------
    loadings : pd.DataFrame
        DataFrame of shape (n_features, 5) with columns ['PC1','PC2','PC3','PC4','PC5'].
    n_clusters : int
        Number of clusters for KMeans.
    random_state : int
        Random seed for reproducibility.
    plot : bool
        If True, scatter-plot PC1 vs PC2 colored by cluster.

    Returns
    -------
    pd.DataFrame
        A copy of `loadings` with an extra 'cluster' column.
    """
    # 1. Fit KMeans
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(loadings.values)

    # 2. Attach cluster labels
    clustered = loadings.copy()
    clustered['cluster'] = labels

    # 3. Print cluster membership
    for c in sorted(clustered['cluster'].unique()):
        members = clustered[clustered['cluster'] == c].index.tolist()
        print(f"Cluster {c} ({len(members)} features):")
        print("  ", members)

    # 4. Plot PC1 vs PC2 if requested
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            clustered['PC1'],
            clustered['PC2'],
            c=clustered['cluster'],
            cmap='tab10',
            alpha=0.7
        )
        # annotate points with feature names
        for feature, x, y in zip(clustered.index, clustered['PC1'], clustered['PC2']):
            ax.annotate(feature, (x, y), fontsize=8, alpha=0.6)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Feature Clusters (k={n_clusters}) in PC1–PC2 space')
        plt.grid(True)
        plt.show()

    return clustered