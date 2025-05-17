import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, ConfusionMatrixDisplay

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