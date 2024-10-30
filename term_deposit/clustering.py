# %% [markdown]
# ## Set up

# %% Imports
# Data Manipulation & Analysis
import numpy as np
import pandas as pd

# File & System Operations
import pathlib
import os
import sys
import toml

# Data Preprocessing & Transformation (scikit-learn)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Dimensionality Reduction & Manifold Learning (scikit-learn)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Clustering (scikit-learn)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Data Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import seaborn as sns
import plotly.express as px

# Outlier Detection (scipy)
from scipy.stats import zscore

# Non-linear Dimensionality Reduction
import umap.umap_ as umap
from kneed import KneeLocator



# %% Preprocessing Pipeline Setup
def setup_preprocessing_pipeline(numerical_features, categorical_features):
    numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5, weights='uniform')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
        ]
    )

# %% Clustering Function
def apply_clustering(preprocessed_data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(preprocessed_data)
    return labels, kmeans

# %% Visualization Functions
def plot_clusters_3d(data, labels, method='PCA'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', alpha=0.6)
    plt.title(f'3D Clustering Visualization using {method}')
    plt.show()

