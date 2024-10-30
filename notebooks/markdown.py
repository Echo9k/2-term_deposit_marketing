# %% [markdown]
# ## Set up

# %%
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

# %%
while os.path.basename(cwd := os.getcwd()) != "2-term_deposit_marketing":
    os.chdir("..")
    print(f"{cwd=:}")

sys.path.append("./term_deposit")
import term_deposit as td

# Load the configuration file
config = toml.load('config.toml')
target_col = 'y'

# %% [markdown]
# ## Load and transform the data

# %%
data = pd.read_csv(config['paths']["data"]['raw'])


# Separate the features and the target variable
features = data.drop(columns=[target_col])
target = data[target_col]

# Identifying numerical and categorical features
numerical_features = ['age', 'balance', 'duration', 'campaign']
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact']

# Define transformers for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5, weights='uniform')),  # KNN Imputer for missing values
    ('scaler', StandardScaler())  # Standardize numerical features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),  # Impute with 'unknown'
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Combine both transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Fit and transform the dataset using the preprocessor
processed_features = preprocessor.fit_transform(features)

# Convert processed features back into a DataFrame for inspection
processed_df = pd.DataFrame(processed_features, columns=preprocessor.get_feature_names_out())

# Feature engineering
processed_df["age_balance"] = processed_df["num__age"] * processed_df["num__balance"]
processed_df["age_duration"] = processed_df["num__age"] * processed_df["num__duration"]
processed_df["age_campaign"] = processed_df["num__age"] * processed_df["num__campaign"]
processed_df["balance_duration"] = processed_df["num__balance"] * processed_df["num__duration"]
processed_df["balance_campaign"] = processed_df["num__balance"] * processed_df["num__campaign"]

# %% [markdown]
# ## Modeling

# %%
# Determine the optimal number of clusters using the Elbow method
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(processed_features)
    inertia.append(kmeans.inertia_)

# Use KneeLocator to find the elbow point
kneedle = KneeLocator(k_range, inertia, curve='convex', direction='decreasing')
optimal_k = kneedle.elbow

# Plotting the Elbow method graph with the knee point
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k (t-SNE Data)')
plt.legend()
plt.grid(True)
plt.show()

optimal_k

# %%
# Optional: Apply KMeans clustering to the processed features
kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust the number of clusters as needed
clusters = kmeans.fit_predict(processed_features)

# %%
# Applying 3D PCA
pca_3d = PCA(n_components=3)
pca_result_3d = pca_3d.fit_transform(processed_df)
vis.plot_3d_interactive(pca_result_3d, clusters, 'PCA')

# # Applying 3D t-SNE
# tsne_3d = TSNE(n_components=3, random_state=42)
# tsne_result_3d = tsne_3d.fit_transform(processed_df)
# vis.plot_3d_interactive(tsne_result_3d, clusters, 't-SNE')

# # Applying 3D UMAP
# umap_3d = umap.UMAP(n_components=3, random_state=42)
# umap_result_3d = umap_3d.fit_transform(processed_df)
# vis.plot_3d_interactive(umap_result_3d, clusters, 'UMAP')


# %%
# Adjusting the numerical features to match the transformed column names in the processed DataFrame.

numerical_transformed_features = [f'num__{feature}' for feature in numerical_features]

# Apply Z-score to the correctly named numerical columns
z_scores = np.abs(zscore(processed_df[numerical_transformed_features]))
outliers = (z_scores > 3).any(axis=1)  # Identify rows with any z-score greater than 3

# Calculate the percentage of outliers
outlier_percentage = (outliers.sum() / len(processed_df)) * 100

# Remove outliers for rows where any feature exceeds the Z-score threshold
cleaned_df = processed_df[~outliers]

# Display the percentage of outliers detected
outlier_percentage

# %%
# Applying PCA to the cleaned dataset
pca = PCA()
pca.fit(cleaned_df)

# Calculate the explained variance ratio for each principal component
explained_variance_ratio = pca.explained_variance_ratio_

# Calculate the cumulative explained variance
cumulative_explained_variance = explained_variance_ratio.cumsum()

# Plotting the explained variance and cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', label='Individual Explained Variance')
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='x', linestyle='--', label='Cumulative Explained Variance')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Determine the number of components that account for 85% of the variance
n_components_85_variance = np.argmax(cumulative_explained_variance >= 0.85) + 1

# Apply PCA with the determined number of components
pca_85 = PCA(n_components=n_components_85_variance)
pca_85_result = pca_85.fit_transform(cleaned_df)

# Convert the result back to a DataFrame for inspection
pca_85_df = pd.DataFrame(
    pca_85_result,
    columns=[f'PC{i+1}' for i in range(n_components_85_variance)],
    index=cleaned_df.index
    )

# Display the number of components and the head of the DataFrame
display(pca_85_df.head())
n_components_85_variance

# %%
# Determine the optimal number of clusters using the Elbow method
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_85_result)
    inertia.append(kmeans.inertia_)

# Use KneeLocator to find the elbow point
kneedle = KneeLocator(k_range, inertia, curve='convex', direction='decreasing')
optimal_k = kneedle.elbow

# Plotting the Elbow method graph with the knee point
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k (t-SNE Data)')
plt.legend()
plt.grid(True)
plt.show()

optimal_k

# %%
# Applying t-SNE for dimensionality reduction to 2 components
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_results = tsne.fit_transform(pca_85_df)

# Applying UMAP for dimensionality reduction to 2 components
umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_results = umap_reducer.fit_transform(pca_85_df)

# Plotting the t-SNE results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=5, alpha=0.7)
plt.title('t-SNE Results')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

# Plotting the UMAP results
plt.subplot(1, 2, 2)
plt.scatter(umap_results[:, 0], umap_results[:, 1], s=5, alpha=0.7)
plt.title('UMAP Results')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')

plt.tight_layout()
plt.show()

# %%
# Determining the optimal number of clusters using the Elbow method
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(tsne_results)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow method graph
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k (t-SNE Data)')
plt.grid(True)
plt.show()

# %%
# Define a range of cluster numbers to test
k_range = range(2, 11)

# Lists to store silhouette scores and Calinski-Harabasz scores
silhouette_scores = []
calinski_harabasz_scores = []

# Compute silhouette score and Calinski-Harabasz score for each k
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(tsne_results)
    
    # Calculate the silhouette score
    silhouette_avg = silhouette_score(tsne_results, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    
    # Calculate the Calinski-Harabasz index
    calinski_harabasz_avg = calinski_harabasz_score(tsne_results, cluster_labels)
    calinski_harabasz_scores.append(calinski_harabasz_avg)

# Plotting the silhouette scores and Calinski-Harabasz scores
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Silhouette Score Plot
ax[0].plot(k_range, silhouette_scores, marker='o')
ax[0].set_xlabel('Number of Clusters (k)')
ax[0].set_ylabel('Silhouette Score')
ax[0].set_title('Silhouette Score for Optimal k (t-SNE Data)')
ax[0].grid(True)

# Calinski-Harabasz Index Plot
ax[1].plot(k_range, calinski_harabasz_scores, marker='o', color='orange')
ax[1].set_xlabel('Number of Clusters (k)')
ax[1].set_ylabel('Calinski-Harabasz Score')
ax[1].set_title('Calinski-Harabasz Index for Optimal k (t-SNE Data)')
ax[1].grid(True)

plt.tight_layout()
plt.show()

# %%
# Set the number of clusters based on the analysis
n_clusters = 7

# Applying K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(pca_85_df)

# Calculate the silhouette score and Calinski-Harabasz index for the clustering
silhouette_avg = silhouette_score(pca_85_df, kmeans_labels)
calinski_harabasz_avg = calinski_harabasz_score(pca_85_df, kmeans_labels)

# Adding the cluster labels to the dataset
pca_85_df.loc[:, 'Cluster'] = kmeans_labels

# Display the silhouette score, Calinski-Harabasz index, and the head of the dataset with cluster labels
silhouette_avg, calinski_harabasz_avg
# display(pca_85_df.head())

# 3D plot of the clusters using PCA
pca_result_3d = pca_3d.fit_transform(pca_85_df)
vis.plot_3d_interactive(pca_result_3d, kmeans_labels, 'PCA')

# %%
pca_3d.get_params()

# %%
data_clustered = data.join(pca_85_df["Cluster"])
data_clustered.to_parquet("./data/processed/pca_result.parquet", index=False, compression="brotli")

# %%
# 3D plot of the clusters using t-SNE
# tsne_result_3d = tsne_3d.fit_transform(pca_85_df)
vis.plot_3d_interactive(tsne_result_3d, kmeans_labels, 'PCA')

# %%
# Configure UMAP to have 3 components
umap_3d = umap.UMAP(n_components=3)


umap_result_3d = umap_3d.fit_transform(pca_85_df)
vis.plot_3d_interactive(umap_result_3d, kmeans_labels, 'PCA')

# %%
# Group the data by clusters and calculate descriptive statistics
cluster_summary = data_clustered.groupby('Cluster').describe()

# Display the summary statistics for each cluster
display(cluster_summary)

# Visualize the distribution of numerical features across clusters
numerical_columns = ['age', 'balance', 'duration', 'campaign']
for column in numerical_columns:
    plt.figure(figsize=(10, 6))
    for cluster in data_clustered['Cluster'].unique():
        subset = data_clustered[data_clustered['Cluster'] == cluster]
        sns.kdeplot(subset[column], label=f'Cluster {cluster}', shade=True)
    plt.title(f'Distribution of {column} by Cluster')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Visualize the distribution of categorical features across clusters
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact']
for column in categorical_columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data_clustered, x=column, hue='Cluster')
    plt.title(f'Distribution of {column} by Cluster')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.legend(title='Cluster')
    plt.xticks(rotation=45)
    plt.show()


