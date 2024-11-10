# Data Manipulation & Analysis
import numpy as np
import pandas as pd

# File & System Operations
import pathlib
import os
import sys
import toml


from sklearn.pipeline import Pipeline
from hyperopt import STATUS_OK
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, classification_report, confusion_matrix

# %% Configuration Handling
def load_config(config_path='config.toml'):
    return toml.load(config_path)

# %% Load and Transform Data
def load_data(config):
    data = pd.read_csv(config['paths']["data"]['raw'])
    target_col = config['data']["separation"]["target"]
    features = data.drop(columns=[target_col])
    target = data[target_col]
    return features, target

def create_numerical_transformer():
    """Creates a pipeline for processing numerical features."""
    return Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5, weights='uniform')),  # KNN Imputer for missing values
        ('scaler', StandardScaler())  # Standardize numerical features
    ])

def create_categorical_transformer():
    """Creates a pipeline for processing categorical features."""
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),  # Impute with 'unknown'
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
    ])

def create_preprocessor(numerical_features, categorical_features):
    """Creates a ColumnTransformer combining numerical and categorical transformers."""
    numerical_transformer = create_numerical_transformer()
    categorical_transformer = create_categorical_transformer()

    return ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
        ]
    )

def add_interaction_features(processed_data, feature_names):
    """
    Adds interaction features to the processed dataset.
    
    Args:
        processed_data: The processed dataset as a NumPy array or DataFrame.
        feature_names: List of feature names after transformation.
    
    Returns:
        A DataFrame with new interaction features.
    """
    processed_df = pd.DataFrame(processed_data, columns=feature_names)
    
    # Adding interaction features
    processed_df["age_balance"] = processed_df["num__age"] * processed_df["num__balance"]
    processed_df["age_duration"] = processed_df["num__age"] * processed_df["num__duration"]
    processed_df["age_campaign"] = processed_df["num__age"] * processed_df["num__campaign"]
    processed_df["balance_duration"] = processed_df["num__balance"] * processed_df["num__duration"]
    processed_df["balance_campaign"] = processed_df["num__balance"] * processed_df["num__campaign"]
    
    return processed_df

# Define a function to log parameters, metrics, and model to MLflow
def log_results(y_test, y_pred, model, params):
    # Log hyperparameters
    mlflow.log_param('n_estimators', params.get('n_estimators', 'N/A'))  # Handle missing params for TPOT
    mlflow.log_param('max_depth', params.get('max_depth', 'N/A'))  # Handle missing params for TPOT
    mlflow.log_param('min_samples_split', params.get('min_samples_split', 'N/A'))  # Handle missing params for TPOT

    # Log key metrics
    mlflow.log_metric('precision', precision_score(y_test, y_pred))
    mlflow.log_metric('recall', recall_score(y_test, y_pred))
    mlflow.log_metric('f1_score', f1_score(y_test, y_pred))
    mlflow.log_metric('average_precision', average_precision_score(y_test, y_pred))

    # Log classification report as a text artifact
    mlflow.log_text(classification_report(y_test, y_pred), artifact_file="classification_report.txt")

    # Log confusion matrix as a text artifact
    mlflow.log_text(str(confusion_matrix(y_test, y_pred)), artifact_file="confusion_matrix.txt")

    # Log the model
    mlflow.sklearn.log_model(model, "model")

# Define the objective function
from hyperopt import STATUS_OK
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import mlflow

def objective(params):
    with mlflow.start_run(nested=True):  # Nested run for each evaluation
        # Extract hyperparameters
        n_estimators = int(params['n_estimators'])
        max_depth = int(params['max_depth'])
        min_samples_split = int(params['min_samples_split'])

        # Define and train the model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Make predictions and calculate metrics
        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred, average='binary')  # Adjust if multi-class

        # Log hyperparameters and results to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("precision", precision)
        log_results(y_test, y_pred, model, params)

        # Return a dictionary with status and loss (to minimize)
        return {'loss': -precision, 'status': STATUS_OK}
