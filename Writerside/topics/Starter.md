# Starter Guide

<!-- This starter page introduces the project structure, environment setup, and dependencies for smooth onboarding. -->

## **Project Overview**
This project aims to develop a machine learning model to predict term deposit subscriptions in marketing campaigns. To explore different modeling approaches, we utilize multiple libraries across separate environments, minimizing dependency conflicts.

## **Table of Contents**
1. [Project Structure](#project-structure)
2. [Setting Up Dependencies](#setting-up-dependencies)
3. [Environment-Specific Notebooks](#environment-specific-notebooks)
4. [Getting Started](#getting-started)
5. [Further Information](#further-information)

## **Project Structure**

Below is an overview of the main directories and files:

```plaintext
.
├── Makefile
├── README.md
├── config.toml
├── data/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── environment-causal.yaml
├── environment-cluster.yaml
├── environment-post.yaml
├── environment-pre.yaml
├── mlruns/
├── models/
├── notebooks/
├── supporting_files/
└── term_deposit/
```

## **Getting Started**

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Set Up the Required Environment**
   Choose and activate the environment based on the notebook you plan to run.

3. **Run Notebooks in Jupyter**
   Open Jupyter and select the appropriate environment kernel for each notebook:
   ```bash
   jupyter notebook
   ```

4. **Install Jupyter Kernel for Each Environment**
   ```bash
   python -m ipykernel install --user --name=<environment-name>
   ```

---
<details>
<summary><strong>Data Directory (`data/`)</strong></summary>
Contains datasets at various stages:
- **raw/**: Original, unprocessed data files.
- **interim/**: Intermediate data for transformation steps.
- **processed/**: Final data used for modeling.
</details>

<details>
<summary><strong>Notebooks Directory (`notebooks/`)</strong></summary>
Contains project notebooks for EDA, modeling, and analysis:
- `1-exploration.ipynb`: Exploratory analysis
- `2-number_of_calls_prediction.ipynb`: Regression model for predicting calls needed
- `3-causal_analysis.ipynb`: Causal analysis of variables
- `4-pre_call-predict_first5_calls_precision.ipynb`: Pre-call modeling
- `5-post_call-prediction_model.ipynb`: Post-call reclassification
- `6-clustering.ipynb`: Clustering for customer segmentation
</details>

---

## **Setting Up Dependencies** {collapsible="true"}

### Why Separate Environments?

To explore different libraries without dependency conflicts, each stage uses a dedicated Conda environment. Install the environments below:

<tabs>
<tab title="Pre-Call Modeling">
<code-block lang="bash">
conda env create -f environment-pre.yaml
conda activate environment-pre
</code-block>
</tab>

<tab title="Post-Call Modeling">
<code-block lang="bash">
conda env create -f environment-post.yaml
conda activate environment-post
</code-block>
</tab>

<tab title="Causal Analysis">
<code-block lang="bash">
conda env create -f environment-causal.yaml
conda activate environment-causal
</code-block>
</tab>

<tab title="Clustering">
<code-block lang="bash">
conda env create -f environment-cluster.yaml
conda activate environment-cluster
</code-block>
</tab>
</tabs>

---

### **Notebook Overview** {collapsible="true"}

1. **Exploration** (`1-exploration.ipynb`):
Perform exploratory data analysis (EDA) to understand the data structure and patterns.
    - **Environment**: `environment-pre.yaml`
    - **Libraries**: Pandas, Matplotlib, Seaborn

2. **Call Prediction** (`2-number_of_calls_prediction.ipynb`):
Predict the number of calls required for a customer subscription.
    - **Environment**: `environment-pre.yaml`
    - **Libraries**: Scikit-learn

3. **Causal Analysis** (`3-causal_analysis.ipynb`):
Assess causal relationships among variables using causal inference.
    - **Environment**: `environment-causal.yaml`
    - **Libraries**: DoWhy, EconML

4. **Pre-Call Modeling** (`4-pre_call-predict_first5_calls_precision.ipynb`):
Customer classification prior to contact.
    - **Environment**: `environment-pre.yaml`
    - **Libraries**: Scikit-learn, MLflow, Hyperopt

5. **Post-Call Prediction** (`5-post_call-prediction_model.ipynb`):
Reclassification after contact with additional customer information.
    - **Environment**: `environment-post.yaml`
    - **Libraries**: PyCaret, H2O

6. **Clustering** (`6-clustering.ipynb`):
Segment customers based on demographic and behavioral features.
    - **Environment**: `environment-cluster.yaml`
    - **Libraries**: Scikit-learn, K-Means, PCA

---


## **Further Information**

For any questions or troubleshooting, consult the `README.md` or contact [Projects@emailmaks.uk](mailto:Projects@emailmaks.uk).

