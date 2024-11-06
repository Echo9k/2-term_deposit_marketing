# Term Deposit Marketing Campaign Analysis

This project develops a machine learning system to improve the effectiveness of a bank's marketing campaign by predicting the likelihood of customer subscription to term deposits. We explore various libraries and modeling approaches to identify customers most likely to subscribe, helping the bank optimize its marketing strategy.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Key Components](#key-components)
- [Environment Setup](#environment-setup)
- [Running the Project](#running-the-project)
- [Results](#results)
- [Documentation Links](#documentation-links)

---

## **Project Overview**

The goal of this project is to build an adaptive, interpretable model that can analyze call center data and provide insights to enhance the success rate of a term deposit marketing campaign. We approach the project in multiple stages, from exploratory analysis and causal inference to pre- and post-call classification modeling and customer segmentation.

**Key Objectives:**
- **Primary Goal**: Accurately predict term deposit subscriptions, prioritizing recall for class 1 (subscribers).
- **Secondary Goals**:
    - Understand feature importance and causal relationships between variables.
    - Develop pre- and post-call classification models to optimize marketing efforts.
    - Segment customers for targeted marketing strategies.

---

## **Key Components**

### **1. Exploratory Data Analysis (EDA)**
Initial data exploration and visualization to understand feature distributions and detect patterns.

### **2. Call Prediction Modeling**
Predict the number of calls required for a customer to subscribe, using regression to explore feature engineering options.

### **3. Causal Analysis**
Identify interrelationships between variables using causal inference techniques to understand factors influencing subscriptions.

### **4. Pre-Call Modeling**
Classification model to prioritize customers before contact, helping to optimize marketing efforts.

### **5. Post-Call Prediction**
Classification model that uses additional customer information gathered post-contact to refine subscription likelihood predictions.

### **6. Customer Segmentation**
Clustering analysis to identify distinct customer groups for targeted marketing strategies.

---

## **Environment Setup**

To avoid dependency conflicts, we use separate Conda environments for different stages of the project. The required environments and their corresponding files are:

- **Pre-Call Modeling**: `environment-pre.yaml`
- **Post-Call Modeling**: `environment-post.yaml`
- **Causal Analysis**: `environment-causal.yaml`
- **Clustering**: `environment-cluster.yaml`

### **Setting Up an Environment**

1. **Create the environment**:
   ```bash
   conda env create -f <environment-file>.yaml
   ```
   Replace `<environment-file>` with the appropriate environment file (e.g., `environment-pre.yaml`).

2. **Activate the environment**:
   ```bash
   conda activate <environment-name>
   ```

3. **Add Jupyter Kernel for Each Environment** (optional):
   ```bash
   python -m ipykernel install --user --name=<environment-name>
   ```

---

## **Running the Project**

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Data Preparation**: Ensure that raw data files are placed in the `data/raw/` directory. Interim and processed data will be generated in the `data/interim/` and `data/processed/` folders as you progress.

3. **Run Notebooks**:
   Open Jupyter Notebook or JupyterLab and select the corresponding environment kernel for each notebook.

4. **Track Experiments**: MLflow and other tracking tools are used to log experiments. Check the `mlruns/` folder for details.

---

## **Results**

Results are provided for each analysis stage, including key insights and performance metrics. For a detailed breakdown, see the [Results Documentation](results.md).

### **Summary of Key Results**:
- **EDA**: Initial insights on feature distributions and customer demographics.
- **Call Prediction**: Metrics on the regression model predicting the number of calls.
- **Causal Analysis**: Identified causal relationships that impact customer subscriptions.
- **Pre-Call Model**: Precision-focused model for prioritizing likely subscribers.
- **Post-Call Model**: Improved recall for subscribers after gathering additional customer information.
- **Clustering**: Segmentation of customer groups for tailored marketing strategies.

---

## **Documentation Links**

- [Starter Guide](starter.md): Quick-start setup and project structure.
- [Project Overview](overview.md): Detailed goals, data, and metric explanations.
- [Results Documentation](results.md): Comprehensive results for each analysis phase.

For further technical details, see the [Appendix](appendix.md) for troubleshooting, parameter settings, and advanced configurations.

---

## **Contributing**

For questions or suggestions, please reach out to [project-support@example.com](mailto:project-support@example.com). Contributions are welcome via pull requests.