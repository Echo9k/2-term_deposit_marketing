# Practical implications

This section outlines the results of each stage in the Term Deposit Marketing Campaign Analysis, including the baseline subscription rate, pre-call model performance, and post-call model refinement.

---

## **Baseline**

- **Subscription Rate**: The initial data shows a **41% subscription rate on the first call**, which serves as our baseline for assessing model improvements.

---

## **Pre-Call Model Results**

In this phase, we developed a model to predict whether a customer would be identified as a potential subscriber before making contact. Here are the key performance metrics:

### **1. Precision and Recall**

- **Class 0 (Non-identified Customers)**:
    - **Precision**: 0.12, indicating that only 12% of the customers predicted as non-subscribers were correctly identified.
    - **Recall**: 0.53, meaning the model correctly identifies 53% of non-subscribing customers in total calls.

- **Class 1 (Identified Customers)**:
    - **Precision**: 0.90, showing high precision in identifying potential subscribers.
    - **Recall**: 0.53, with the model correctly identifying 53% of actual subscribers.

### **2. F1-Score**

- **Class 0**: F1-Score of 0.19, reflecting a significant imbalance in the model's ability to correctly identify non-subscribers.
- **Class 1**: F1-Score of 0.67, indicating relatively strong precision but limited ability to recall all subscribers.

### **3. Overall Accuracy**

- **Accuracy**: 53%, showing that the model is correct in just over half of all cases, highlighting potential areas for improvement, particularly in recall for both classes.

---

## **Post-Call Model Results**

Following initial customer contact, additional data was used to refine the model and improve classification accuracy. The results below represent an average performance across multiple folds in cross-validation.

| Metric                | Fold 8 Result |
|-----------------------|---------------|
| **Accuracy**          | 0.7200        |
| **AUC**               | 0.8662        |
| **Recall (Overall)**  | 0.9113        |
| **Precision**         | 0.1945        |
| **F1-Score**          | 0.3206        |
| **Kappa**             | 0.2284        |
| **MCC**               | 0.3375        |
| **Recall (Class 1)**  | 0.0000        |

### Key Observations

- **Accuracy** increased significantly to **72%**, indicating a substantial improvement over the pre-call model.
- **AUC (Area Under Curve)** of **0.8662** demonstrates a strong overall classification ability across classes.
- **Recall (Overall)** is high at **91.13%**, showing that the model effectively identifies a large proportion of actual subscribers after gathering post-call data.
- **Class 1 Recall** remains challenging, as reflected in a recall of **0.0000** in this fold, signaling the need for further refinement to better capture all subscriber cases.