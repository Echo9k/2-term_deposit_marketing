# Project Overview

### **Objective**
The project aims to develop an adaptive, interpretable machine learning model that can predict the likelihood of customers subscribing to term deposit products in a bank's marketing campaigns. By analyzing call center data, we strive to enhance campaign success rates and provide actionable insights for clients to optimize their marketing strategies.

### **Scope**
This system focuses on identifying potential subscribers based on demographic, financial, and contact information provided in the campaign dataset. The model is designed to support informed decision-making for targeted marketing, helping clients increase engagement and subscription rates for financial products like term deposits.

---

## **Data Source and Attributes**

### **Data Description**
The dataset comes from a European bank’s direct marketing campaign, involving phone calls to potential clients for term deposit promotion. Term deposits are fixed-maturity, non-withdrawable financial instruments, and the dataset used maintains client privacy through anonymized records.

### **Attributes**
Key attributes in the dataset include:

- **Demographic Information**: Age, job type, marital status, education
- **Financial Indicators**: Default status, average yearly balance, housing loan, personal loan status
- **Contact Details**: Contact type, last contact day and month, duration of the last call
- **Campaign Engagement**: Number of contacts in the current campaign

### **Target Variable**
- **Subscription (y)**: Binary indicator of term deposit subscription (yes/no)

---

## **Project Goals and Performance Metrics**

### **Primary Objective**
The model’s primary objective is to predict the likelihood of term deposit subscription accurately, with a focus on **maximizing recall for class 1 (subscribers)** as our key performance metric. Achieving high recall ensures that the model effectively identifies customers most likely to subscribe, even at the expense of some precision. Validation will be performed using 5-fold cross-validation, with results averaged across all folds.

### **Secondary Objectives**
1. **Customer Segmentation**: Identify and prioritize customer segments more inclined to subscribe.
2. **Feature Importance**: Pinpoint key factors influencing subscription decisions to help clients focus their marketing strategies effectively.