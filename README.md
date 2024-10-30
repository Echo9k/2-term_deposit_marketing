# **Term Deposit Marketing Campaign Analysis**

## **Project Overview**

This project aims to develop a sophisticated machine learning system that analyzes call center data to enhance the success rate of marketing campaigns. Specifically, we are focused on increasing the likelihood of customer subscriptions to financial products, such as term deposits, offered by our clients. The ultimate goal is to create an adaptive, interpretable model that not only achieves high predictive accuracy but also provides actionable insights for our clients to optimize their marketing strategies.

## **Data Source**

The dataset originates from the direct marketing efforts of a European banking institution. The marketing campaign involves multiple phone calls made to customers to promote a term deposit product. Term deposits are short-term financial instruments with fixed maturity dates, typically ranging from one month to a few years. It is crucial for customers to understand that funds deposited in a term deposit cannot be withdrawn until the maturity date. To protect customer privacy, all personally identifiable information has been anonymized.

### **Data Attributes**

The dataset includes the following attributes:

- **Age**: Customer's age (numeric)
- **Job**: Type of job (categorical)
- **Marital Status**: Marital status of the customer (categorical)
- **Education**: Level of education (categorical)
- **Default**: Indicates if the customer has credit in default (binary)
- **Balance**: Average yearly balance in euros (numeric)
- **Housing Loan**: Indicates if the customer has a housing loan (binary)
- **Personal Loan**: Indicates if the customer has a personal loan (binary)
- **Contact Type**: Type of communication used to contact the customer (categorical)
- **Last Contact Day**: Day of the last contact within the month (numeric)
- **Last Contact Month**: Month of the last contact within the year (categorical)
- **Duration**: Duration of the last contact in seconds (numeric)
- **Campaign Contacts**: Number of contacts made during this campaign, including the last contact (numeric)

### **Target Variable**

- **Subscription (y)**: Indicates whether the customer subscribed to a term deposit (binary: yes/no)

## **Project Goals**

The primary objective is to accurately predict whether a customer will subscribe to a term deposit based on the provided attributes.

### **Performance Metrics**

The success of the model will be evaluated based on its accuracy. Our target is to achieve an accuracy rate of 81% or higher, validated through 5-fold cross-validation, with the final performance score reported as the average across all folds.

## **Additional Objectives**

- **Customer Segmentation**: Identify and prioritize customer segments that are more likely to subscribe to the term deposit.
  
- **Feature Importance**: Determine which features most significantly influence a customer's decision to subscribe, enabling our clients to focus their marketing efforts more effectively.
