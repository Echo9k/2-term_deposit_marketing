# Regression-OLS

## Overall Model Performance

- **R-squared:** This indicates that approximately 13.2% of the variation in the campaign outcome (dependent variable) can be explained by the independent variables in the model.

- **F-statisti**: This suggests that the overall model is statistically significant, meaning that at least one of the independent variables is significantly related  to the campaign outcome.

- **Log-Likelihood, AIC, and BIC:** These are measures of model fit and complexity. Lower values generally indicate a better-fitting model.

## How to interpret Individual Variable Effects

The coefficients and their p-values in the table represent the estimated effect of each independent variable on the campaign outcome, while controlling for the other variables in the model.

### Interpretation {collapsible="true"}
**Significant Variables:** Variables with a p-value less than 0.05 (typically) are considered statistically significant. These variables have a non-zero impact on the campaign outcome.

**Coefficient Interpretation:**

- A positive coefficient indicates that an increase in the variable is associated with an increase in the campaign outcome.
- A negative coefficient indicates that an increase in the variable is associated with a decrease in the campaign outcome.


### **Statistically Significant Variables (p < 0.05)**

Let's focus on the variables that are statistically significant and interpret their coefficients.

#### **1. Job Category** {collapsible="true"}

- job_housemaid
  - **Coefficient**: 0.6838
  - **p-value**: 0.033
  - **Interpretation**: Being a housemaid is associated with an increase of approximately 0.68 calls compared to the reference job category (likely the omitted category in dummy variables, such as 'management' or 'unknown').

#### **2. Marital Status** {collapsible="true"}

- marital_divorced

  - **Coefficient**: 0.2601
  - **p-value**: 0.021

- marital_married

  - **Coefficient**: 0.4325
  - **p-value**: 0.000

- marital_single

  - **Coefficient**: 0.3986
  - **p-value**: 0.000

- Interpretation:

  - All marital statuses listed are associated with an increase in the number of calls compared to the reference category.
  - Married customers require approximately 0.43 more calls, singles about 0.40 more calls, and divorced customers about 0.26 more calls than the reference group.

#### **3. Education Level** {collapsible="true"}

- education_primary

  - **Coefficient**: 0.3056
  - **p-value**: 0.022

- education_secondary

  - **Coefficient**: 0.2611
  - **p-value**: 0.005

- Interpretation:

  - Customers with primary education require approximately 0.31 more calls, and those with secondary education require about 0.26 more calls than the reference group (possibly 'tertiary' or 'unknown').

#### **4. Contact Method** {collapsible="true"}

- contact_cellular

  - **Coefficient**: -0.2438
  - **p-value**: 0.010

- contact_telephone

  - **Coefficient**: 0.9783
  - **p-value**: 0.000

- contact_unknown

  - **Coefficient**: 0.3568
  - **p-value**: 0.004

- Interpretation:

  - **Cellular**: Using a cellular phone is associated with a decrease of approximately 0.24 calls compared to the reference category.
  - **Telephone**: Using a landline telephone is associated with an increase of about 0.98 calls.
  - **Unknown**: When the contact method is unknown, there's an increase of about 0.36 calls.
  - **Implication**: Cellular contacts may be more efficient, requiring fewer calls to reach the customer.

#### **5. Month of Contact** {collapsible="true"}

- month_aug

  - **Coefficient**: 2.1689
  - **p-value**: 0.000

- month_jul

  - **Coefficient**: 1.6041
  - **p-value**: 0.000

- month_jun

  - **Coefficient**: 0.5736
  - **p-value**: 0.024

- month_oct

  - **Coefficient**: -1.8611
  - **p-value**: 0.000

- Interpretation:

  - **August**: Requires about 2.17 more calls compared to the reference month.
  - **July**: Requires about 1.60 more calls.
  - **June**: Requires about 0.57 more calls.
  - **October**: Requires approximately 1.86 fewer calls.
  - **Implication**: Certain months require significantly more or fewer calls, possibly due to seasonal factors or campaign timing.

#### **6. Day Encoded Variables** {collapsible="true"}

- day_encoded_high

  - **Coefficient**: 0.2842
  - **p-value**: 0.013

- day_encoded_low

  - **Coefficient**: 0.6004
  - **p-value**: 0.000

- day_encoded_medium

  - **Coefficient**: 0.2067
  - **p-value**: 0.015

- Interpretation:

  - Higher encoded days are associated with an increase in the number of calls.
  - **Low**: Highest increase (~0.60 calls)
  - **High**: Moderate increase (~0.28 calls)
  - **Medium**: Slight increase (~0.21 calls)
  - **Implication**: The day of contact has a significant impact on the number of calls required.

#### **7. Other Variables** {collapsible="true"}

- job_entrepreneur
  - **Coefficient**: 0.3922
  - **p-value**: 0.110 (Not significant at 0.05 level but close)

- education_tertiary
  - **Coefficient**: 0.1951
  - **p-value**: 0.084 (Not significant at 0.05 level but close)

- month_feb
  - **Coefficient**: 0.4438
  - **p-value**: 0.073 (Not significant at 0.05 level but close)

- Interpretation:
  - These variables show a trend but are not statistically significant at the 5% level.

### **Variables Not Statistically Significant**

Variables with p-values greater than 0.05 are not considered statistically significant and may not have a meaningful effect on the dependent variable within this model:

- **Age**
- **Default**
- **Balance**
- **Housing**
- **Loan**
- **Most Job Categories**
- **Education Unknown**
- **Months**: April, December, January, March, May, November
- **Job Categories**: Management, Retired, Self-employed, Services, Student, Technician, Unemployed, Unknown

### **Model Diagnostics** {collapsible="true"}

- **Omnibus Test**: Indicates non-normality of residuals (p-value close to 0)
- **Durbin-Watson Statistic**: 2.065 (close to 2, suggesting no autocorrelation in residuals)
- **Jarque-Bera (JB) Test**: High value (58938.851), indicating non-normality
- **Skewness**: 3.670 (positive skew)
- **Kurtosis**: 26.599 (leptokurtic distribution)

**Implications**:

- **Non-Normal Residuals**: The residuals are not normally distributed, which can affect the validity of the model's inference.
- **Skewness and Kurtosis**: Suggest that the data have outliers or are not symmetrically distributed.