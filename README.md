# **Personalized Discount Allocation Using Causal Inference and Machine Learning**

## **Overview**
This project tackles a critical challenge for an e-commerce company: optimizing discount strategies to maximize profits. By leveraging causal inference and machine learning, we identified specific customer segments that respond positively to discounts, enabling the company to allocate discounts strategically and improve profitability.

### **Key Takeaways**
- **Insights**: Discounts decrease profits on average, but certain customer segments show a positive response.
- **Strategic Segments Identified**: Younger customers with higher predicted sales and specific regional profiles are more receptive to discounts.
- **Impact**: Implementing a targeted discount strategy can enhance profits by focusing on high-impact customer segments.
- **Next Steps**: Deploy personalized discounts to identified segments and continuously refine the strategy based on observed results.

---

## **Business Problem**
The company wants to understand if discounts can be profitable and, if so, identify which customers to target. The profitability equation is:
$$
Profits_i = Sales_i \times 0.05 - Discount_i
$$
The challenge is to move beyond average effects and uncover customer-level insights for personalized discounting.

### **Why It Matters**
- **Cost Efficiency**: Reduces wasteful spending on unprofitable discounts.
- **Customer Engagement**: Increases satisfaction by offering relevant incentives.
- **Competitive Advantage**: Enhances profitability through data-driven personalization.

---

## **Findings and Insights**

### 1. **Overall Impact of Discounts**
**Visual**: Scatter plot showing the relationship between discount and profits for randomized vs. non-randomized data.

- **Non-Randomized Data**: Suggested a misleading positive relationship due to confounding factors.
- **Randomized Data**: Revealed that, on average, increasing discounts reduces profits.

**Key Insight**: Discounts are not universally beneficial and should not be offered indiscriminately.

---

### 2. **Identification of High-Impact Customer Segments**
**Visual**: Heterogeneous Effects Plot highlighting high-TE (treatment effect) customer clusters.

Through advanced modeling, we identified specific customer segments that respond positively to discounts:

- **Age Group**: Younger customers (below 35 years old) showed higher positive treatment effects.
- **Predicted Sales**: Customers with higher predicted sales volumes are more likely to increase profits when given discounts.
- **Geographic Regions**: Certain regions, represented by higher average sales in their state, responded better to discounts.

**Key Insight**: Targeting discounts to younger customers with high predicted sales in specific regions maximizes profitability.

---

### 3. **Evaluating the Personalized Strategy**
**Visual 1**: **Cumulative Elasticity Curve** demonstrating the model’s ability to prioritize high-TE customers.

- The curve shows that focusing on the top 60% of customers, sorted by predicted treatment effects, results in a positive elasticity, indicating increased profits from discounts.
- Customers below the 40th percentile exhibited negative treatment effects, suggesting discounts to them would decrease profits.

**Visual 2**: **Cumulative Gain Curve** compares the effectiveness of the treatment effect model with that of using customer age as a heuristic.

- The model outperforms age-based targeting by better distinguishing customers who will generate incremental profits when offered discounts.

**Key Insight**: The model-based approach significantly improves the identification of profitable customers over simple heuristics.

---

## **Actionable Recommendations**

1. **Implement a Targeted Discount Strategy**:
   - **Focus on Customers**:
     - **Age**: Under 35 years old.
     - **Predicted Sales**: Upper 50th percentile of sales predictions.
     - **Regions**: States with historically higher average sales.
   - **Discount Allocation**:
     - Offer discounts primarily to customers who fall into the high-TE segments identified.
     - Refrain from offering discounts to customers with predicted negative treatment effects.

2. **Monitor and Optimize**:
   - **Performance Tracking**:
     - Measure the impact on profits and customer engagement.
     - Use A/B testing to validate the effectiveness of the targeted strategy.
   - **Continuous Refinement**:
     - Update the model with new data to improve predictions.
     - Incorporate additional customer features such as purchase frequency, product categories, and engagement metrics.

3. **Explore Complementary Strategies**:
   - **Non-Monetary Incentives**:
     - For customers with negative treatment effects, consider loyalty programs or personalized recommendations.
   - **Cross-Selling and Upselling**:
     - Leverage insights to promote complementary products to high-TE customers.

---

## **Impact**

By implementing these recommendations:

- **Profit Increase**: The company can expect an increase in profits by concentrating discounts where they are most effective.
- **Cost Reduction**: Eliminating unprofitable discounts reduces unnecessary expenditure.
- **Customer Satisfaction**: Personalized offers enhance the shopping experience, potentially increasing customer loyalty.

---

## **Methodology Overview**

### **Data Analysis and Modeling**

- **Data Sources**: Combined randomized and non-randomized customer data.
- **Feature Engineering**: Encoded customer states numerically based on average sales, categorized customers into sales prediction bins.
- **Debiasing Techniques**: Applied the Frisch-Waugh-Lovell (FWL) theorem and double/debiased machine learning to control for confounding variables.
- **Machine Learning Model**: Used LightGBM regressor to estimate the Conditional Average Treatment Effects (CATE) at the customer level.

### **Validation**

- **Cumulative Elasticity and Gain Curves**: Assessed the model's ability to identify customers who would positively respond to discounts.
- **Comparison with Baseline**: Demonstrated that the model outperforms simple heuristics like customer age.

---

## **Visual Aids**

1. **Scatter Plot**: Shows the true causal relationship between discounts and profits using randomized data.
2. **Heterogeneous Effects Plot**: Illustrates the distribution of high and low treatment effects among customers.
3. **Cumulative Elasticity Curve**: Visualizes the positive impact of targeting high-TE customers.
4. **Cumulative Gain Curve**: Highlights the increased profits from the model-based strategy compared to the baseline.
- **Scalability**: Expand the personalized approach to other promotional strategies.

---

This enhanced README provides specific insights into the customer segments that are strategic for the company's discounting strategy, based on the analysis conducted. It emphasizes actionable recommendations and the potential impact on the business, aligning with your goal to communicate effectively with executives and stakeholders.

Let me know if there's anything else you'd like to add or modify!
