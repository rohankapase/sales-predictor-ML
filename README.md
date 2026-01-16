# Sales Prediction using Machine Learning (Python)

### Project Overview

This project is a data science solution designed to predict future sales based on advertising expenditures. Businesses often struggle to allocate their marketing budgets across different platforms. This tool uses **Linear Regression** to analyze how much impact spending on **TV, Radio, and Newspaper** advertisements has on the total sales volume, allowing for data-driven budget optimization.

### Key Features

* **Correlation Analysis:** Identifies the relationship between specific advertising channels and sales growth.
* **Predictive Modeling:** Uses a trained regression model to forecast revenue for any given budget combination.
* **Data Visualization:** Includes heatmaps and scatter plots to visualize trends and model performance.
* **Performance Metrics:** Evaluates accuracy using R-Squared (R2) score and Mean Squared Error (MSE).

### Tech Stack

* **Language:** Python 3.x
* **Libraries:** Pandas (Data Manipulation), NumPy (Numerical Analysis), Scikit-Learn (Machine Learning), Matplotlib & Seaborn (Visualization).

### Machine Learning Pipeline

1. **Exploratory Data Analysis (EDA):** Checking for missing values and visualizing distributions.
2. **Feature Selection:** Analyzing which platforms (TV, Radio, or Newspaper) are the strongest predictors.
3. **Data Splitting:** Dividing the dataset into Training (80%) and Testing (20%) sets.
4. **Model Training:** Implementing the **Linear Regression** algorithm to find the "Best Fit Line."
5. **Testing & Validation:** Predicting sales on the test set and comparing them with actual values.

### Results

* **Model Accuracy:** The system achieved an **R2 Score of ~89.94%**, indicating a high level of reliability.
* **Insight:** TV advertising was found to have the strongest correlation with sales in this dataset.

### How to Run

1. **Clone this repository:**
```bash
git clone https://github.com/rohankapase/sales-predictor-ML.git

```


2. **Install dependencies:**
```bash
pip install pandas scikit-learn matplotlib seaborn

```


3. **Run the script:**
```bash
python sales_prediction.py

```



### Conclusion

By leveraging this model, a product-based business can effectively manipulate their advertising costs to maximize sales and ensure a higher Return on Investment (ROI).

---
