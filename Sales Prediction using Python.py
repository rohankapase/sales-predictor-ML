import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# 1. LOAD DATASET
# Loading the advertising data from the CSV file
df = pd.read_csv('Advertising.csv')

# 2. DATA CLEANING
# Remove the unnecessary index column 'Unnamed: 0' if it exists
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# 3. EXPLORATORY DATA ANALYSIS (EDA)
# Correlation heatmap helps to see which advertising platform affects Sales the most
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Analysis: Platforms vs Sales")
plt.show()

# 4. PREPARING DATA
# X contains independent variables (features), y contains the dependent variable (target)
X = df[['TV', 'Radio', 'Newspaper']] 
y = df['Sales']

# Splitting data: 80% for training and 20% for testing the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. MODEL TRAINING
# Initializing and training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)



# 6. MODEL EVALUATION
# Making predictions on the test set to calculate accuracy
predictions = model.predict(X_test)
accuracy = metrics.r2_score(y_test, predictions)

print(f"Model Accuracy (R2 Score): {accuracy * 100:.2f}%")
print(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, predictions):.2f}")

# 7. VISUALIZING ACTUAL VS PREDICTED SALES
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', edgecolor='white', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2) # Diagonal line
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales Performance")
plt.show()

# 8. PREDICTING FOR A NEW CAMPAIGN
# We use a DataFrame here to avoid the "feature names" warning
new_campaign_data = pd.DataFrame([[200, 50, 10]], columns=['TV', 'Radio', 'Newspaper'])
predicted_val = model.predict(new_campaign_data)

print("\n--- New Campaign Prediction ---")
print(f"Budget Plan: TV=200, Radio=50, Newspaper=10")
print(f"Estimated Sales Output: {predicted_val[0]:.2f} units")