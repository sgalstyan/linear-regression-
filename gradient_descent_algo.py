import pandas as pd

# Load the dataset
data = pd.read_csv('data/economic_indicators_dataset_2010_2023.csv')

# Extract the relevant columns (Inflation Rate as X and Stock Index Value as y)
X = data['Inflation Rate (%)'].values
y = data['Stock Index Value'].values

# Initialize parameters
beta_0 = 0  # Intercept
beta_1 = 0  # Slope
alpha = 0.01  # Learning rate
iterations = 1000
n = len(X)

# Gradient Descent algorithm
for i in range(iterations):
    y_pred = beta_0 + beta_1 * X
    error = y_pred - y
    beta_0 -= alpha * (2/n) * sum(error)
    beta_1 -= alpha * (2/n) * sum(error * X)

print(f"Final values: Intercept = {beta_0}, Slope = {beta_1}")
