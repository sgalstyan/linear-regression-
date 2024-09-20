import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from CSV
data = pd.read_csv('data/student_performance.csv')

# Extract the relevant columns (Inflation Rate as X and Stock Index Value as y)
X = data[['AttendanceRate']].values  # Ensure X is 2D
y = data['FinalGrade'].values  # y is 1D

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Linear Regression model
model = LinearRegression()

# Train the model (fit the model to the training data)
model.fit(X_train, y_train)

# Make predictions using the test data
y_pred = model.predict(X_test)

# Print the model coefficients
print(f"Slope (beta_1): {model.coef_[0]}")
print(f"Intercept (beta_0): {model.intercept_}")

# Plot the data points and regression line
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X_train, model.predict(X_train), color='red', label='Regression Line')
plt.xlabel('AttendanceRate')
plt.ylabel('FinalGrade')
plt.title('Simple Linear Regression: AttendanceRate vs FinalGrade')
plt.legend()
plt.show()

# Calculate MSE and R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
