import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Original dataset: house sizes (square feet) and their prices
house_sizes = np.array([500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]).reshape(-1, 1)
house_prices = np.array([150000, 185000, 200000, 210000, 240000, 260000, 275000, 290000, 310000, 330000, 350000])

# 1. Expand the Dataset
additional_sizes = np.array([3250, 3500, 3750, 4000, 4250]).reshape(-1, 1)
additional_prices = np.array([370000, 390000, 410000, 430000, 450000])
house_sizes = np.vstack((house_sizes, additional_sizes))
house_prices = np.append(house_prices, additional_prices)

# 2. Calculate Model Accuracy
def calculate_mae(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Mean Absolute Error: {mae}")

# 3. Modify Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(house_sizes, house_prices, test_size=0.3, random_state=42)

# Train a Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict on the test set
y_pred_linear = linear_model.predict(X_test)

# Calculate MAE for Linear Regression
calculate_mae(y_test, y_pred_linear)
# 4. Customize the Plot
plt.scatter(house_sizes, house_prices, color='green', label='Data points')
plt.plot(house_sizes, linear_model.predict(house_sizes), color='black', label='Linear regression line')
plt.xlabel('House Size (square feet)')
plt.ylabel('House Price ($)')
plt.title('House Price Prediction')
plt.legend()
plt.grid(True)
plt.show()

# 5. Implement Polynomial Regression
# Fit a polynomial regression model of degree 2
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, y_train)

# Predict on the test set with Polynomial Regression
y_pred_poly = poly_model.predict(X_test)

# Calculate MAE for Polynomial Regression
calculate_mae(y_test, y_pred_poly)

# Visualize Polynomial Regression Line
plt.scatter(house_sizes, house_prices, color='green', label='Data points')
plt.plot(house_sizes, linear_model.predict(house_sizes), color='black', label='Linear regression line')
plt.plot(house_sizes, poly_model.predict(house_sizes), color='red', label='Polynomial regression line')
plt.xlabel('House Size (square feet)')
plt.ylabel('House Price ($)')
plt.title('House Price Prediction')
plt.legend()
plt.grid(True)
plt.show()

# Observations
print("Test set predictions (Linear Regression):", y_pred_linear)
print("Test set predictions (Polynomial Regression):", y_pred_poly)
print("Actual prices:", y_test)