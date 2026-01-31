import numpy as np

# Create arrays from Python lists
actual = np.array([2, 4, 5, 4, 5, 7, 9])
predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])

# Calculate residuals
residuals = actual - predicted

# Mean Absolute Error (MAE)
mae = np.mean(np.abs(residuals))

# Mean Squared Error (MSE)
mse = np.mean(residuals**2)

# R-squared (coefficient of determination)
ss_res = np.sum(residuals**2)              # residual sum of squares
ss_tot = np.sum((actual - np.mean(actual))**2)  # total sum of squares
r_squared = 1 - (ss_res / ss_tot)

# Print results
print("Actual:", actual)
print("Predicted:", predicted)
print("Residuals:", residuals)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r_squared)
