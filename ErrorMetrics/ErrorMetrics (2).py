import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create arrays
actual = np.array([2, 4, 5, 4, 5, 7, 9])
predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])

# Metrics
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
r_squared = r2_score(actual, predicted)

# Find worst prediction (largest absolute residual)
residuals = actual - predicted
worst_idx = np.argmax(np.abs(residuals))

# Plot
plt.figure(figsize=(6, 6))

# All points
plt.scatter(actual, predicted, s=60, alpha=0.8, label="Data")

# Worst point in red
plt.scatter(actual[worst_idx], predicted[worst_idx],
            s=120, color="red", edgecolor="black", label="Worst Prediction")

# Perfect fit line
plt.plot([actual.min(), actual.max()],
         [actual.min(), actual.max()],
         linestyle="--", linewidth=2, label="Perfect Fit (y = x)")

# Labels & title
plt.xlabel("Actual Values", fontsize=11)
plt.ylabel("Predicted Values", fontsize=11)
plt.title("Predicted vs Actual", fontsize=13)

# Grid + legend
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()

# Metrics text box
plt.text(0.05, 0.95,
         f"MAE = {mae:.2f}\nMSE = {mse:.2f}\nR² = {r_squared:.2f}",
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", alpha=0.9))

# Save figure
plt.tight_layout()
plt.savefig("predicted_vs_actual_clean.png", dpi=300)
plt.close()

