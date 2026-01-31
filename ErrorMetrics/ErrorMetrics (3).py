import numpy as np
import matplotlib.pyplot as plt

# Your data
actual = np.array([2, 4, 5, 4, 5, 7, 9])
predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])
residuals = actual - predicted

# Residuals plot: Residuals vs Predicted
plt.figure()
plt.scatter(predicted, residuals)

# Dotted vertical lines from each point to y = 0
plt.vlines(predicted, 0, residuals, linestyles="dotted")

# Horizontal zero line
plt.axhline(0)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs Predicted")

# Save the figure
plt.savefig("residuals_plot_with_dotted_lines.png", dpi=300, bbox_inches="tight")
plt.close()
