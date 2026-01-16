import matplotlib.pyplot as plt
import numpy as np

# Number of carbons in the first 10 linear alkanes
carbons = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Accurate melting points in °C
melting_points = np.array([-182.5, -183.3, -187.7, -138.3, -129.7, -95.0, -90.6, -56.8, -53.0, -30.0])

# Create figure
plt.figure(figsize=(10, 6))

# Scatter plot
plt.scatter(carbons, melting_points, color='blue', label='Melting Point', zorder=5)

# Linear trend line (for demonstration, even if not perfect)
coeffs = np.polyfit(carbons, melting_points, 1)
trendline = np.poly1d(coeffs)
plt.plot(carbons, trendline(carbons), color='orange', linestyle='--', label='Trend Line')

# Trend line equation in bottom right
x_pos = max(carbons) - 0.5
y_pos = min(melting_points) + 5
equation_text = f'y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}'
plt.text(x_pos, y_pos, equation_text, fontsize=12, color='orange',
         ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Titles and labels
plt.title("Melting Points vs. Number of Carbons for Linear Alkanes", fontsize=14, fontweight='bold')
plt.xlabel("Number of Carbon Atoms", fontsize=12)
plt.ylabel("Melting Point (°C)", fontsize=12)

# Grid and legend
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(carbons)
plt.legend()

plt.show()



