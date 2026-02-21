import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ===============================
# 1. Generate Data
# ===============================
np.random.seed(42)

x = np.linspace(0, 10, 25)
X = x.reshape(-1, 1)

y_true = 2 * x + 5
noise = np.random.normal(0, 2, len(x))
y = y_true + noise

# ===============================
# 2. Fit Linear Regression
# ===============================
model = LinearRegression()
model.fit(X, y)

print("Learned slope:", model.coef_[0])
print("Learned intercept:", model.intercept_)

y_pred = model.predict(X)

# ===============================
# 3. Plot & Save Regression Figure
# ===============================
plt.figure(figsize=(8, 6))

plt.scatter(x, y, label="Noisy Data")
plt.plot(x, y_true, label="True Line (y = 2x + 5)")
plt.plot(x, y_pred, label="Fitted Regression")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit to Noisy Data")
plt.legend()

plt.savefig("linear_regression_plot.png", dpi=300, bbox_inches="tight")
plt.close()

# ===============================
# 4. Build Parameter Grid
# ===============================
m_values = np.linspace(0, 4, 200)
b_values = np.linspace(0, 10, 200)

M, B = np.meshgrid(m_values, b_values)

# ===============================
# 5. Compute Loss Landscape (Vectorized)
# ===============================
x_expanded = x.reshape(1, 1, -1)

Y_pred_grid = M[:, :, np.newaxis] * x_expanded + B[:, :, np.newaxis]
loss = np.mean((y - Y_pred_grid) ** 2, axis=2)

# ===============================
# 6. Plot & Save Loss Landscape
# ===============================
plt.figure(figsize=(8, 6))

contour = plt.contourf(M, B, loss, levels=50, cmap="plasma_r")
plt.colorbar(contour)

# Mark optimal parameters
plt.scatter(model.coef_[0], model.intercept_)

plt.xlabel("Slope (m)")
plt.ylabel("Intercept (b)")
plt.title("Loss Landscape (MSE)\nYellow = Low Error | Purple = High Error")

plt.savefig("loss_landscape.png", dpi=300, bbox_inches="tight")
plt.close()

print("Both plots saved successfully.")