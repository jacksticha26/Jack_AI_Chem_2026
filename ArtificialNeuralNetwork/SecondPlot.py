import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Data Setup
data = {
    "Compound": ["Methane", "Water", "Propane", "Ethanol", "Formic Acid", "Acetic Acid", "Butane", "Acetone", "Benzene", "Toluene", "Octane"],
    "MW": [16, 18, 44, 46, 46, 60, 58, 58, 78, 92, 114],
    "BP": [-161, 100, -42, 78, 101, 118, -1, 56, 80, 111, 125]
}
df = pd.DataFrame(data).sort_values(by="MW")
X = df[["MW"]]
y = df["BP"]

# 2. Linear Regression (For comparison)
lr_model = LinearRegression()
lr_model.fit(X, y)
y_pred_lr = lr_model.predict(X)

# 3. 3-Layer Neural Network (MLP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Configuring with 3 layers: 10, 8, 6 neurons
mlp = MLPRegressor(
    hidden_layer_sizes=(10, 8, 6),
    activation='relu',
    max_iter=5000,
    early_stopping=False,
    n_iter_no_change=5001, # Ensure it hits 5000 epochs
    random_state=42,
    solver='lbfgs'
)
mlp.fit(X_scaled, y)
y_pred_mlp = mlp.predict(X_scaled)

# 4. Metrics Calculation
def get_metrics(actual, pred):
    return {
        "MAE": mean_absolute_error(actual, pred),
        "MSE": mean_squared_error(actual, pred),
        "R2": r2_score(actual, pred)
    }

lr_metrics = get_metrics(y, y_pred_lr)
mlp_metrics = get_metrics(y, y_pred_mlp)

# Print Summary
print("--- Model Comparison Summary ---")
print(f"Linear Regression  | MAE: {lr_metrics['MAE']:.2f} | R2: {lr_metrics['R2']:.4f}")
print(f"3-Layer MLP (10,8,6)| MAE: {mlp_metrics['MAE']:.2f} | R2: {mlp_metrics['R2']:.4f}")
print(f"Epochs Completed: {mlp.n_iter_}")

# 5. Visualization
plt.figure(figsize=(12, 7))

# Plot actual data points
plt.scatter(X, y, color='black', s=60, label='Actual Data', zorder=5)

# Plot Linear Regression Line
plt.plot(X, y_pred_lr, color='#3498db', linestyle='--', label='Linear Regression', linewidth=2)

# Plot Neural Network Curve
# (We create a smooth line for the MLP by predicting over a range)
X_smooth = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
X_smooth_scaled = scaler.transform(X_smooth)
y_smooth_mlp = mlp.predict(X_smooth_scaled)
plt.plot(X_smooth, y_smooth_mlp, color='#e74c3c', label='3-Layer MLP (10, 8, 6)', linewidth=3)

plt.xlabel("Molecular Weight (MW)", fontsize=12)
plt.ylabel("Boiling Point (BP) °C", fontsize=12)
plt.title("Chemical Property Prediction: Linear vs. 3-Layer Deep Learning", fontsize=14)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Save the final comparison
plt.savefig("3layer_mlp_comparison.png", dpi=300, bbox_inches='tight')
plt.show()