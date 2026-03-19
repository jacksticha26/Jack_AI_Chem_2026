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

# 2. Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X, y)
y_pred_lr = lr_model.predict(X)

# 3. Train Neural Network (MLP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mlp = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation='relu',
    max_iter=5000,
    early_stopping=False,
    n_iter_no_change=5001, # Force 5000 epochs
    random_state=42,
    solver='lbfgs'
)
mlp.fit(X_scaled, y)
y_pred_mlp = mlp.predict(X_scaled)

# 4. Calculate Metrics
metrics = {
    "Metric": ["MAE", "MSE", "R2 Score"],
    "Linear": [mean_absolute_error(y, y_pred_lr), mean_squared_error(y, y_pred_lr), r2_score(y, y_pred_lr)],
    "Neural Net": [mean_absolute_error(y, y_pred_mlp), mean_squared_error(y, y_pred_mlp), r2_score(y, y_pred_mlp)]
}
metrics_df = pd.DataFrame(metrics)

print("--- Model Metrics ---")
print(metrics_df.to_string(index=False))
print(f"\nNeural Network Epochs Used: {mlp.n_iter_}")

# 5. Visualization 1: Model Comparison
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', label='Actual Data', zorder=5)
plt.plot(X, y_pred_lr, color='blue', label='Linear Regression', linewidth=2)
plt.plot(X, y_pred_mlp, color='red', label='Neural Network (MLP)', linewidth=2)
plt.xlabel("Molecular Weight")
plt.ylabel("Boiling Point (°C)")
plt.title("Linear Regression vs. Neural Network Comparison")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("model_comparison.png")
plt.show()

# 6. Visualization 2: Understanding ReLU
# Let's show how ReLU affects the scaled input data
x_range = np.linspace(-3, 3, 100)
relu_output = np.maximum(0, x_range)

plt.figure(figsize=(8, 5))
plt.plot(x_range, relu_output, color='purple', linewidth=3)
plt.title("ReLU Activation Function: $f(x) = \max(0, x)$")
plt.xlabel("Input to Neuron (Scaled MW)")
plt.ylabel("Neuron Output")
plt.grid(True)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.savefig("relu_explanation.png")
plt.show()