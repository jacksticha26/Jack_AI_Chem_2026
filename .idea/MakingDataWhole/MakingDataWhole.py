import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error
import os

# ------------------------------
# Step 1: Load Titanic dataset
# ------------------------------
titanic = sns.load_dataset("titanic")

# ------------------------------
# Step 2: Keep numeric + safe categorical columns
# ------------------------------
drop_cols = ["deck", "embark_town", "alive", "class", "who"]
titanic_safe = titanic.drop(columns=drop_cols)

# Identify categorical columns using string dtype to avoid pandas warning
categorical_cols = titanic_safe.select_dtypes(include="string").columns.tolist()

# Fill missing categorical values and encode
titanic_encoded = titanic_safe.copy()
titanic_encoded[categorical_cols] = titanic_encoded[categorical_cols].fillna("missing")
encoder = OrdinalEncoder()
titanic_encoded[categorical_cols] = encoder.fit_transform(titanic_encoded[categorical_cols])

# ------------------------------
# Step 3: Create mask for evaluation
# ------------------------------
# Only keep rows where age is known
known_age = titanic_encoded[~titanic_encoded["age"].isna()].copy()

# Randomly hide 20% of ages to simulate missing values
np.random.seed(42)
mask = np.random.rand(len(known_age)) < 0.2
test_set = known_age.copy()
test_set.loc[mask, "age"] = np.nan

# ------------------------------
# Step 4: Apply KNN imputer
# ------------------------------
imputer = KNNImputer(n_neighbors=5)
imputed_array = imputer.fit_transform(test_set)
imputed_df = pd.DataFrame(imputed_array, columns=test_set.columns)

# ------------------------------
# Step 5: Compare predicted vs actual ages and calculate MAE
# ------------------------------
predicted_ages = imputed_df["age"][mask]
actual_ages = known_age["age"][mask]
mae = mean_absolute_error(actual_ages, predicted_ages)
print(f"Mean Absolute Error (MAE) of KNN imputation: {mae:.2f}")

# ------------------------------
# Step 6: Plot actual vs predicted ages with MAE
# ------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(actual_ages, predicted_ages, color="orange", label="Predicted Ages")
plt.plot([0, 80], [0, 80], color="blue", linestyle="--", label="Perfect Prediction")
plt.xlabel("Actual Age")
plt.ylabel("KNN Predicted Age")
plt.title("Actual vs KNN-Predicted Ages")
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.grid(True)

# Display MAE on chart
plt.text(5, 75, f"MAE: {mae:.2f}", fontsize=12, color="red", bbox=dict(facecolor='white', alpha=0.6))
plt.legend()

# ------------------------------
# Step 7: Save plot to MakinDataWhole directory
# ------------------------------
save_dir = "MakinDataWhole"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "age_knn_actual_vs_predicted.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to {save_path}")

plt.show()
