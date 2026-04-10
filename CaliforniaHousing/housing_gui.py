import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Train model
# -----------------------------
housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["MedHouseValue"] = housing.target

features = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup"
]

X = df[features]
y = df["MedHouseValue"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# -----------------------------
# Tkinter GUI
# -----------------------------
root = tk.Tk()
root.title("California Housing Price Predictor")

entries = {}

def create_input(label_text, row):
    label = ttk.Label(root, text=label_text)
    label.grid(row=row, column=0, padx=10, pady=5, sticky="w")

    entry = ttk.Entry(root)
    entry.grid(row=row, column=1, padx=10, pady=5)

    entries[label_text] = entry


# Inputs
create_input("Median Income (USD per year)", 0)
create_input("House Age (years)", 1)
create_input("Average Rooms", 2)
create_input("Average Bedrooms", 3)
create_input("Population", 4)
create_input("Average Occupancy", 5)

# -----------------------------
# Prediction function
# -----------------------------
def predict_price():
    try:
        medinc_dollars = float(entries["Median Income (USD per year)"].get())
        medinc_scaled = medinc_dollars / 10000

        values = [
            medinc_scaled,
            float(entries["House Age (years)"].get()),
            float(entries["Average Rooms"].get()),
            float(entries["Average Bedrooms"].get()),
            float(entries["Population"].get()),
            float(entries["Average Occupancy"].get())
        ]

        input_df = pd.DataFrame([values], columns=features)
        prediction = model.predict(input_df)[0]

        price = prediction * 100000

        result_label.config(text=f"Predicted Price: ${price:,.0f}")

    except ValueError:
        result_label.config(text="Error: Please enter valid numeric values.")


# -----------------------------
# Reset function
# -----------------------------
def reset_fields():
    for entry in entries.values():
        entry.delete(0, tk.END)

    result_label.config(text="Predicted Price: $---")


# -----------------------------
# Buttons
# -----------------------------
predict_button = ttk.Button(root, text="Predict Price", command=predict_price)
predict_button.grid(row=6, column=0, pady=10)

reset_button = ttk.Button(root, text="Reset", command=reset_fields)
reset_button.grid(row=6, column=1, pady=10)


# -----------------------------
# Output label
# -----------------------------
result_label = ttk.Label(root, text="Predicted Price: $---", font=("Arial", 14))
result_label.grid(row=7, column=0, columnspan=2, pady=20)

root.mainloop()