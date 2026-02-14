import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans

# --- Load CSV safely ---
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "noble_gases_elements.csv")

df = pd.read_csv(file_path)

# --- Features used for clustering ---
X = df[["atomic_radius_pm",
        "first_ionization_energy_kJ_per_mol"]]

# ===============================
# LOOP: keep running until user stops
# ===============================
while True:

    # --- Ask user for k value ---
    k_input = input("\nEnter number of clusters (k) or 'q' to quit: ")

    if k_input.lower() == "q":
        print("Stopping clustering.")
        break

    k = int(k_input)

    # --- Run K-Means ---
    kmeans = KMeans(n_clusters=k, random_state=42)
    df["cluster"] = kmeans.fit_predict(X)

    # Cluster centers
    centers = kmeans.cluster_centers_

    # --- Create figure with light grey background ---
    plt.figure(figsize=(8,5), facecolor="lightgrey")
    ax = plt.gca()
    ax.set_facecolor("#f0f0f0")

    # --- Scatter points ---
    plt.scatter(
        df["atomic_radius_pm"],
        df["first_ionization_energy_kJ_per_mol"],
        c=df["cluster"],
        cmap="viridis",
        s=100
    )

    # --- Cluster centers (transparent black X) ---
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=300,
        color="black",
        alpha=0.6,
        edgecolors="white",
        linewidths=1.5,
        label="Cluster Centers"
    )

    # --- Element labels (offset so visible) ---
    for _, row in df.iterrows():
        plt.text(
            row["atomic_radius_pm"] + 3,
            row["first_ionization_energy_kJ_per_mol"] + 5,
            row["symbol"],
            fontsize=10,
            ha="left",
            va="bottom"
        )

    # --- Labels & formatting ---
    plt.xlabel("Atomic Radius (pm)")
    plt.ylabel("First Ionization Energy (kJ/mol)")
    plt.title(f"Noble Gases — K-Means Clustering (k={k})")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.show()
