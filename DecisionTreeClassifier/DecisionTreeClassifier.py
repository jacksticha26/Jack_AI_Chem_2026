import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# -----------------------------
# Data
# -----------------------------
data = {
    "Molecular Weight": [180, 250, 80, 300, 150, 400, 90, 200, 130, 275, 135, 220],
    "Hydrogen Bond Donors": [5, 2, 1, 1, 4, 3, 0, 2, 3, 1, 1, 3],
    "Hydrogen Bond Acceptors": [6, 3, 2, 2, 5, 4, 1, 3, 4, 2, 3, 2],
    "Water Solubility": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

# -----------------------------
# Features and Target
# -----------------------------
X = df[["Molecular Weight", "Hydrogen Bond Donors", "Hydrogen Bond Acceptors"]]
y = df["Water Solubility"]

# -----------------------------
# Train Model
# -----------------------------
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# -----------------------------
# Create Figure with Slightly Dark Gray Background
# -----------------------------
fig = plt.figure(figsize=(18, 10), facecolor="#c0c0c0")  # slightly darker gray
ax = plt.gca()
ax.set_facecolor("#c0c0c0")

# -----------------------------
# Plot Decision Tree (normal borders)
# -----------------------------
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Soluble", "Soluble"],
    filled=True,
    rounded=True,
    fontsize=9
)

# -----------------------------
# Annotate Tree Depth
# -----------------------------
tree_depth = model.get_depth()
plt.text(
    x=0, y=1.05,  # above the plot
    s=f"Tree Depth: {tree_depth}",
    fontsize=14,
    fontweight="bold",
    transform=plt.gca().transAxes,
    verticalalignment="top"
)

plt.title("Decision Tree (All Features)", fontsize=14, pad=20)
plt.show()
