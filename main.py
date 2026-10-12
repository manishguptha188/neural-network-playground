
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 1. Generate our dataset
X, y = make_moons(
    n_samples=500,
    noise=0.20,
    random_state=42
)

# 2. Inspect the data
print("Input shape:", X.shape)
print("Output shape:", y.shape)
print("First 5 inputs:\n", X[:5])
print("First 5 labels:", y[:5])

# 3. Visualize the dataset
plt.figure(figsize=(8, 6))

plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    label="Class 0",
    alpha=0.8
)

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    label="Class 1",
    alpha=0.8
)

plt.title("My First Neural Network Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
