import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

# ReLU
def relu(x):
    return np.maximum(0, x)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x, relu(x), color='red', linewidth=2)
axes[0].set_title('ReLU — max(0, x)')
axes[0].axhline(0, color='black', linewidth=0.5)
axes[0].axvline(0, color='black', linewidth=0.5)
axes[0].grid(True, alpha=0.3)

axes[1].plot(x, sigmoid(x), color='blue', linewidth=2)
axes[1].set_title('Sigmoid — 1/(1+e^-x)')
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].axvline(0, color='black', linewidth=0.5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/activations.png')
plt.show()
print("Saved to outputs/activations.png")