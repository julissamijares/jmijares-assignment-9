import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function

        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))

    def activate(self, x):
        if self.activation_fn == 'tanh':
            return np.tanh(x), 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return np.maximum(0, x), (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            s = 1 / (1 + np.exp(-x))
            return s, s * (1 - s)

    def forward(self, X):
        # Forward pass through the network
        self.Z1 = X @ self.W1 + self.b1
        self.A1, self.A1_grad = self.activate(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = 1 / (1 + np.exp(-self.Z2))  # Sigmoid output
        return self.A2

    def backward(self, X, y):
        # Compute gradients using chain rule
        m = y.shape[0]
        dZ2 = self.A2 - y
        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.A1_grad
        dW1 = (X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights with gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
    
    # Plot hidden features
    hidden_features = X @ mlp.W1 + mlp.b1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)

    # Decision boundary in input space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = (mlp.forward(grid) > 0.5).astype(int)
    ax_input.contourf(xx, yy, predictions.reshape(xx.shape), alpha=0.5, cmap='bwr')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')

    # Gradient visualization
    for i in range(mlp.W1.shape[0]):
        for j in range(mlp.W1.shape[1]):
            gradient_magnitude = abs(mlp.W1[i, j])
            circle = Circle((i, j), radius=0.1, alpha=0.5, color='blue' if gradient_magnitude < 0.5 else 'red')
            ax_gradient.add_patch(circle)
    ax_gradient.set_xlim(-0.5, mlp.W1.shape[0] - 0.5)
    ax_gradient.set_ylim(-0.5, mlp.W1.shape[1] - 0.5)

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(
        fig,
        partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y),
        frames=step_num // 10,
        repeat=False
    )

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
