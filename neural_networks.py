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

from sklearn.decomposition import PCA
import numpy as np

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # --- Hidden Space Visualization ---
    hidden_features = mlp.A1

    ax_hidden.clear()
    ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")
    
    # 3D scatter of the hidden features
    ax_hidden.scatter(
        hidden_features[:, 0],
        hidden_features[:, 1],
        hidden_features[:, 2],
        c=y.ravel(),
        cmap="bwr",
        alpha=0.7,
    )
    
    # Set the axis limits to [-1.5, 1.5] for x, y, z axes
    ax_hidden.set_xlim([-1.5, 1.5])
    ax_hidden.set_ylim([-1.5, 1.5])
    ax_hidden.set_zlim([-1.5, 1.5])

    # --- First Plane: Flat decision boundary based on first hidden unit ---
    x_plane, y_plane = np.meshgrid(
        np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50)
    )
    z_plane_1 = (
        -(mlp.W1[0, 0] * x_plane + mlp.W1[1, 0] * y_plane + mlp.b1[0, 0])
    )  # Plane based on first hidden node weights
    ax_hidden.plot_surface(
        x_plane,
        y_plane,
        z_plane_1,
        alpha=0.5,
        cmap="bwr",
        rstride=100,
        cstride=100
    )

    # --- Second Plane: 3D plane surrounding the points (best fit plane) ---
    # Apply PCA to the hidden features to get the principal components
    pca = PCA(n_components=3)
    pca.fit(hidden_features)

    # The normal vector to the plane is the first principal component
    normal_vector = pca.components_[2]  # Normal vector is the 3rd component

    # Define the point in the plane (mean of the data points)
    mean_point = np.mean(hidden_features, axis=0)

    # Create a grid of points to display the plane
    x_plane, y_plane = np.meshgrid(
        np.linspace(-1.5, 1.5, 50), np.linspace(-1.5, 1.5, 50)
    )
    z_plane_2 = mean_point[2] - (normal_vector[0] * (x_plane - mean_point[0]) + normal_vector[1] * (y_plane - mean_point[1])) / normal_vector[2]

    ax_hidden.plot_surface(
        x_plane,
        y_plane,
        z_plane_2,
        alpha=0.5,
        cmap="coolwarm",
        rstride=100,
        cstride=100
    )

    # --- Input Space Visualization ---
    ax_input.clear()
    ax_input.set_title(f"Input Space at Step {frame * 10}")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = (mlp.forward(grid) > 0.5).astype(int)
    ax_input.contourf(xx, yy, predictions.reshape(xx.shape), alpha=0.5, cmap="bwr")
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolor="k")

    # --- Gradients Visualization ---
    ax_gradient.clear()
    ax_gradient.set_title(f"Gradients at Step {frame * 10}")
    input_nodes = [(0, 0), (0, 1)]  # Positions for x1, x2
    hidden_nodes = [(0.5, 0), (0.5, 0.5), (0.5, 1)]  # Positions for h1, h2, h3
    output_node = (1, 0)  # Position for y

    # Plot nodes
    for i, (x, y) in enumerate(input_nodes):
        ax_gradient.add_patch(Circle((x, y), radius=0.05, color="blue"))
        ax_gradient.text(x, y, f"x{i+1}", ha="center", va="center", fontsize=8)
    for i, (x, y) in enumerate(hidden_nodes):
        ax_gradient.add_patch(Circle((x, y), radius=0.05, color="green"))
        ax_gradient.text(x, y, f"h{i+1}", ha="center", va="center", fontsize=8)
    ax_gradient.add_patch(Circle(output_node, radius=0.05, color="red"))
    ax_gradient.text(*output_node, "y", ha="center", va="center", fontsize=8)

    # Plot edges with gradient-based thickness (reduced intensity)
    gradients_input_hidden = mlp.W1  # Gradients from input to hidden
    gradients_hidden_output = mlp.W2  # Gradients from hidden to output
    for i, (x1, y1) in enumerate(input_nodes):
        for j, (x2, y2) in enumerate(hidden_nodes):
            gradient = gradients_input_hidden[i, j]
            ax_gradient.plot(
                [x1, x2], [y1, y2], "k-", linewidth=1 + 2 * abs(gradient), alpha=0.7
            )
    for i, (x1, y1) in enumerate(hidden_nodes):
        x2, y2 = output_node
        gradient = gradients_hidden_output[i, 0]
        ax_gradient.plot(
            [x1, x2], [y1, y2], "k-", linewidth=1 + 2 * abs(gradient), alpha=0.7
        )
    ax_gradient.set_xlim(-0.1, 1.1)
    ax_gradient.set_ylim(-0.1, 1.1)
    ax_gradient.axis("off")


# Main visualization function
def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use("agg")
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection="3d")
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(
        fig,
        partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y),
        frames=step_num // 10,
        repeat=False,
    )

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer="pillow", fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
