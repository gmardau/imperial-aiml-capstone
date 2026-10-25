import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# --- Load the initial data ---
inputs = {n: np.load(f"../initial_data/function_{n}/initial_inputs.npy") for n in range(1,9)}
outputs = {n: np.load(f"../initial_data/function_{n}/initial_outputs.npy") for n in range(1,9)}

# --- Select the function number from command line argument ---
function_no = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# --- Prepare the data for the selected function ---
X = inputs[function_no]
y = outputs[function_no]
N = X.shape[1]

# --- Sort the inputs by output value for the selected function ---
sorted_indices = np.argsort(y)
X = X[sorted_indices]
y = y[sorted_indices]

# --- Print the inputs and outputs for the selected function ---
for Xi, yi in zip(X, y):
    print(Xi, yi)

# --- Plot the outputs with colors ---
plt.scatter(range(len(y)), y, c=y, cmap='rainbow', s=50, edgecolor='black')
plt.xlabel("Index")
plt.ylabel("y")
plt.title(f'Outputs for function {function_no}')
plt.show()

# --- Model parameters ---
RBF_LENGTH_SCALE = 0.2
UCB_KAPPA = 2

# --- Fit the Gaussian Process model ---
model = GaussianProcessRegressor(kernel = RBF(length_scale=RBF_LENGTH_SCALE, length_scale_bounds='fixed'))
model.fit(X, y)

# --- Create an X grid for predictions and plotting ---
n_grid = int(10000 ** (1/N))
axes = [np.linspace(0, 1, n_grid) for _ in range(N)]
mesh = np.meshgrid(*axes, indexing='ij')
x_grid = np.column_stack([m.ravel() for m in mesh])

# --- Make predictions on the grid ---
post_mean, post_std = model.predict(x_grid, return_std=True)
post_mean = post_mean.reshape(*(n_grid,) * N)
post_std = post_std.reshape(*(n_grid,) * N)

# --- Obtain the acquisition function (Upper Confidence Bound) and find its maximum ---
acquisition = post_mean + UCB_KAPPA * post_std
acquisition_max_index = np.unravel_index(np.argmax(acquisition), acquisition.shape)
acquisition_max_x = [mesh[i][acquisition_max_index] for i in range(N)]
print(f"Acquisition function maximum at X = {[float(f'{acquisition_max_x[i]:.8f}') for i in range(N)]}")

# --- Plotting for 2D cases ---
if N == 1:
    plt.figure(figsize=(10, 8))

    # Scatter of the actual data points
    plt.plot(X, y, 'o', label='Observed data')

    # Posterior mean
    plt.plot(axes[0], post_mean, label='Posterior mean')

    # Upper and lower std bounds
    plt.fill_between(axes[0], post_mean - post_std, post_mean + post_std, alpha=0.2, label='Posterior std')

    # Set labels and title
    plt.xlabel('x')
    plt.ylabel('output')
    plt.title(f'GP posterior for function {function_no}')
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

# --- Plotting for 3D cases ---
if N == 2:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter of the actual data points
    ax.scatter(X[:, 0], X[:, 1], y, color='black', s=50, depthshade=True, label='Observed data')

    # Posterior mean surface
    ax.plot_surface(mesh[0], mesh[1], post_mean, cmap='rainbow', alpha=0.8, linewidth=0, antialiased=True, label='Posterior mean')

    # Upper and lower std bounds surfaces
    ax.plot_surface(mesh[0], mesh[1], post_mean + post_std, color='lightgray', alpha=0.2, linewidth=0, label='Posterior std upper bound')
    ax.plot_surface(mesh[0], mesh[1], post_mean - post_std, color='darkgray', alpha=0.2, linewidth=0, label='Posterior std lower bound')

    # Acquisition function surface
    ax.plot_surface(mesh[0], mesh[1], acquisition, color='green', alpha=0.2, linewidth=0, label='Acquisition function')

    # Plot the acquisition function maximum point
    ax.scatter(acquisition_max_x[0], acquisition_max_x[1], acquisition[acquisition_max_index], color='green', s=100, label='Acquisition max')

    # Set labels and title
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('output')
    ax.set_title(f'GP posterior for function {function_no}')
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

# --- Plotting for 4D cases ---
if N == 3:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter of the actual data points with colour and size based on output value
    sizes = 50 * (y - y.min()) / (y.max() - y.min()) + 10  # Scale sizes between 10 and 60
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=sizes, c=y, cmap='rainbow', depthshade=True, label='Observed data')

    # Plot the acquisition function as a scatter plot
    ax.scatter(mesh[0], mesh[1], mesh[2], c=acquisition, cmap='viridis', s=50, alpha=0.5, edgecolors='none', label='Acquisition function')

    # Plot the acquisition function maximum point
    ax.scatter(acquisition_max_x[0], acquisition_max_x[1], acquisition_max_x[2], color='green', s=100, label='Acquisition max')

    # Set labels and title
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title(f'Observed data for function {function_no}')
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

# --- Plotting for higher-dimensional cases (5D+) ---
if N > 3:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalize y for colormap
    norm = plt.Normalize(y.min(), y.max())
    cmap = plt.cm.rainbow

    x_axis = np.arange(N)

    # Plot each point as a line in the parallel coordinates plot
    for i in range(X.shape[0]):
        ax.plot(x_axis, X[i, :], color=cmap(norm(y[i])), alpha=0.6)

    # Plot the acquisition function maximum point as a line in the parallel coordinates plot
    ax.plot(x_axis, acquisition_max_x, color='green', linewidth=2, label='Acquisition max')

    # Set labels and title
    ax.set_xticks(x_axis)
    ax.set_xticklabels([f'x[{d}]' for d in range(N)])
    ax.set_title(f'Parallel coordinates for function {function_no}')
    ax.set_ylabel('Input value')
    ax.legend()

    # Colorbar to show what color maps to what y value
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='y')

    # Display the plot
    plt.tight_layout()
    plt.show()