import sys

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# --- Load the initial data ---
inputs = {n: np.load(f"../initial_data/function_{n}/initial_inputs.npy") for n in range(1,9)}
outputs = {n: np.load(f"../initial_data/function_{n}/initial_outputs.npy") for n in range(1,9)}

# --- Add data from previous weeks ---
week_no = 2
for week in range(1, week_no):
    with open(f"../weekly_data/week{week}_inputs.txt", 'r') as f:
        for n,line in enumerate(f.readlines(), start=1):
            new_values = np.array([float(x) for x in line.strip().split(',')])
            inputs[n] = np.vstack((inputs[n], new_values))
    with open(f"../weekly_data/week{week}_outputs.txt", 'r') as f:
        for n,line in enumerate(f.readlines(), start=1):
            new_values = np.array([float(x) for x in line.strip().split(',')])
            outputs[n] = np.append(outputs[n], new_values)

# --- Select the function number from command line argument ---
function_no = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# --- Prepare the data for the selected function ---
X = inputs[function_no]
y = outputs[function_no]
N = X.shape[1]

# --- Sort the inputs by output value for the selected function ---
sorted_indices = np.argsort(y)
# sorted_indices = np.arange(len(y)) # Uncomment this line to disable sorting and keep the original order of the data
X = X[sorted_indices]
y = y[sorted_indices]

# --- Get the indices of the initial and weekly data points to separate them during plotting ---
rank_index = np.argsort(sorted_indices)
initial_indices = rank_index[:-(week_no-1)] if week_no > 1 else rank_index
weekly_indices = rank_index[-(week_no-1):] if week_no > 1 else []

# --- Print the inputs and outputs for the selected function ---
for Xi, yi in zip(X, y):
    print(Xi, yi)

# --- Plot the outputs with colors ---
fig = plt.figure(figsize=(8, 4))
ax = fig.subplots()
ax.scatter(initial_indices, y[initial_indices], c=y[initial_indices], vmin=y.min(), vmax=y.max(), marker='o', cmap='rainbow', s=50, edgecolor='black')
ax.scatter(weekly_indices, y[weekly_indices], c=y[weekly_indices], vmin=y.min(), vmax=y.max(), marker='s', cmap='rainbow', s=50, edgecolor='black')
ax.set_xlabel("Index")
ax.set_ylabel("y")
ax.set_title(f'Outputs for function {function_no}')
plt.show()

# --- Plot the outputs for each dimension separately ---
if N > 1:
    nrows = int(np.ceil(np.sqrt(N)))
    ncols = int(np.ceil(N / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3*nrows))
    axes = axes.flatten()
    for i in range(N):
        axes[i].scatter(X[initial_indices, i], y[initial_indices], c=y[initial_indices], vmin=y.min(), vmax=y.max(), marker='o', cmap='rainbow', s=50, edgecolor='black')
        axes[i].scatter(X[weekly_indices, i], y[weekly_indices], c=y[weekly_indices], vmin=y.min(), vmax=y.max(), marker='s', cmap='rainbow', s=50, edgecolor='black')
        axes[i].set_xlabel("Index")
        axes[i].set_ylabel(f"x[{i+1}]")
        axes[i].set_title(f'Inputs for dimension {i+1} of function {function_no}')
    plt.tight_layout()
    plt.show()

# --- Create an X grid for predictions and plotting ---
n_grid = int(1000 ** (1/N))
axes = [np.linspace(0, 1, n_grid) for _ in range(N)]
mesh = np.meshgrid(*axes, indexing='ij')
x_grid = np.column_stack([m.ravel() for m in mesh])

# --- Model parameters ---
RBF_LENGTH_SCALE = 0.2 # lower values make the GP more sensitive to changes in the function (i.e. sharper), while higher values make it smoother.
UCB_KAPPA = 2 # lower values make it more exploitative, while higher values make the acquisition function more exploratory

# --- Fit the Gaussian Process model ---
model = GaussianProcessRegressor(kernel = RBF(length_scale=RBF_LENGTH_SCALE, length_scale_bounds='fixed'))
model.fit(X, y)

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
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()

    # Scatter of the actual data points (spheres for initial data, cubes for weekly data)
    ax.scatter(X[initial_indices], y[initial_indices], color='black', marker='o', s=50, label='Initial data')
    ax.scatter(X[weekly_indices], y[weekly_indices], color='black', marker='s', s=50, label='Weekly data')

    # Posterior mean
    ax.plot(axes[0], post_mean, label='Posterior mean', color='orange')

    # Upper and lower std bounds
    ax.fill_between(axes[0], post_mean - post_std, post_mean + post_std, alpha=0.2, color='blue', label='Posterior std')

    # Acquisition function
    ax.plot(axes[0], acquisition, color='green', linewidth=2, label='Acquisition function')

    # Plot the acquisition function maximum point
    ax.scatter([acquisition_max_x[0]], [acquisition[acquisition_max_index]], marker='*', s=100, color='green', label='Acquisition max')

    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('output')
    ax.set_title(f'GP posterior for function {function_no}')
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

# --- Plotting for 3D cases ---
if N == 2:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter of the actual data points
    # Make weekly input points a cube rather than a sphere for better visibility
    ax.scatter(X[initial_indices, 0], X[initial_indices, 1], y[initial_indices], marker='o', color='black', s=50, depthshade=True, label='Initial data')
    ax.scatter(X[weekly_indices, 0], X[weekly_indices, 1], y[weekly_indices], marker='s', color='black', s=50, depthshade=True, label='Weekly data')

    # Posterior mean surface
    ax.plot_surface(mesh[0], mesh[1], post_mean, cmap='rainbow', alpha=0.8, linewidth=0, antialiased=True, label='Posterior mean')

    # Upper and lower std bounds surfaces
    ax.plot_surface(mesh[0], mesh[1], post_mean + post_std, color='lightgray', alpha=0.2, linewidth=0, label='Posterior std upper bound')
    ax.plot_surface(mesh[0], mesh[1], post_mean - post_std, color='darkgray', alpha=0.2, linewidth=0, label='Posterior std lower bound')

    # Acquisition function surface
    ax.plot_surface(mesh[0], mesh[1], acquisition, color='green', alpha=0.2, linewidth=0, label='Acquisition function')

    # Plot the acquisition function maximum point
    ax.scatter(acquisition_max_x[0], acquisition_max_x[1], acquisition[acquisition_max_index], color='green', s=100, marker='*', label='Acquisition max')

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
    sizes = 40 * (y - y.min()) / (y.max() - y.min()) + 10  # Scale sizes between 10 and 50
    ax.scatter(X[initial_indices, 0], X[initial_indices, 1], X[initial_indices, 2], s=sizes[initial_indices], c=y[initial_indices], cmap='rainbow', marker='o', depthshade=True, label='Initial data')
    ax.scatter(X[weekly_indices, 0], X[weekly_indices, 1], X[weekly_indices, 2], s=sizes[weekly_indices], c=y[weekly_indices], cmap='rainbow', marker='s', depthshade=True, label='Weekly data')

    # Plot the acquisition function as a scatter plot
    ax.scatter(mesh[0], mesh[1], mesh[2], c=acquisition, cmap='viridis', s=50, alpha=0.5, edgecolors='none', label='Acquisition function')

    # Plot the acquisition function maximum point
    ax.scatter(acquisition_max_x[0], acquisition_max_x[1], acquisition_max_x[2], color='green', s=100, marker='*', label='Acquisition max')

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
        ax.plot(x_axis, X[i, :], color=cmap(norm(y[i])), linewidth=1 if i in initial_indices else 1.5, alpha=0.6)

    # Plot the acquisition function maximum point as a dashed line in the parallel coordinates plot
    ax.plot(x_axis, acquisition_max_x, color='green', linewidth=2, linestyle='--', label='Acquisition max')

    # Set labels and title
    ax.set_xticks(x_axis)
    ax.set_xticklabels([f'x[{d}]' for d in range(1, N+1)])
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