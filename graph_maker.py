# Data with None for values < 1
from matplotlib import pyplot as plt


particles = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
cell_size_10 = [18000, 16500, 6200, 250, 4]
cell_size_50 = [18000, 12500, 950, 6, None]
cell_size_100 = [18000, 4300, 90, 0.5, None]

# Plotting
plt.figure(figsize=(10, 6))

# Function to handle None values in plotting
def plot_with_none(x, y, label):
    x_filtered = [x[i] for i in range(len(y)) if y[i] is not None]
    y_filtered = [y[i] for i in range(len(y)) if y[i] is not None]
    plt.plot(x_filtered, y_filtered, marker='o', label=label)

plot_with_none(particles, cell_size_10, 'Cell size 10')
plot_with_none(particles, cell_size_50, 'Cell size 50')
plot_with_none(particles, cell_size_100, 'Cell size 100')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Number of Particles')
plt.ylabel('Performance (it/s)')
plt.title('Performance vs Number of Particles for Different Cell Sizes')
plt.legend()
plt.grid(True, which="both", ls="--")

plt.show()