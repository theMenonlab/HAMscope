import re
import matplotlib.pyplot as plt
import numpy as np

# Path to the loss log file
log_file = "/home/al/Documents/hyperspectral_miniscope_paper/loss_log.txt"

# User input: total number of epochs
total_epochs = 50

# Regular expression to extract G_L1 values
pattern = r"G_L1: ([\d\.]+)"

# Lists to store extracted values
g_l1_values = []

# Read and parse the log file
with open(log_file, "r") as file:
    for line in file:
        match = re.search(pattern, line)
        if match:
            g_l1_values.append(float(match.group(1)))

# Generate evenly spaced epoch values
num_points = len(g_l1_values)
epochs = np.linspace(1, total_epochs, num_points)

# Plot G_L1 over inferred epochs
plt.figure(figsize=(10, 5))
plt.plot(epochs, g_l1_values, marker="o", linestyle="-", color="r", markersize=3)
plt.xlabel("Epochs")
plt.ylabel("G_L1 Loss")
plt.title("G_L1 Loss over Epochs")
plt.grid(True)
plt.show()


