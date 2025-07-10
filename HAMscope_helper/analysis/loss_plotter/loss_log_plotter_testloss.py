import re
import matplotlib.pyplot as plt
import numpy as np

# Path to the loss log file
log_file = "/home/al/Documents/hyperspectral_miniscope_paper/loss_log.txt"

# Test loss data
test_epochs = [5, 10, 15, 20, 25, 30]
test_loss = [0.0189, 0.0162, 0.0137, 0.0133, 0.0130, 0.0125]

# User input: total number of epochs
total_epochs = 30

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

# Scale the training loss by 0.01 to match test loss scale
scaled_g_l1_values = [x * 0.01 for x in g_l1_values]

# Plot G_L1 over inferred epochs
plt.figure(figsize=(10, 5))

# Plot training loss
plt.plot(epochs, scaled_g_l1_values, marker="o", linestyle="-", color="r", markersize=3, label="Training Loss")

# Add test loss markers
plt.plot(test_epochs, test_loss, marker="s", linestyle="", color="blue", markersize=8, label="Test Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Test Loss over Epochs")
plt.grid(True)
plt.legend()
plt.show()


