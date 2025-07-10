import tifffile
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Hard-coded channel selection variable
selected_channel = 20  # Change this value to visualize a different channel, 0 is 700, 12 is 600, 23 is 500,

# Normalization parameters
clip_min = 0 # Minimum value to clip to
clip_max = 0.4  # Maximum value to clip to

tif_path = '/media/al/Extreme SSD/20250701_usaf/deconvolution_results/batch_reg/hs_gen_6_deconvolved.tif'
output_path = f'/media/al/Extreme SSD/20250701_usaf/deconvolution_results/batch_reg/visualization/20250701_usaf_deconvolution_hs_gen_6_deconvolved_{selected_channel}_clip{clip_min}to{clip_max}.png'


# Ensure the output directory exists
output_dir = os.path.dirname(output_path)
os.makedirs(output_dir, exist_ok=True)

# Load the hyperspectral TIF stack
print(f"Loading TIF stack from: {tif_path}")
hyperspectral_data = tifffile.imread(tif_path)

# Extract the selected channel
# Assuming dimensions are [channels, height, width] or [height, width, channels]
if hyperspectral_data.ndim == 3:
    if hyperspectral_data.shape[0] < hyperspectral_data.shape[1] and hyperspectral_data.shape[0] < hyperspectral_data.shape[2]:
        # Format is likely [channels, height, width]
        selected_channel_data = hyperspectral_data[selected_channel, :, :]
        print(f"Extracted channel {selected_channel} from shape {hyperspectral_data.shape} (assuming [channels, height, width])")
    else:
        # Format is likely [height, width, channels]
        selected_channel_data = hyperspectral_data[:, :, selected_channel]
        print(f"Extracted channel {selected_channel} from shape {hyperspectral_data.shape} (assuming [height, width, channels])")
else:
    raise ValueError(f"Unexpected data dimensions: {hyperspectral_data.shape}")

# Normalize data using min/max clipping
current_mean = np.mean(selected_channel_data)
current_std = np.std(selected_channel_data)
original_min = np.min(selected_channel_data)
original_max = np.max(selected_channel_data)

# Clip data to specified range
clipped_data = np.clip(selected_channel_data, clip_min, clip_max)

# Normalize clipped data to 0-255 range
channel_norm = ((clipped_data - clip_min) / (clip_max - clip_min) * 255).astype(np.uint8)

# Generate histogram comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original data histogram
ax1.hist(selected_channel_data.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
ax1.axvline(clip_min, color='red', linestyle='--', label=f'Clip Min: {clip_min}')
ax1.axvline(clip_max, color='red', linestyle='--', label=f'Clip Max: {clip_max}')
ax1.set_title(f'Original Data\nRange: {original_min:.0f} - {original_max:.0f}\nMean: {current_mean:.2f}, Std: {current_std:.2f}')
ax1.set_xlabel('Pixel Value')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Normalized data histogram
final_mean = np.mean(channel_norm)
final_std = np.std(channel_norm)
ax2.hist(channel_norm.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
ax2.set_title(f'Clipped & Normalized Data\nRange: 0 - 255\nMean: {final_mean:.2f}, Std: {final_std:.2f}')
ax2.set_xlabel('Pixel Value (0-255)')
ax2.set_ylabel('Frequency')
ax2.set_xlim(0, 255)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
histogram_path = output_path.replace('.png', '_histogram.png')
#plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved histogram to: {histogram_path}")

# Apply a colormap (viridis)
from matplotlib import cm
colored_img = cm.viridis(channel_norm)
# Convert from RGBA to RGB and back to 0-255 range
colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)

# Create and save colorbar figure
fig_colorbar, ax_colorbar = plt.subplots(figsize=(8, 2))
colorbar = ax_colorbar.imshow(np.linspace(0, 1, 256).reshape(1, -1), 
                              cmap='viridis', aspect='auto')
ax_colorbar.set_xlim(0, 255)
ax_colorbar.set_xticks([0, 255])
ax_colorbar.tick_params(axis='x', labelsize=30)
ax_colorbar.set_xticklabels([f'{clip_min}', f'{clip_max}'])
ax_colorbar.set_yticks([])

# Save colorbar
colorbar_path = output_path.replace('.png', '_colorbar.png')
plt.savefig(colorbar_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved colorbar to: {colorbar_path}")

# Save directly using PIL without any borders or resolution changes
img = Image.fromarray(colored_img)
img.save(output_path)
print(f"Saved visualization to: {output_path}")

# Optionally display the image
img.show()