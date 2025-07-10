import numpy as np
from PIL import Image
import os
import random

# Create output directory
output_dir = "/media/al/Extreme SSD/20250425_dataset/100_points_512/split_1/test/miniscope"
os.makedirs(output_dir, exist_ok=True)

# Image parameters
width, height = 512, 512
num_images = 99

print(f"Generating {num_images} images...")

# Create a summary array to accumulate all points
summary_array = np.zeros((height, width), dtype=np.uint16)  # Use uint16 to handle potential overflow

for i in range(num_images):
    # Create a black image (all pixels = 0)
    img_array = np.zeros((height, width), dtype=np.uint8)
    
    # Set one random pixel to white (255)
    random_x = random.randint(0, width - 1)
    random_y = random.randint(0, height - 1)
    img_array[random_y, random_x] = 255
    
    # Add this point to the summary array
    summary_array[random_y, random_x] += 255
    
    # Create PIL Image and save
    img = Image.fromarray(img_array, mode='L')  # 'L' mode for grayscale
    filename = f"image_{i+1:03d}.png"
    filepath = os.path.join(output_dir, filename)
    img.save(filepath)
    
    if (i + 1) % 10 == 0:
        print(f"Generated {i + 1}/{num_images} images")

# Create and save the summary image with all points
# Convert to uint8 and clip values to 255 max
summary_img_array = np.clip(summary_array, 0, 255).astype(np.uint8)
summary_img = Image.fromarray(summary_img_array, mode='L')
summary_filepath = os.path.join(output_dir, "summary_all_points.png")
summary_img.save(summary_filepath)

print(f"All {num_images} images generated in '{output_dir}' folder!")
print(f"Summary image with all points saved as 'summary_all_points.png'")