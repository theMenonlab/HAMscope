from PIL import Image
import numpy as np

# Load the uploaded image
image_path = '/home/al/hyperspectral_pix2pix/datasets/CAVE/test/watercolors_ms/watercolors_ms/watercolors_ms_08.png'  # Replace with the actual path to your image
image = Image.open(image_path)

# Convert the image to a numpy array
image_np = np.array(image)

# Check the bit depth by examining the dtype and value range
bit_depth = image_np.dtype
min_val = image_np.min()
max_val = image_np.max()

print(f'Bit depth: {bit_depth}')
print(f'Min value: {min_val}')
print(f'Max value: {max_val}')
