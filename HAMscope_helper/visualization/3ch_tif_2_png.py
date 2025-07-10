import os
import tifffile
from PIL import Image
import numpy as np

def convert_tiffs_to_png(input_dir, output_dir):
    """Convert 3-channel TIFF files to PNG format"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all TIFF files from input directory
    tiff_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff'))]
    
    for tiff_file in tiff_files:
        input_path = os.path.join(input_dir, tiff_file)
        
        try:
            # Load TIFF with tifffile
            img_array = tifffile.imread(input_path)

            if img_array.shape[0] == 6:
                # If the image has 6 channels, we assume the first 3 are RGB
                img_array = img_array[:3, :, :]
            
            # Check if image has 3 channels (handle both channel formats)
            if img_array.ndim == 3 and img_array.shape[0] == 3:
                # Transpose from (C, H, W) to (H, W, C) format
                img_array = np.transpose(img_array, (1, 2, 0))
            elif img_array.ndim == 3 and img_array.shape[2] == 3:
                # Already in (H, W, C) format
                pass
            else:
                print(f"Skipped {tiff_file}: Not a 3-channel image (shape: {img_array.shape})")
                continue
            
            # Ensure data type is suitable for PIL (uint8)
            if img_array.dtype != np.uint8:
                # Normalize to 0-255 range if needed
                img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(img_array)
            
            # Create output filename
            base_name = os.path.splitext(tiff_file)[0]
            output_path = os.path.join(output_dir, f"{base_name}.png")
            
            # Save as PNG
            pil_image.save(output_path)
            print(f"Converted: {tiff_file} -> {base_name}.png")
                
        except Exception as e:
            print(f"Error processing {tiff_file}: {str(e)}")

# Hard-coded input and output directories
INPUT_DIR = "/media/al/Extreme SSD/20250425_results/results/misc_test_dataset_layernorm/results/20250425_0gan_single_reg_hs_nmf_fig3/test_latest/images"
OUTPUT_DIR = "/media/al/Extreme SSD/20250425_results/results/misc_test_dataset_layernorm/results/20250425_0gan_single_reg_hs_nmf_fig3/test_latest/images_png"
if __name__ == "__main__":
    convert_tiffs_to_png(INPUT_DIR, OUTPUT_DIR)
    print("Conversion complete!")