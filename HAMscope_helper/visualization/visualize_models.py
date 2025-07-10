import tifffile
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm

# Configuration
# Base directory to search for TIF files
base_dir = '/media/al/20250701_AI/hs_results/20250707_results'
OUTPUT_BASE_DIR = "/media/al/20250701_AI/hs_results/output"

# Hyperspectral visualization parameters
SELECTED_CHANNEL = 20  # Channel to visualize for hyperspectral
CLIP_MIN = 0.03  # Minimum value to clip to
CLIP_MAX = 0.4   # Maximum value to clip to

# RGB visualization parameters (per channel)
RGB_CLIP_MIN_R = 0.02
RGB_CLIP_MAX_R = 0.5
RGB_CLIP_MIN_G = 0.02
RGB_CLIP_MAX_G = 0.2
RGB_CLIP_MIN_B = 0.02
RGB_CLIP_MAX_B = 0.3

RAW_OUTPUT_PATH = os.path.join(OUTPUT_BASE_DIR, "hs_raw_6_rgb.png")

def find_tif_images(base_dir):
    """Recursively find all hs_gen_6.tif files under base_dir."""
    tif_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "hs_gen_6.tif":
                tif_paths.append(os.path.join(root, file))
    return tif_paths

def extract_model_name(tif_path):
    """Extract model name from the TIF path for output filename"""
    parts = tif_path.split('/')
    for part in parts:
        if '20250425_' in part and 'results' not in part:
            return part
    return "unknown_model"

def load_and_normalize_tif(tif_path):
    """Load TIF file and normalize channel order to [height, width, channels]"""
    data = tifffile.imread(tif_path)
    
    # Move channel axis to last if needed
    if data.ndim == 3 and data.shape[0] < data.shape[-1]:
        data = np.moveaxis(data, 0, -1)
    
    print(f'Original shape: {data.shape}')
    print(f'Data range: {data.min():.4f} - {data.max():.4f}')
    
    return data

def process_hyperspectral_tif(tif_path, data, selected_channel, clip_min, clip_max):
    """Process hyperspectral TIF (30 or 60 channels) and create single-channel visualization"""
    model_name = extract_model_name(tif_path)
    output_path = os.path.join(OUTPUT_BASE_DIR, f"{model_name}_hs_gen_6_{selected_channel}_clip{clip_min}to{clip_max}.png")
    
    try:
        # Take only first 30 channels
        if data.shape[-1] > 30:
            data = data[..., :30]
            print(f"Trimmed to first 30 channels: {data.shape}")
        
        # Extract the selected channel
        selected_channel_data = data[:, :, selected_channel]
        print(f"Extracted channel {selected_channel} from shape {data.shape}")
        
        # Calculate statistics
        current_mean = np.mean(selected_channel_data)
        current_std = np.std(selected_channel_data)
        original_min = np.min(selected_channel_data)
        original_max = np.max(selected_channel_data)
        
        # Clip data to specified range
        clipped_data = np.clip(selected_channel_data, clip_min, clip_max)
        
        # Normalize clipped data to 0-255 range
        channel_norm = ((clipped_data - clip_min) / (clip_max - clip_min) * 255).astype(np.uint8)
        
        # Apply a colormap (viridis)
        colored_img = cm.viridis(channel_norm)
        # Convert from RGBA to RGB and back to 0-255 range
        colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)
        
        # Save visualization
        img = Image.fromarray(colored_img)
        img.save(output_path)
        print(f"Saved hyperspectral visualization to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing hyperspectral {tif_path}: {str(e)}")
        return False

def process_rgb_tif(tif_path, data):
    """Process RGB TIF (3 or 6 channels) and create RGB visualization"""
    model_name = extract_model_name(tif_path)
    output_path = os.path.join(OUTPUT_BASE_DIR, f"{model_name}_hs_gen_6_rgb.png")
    
    try:
        # Take only first 3 channels
        if data.shape[-1] > 3:
            data = data[..., :3]
            print(f"Trimmed to first 3 channels: {data.shape}")
        
        rgb = data[..., 0:3]
        
        # Clip each channel separately
        red_clipped = np.clip(rgb[..., 0], RGB_CLIP_MIN_R, RGB_CLIP_MAX_R)
        green_clipped = np.clip(rgb[..., 1], RGB_CLIP_MIN_G, RGB_CLIP_MAX_G)
        blue_clipped = np.clip(rgb[..., 2], RGB_CLIP_MIN_B, RGB_CLIP_MAX_B)
        
        # Normalize each channel separately to 0-255
        red_norm = ((red_clipped - RGB_CLIP_MIN_R) / (RGB_CLIP_MAX_R - RGB_CLIP_MIN_R) * 255).astype(np.uint8)
        green_norm = ((green_clipped - RGB_CLIP_MIN_G) / (RGB_CLIP_MAX_G - RGB_CLIP_MIN_G) * 255).astype(np.uint8)
        blue_norm = ((blue_clipped - RGB_CLIP_MIN_B) / (RGB_CLIP_MAX_B - RGB_CLIP_MIN_B) * 255).astype(np.uint8)
        
        # Stack the normalized channels back together
        rgb_norm = np.stack([red_norm, green_norm, blue_norm], axis=-1)

        img = Image.fromarray(rgb_norm, mode='RGB')
        img.save(output_path)
        print(f"Saved RGB visualization to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing RGB {tif_path}: {str(e)}")
        return False

def process_raw_image(base_dir):
    """Process hs_raw_6.tif from the first directory found."""
    # Find the first subdirectory
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        print("No subdirectories found")
        return False
    
    first_subdir = subdirs[0]
    raw_path = os.path.join(base_dir, first_subdir, "test_latest", "images", "hs_raw_6.tif")
    
    if not os.path.exists(raw_path):
        print(f"hs_raw_6.tif not found at: {raw_path}")
        return False
    
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    try:
        data = load_and_normalize_tif(raw_path)
        print(f'Raw image shape: {data.shape}')

        # Handle both 3-channel and 6-channel raw images
        if data.shape[-1] == 3:
            rgb = data[..., 0:3]
        elif data.shape[-1] == 6:
            rgb = data[..., 0:3]
        else:
            raise ValueError(f"Raw file has {data.shape[-1]} channels, expected 3 or 6.")
        
        # Use same clip values as generated images
        clip_min_r = 0.02
        clip_max_r = 0.7
        clip_min_g = 0.02
        clip_max_g = 0.3
        clip_min_b = 0.02
        clip_max_b = 0.5
        
        # Clip each channel separately
        red_clipped = np.clip(rgb[..., 0], clip_min_r, clip_max_r)
        green_clipped = np.clip(rgb[..., 1], clip_min_g, clip_max_g)
        blue_clipped = np.clip(rgb[..., 2], clip_min_b, clip_max_b)
        
        # Normalize each channel separately to 0-255
        red_norm = ((red_clipped - clip_min_r) / (clip_max_r - clip_min_r) * 255).astype(np.uint8)
        green_norm = ((green_clipped - clip_min_g) / (clip_max_g - clip_min_g) * 255).astype(np.uint8)
        blue_norm = ((blue_clipped - clip_min_b) / (clip_max_b - clip_min_b) * 255).astype(np.uint8)
        
        # Stack the normalized channels back together
        rgb_norm = np.stack([red_norm, green_norm, blue_norm], axis=-1)

        img = Image.fromarray(rgb_norm, mode='RGB')
        img.save(RAW_OUTPUT_PATH)
        print(f"Saved raw RGB PNG to: {RAW_OUTPUT_PATH}")
        return True
    except Exception as e:
        print(f"Error processing raw image: {str(e)}")
        return False

def process_single_tif(tif_path):
    """Process a single TIF file - detect type and route to appropriate function"""
    try:
        # Load and normalize the TIF data
        data = load_and_normalize_tif(tif_path)
        
        # Determine processing type based on number of channels
        num_channels = data.shape[-1]
        
        if num_channels >= 30:
            # Hyperspectral processing (30+ channels)
            print(f"Processing as HYPERSPECTRAL ({num_channels} channels)")
            return process_hyperspectral_tif(tif_path, data, SELECTED_CHANNEL, CLIP_MIN, CLIP_MAX)
        elif num_channels >= 3:
            # RGB processing (3-6 channels)
            print(f"Processing as RGB ({num_channels} channels)")
            return process_rgb_tif(tif_path, data)
        else:
            print(f"Unsupported number of channels: {num_channels}")
            return False
            
    except Exception as e:
        print(f"Error loading {tif_path}: {str(e)}")
        return False

# Main processing
def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # Find all TIF files
    tif_paths = find_tif_images(base_dir)
    print(f"Found {len(tif_paths)} hs_gen_6.tif files in {base_dir}")
    
    # Process the raw image first
    print("\nProcessing raw image...")
    process_raw_image(base_dir)
    
    # Process all found TIF files
    successful = 0
    failed = 0
    hyperspectral_count = 0
    rgb_count = 0
    
    print(f"\nProcessing {len(tif_paths)} TIF files...")
    
    for tif_path in tif_paths:
        if os.path.exists(tif_path):
            print(f"\n--- Processing: {extract_model_name(tif_path)} ---")
            
            # Determine file type first for counting
            try:
                temp_data = load_and_normalize_tif(tif_path)
                if temp_data.shape[-1] >= 30:
                    hyperspectral_count += 1
                else:
                    rgb_count += 1
            except:
                pass
            
            # Process the file
            if process_single_tif(tif_path):
                successful += 1
            else:
                failed += 1
        else:
            print(f"File not found: {tif_path}")
            failed += 1
    
    print(f"\n=== Processing Complete ===")
    print(f"Total files processed: {len(tif_paths)}")
    print(f"Hyperspectral files: {hyperspectral_count}")
    print(f"RGB files: {rgb_count}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()