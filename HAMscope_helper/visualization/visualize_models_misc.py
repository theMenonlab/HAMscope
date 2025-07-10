import tifffile
import numpy as np
import os
from PIL import Image
from matplotlib import cm

#raw = False  # Set to True if processing raw images, False for processed hyperspectral images
#clip_min_3ch = 0.0
#clip_max_3ch = 0.3
#clip_max_3ch = 0.05
#clip_min = 0.0
#clip_max = 0.15
#clip_max = 0.015
#img_numbers = [0, 100, 200]
#input_base_dir = "/media/al/Extreme SSD/20250425_results/results/misc_test_dataset_layernorm/results/20250425_0gan_single_reg_hs_dasmeet_movie"
#input_base_dir = "/media/al/Extreme SSD/20250425_results/results/misc_test_dataset_layernorm/results/20250425_0gan_single_reg_hs_nmf_dasmeet_movie"

#img_numbers = [0, 50, 99]
#input_base_dir = "/media/al/Extreme SSD/20250425_results/results/misc_test_dataset_layernorm/results/20250425_0gan_single_reg_hs_fig5A"
#input_base_dir = "/media/al/Extreme SSD/20250425_results/results/misc_test_dataset_layernorm/results/20250425_0gan_single_reg_hs_nmf_fig5A"

#selected_channels = [0, 12, 23]  # 700, 600, 500,
#selected_channels = [30, 42, 53]  # 700, 600, 500,
#output_base_dir = "/media/al/Extreme SSD/20250425_results/results/misc_test_dataset_layernorm/output_raw/gen"

#raw = False  # Set to True if processing raw images, False for processed hyperspectral images
#clip_min_3ch = 0.0
#clip_max_3ch = 0.03
#clip_max_3ch = 0.3
#clip_max_3ch = 700000
#clip_min = 0.0
#clip_max = 0.2
#clip_min = 0.0
#clip_max = 1
#clip_max = 0.015

#img_numbers = [0, 50, 99]
#img_numbers = [1, 51, 201]
#input_base_dir = "/media/al/Extreme SSD/20250425_results/results/20250522_2_timelapse/results/20250425_0gan_single_reg_hs_522time"
#input_base_dir = "/media/al/Extreme SSD/20250425_results/results/20250522_2_timelapse/results/20250425_0gan_single_reg_hs_nmf_522_time"
#input_base_dir = "/media/al/Extreme SSD/20250522_timelapse_2/nmf_fixed"

#selected_channels = [0, 12, 23]  # 700, 600, 500,
#selected_channels = [30, 42, 53]  # 700, 600, 500,
#output_base_dir = "/media/al/Extreme SSD/20250425_results/results/misc_test_dataset_layernorm/output_raw"

#raw = True  # Set to True if processing raw images, False for processed hyperspectral images
#clip_min_3ch = 0.0
#clip_max_3ch = 300000
#clip_min = 0.0
#clip_max = 60000

#img_numbers = [8, 288, 631]
#img_numbers = [5, 285, 631]
#input_base_dir = "/media/al/Extreme SSD/20250519_timelapse/ham_aligned_every_7th"
#input_base_dir = "/media/al/Extreme SSD/20250519_timelapse/nmf_every_7th"

#clip_min = 0.0
#clip_max = 0.3
#clip_min = 0.0
#clip_max = 0.9
#clip_min = 0.0
#clip_max = 0.03
#clip_min_3ch = 0.0
#clip_max_3ch = 0.7
#img_numbers = [9, 24, 36]
#raw = False
#input_base_dir = "/media/al/Extreme SSD/20250425_results/results/misc_test_dataset_layernorm/results/20250425_0gan_single_reg_hs_suberine"
#input_base_dir = "/media/al/Extreme SSD/20250425_results/results/misc_test_dataset_layernorm/results/20250425_0gan_single_reg_hs_nmf_suberine"


#selected_channels = [0, 12, 23]  # 700, 600, 500,
#selected_channels = [30, 42, 53]  # 700, 600, 500,
#output_base_dir = "//media/al/Extreme SSD/20250425_results/results/misc_test_dataset_layernorm/output_raw"

#clip_min = 0.0
#clip_max = 0.25
#clip_min = 0.02
#clip_max = 0.15
#clip_min_3ch = 0.0
#clip_max_3ch = 0.4
#img_numbers = [14, 18, 32]
#input_base_dir = "/media/al/Extreme SSD/20250425_results/results/misc_test_dataset_layernorm/results/20250425_0gan_single_reg_hs_nmf_fig4wt"
#input_base_dir = "/media/al/Extreme SSD/20250425_results/results/misc_test_dataset_layernorm/results/20250425_0gan_single_reg_hs_fig4wt"
#input_base_dir = "/media/al/Extreme SSD/20250425_results/results/misc_test_dataset_layernorm/results/20250425_0gan_single_reg_hs_fig4mut"
#input_base_dir = "/media/al/Extreme SSD/20250425_results/results/misc_test_dataset_layernorm/results/20250425_0gan_single_reg_hs_nmf_fig4mut"

# Hard-coded channel selection variable
#selected_channels = [29, 17, 6]  # 450, 550, 650,
#selected_channels = [59, 47, 36]  # 450, 550, 650,



#clip_min = 0.0
#clip_max = 50000
#clip_min_3ch = 0.0
#clip_max_3ch = 50000
#img_numbers = [15, 19, 33]
#raw = True
#input_base_dir = "/media/al/Extreme SSD/20250410_dasmeet/20250410_dasmeet_aligned/wt/aligned_manual"
#input_base_dir = "/media/al/Extreme SSD/20250410_dasmeet/20250410_dasmeet_aligned/mut/aligned_manual"

# Hard-coded channel selection variable
#selected_channels = [1, 3, 4]  # 450, 550, 650,
#output_base_dir = "/media/al/Extreme SSD/20250425_results/results/misc_test_dataset_layernorm/output_raw"



def process_3_channel_image(img_array, output_path, base_name):
    """Process 3-channel image and save as PNG"""
    try:
        # Take first 3 channels if more than 3 exist
        if img_array.ndim == 3 and img_array.shape[0] >= 3:
            img_array = img_array[:3, :, :] # mean
            #img_array = img_array[3:, :, :] # scale
            # Transpose from (C, H, W) to (H, W, C) format
            img_array = np.transpose(img_array, (1, 2, 0))
        else:
            print(f"Invalid shape for 3-channel processing: {img_array.shape}")
            return False
        
        # BGR to rGB conversion
        #img_array = img_array[..., ::-1]  # Reverse the last dimension (BGR to RGB)
        
        # Clip data to specified range
        clipped_data = np.clip(img_array, clip_min_3ch, clip_max_3ch)
        
        # Normalize clipped data to 0-255 range
        img_normalized = ((clipped_data - clip_min_3ch) / (clip_max_3ch - clip_min_3ch) * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(img_normalized)
        
        # Create filename with clip parameters
        output_dir = os.path.dirname(output_path)
        filename_with_clip = f"{base_name}_3ch_clip[{clip_min_3ch},{clip_max_3ch}].png"
        final_output_path = os.path.join(output_dir, filename_with_clip)
        
        # Save as PNG
        pil_image.save(final_output_path)
        print(f"Converted 3-channel image -> {os.path.basename(final_output_path)}")
        return True
        
    except Exception as e:
        print(f"Error processing 3-channel image: {str(e)}")
        return False

def process_hyperspectral_image(img_data, output_dir, base_name):
    """Process hyperspectral image and save selected channels"""
    try:
        # Check if image has enough channels for hyperspectral processing
        if img_data.ndim == 3 and img_data.shape[0] > 3:
            # Format is [channels, height, width]
            for i, channel in enumerate(selected_channels):
                if channel < img_data.shape[0]:
                    selected_channel_data = img_data[channel, :, :]
                    
                    # Normalize data using min/max clipping
                    clipped_data = np.clip(selected_channel_data, clip_min, clip_max)
                    channel_norm = ((clipped_data - clip_min) / (clip_max - clip_min) * 255).astype(np.uint8)
                    
                    # Apply viridis colormap
                    colored_img = cm.viridis(channel_norm)
                    colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)
                    
                    # Save the image
                    output_path = os.path.join(output_dir, f"{base_name}_ch{channel}_clip[{clip_min},{clip_max}].png")
                    img = Image.fromarray(colored_img)
                    img.save(output_path)
                    print(f"Saved hyperspectral channel: {base_name}_channel_{channel}.png")
            return True
        else:
            print(f"Not enough channels for hyperspectral processing (shape: {img_data.shape})")
            return False
            
    except Exception as e:
        print(f"Error processing hyperspectral image: {str(e)}")
        return False

def process_model_directory():
    """Process all model directories and generate figures"""
    
    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get model name from the input directory
    model_name = os.path.basename(input_base_dir)
    
    print(f"\nProcessing model: {model_name}")
    
    # Path to images directory
    if raw:
        images_dir = input_base_dir
    else:
        images_dir = os.path.join(input_base_dir, "test_latest", "images")
    #images_dir = os.path.join(input_base_dir, "test_latest", "images")
    
    print(f"Images directory: {images_dir}")
    
    if not os.path.exists(images_dir):
        print(f"Images directory not found for {model_name}")
        return
    
    # Create output directories
    model_output_dir = os.path.join(output_base_dir, model_name)
    hyperspectral_output_dir = os.path.join(model_output_dir, "hyperspectral_channels")
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(hyperspectral_output_dir, exist_ok=True)
    
    # Process specified image numbers
    for img_num in img_numbers:

       
        if raw:
            #img_filename = f"pos{img_num}_aligned_stack.tif"
            #img_filename = f"pos{img_num}_aligned_stack_nmf.tif"
            img_filename = f"hs_raw_{img_num}.tif"
        else:
            img_filename = f"hs_gen_{img_num}.tif"
        img_path = os.path.join(images_dir, img_filename)
        
        if os.path.exists(img_path):
            try:
                # Load image to check number of channels
                img_data = tifffile.imread(img_path)
                print(f"Loaded image: {img_path} with shape: {img_data.shape}")
                base_name = os.path.splitext(img_filename)[0]
                
                print(f"Processing {img_filename} with shape: {img_data.shape}")
                if raw:
                    if img_data.ndim == 3 and img_data.shape[0] <= 3:
                        # Process as 3-channel image only
                        output_path = os.path.join(model_output_dir, f"{base_name}.png")
                        process_3_channel_image(img_data, output_path, base_name)
                        
                    elif img_data.ndim == 3 and img_data.shape[0] > 3:
                        # Process as hyperspectral image (save selected channels in subfolder)
                        process_hyperspectral_image(img_data, hyperspectral_output_dir, base_name)
                else:
                    if img_data.ndim == 3 and img_data.shape[0] <= 6:
                        # Process as 3-channel image only
                        output_path = os.path.join(model_output_dir, f"{base_name}.png")
                        process_3_channel_image(img_data, output_path, base_name)
                        
                    elif img_data.ndim == 3 and img_data.shape[0] > 6:
                        # Process as hyperspectral image (save selected channels in subfolder)
                        process_hyperspectral_image(img_data, hyperspectral_output_dir, base_name)
                    
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
        else:
            print(f"Image not found: {img_path}")

if __name__ == "__main__":
    process_model_directory()
    print(f"\nAll figures saved to: {output_base_dir}")