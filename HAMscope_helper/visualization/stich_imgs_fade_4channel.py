import os
import glob
import numpy as np
import tifffile
from PIL import Image

def map_4channel_to_rgb(img_4ch, color_mapping=None):
    """
    Maps 4 input channels to RGB using customizable color assignments.
    
    Args:
        img_4ch: Input image with 4 channels
        color_mapping: Dict or list defining color for each channel.
                      Can be:
                      - Dict: {channel_index: (R, G, B)} where values are 0-1
                      - List: [(R, G, B), (R, G, B), (R, G, B), (R, G, B)] for channels 0,1,2,3
                      - None: Uses default color scheme
    
    Default color scheme:
        Channel 0 -> Red (1, 0, 0)
        Channel 1 -> Yellow (1, 1, 0) 
        Channel 2 -> Magenta (1, 0, 1)
        Channel 3 -> Blue (0, 0, 1)
    """
    if len(img_4ch.shape) != 3 or img_4ch.shape[2] != 4:
        return img_4ch  # Return unchanged if not 4-channel
    
    # Default color mapping (normalized 0-1 values)
    if color_mapping is None:
        color_mapping = {
            0: (1.0, 0.0, 0.0),  # Red,
            1: (1.0, 1.0, 0.0),  # Yellow
            2: (0.0, 1.0, 0.0),  # green
            3: (0.0, 0.0, 1.0),  # Blue
        }
    
    # Convert list to dict if needed
    if isinstance(color_mapping, list):
        color_mapping = {i: color_mapping[i] for i in range(len(color_mapping))}
    
    height, width, _ = img_4ch.shape
    rgb_output = np.zeros((height, width, 3), dtype=np.float32)
    
    # Get all channels and normalize them individually
    channels = []
    for i in range(4):
        ch = img_4ch[:, :, i].astype(np.float32)
        ch_max = np.max(ch)
        if ch_max > 0:
            ch = ch / ch_max
        channels.append(ch)
    
    # Mix each channel with its assigned color
    for ch_idx, (r, g, b) in color_mapping.items():
        if ch_idx < 4:  # Safety check
            rgb_output[:, :, 0] += channels[ch_idx] * r  # Red component
            rgb_output[:, :, 1] += channels[ch_idx] * g  # Green component  
            rgb_output[:, :, 2] += channels[ch_idx] * b  # Blue component
    
    # Normalize the final output to prevent overflow
    max_val = np.max(rgb_output)
    if max_val > 1.0:
        rgb_output = rgb_output / max_val
    
    # Scale to the original data range (find max across all input channels)
    original_max = np.max(img_4ch)
    rgb_output = rgb_output * original_max
    
    return rgb_output

# Hardcoded input and output directories
input_dir = "/media/al/Extreme SSD/20250425_dataset/example_depth_stitch/nmf_norm_4ch_roi2"
output_dir = "/media/al/Extreme SSD/20250425_dataset/example_depth_stitch/nmf_norm_4ch_roi2"

# Offset parameter (in pixels) to fine-tune the crop width
offset_pixels = 4
microns_per_image = 940
microns_step = 100

# New parameter: Width of the fade/blend zone in pixels
fade_pixel_width = 15  # Example: 20 pixels fade width

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all image files in the input directory and sort them
image_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
image_files.reverse()  # Images go from right to left

if not image_files:
    print(f"No images found in {input_dir}")
    exit()

print(f"Found {len(image_files)} images to stitch (right to left)")

# Load the first image to get dimensions
try:
    first_image = tifffile.imread(image_files[0])
except FileNotFoundError:
    print(f"Error: Could not open the first image: {image_files[0]}")
    exit()
except Exception as e:
    print(f"Error loading first image {image_files[0]}: {e}")
    exit()
    
# tifffile returns numpy array with shape (height, width, channels)
height, width = first_image.shape[0], first_image.shape[1]
print(f'first image shape: {first_image.shape}')
print(f'first image max: {np.max(first_image)}')

# if the first dim is smallest, move it to the last
if len(first_image.shape) == 3 and first_image.shape[0] < first_image.shape[1]:
    # Move channels to last dimension
    first_image = np.moveaxis(first_image, 0, -1)
    # switch red and blue channels
    # first_image = first_image[:, :, [2, 1, 0]]
    img = map_4channel_to_rgb(first_image)
    height, width = first_image.shape[0], first_image.shape[1]
    print(f'first image shape after: {first_image.shape}')
    print(f'first image max after: {np.max(first_image)}')

# Calculate crop amount based on overlap
microns_overlap = microns_per_image - microns_step
crop_microns = microns_overlap / 2
crop_pixels_calculated = int(width * crop_microns / microns_per_image)

adjusted_crop_pixels = crop_pixels_calculated + offset_pixels
print(f"Calculated crop per side: {crop_pixels_calculated} pixels.")
print(f"Offset: {offset_pixels} pixels.")
print(f"Final adjusted crop from each side (before fading considerations): {adjusted_crop_pixels} pixels.")

if adjusted_crop_pixels < 0:
    print(f"Warning: adjusted_crop_pixels ({adjusted_crop_pixels}) is negative.")
if 2 * adjusted_crop_pixels >= width:
    print(f"Error: The total crop amount (2 * {adjusted_crop_pixels} = {2*adjusted_crop_pixels}) is greater than or equal to image width ({width}).")
    print("This means there's no image content left after cropping. Adjust offset_pixels or check image dimensions.")
    exit()

nominal_segment_width = width - 2 * adjusted_crop_pixels
if nominal_segment_width <= 0:
    print(f"Error: Nominal segment width ({nominal_segment_width}) is zero or negative. Check crop parameters.")
    exit()
print(f"Nominal width of each image segment after cropping: {nominal_segment_width} pixels.")

# Helper function to convert float array to uint8 for PIL
def prepare_for_pil(img_array):
    # Check if the array is float type
    if np.issubdtype(img_array.dtype, np.floating):
        # Get the data range
        data_max = np.max(img_array)
        
        # Scale based on the range
        if data_max > 255:
            # Scale down values above 255
            img_array = (img_array / data_max * 255).astype(np.uint8)
        else:
            # Just convert directly if values are already in 0-255 range
            img_array = img_array.astype(np.uint8)
    
    return img_array

if fade_pixel_width <= 0:
    print("Fading is disabled (fade_pixel_width <= 0). Using original stitching logic.")
    # Original stitching logic (without fading)
    cropped_images = []
    for i, img_path in enumerate(image_files):
        try:
            print(f"Processing image {i+1}/{len(image_files)}")
            img = tifffile.imread(img_path)
            
            # Move channels to last dim if needed
            if len(img.shape) == 3 and img.shape[0] < img.shape[1]:
                # Move channels to last dimension
                img = np.moveaxis(img, 0, -1)
                
                # switch red and blue channels
                #img = img[:, :, [2, 1, 0]]
            img = map_4channel_to_rgb(img)
            
            # Special case for first (rightmost) and last (leftmost) images
            if i == 0:  # Rightmost image - keep LEFT edge
                cropped = img[:, 0:width - adjusted_crop_pixels]
            elif i == len(image_files) - 1:  # Leftmost image - keep RIGHT edge
                cropped = img[:, adjusted_crop_pixels:]
            else:  # Middle images - crop both sides
                cropped = img[:, adjusted_crop_pixels:width - adjusted_crop_pixels]
            
            cropped_images.append(cropped)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue
    
    if not cropped_images:
        print("No images were successfully cropped. Exiting.")
        exit()

    # Calculate final width
    final_stitched_width = nominal_segment_width * (len(cropped_images) - 2)  # Middle images
    if len(cropped_images) >= 1:
        final_stitched_width += cropped_images[0].shape[1]  # First image
    if len(cropped_images) >= 2:
        final_stitched_width += cropped_images[-1].shape[1]  # Last image
    
    # Convert all images to uint8 for PIL
    pil_ready_images = [prepare_for_pil(img) for img in cropped_images]
    
    # Create stitched image canvas
    stitched_image_canvas = Image.new('RGB', (final_stitched_width, height))

    current_paste_x = 0
    for img in pil_ready_images:
        img_pil = Image.fromarray(img)
        stitched_image_canvas.paste(img_pil, (current_paste_x, 0))
        current_paste_x += img_pil.width

else:
    print(f"Fading enabled with fade_pixel_width = {fade_pixel_width} pixels.")
    if fade_pixel_width > nominal_segment_width:
        print(f"Warning: fade_pixel_width ({fade_pixel_width}) is greater than nominal_segment_width ({nominal_segment_width}).")
        print("The entire segment might be consumed by the fade. Results might be unexpected.")

    original_images = []
    for img_path in image_files:
        try:
            img = tifffile.imread(img_path)
            # Move channels to last position if needed
            if len(img.shape) == 3 and img.shape[0] < img.shape[1]:
                img = np.moveaxis(img, 0, -1)
                # switch red and blue channels
                #img = img[:, :, [2, 1, 0]]
            img = map_4channel_to_rgb(img)
                
            original_images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue

    if not original_images:
        print("No images loaded successfully for fading. Exiting.")
        exit()
    
    if len(original_images) == 1:
        print("Only one image found. Saving it directly without cropping (full width).")
        stitched_image = prepare_for_pil(original_images[0])
        stitched_image_canvas = Image.fromarray(stitched_image)
    else:
        # Calculate the final image width
        first_img_contribution = width - adjusted_crop_pixels  # Keep full left edge of first image
        last_img_contribution = width - adjusted_crop_pixels   # Keep full right edge of last image
        middle_imgs_contribution = (len(original_images) - 2) * (nominal_segment_width - fade_pixel_width)
        
        final_stitched_width = first_img_contribution + middle_imgs_contribution + last_img_contribution
        if final_stitched_width <= 0:
            print(f"Error: Calculated final_stitched_width ({final_stitched_width}) is zero or negative. Check parameters.")
            exit()

        print(f"Final stitched image dimensions: {final_stitched_width} x {height}")
        stitched_image = np.zeros((height, final_stitched_width, 3), dtype=np.float32)
        
        current_x_on_canvas = 0

        for i in range(len(original_images)):
            img_curr = original_images[i]

            if i == 0: # Rightmost image
                # Keep full LEFT edge, only crop right side
                segment_to_paste = img_curr[:, 0:width-adjusted_crop_pixels]
                print(f'segment_to_paste shape: {segment_to_paste.shape}')

                segment_width = segment_to_paste.shape[1]
                stitched_image[:, current_x_on_canvas:current_x_on_canvas + segment_width] = segment_to_paste
                current_x_on_canvas += segment_width
            elif i == len(original_images) - 1: # Leftmost image
                # For the last image, we need to blend its left edge with the previous image
                img_prev = original_images[i-1]

                # Extract strips from original images for blending
                prev_strip_orig_start = width - adjusted_crop_pixels - fade_pixel_width
                prev_strip_orig_end = width - adjusted_crop_pixels
                
                curr_strip_orig_start = adjusted_crop_pixels
                curr_strip_orig_end = adjusted_crop_pixels + fade_pixel_width

                # Safety checks and clamping
                prev_strip_orig_start_clamped = max(0, prev_strip_orig_start)
                prev_strip_orig_end_clamped = min(width, prev_strip_orig_end)
                curr_strip_orig_start_clamped = max(0, curr_strip_orig_start)
                curr_strip_orig_end_clamped = min(width, curr_strip_orig_end)

                if prev_strip_orig_start_clamped >= prev_strip_orig_end_clamped or \
                   curr_strip_orig_start_clamped >= curr_strip_orig_end_clamped:
                    print(f"Warning: Cannot extract valid strips for blending at last seam. Skipping blend.")
                    current_x_on_canvas -= fade_pixel_width
                    curr_main_body = img_curr[:, adjusted_crop_pixels:]  # Keep full right side
                    if curr_main_body.size > 0:
                        curr_main_body_width = curr_main_body.shape[1]
                        stitched_image[:, current_x_on_canvas:current_x_on_canvas + curr_main_body_width] = curr_main_body
                    continue

                strip_prev = img_prev[:, prev_strip_orig_start_clamped:prev_strip_orig_end_clamped]
                strip_curr = img_curr[:, curr_strip_orig_start_clamped:curr_strip_orig_end_clamped]
                
                actual_blend_w = min(strip_prev.shape[1], strip_curr.shape[1])

                if actual_blend_w <= 0:
                    print(f"Warning: Actual blend width is {actual_blend_w} at last seam. Skipping blend.")
                    current_x_on_canvas -= fade_pixel_width
                    curr_main_body = img_curr[:, adjusted_crop_pixels:]  # Keep full right side
                    if curr_main_body.size > 0:
                        curr_main_body_width = curr_main_body.shape[1]
                        stitched_image[:, current_x_on_canvas:current_x_on_canvas + curr_main_body_width] = curr_main_body
                    continue
                
                strip_prev = strip_prev[:, :actual_blend_w]
                strip_curr = strip_curr[:, :actual_blend_w]

                blended_strip_float = np.zeros((height, actual_blend_w, 3), dtype=np.float32)
                for col in range(actual_blend_w):
                    alpha = (col + 0.5) / actual_blend_w
                    # Fix: properly handle the channel dimension with [:, col, :]
                    blended_strip_float[:, col, :] = (strip_prev[:, col, :] * (1.0 - alpha) + 
                                                     strip_curr[:, col, :] * alpha)
                
                blend_paste_x_on_canvas = current_x_on_canvas - actual_blend_w
                stitched_image[:, blend_paste_x_on_canvas:blend_paste_x_on_canvas + actual_blend_w] = blended_strip_float

                # Paste the right part of the last image (including its full right edge)
                curr_remaining_body_start_orig = adjusted_crop_pixels + actual_blend_w
                curr_remaining_body = img_curr[:, curr_remaining_body_start_orig:]
                curr_remaining_body_width = curr_remaining_body.shape[1]
                paste_x_for_curr_remaining = blend_paste_x_on_canvas + actual_blend_w
                stitched_image[:, paste_x_for_curr_remaining:paste_x_for_curr_remaining + curr_remaining_body_width] = curr_remaining_body
                current_x_on_canvas = paste_x_for_curr_remaining + curr_remaining_body_width
            else: # Middle images 
                # Use existing blending logic for middle images
                img_prev = original_images[i-1]

                # Extract strips from original images for blending
                prev_strip_orig_start = width - adjusted_crop_pixels - fade_pixel_width
                prev_strip_orig_end = width - adjusted_crop_pixels
                
                curr_strip_orig_start = adjusted_crop_pixels
                curr_strip_orig_end = adjusted_crop_pixels + fade_pixel_width

                # Defensive slicing
                prev_strip_orig_start_clamped = max(0, prev_strip_orig_start)
                prev_strip_orig_end_clamped = min(width, prev_strip_orig_end)
                curr_strip_orig_start_clamped = max(0, curr_strip_orig_start)
                curr_strip_orig_end_clamped = min(width, curr_strip_orig_end)

                if prev_strip_orig_start_clamped >= prev_strip_orig_end_clamped or \
                   curr_strip_orig_start_clamped >= curr_strip_orig_end_clamped:
                    print(f"Warning: Cannot extract valid strips for blending at seam {i}. Skipping blend.")
                    current_x_on_canvas -= fade_pixel_width
                    curr_main_body = img_curr[:, adjusted_crop_pixels:width - adjusted_crop_pixels]
                    if curr_main_body.size > 0:
                        curr_main_body_width = curr_main_body.shape[1]
                        stitched_image[:, current_x_on_canvas:current_x_on_canvas + curr_main_body_width] = curr_main_body
                        current_x_on_canvas += curr_main_body_width
                    continue

                strip_prev = img_prev[:, prev_strip_orig_start_clamped:prev_strip_orig_end_clamped]
                strip_curr = img_curr[:, curr_strip_orig_start_clamped:curr_strip_orig_end_clamped]
                
                actual_blend_w = min(strip_prev.shape[1], strip_curr.shape[1])

                if actual_blend_w <= 0:
                    print(f"Warning: Actual blend width is {actual_blend_w} at seam {i}. Skipping blend.")
                    current_x_on_canvas -= fade_pixel_width
                    curr_main_body = img_curr[:, adjusted_crop_pixels:width - adjusted_crop_pixels]
                    if curr_main_body.size > 0:
                        curr_main_body_width = curr_main_body.shape[1]
                        stitched_image[:, current_x_on_canvas:current_x_on_canvas + curr_main_body_width] = curr_main_body
                        current_x_on_canvas += curr_main_body_width
                    continue
                
                strip_prev = strip_prev[:, :actual_blend_w]
                strip_curr = strip_curr[:, :actual_blend_w]

                blended_strip_float = np.zeros((height, actual_blend_w, 3), dtype=np.float32)
                for col in range(actual_blend_w):
                    alpha = (col + 0.5) / actual_blend_w
                    # Fix: properly handle the channel dimension with [:, col, :]
                    blended_strip_float[:, col, :] = (strip_prev[:, col, :] * (1.0 - alpha) + 
                                                     strip_curr[:, col, :] * alpha)
                
                blend_paste_x_on_canvas = current_x_on_canvas - actual_blend_w
                stitched_image[:, blend_paste_x_on_canvas:blend_paste_x_on_canvas + actual_blend_w] = blended_strip_float

                # Middle image - crop both sides
                curr_remaining_body_start_orig = adjusted_crop_pixels + actual_blend_w
                curr_remaining_body_end_orig = width - adjusted_crop_pixels
                
                paste_x_for_curr_remaining = blend_paste_x_on_canvas + actual_blend_w

                if curr_remaining_body_start_orig < curr_remaining_body_end_orig:
                    curr_remaining_body = img_curr[:, curr_remaining_body_start_orig:curr_remaining_body_end_orig]
                    curr_remaining_body_width = curr_remaining_body.shape[1]
                    stitched_image[:, paste_x_for_curr_remaining:paste_x_for_curr_remaining + curr_remaining_body_width] = curr_remaining_body
                    current_x_on_canvas = paste_x_for_curr_remaining + curr_remaining_body_width
                else:
                    current_x_on_canvas = paste_x_for_curr_remaining
        
        # Convert the entire stitched image to uint8 for PIL
        stitched_image_uint8 = prepare_for_pil(stitched_image)
        stitched_image_canvas = Image.fromarray(stitched_image_uint8)

# Save the stitched image
output_filename = "stitched_panorama_faded.tif" if fade_pixel_width > 0 else "stitched_panorama_original.tif"
output_path = os.path.join(output_dir, output_filename)

try:
    stitched_image_canvas.save(output_path)
    print(f"Stitched image saved to {output_path}")
except Exception as e:
    print(f"Error saving stitched image: {e}")

