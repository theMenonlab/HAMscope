from scipy.ndimage import rotate
import scipy.io
import numpy as np
import cv2
import os
from glob import glob
from tifffile import imsave
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def natural_sort_key(s):
    """Sort strings containing numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def get_position_number(filename):
    """Extract position number from filename."""
    match = re.search(r'pos(\d+)', filename)
    return int(match.group(1)) if match else None

def get_filter_number(filename):
    """Extract filter number from filename."""
    match = re.search(r'filt(\d+)', filename)
    return int(match.group(1)) if match else None

# For comparing Miniscope with different Hamamatsu channels
def slider_alignment(miniscope_img, ham_stack, pos_num=""):
    """
    Create an interactive visualization to compare Miniscope with different Hamamatsu channels.
    
    Args:
        miniscope_img: Miniscope image array
        ham_stack: Hamamatsu image stack (channels, height, width)
        pos_num: Position number for title
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)
    
    # Make sure images are properly normalized
    if miniscope_img.max() > miniscope_img.min():  # Avoid division by zero
        miniscope_img = (miniscope_img.astype(np.float32) - miniscope_img.min()) / (miniscope_img.max() - miniscope_img.min())
    else:
        miniscope_img = np.zeros_like(miniscope_img, dtype=np.float32)
    
    n_channels = ham_stack.shape[0]
    ham_norm = np.zeros_like(ham_stack, dtype=np.float32)
    for c in range(n_channels):
        if ham_stack[c].max() > ham_stack[c].min():
            ham_norm[c] = (ham_stack[c].astype(np.float32) - ham_stack[c].min()) / (ham_stack[c].max() - ham_stack[c].min())
        else:
            ham_norm[c] = np.zeros_like(ham_stack[c], dtype=np.float32)
    
    print(f"Miniscope shape: {miniscope_img.shape}, min: {miniscope_img.min()}, max: {miniscope_img.max()}")
    print(f"Hamamatsu shape: {ham_norm.shape}, min: {ham_norm.min()}, max: {ham_norm.max()}")

    # Create two separate image objects
    mini_display = ax.imshow(miniscope_img, cmap='gray')
    ham_display = ax.imshow(ham_norm[2], cmap='jet', alpha=0.0)  # Start with channel 3 and hidden
    
    ax.set_title(f"Miniscope (Position {pos_num})")
    ax.axis('off')

    # Create slider axes
    ax_slider_blend = plt.axes([0.2, 0.15, 0.6, 0.03])
    ax_slider_channel = plt.axes([0.2, 0.1, 0.6, 0.03])
    
    # Create sliders
    slider_blend = Slider(ax_slider_blend, 'Blend', 0, 1, valinit=0.0)
    slider_channel = Slider(ax_slider_channel, 'Ham Channel', 0, n_channels-1, 
                           valinit=2, valstep=1, valfmt='%d')
    
    # Update function for sliders
    def update(val):
        blend = slider_blend.val
        channel = int(slider_channel.val)
        
        if blend < 0.01:
            # Only show Miniscope
            mini_display.set_data(miniscope_img)
            mini_display.set_alpha(1.0)
            ham_display.set_alpha(0.0)
            ax.set_title(f"Miniscope (Position {pos_num})")
        elif blend > 0.99:
            # Only show Hamamatsu
            ham_display.set_data(ham_norm[channel])
            ham_display.set_alpha(1.0)
            mini_display.set_alpha(0.0)
            ax.set_title(f"Hamamatsu Channel {channel+1} (Position {pos_num})")
        else:
            # Blend mode
            mini_display.set_data(miniscope_img)
            mini_display.set_alpha(1.0)
            ham_display.set_data(ham_norm[channel])
            ham_display.set_alpha(blend)
            ax.set_title(f"Miniscope with Hamamatsu Ch.{channel+1} overlay (alpha={blend:.2f})")
        
        fig.canvas.draw_idle()

    # Connect sliders to update function
    slider_blend.on_changed(update)
    slider_channel.on_changed(update)
    
    plt.show()

def process_mat_image(mat_filename, var_name, top_left, crop_size, interpolate_size, angle):
    """Process a .mat file with fixed parameters."""
    # Load the .mat file
    data = scipy.io.loadmat(mat_filename)
    
    # Extract the image
    img = data[var_name]

    # Fix broken pixels if present
    if img.shape[0] > 400 and img.shape[1] > 400:
        img[303, 389] = img[303, 388]
        img[443, 357] = img[443, 358]
        img[378, 320] = img[378, 321]

    # Normalize the image
    #img = (img - img.min()) / (img.max() - img.min())
    
    # Define cropping parameters
    x, y = top_left
    width, height = crop_size
    
    # Crop the image
    cropped = img[y:y+height, x:x+width]

    # Rotate the cropped region
    rotated_crop = rotate(cropped, angle, reshape=False, mode='nearest')

    # Flip the image about the vertical axis
    flipped_img = np.fliplr(rotated_crop)

    # Resize using bicubic interpolation
    processed_img = cv2.resize(flipped_img, interpolate_size, interpolation=cv2.INTER_CUBIC)
    
    return processed_img

def apply_fixed_transform(image, transform_params):
    """Apply fixed transformation parameters to an image."""
    # Get transformation parameters
    tx, ty, sx, sy, angle = transform_params
    
    # Convert to OpenCV transformation matrix
    # Calculate center of the image
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    
    # Calculate translation in pixels
    # Map tx from [0,1] to actual pixel shift
    tx_pixels = (tx * 2 - 1) * width / 2
    ty_pixels = (ty * 2 - 1) * height / 2
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle * 180, sx)
    
    # Apply translation
    rotation_matrix[0, 2] += tx_pixels
    rotation_matrix[1, 2] += ty_pixels
    
    # Apply the transformation
    transformed = cv2.warpAffine(image, rotation_matrix, (width, height), 
                               flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return transformed

def batch_process_images(hamamatsu_dir, output_dir, miniscope_dir=None, transform_params=None, 
                       crop_params=None, show=False):
    """
    Process images with fixed parameters and compare with Miniscope images.
    
    Args:
        hamamatsu_dir: Directory containing .mat files
        output_dir: Directory to save processed stacks
        miniscope_dir: Directory containing Miniscope images for comparison
        transform_params: List of transformation parameters [tx, ty, sx, sy, angle]
        crop_params: Dictionary of crop parameters
        show: Whether to show visualizations
    """
    # Default parameters if none provided
    if transform_params is None:
        transform_params = [0.5, 0.5, 1.0, 1.0, 0.0]
        
    if crop_params is None:
        crop_params = {
            'top_left': (0, 0),
            'crop_size': (512, 512),
            'interpolate_size': (608, 608),
            'angle': 0
        }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .mat files
    all_hamamatsu_files = glob(os.path.join(hamamatsu_dir, 'pos*_filt*.mat'))
    
    # Group files by position
    position_files = {}
    for file in all_hamamatsu_files:
        pos_num = get_position_number(os.path.basename(file))
        if pos_num not in position_files:
            position_files[pos_num] = []
        position_files[pos_num].append(file)
    
    # Get Miniscope images if directory is provided
    miniscope_files = []
    if miniscope_dir and show:
        miniscope_files = sorted(glob(os.path.join(miniscope_dir, '*.png')), key=natural_sort_key)
        print(f"Found {len(miniscope_files)} Miniscope images")
    
    # Create position to miniscope mapping based on natural sort order
    position_to_mini = {}
    sorted_positions = sorted(position_files.keys())
    if miniscope_files and len(miniscope_files) >= len(sorted_positions):
        for i, pos in enumerate(sorted_positions):
            if i < len(miniscope_files):
                position_to_mini[pos] = miniscope_files[i]
        print(f"Paired {len(position_to_mini)} positions with Miniscope files")
    
    # Process each position
    total_positions = len(position_files)
    
    for i, pos_num in enumerate(sorted(position_files.keys())):
        print(f"\nProcessing Position {pos_num} ({i+1}/{total_positions})")
        
        ham_files = sorted(position_files[pos_num], key=natural_sort_key)
        
        # Process each filter for this position
        processed_images = []
        
        for ham_path in ham_files:
            print(f"Processing {os.path.basename(ham_path)}")
            
            # Process the image
            processed_ham = process_mat_image(
                ham_path, 'objCCM',
                top_left=crop_params['top_left'],
                crop_size=crop_params['crop_size'],
                interpolate_size=crop_params['interpolate_size'],
                angle=crop_params['angle']
            )
            
            # Apply the fixed transformation
            aligned_img = apply_fixed_transform(processed_ham, transform_params)

            # if the number of hamamatsu channels is grater than 6, rotate the image 180 degrees
            if len(ham_files) > 6:
                aligned_img = rotate(aligned_img, 180, reshape=False, mode='nearest')
            
            # Convert to uint16 for saving
            #aligned_img = (aligned_img * 65535).astype(np.uint16)
            
            # Add to stack
            processed_images.append(aligned_img)
        
        # Save the stack
        image_stack = np.stack(processed_images)

        stack_min = image_stack.min()
        stack_max = image_stack.max()
        #print(f"Stack min: {stack_min}, max: {stack_max}")
        #image_stack = (image_stack - stack_min) / (stack_max - stack_min)
        # print data type
        #print(f"Image stack data type before scaling: {image_stack.dtype}")
        #image_stack = (image_stack * 65535).astype(np.uint16)
        #print(f"Stack shape: {image_stack.shape}, min: {stack_min}, max: {stack_max}")

        output_path = os.path.join(output_dir, f'pos{pos_num}_aligned_stack.tif')
        imsave(output_path, image_stack)
        print(f"Saved stack to {output_path}")
        
        # Show visualizations if requested
        if show:
            # Get corresponding Miniscope image if available
            mini_img = None
            if pos_num in position_to_mini:
                mini_path = position_to_mini[pos_num]
                mini_img = cv2.imread(mini_path, cv2.IMREAD_GRAYSCALE)
                print(f"Using Miniscope reference: {os.path.basename(mini_path)}")
                
                if mini_img is None:
                    print(f"WARNING: Failed to load Miniscope image {mini_path}")
                else:
                    # Print shape to debug
                    print(f"Loaded Miniscope image with shape {mini_img.shape}")
            
            # Show multichannel view of hyperspectral data
            display_stack = np.zeros_like(image_stack, dtype=np.float32)
            for c in range(image_stack.shape[0]):
                img = image_stack[c].astype(np.float32)
                display_stack[c] = (img - img.min()) / (img.max() - img.min())
                
            
            # Compare with Miniscope image if available
            if mini_img is not None:
                # Use channel 3 (index 2) for comparison with Miniscope
                ham_compare = display_stack[2]  # Channel 3 (index 2)
                
                # Normalize Miniscope image
                mini_norm = mini_img.astype(np.float32)
                if mini_norm.max() > mini_norm.min():
                    mini_norm = (mini_norm - mini_norm.min()) / (mini_norm.max() - mini_norm.min())
                else:
                    print("Warning: Miniscope image has no contrast")
                
                # Make sure sizes match
                if mini_norm.shape != ham_compare.shape:
                    print(f"Resizing miniscope image from {mini_norm.shape} to {ham_compare.shape}")
                    mini_norm = cv2.resize(mini_norm, (ham_compare.shape[1], ham_compare.shape[0]))
                
                # DEBUG: Save images to verify content
                plt.figure(figsize=(12, 6))
                plt.subplot(121)
                plt.imshow(mini_norm, cmap='gray')
                plt.title("Miniscope (Debug)")
                plt.subplot(122)
                plt.imshow(ham_compare, cmap='hot')
                plt.title("Hamamatsu Ch.3 (Debug)")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'debug_pos{pos_num}_images.png'))
                plt.close()
                
                print("Debug image comparison saved")
                
                # Use slider alignment to compare
                slider_alignment(
                    mini_norm, 
                    display_stack,
                    f"Position {pos_num}"
                )
    
    print("\nProcessing complete for all positions!")

if __name__ == "__main__":
    # Set your directories

    hamamatsu_dir = '/media/al/Extreme SSD/20250519_timelapse/hamamatsu'
    output_dir = '/media/al/Extreme SSD/20250519_timelapse/ham_nonorm'
    miniscope_dir = '/media/al/Extreme SSD/20250519_timelapse/miniscope/AL/Experiment0/Poplar/customEntValHere/imageCaptures'

    # Fixed transformation parameters [tx, ty, sx, sy, angle]
    # Using the reference parameters from the original code
    # reference params are x translation, y translation, scale x, scale y, rotation angle
    # tolerances are percentage of change allowed for each parameter
    # 0 is no change in shift (positive is down/right), 
    # 0.5 is no change in scale (larger is bigger hyperspectral image), 
    # 0 is no change in rotation (positive is clockwise)
    # lvbf_test params
    # transform_params = [0.44, 0.68, 0.55, 0.55, -0.02]
    # 20250425_dataset params
    transform_params = [0.57, 0.35, 0.55, 0.55, 0]
    
    # Crop parameters
    crop_params = {
        'top_left': (0, 0),
        'crop_size': (512, 512),
        'interpolate_size': (608, 608),
        'angle': 0
    }
    
    # Run the processing with visualization enabled
    batch_process_images(
        hamamatsu_dir=hamamatsu_dir, 
        output_dir=output_dir,
        miniscope_dir=miniscope_dir,
        transform_params=transform_params, 
        crop_params=crop_params,
        show=False
    )