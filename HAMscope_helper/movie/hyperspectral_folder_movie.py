import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import re
import tifffile
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import medfilt

def numerical_sort(value):
    # Find all the digits in the filename and return them for sorting
    parts = re.compile(r'(\d+)').split(value)
    parts[1::2] = map(int, parts[1::2])  # Convert the extracted digits to integers for proper sorting
    return parts

def wavelength_to_rgb(wavelength):
    """Convert wavelength in nm to approximate RGB color."""
    # Simplified wavelength to RGB conversion
    if wavelength < 380:
        r, g, b = 0, 0, 0
    elif wavelength < 440:
        r = -(wavelength - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif wavelength < 490:
        r = 0.0
        g = (wavelength - 440) / (490 - 440)
        b = 1.0
    elif wavelength < 510:
        r = 0.0
        g = 1.0
        b = -(wavelength - 510) / (510 - 490)
    elif wavelength < 580:
        r = (wavelength - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif wavelength < 645:
        r = 1.0
        g = -(wavelength - 645) / (645 - 580)
        b = 0.0
    elif wavelength < 781:
        r = 1.0
        g = 0.0
        b = 0.0
    else:
        r, g, b = 0, 0, 0
    
    # Intensity correction for extreme wavelengths
    if wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif wavelength > 700:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
    else:
        factor = 1.0
    
    return (r * factor, g * factor, b * factor)

def add_time_and_scale_overlay(frame, frame_index, frame_time_seconds, scale_um_per_pixel, is_rgb=False):
    """Add time label and scale bar to the bottom left of the frame"""
    # Ensure frame is contiguous and in the right format for OpenCV
    frame = np.ascontiguousarray(frame)
    
    # Calculate time in minutes
    time_minutes = (frame_index * frame_time_seconds) / 60
    
    # Font settings - smaller for RGB images
    font = cv2.FONT_HERSHEY_SIMPLEX
    if is_rgb:
        font_scale = 0.5  # Smaller font for RGB images
        thickness = 1
        scale_bar_length_um = 200  # 200 um scale bar for RGB
        time_height = 80
    else:
        font_scale = 2  # Larger font for multi-channel grid layouts
        thickness = 3
        scale_bar_length_um = 800  # 800 um scale bar for multi-channel
        time_height = 130
    
    color = (255, 255, 255)  # White text
    
    # Add time label
    time_text = f"Time: {time_minutes:.1f} min"
    text_size = cv2.getTextSize(time_text, font, font_scale, thickness)[0]
    cv2.putText(frame, time_text, (50, frame.shape[0] - time_height), font, font_scale, color, thickness)
    
    # Add scale bar
    scale_bar_length_pixels = int(scale_bar_length_um / scale_um_per_pixel)
    
    # Draw scale bar
    start_x = 50
    start_y = frame.shape[0] - 20
    end_x = start_x + scale_bar_length_pixels
    end_y = start_y
    
    cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)
    cv2.line(frame, (start_x, start_y - 5), (start_x, start_y + 5), color, thickness)
    cv2.line(frame, (end_x, end_y - 5), (end_x, end_y + 5), color, thickness)
    
    # Add scale bar label
    scale_text = f"{scale_bar_length_um} um"
    cv2.putText(frame, scale_text, (start_x, start_y - 30), font, font_scale * 0.8, color, thickness)
    
    return frame

def generate_wavelength_labels(num_channels, start_wl=450, end_wl=700):
    """Generate wavelength labels equally spaced between start and end wavelengths."""
    wavelengths = np.linspace(start_wl, end_wl, num_channels)
    return [f'{wl:.0f} nm' for wl in wavelengths], wavelengths

def detect_file_type_and_channels(image_files):
    """Detect file type and number of channels in the image set."""
    # Check the first image to determine type and channels
    first_image = image_files[0]
    ext = os.path.splitext(first_image)[1].lower()
    
    if ext == '.tif' or ext == '.tiff':
        # Read the TIFF file to determine number of channels
        with tifffile.TiffFile(first_image) as tif:
            print(f'tif shape: {tif.pages[0].shape}')  # Debugging statement
            # Check if it's a multi-page TIFF (stack)
            if len(tif.pages) > 1:
                # Special case: if 6 channels, treat first 3 as RGB
                if len(tif.pages) == 6:
                    return 'tif_stack_rgb', 3
                else:
                    return 'tif_stack', len(tif.pages)
            else:
                # Single page TIFF - check shape
                image = tif.pages[0].asarray()
                if len(image.shape) == 3:
                    return 'tif', image.shape[2]  # RGB TIFF
                else:
                    return 'tif', 1  # Grayscale TIFF
    elif ext == '.png':
        # Read PNG to determine channels
        image = cv2.imread(first_image, cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 3:
            return 'png', image.shape[2]
        else:
            return 'png', 1
    else:
        # Default for other formats
        image = cv2.imread(first_image)
        if image is not None and len(image.shape) == 3:
            return ext[1:], image.shape[2]
        else:
            return ext[1:], 1

def process_single_channel_images(image_files, output_video_file, fps=30, frame_time_seconds=201, scale_um=940):
    """Process single channel images and create a grayscale video."""
    # Read the first image to get the size
    frame = cv2.imread(image_files[0], cv2.IMREAD_UNCHANGED)
    
    # Normalize if needed (e.g., if it's 16-bit)
    if frame.dtype != np.uint8:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    height, width = frame.shape
    
    # Calculate scale in um per pixel
    scale_um_per_pixel = scale_um / width
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Convert to color video to support overlays
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height), isColor=True)
    
    mean_intensities = []
    
    for frame_index, image_file in enumerate(image_files):
        frame = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        if frame is None:
            print(f"Warning: Could not read image file {image_file}")
            continue
        
        # Normalize if needed
        if frame.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Convert grayscale to color for overlay
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Add time and scale overlay
        frame = add_time_and_scale_overlay(frame, frame_index, frame_time_seconds, scale_um_per_pixel)
        
        mean_intensities.append(cv2.mean(frame)[0])
        video.write(frame)
        print(f"Added frame from {image_file}")
    
    video.release()
    print(f"Video file created: {output_video_file}")
    
    return mean_intensities

def process_three_channel_images(image_files, output_video_file, fps=30, frame_time_seconds=201, scale_um=940):
    """Process 3-channel images and create an RGB video."""
    # Read the first image to get the size
    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape
    
    # Calculate scale in um per pixel
    scale_um_per_pixel = scale_um / width
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height), isColor=True)
    
    # Store mean intensities for each channel
    mean_intensities = {
        'blue': [],
        'green': [],
        'red': []
    }
    
    for frame_index, image_file in enumerate(image_files):
        frame = cv2.imread(image_file)
        if frame is None:
            print(f"Warning: Could not read image file {image_file}")
            continue
        
        # Add time and scale overlay
        frame = add_time_and_scale_overlay(frame, frame_index, frame_time_seconds, scale_um_per_pixel)
        
        # Extract mean intensities for each channel (BGR order in OpenCV)
        means = cv2.mean(frame)
        mean_intensities['blue'].append(means[0])
        mean_intensities['green'].append(means[1])
        mean_intensities['red'].append(means[2])
        
        video.write(frame)
        print(f"Added frame from {image_file}")
    
    video.release()
    print(f"Video file created: {output_video_file}")
    
    return mean_intensities

def process_tiff_stack(image_files, output_video_file, fps=30, max_channels=None, channel_labels=None, frame_time_seconds=201, scale_um=940):
    """Process TIFF stacks with multiple channels and create a video with a grid layout."""
    # Read the first TIFF stack to get sizes and channel count
    with tifffile.TiffFile(image_files[0]) as tif:
        num_channels = len(tif.pages)
        
        # Limit channels if specified
        if max_channels is not None and max_channels > 0:
            num_channels = min(num_channels, max_channels)
            print(f"Limiting to {num_channels} channels as specified")
            
        first_page = tif.pages[0].asarray()
        
        # Handle different data types
        if first_page.dtype != np.uint8:
            first_page = cv2.normalize(first_page, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        height, width = first_page.shape
    
    # Calculate scale in um per pixel
    scale_um_per_pixel = scale_um / width
    
    # Define the layout for 30 channels (6x5 grid)
    if num_channels == 30:
        grid_rows, grid_cols = 5, 6
    elif num_channels == 6:
        grid_rows, grid_cols = 2, 3
    else:
        # Default to a square-ish grid
        grid_cols = int(np.ceil(np.sqrt(num_channels)))
        grid_rows = int(np.ceil(num_channels / grid_cols))
    
    # Calculate the size of the output video frame
    output_height = grid_rows * height
    output_width = grid_cols * width
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (output_width, output_height), isColor=True)
    
    # Generate wavelength labels - corrected to match TIFF stack order (700nm to 450nm)
    if channel_labels is None:
        # TIFF channels go from 700nm to 450nm (long to short wavelength)
        channel_labels, wavelengths = generate_wavelength_labels(num_channels, start_wl=700, end_wl=450)
    else:
        # If custom labels provided, generate corresponding wavelengths (700 to 450)
        wavelengths = np.linspace(700, 450, num_channels)
    
    # Use provided channel labels if available, otherwise use default naming
    default_labels = [f'Channel {i+1}' for i in range(num_channels)]
    
    # Use custom labels if provided and they're enough for all channels
    if channel_labels is not None and len(channel_labels) >= num_channels:
        used_labels = channel_labels[:num_channels]
    else:
        used_labels = default_labels
    
    # Store mean intensities for each channel
    mean_intensities = {used_labels[i]: [] for i in range(num_channels)}
    
    for frame_index, image_file in enumerate(image_files):
        # Create a blank canvas for the grid
        grid_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        with tifffile.TiffFile(image_file) as tif:
            for i, page in enumerate(tif.pages):
                if i >= num_channels:
                    break  # Stop at max_channels or all channels
                
                # Read the channel
                channel_image = page.asarray()
                
                # Normalize if needed
                if channel_image.dtype != np.uint8:
                    channel_image = cv2.normalize(channel_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # Calculate mean intensity
                mean_val = np.mean(channel_image)
                mean_intensities[used_labels[i]].append(mean_val)
                
                # Convert to 3-channel for display
                channel_colored = cv2.cvtColor(channel_image, cv2.COLOR_GRAY2BGR)
                
                # Add text label with smaller font for 30 channels
                label = used_labels[i]
                font_scale = 2 
                thickness = 3
                cv2.putText(channel_colored, label, (5, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                
                # Calculate position in the grid
                row = i // grid_cols
                col = i % grid_cols
                
                # Place in the grid
                y_start = row * height
                y_end = y_start + height
                x_start = col * width
                x_end = x_start + width
                
                grid_image[y_start:y_end, x_start:x_end] = channel_colored
        
        # Add time and scale overlay to the entire grid
        grid_image = add_time_and_scale_overlay(grid_image, frame_index, frame_time_seconds, scale_um_per_pixel)
        
        video.write(grid_image)
        print(f"Added frame from {image_file}")
    
    video.release()
    print(f"Video file created: {output_video_file}")
    
    return mean_intensities, wavelengths if 'wavelengths' in locals() else None

def process_tiff_stack_rgb(image_files, output_video_file, fps=30, frame_time_seconds=201, scale_um=940):
    """Process TIFF stacks with 6 channels, using first 3 as RGB."""
    
    # Define clipping constants for each channel
    RGB_CLIP_MIN_R = 0.02
    RGB_CLIP_MAX_R = 0.5
    RGB_CLIP_MIN_G = 0.02
    RGB_CLIP_MAX_G = 0.2
    RGB_CLIP_MIN_B = 0.02
    RGB_CLIP_MAX_B = 0.3
    
    # Read the first TIFF stack to get sizes
    with tifffile.TiffFile(image_files[0]) as tif:
        first_page = tif.pages[0].asarray()
        height, width = first_page.shape
    
    # Calculate scale in um per pixel
    scale_um_per_pixel = scale_um / width
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height), isColor=True)
    
    # Store mean intensities for RGB channels
    mean_intensities = {
        'red': [],    # Channel 0 -> Red
        'green': [],  # Channel 1 -> Green
        'blue': []    # Channel 2 -> Blue
    }
    
    for frame_index, image_file in enumerate(image_files):
        with tifffile.TiffFile(image_file) as tif:
            # Read first 3 channels
            channels = []
            for i in range(3):
                print(f'shape of tif.pages[{i}]: {tif.pages[i].shape}')  # Debugging statement
                channel_image = tif.pages[i].asarray()
                
                # Keep original data type for proper normalization
                channels.append(channel_image)
                
                # Calculate mean intensity on original data
                mean_val = np.mean(channel_image)
                if i == 0:
                    mean_intensities['red'].append(mean_val)
                elif i == 1:
                    mean_intensities['green'].append(mean_val)
                else:
                    mean_intensities['blue'].append(mean_val)
            
            # Stack channels to create RGB image
            rgb = np.stack(channels, axis=-1)
            
            # Clip each channel separately
            red_clipped = np.clip(rgb[..., 0], RGB_CLIP_MIN_R, RGB_CLIP_MAX_R)
            green_clipped = np.clip(rgb[..., 1], RGB_CLIP_MIN_G, RGB_CLIP_MAX_G)
            blue_clipped = np.clip(rgb[..., 2], RGB_CLIP_MIN_B, RGB_CLIP_MAX_B)
            
            # Normalize each channel separately to 0-255
            red_norm = ((red_clipped - RGB_CLIP_MIN_R) / (RGB_CLIP_MAX_R - RGB_CLIP_MIN_R) * 255).astype(np.uint8)
            green_norm = ((green_clipped - RGB_CLIP_MIN_G) / (RGB_CLIP_MAX_G - RGB_CLIP_MIN_G) * 255).astype(np.uint8)
            blue_norm = ((blue_clipped - RGB_CLIP_MIN_B) / (RGB_CLIP_MAX_B - RGB_CLIP_MIN_B) * 255).astype(np.uint8)
            
            # Stack the normalized channels back together (OpenCV uses BGR order)
            rgb_frame = cv2.merge([blue_norm, green_norm, red_norm])  # BGR order for OpenCV
        
        # Add time and scale overlay with RGB-specific settings
        rgb_frame = add_time_and_scale_overlay(rgb_frame, frame_index, frame_time_seconds, scale_um_per_pixel, is_rgb=True)
        
        video.write(rgb_frame)
        print(f"Added RGB frame from {image_file}")
    
    video.release()
    print(f"RGB video file created: {output_video_file}")
    
    return mean_intensities

def create_video_from_images(image_folder, output_video_file, fps=30, max_channels=None, channel_labels=None, frame_time_seconds=201, scale_um=940, max_frames=None):
    """Main function to create videos from image folders with automatic handling of different formats."""
    image_files = glob.glob(os.path.join(image_folder, '*'))
    
    # Skip if no images found
    if not image_files:
        print(f"No image files found in {image_folder}")
        return {}
    
    # Sort the files using the custom numerical_sort function
    image_files.sort(key=numerical_sort)
    
    # Limit frames if specified
    if max_frames is not None and max_frames > 0:
        image_files = image_files[:max_frames]
        print(f"Limited to {len(image_files)} frames as specified")
    
    print(f"Found {len(image_files)} image files")
    
    # Detect file type and number of channels
    file_type, num_channels = detect_file_type_and_channels(image_files)
    print(f"Detected {file_type} files with {num_channels} channels")
    
    # Process based on file type and number of channels
    if file_type == 'tif_stack_rgb':
        mean_intensities = process_tiff_stack_rgb(image_files, output_video_file, fps, frame_time_seconds, scale_um)
        return mean_intensities, None
    elif file_type == 'tif_stack':
        result = process_tiff_stack(image_files, output_video_file, fps, max_channels, channel_labels, frame_time_seconds, scale_um)
        if isinstance(result, tuple):
            return result  # Returns (mean_intensities, wavelengths)
        else:
            return result, None
    elif num_channels == 1:
        mean_intensities = process_single_channel_images(image_files, output_video_file, fps, frame_time_seconds, scale_um)
        return {'intensity': mean_intensities}, None
    elif num_channels == 3:
        return process_three_channel_images(image_files, output_video_file, fps, frame_time_seconds, scale_um), None
    else:
        print(f"Unsupported number of channels: {num_channels}")
        return {}, None

def plot_intensity_changes_colored(mean_intensities_list, folder_names, wavelengths_list=None, output_file=None, frame_time_seconds=201):
    """Plot intensity changes over time with wavelength-representative colors, optimized for paper publication."""
    # Create figure with larger size for better readability
    plt.figure(figsize=(12, 8))
    
    # Set larger font sizes for paper publication
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10
    })
    
    plt.title('Mean Intensity by Wavelength', fontsize=16, fontweight='bold')
    plt.xlabel('Time (minutes)', fontsize=14, fontweight='bold')
    plt.ylabel('Mean Intensity', fontsize=14, fontweight='bold')
    
    for i, (folder_name, mean_intensities) in enumerate(zip(folder_names, mean_intensities_list)):
        folder_base = os.path.basename(folder_name)
        wavelengths = wavelengths_list[i] if wavelengths_list and i < len(wavelengths_list) else None
        
        # Check if it's a dictionary with multiple channels
        if isinstance(mean_intensities, dict):
            for j, (channel_name, intensity_values) in enumerate(mean_intensities.items()):
                # Remove outliers using IQR method
                if len(intensity_values) > 4:  # Need at least 4 points for IQR
                    intensity_array = np.array(intensity_values)
                    Q1 = np.percentile(intensity_array, 30)
                    Q3 = np.percentile(intensity_array, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Create mask for non-outlier values
                    mask = (intensity_array >= lower_bound) & (intensity_array <= upper_bound)
                    
                    # Apply median filter to smooth remaining outliers
                    if len(intensity_array) > 3:
                        intensity_values = medfilt(intensity_array, kernel_size=min(5, len(intensity_array)//2*2+1))
                    
                    # Apply mask to remove extreme outliers
                    intensity_values = intensity_array[mask] if np.sum(mask) > len(intensity_array) * 0.5 else intensity_array
                    time_indices = np.where(mask)[0] if np.sum(mask) > len(intensity_array) * 0.5 else np.arange(len(intensity_array))
                else:
                    time_indices = np.arange(len(intensity_values))
                
                # Extract wavelength from channel name or use wavelengths array
                if wavelengths is not None and j < len(wavelengths):
                    wl = wavelengths[j]
                elif 'nm' in channel_name:
                    wl = float(channel_name.split()[0])
                else:
                    # Default wavelength assignment - corrected to match TIFF order
                    # For TIFF stacks: channel 0 = 700nm, last channel = 450nm
                    wl = 700 - (j / (len(mean_intensities) - 1)) * 250  # Goes from 700 to 450
                
                # Get color for this wavelength
                rgb_color = wavelength_to_rgb(wl)
                
                # Convert time scale to minutes using the frame_time parameter
                time_scale_minutes = time_indices * (frame_time_seconds / 60)
                
                # Clean up legend label - remove "ham_aligned" and folder path info
                clean_label = channel_name.replace('ham_aligned', '').strip(' -_')
                if clean_label.startswith('Channel'):
                    clean_label = f"{wl:.0f} nm"
                
                plt.plot(time_scale_minutes, intensity_values, 
                         color=rgb_color,
                         linewidth=2.5,  # Increased line width for better visibility
                         alpha=0.9,
                         label=clean_label)
        else:
            # Handle legacy single-channel data with outlier removal
            intensity_array = np.array(mean_intensities)
            if len(intensity_array) > 4:
                Q1 = np.percentile(intensity_array, 25)
                Q3 = np.percentile(intensity_array, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                mask = (intensity_array >= lower_bound) & (intensity_array <= upper_bound)
                intensity_values = intensity_array[mask] if np.sum(mask) > len(intensity_array) * 0.5 else intensity_array
                time_indices = np.where(mask)[0] if np.sum(mask) > len(intensity_array) * 0.5 else np.arange(len(intensity_array))
            else:
                intensity_values = intensity_array
                time_indices = np.arange(len(intensity_array))
                
            time_scale_minutes = time_indices * (frame_time_seconds / 60)
            
            # Clean folder name for legend
            clean_folder = folder_base.replace('ham_aligned', '').strip('_')
            if not clean_folder:
                clean_folder = "Sample"
                
            plt.plot(time_scale_minutes, intensity_values, 
                     linewidth=3,
                     label=clean_folder)
    
    # Improve legend positioning and styling
    legend = plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, 
                       frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Add grid with better styling
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Improve tick formatting
    plt.tick_params(axis='both', which='major', labelsize=12, width=1.5)
    plt.tick_params(axis='both', which='minor', labelsize=10, width=1)
    
    # Add minor ticks
    plt.minorticks_on()
    
    # Tight layout with padding
    plt.tight_layout(pad=2.0)
    
    if output_file:
        base, ext = os.path.splitext(output_file)
        colored_output = f"{base}_colored_publication{ext}"
        plt.savefig(colored_output, dpi=600, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Publication-ready colored plot saved: {colored_output}")
    plt.show()

def plot_intensity_changes_3d(mean_intensities_list, folder_names, wavelengths_list=None, output_file=None, frame_time_seconds=201):
    """Plot intensity changes as 3D surface: channel x time x intensity."""
    fig = plt.figure(figsize=(16, 12))
    print(f'Creating 3D surface plot for {len(mean_intensities_list)} folders')
    
    for i, (folder_name, mean_intensities) in enumerate(zip(folder_names, mean_intensities_list)):
        folder_base = os.path.basename(folder_name)
        wavelengths = wavelengths_list[i] if wavelengths_list and i < len(wavelengths_list) else None
        
        if isinstance(mean_intensities, dict):
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            
            # Prepare data
            channels = list(mean_intensities.keys())
            
            # Find the minimum number of time points across all channels
            time_points = min(len(mean_intensities[ch]) for ch in channels)
            print(f"Using {time_points} time points for 3D plot")
            
            # Convert time scale to minutes using the frame_time parameter
            time_minutes = np.arange(time_points) * (frame_time_seconds / 60)
            
            # Create meshgrid
            T, C = np.meshgrid(time_minutes, np.arange(len(channels)))
            
            # Prepare intensity data - truncate all channels to the same length
            Z = np.array([mean_intensities[ch][:time_points] for ch in channels])
            
            # Create surface plot
            surf = ax.plot_surface(T, C, Z, cmap='viridis', alpha=0.8)
            
            # Labels and title
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Channel')
            ax.set_zlabel('Mean Intensity')
            ax.set_title(f'{folder_base} - 3D Intensity Surface')
            
            # Set channel labels
            ax.set_yticks(np.arange(len(channels)))
            if len(channels) > 10:
                # Show every 5th channel label for readability
                labels = [channels[j] if j % 5 == 0 else '' for j in range(len(channels))]
                ax.set_yticklabels(labels, fontsize=8)
            else:
                ax.set_yticklabels(channels, fontsize=8)
            
            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    if output_file:
        base, ext = os.path.splitext(output_file)
        surface_output = f"{base}_3d_surface{ext}"
        plt.savefig(surface_output, dpi=300, bbox_inches='tight')
        print(f"3D surface plot saved: {surface_output}")
    plt.show()

def plot_intensity_changes(mean_intensities_list, folder_names, wavelengths_list=None, output_file=None, frame_time_seconds=201):
    """Plot intensity changes over time for each channel (original version)."""
    plt.figure(figsize=(12, 8))
    plt.title('Longitudinal Mean Intensity by Channel')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Mean Intensity')
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    for i, (folder_name, mean_intensities) in enumerate(zip(folder_names, mean_intensities_list)):
        folder_base = os.path.basename(folder_name)
        
        # Check if it's a dictionary with multiple channels
        if isinstance(mean_intensities, dict):
            for j, (channel_name, intensity_values) in enumerate(mean_intensities.items()):
                color_idx = j % len(colors)
                marker_idx = i % len(markers)
                label = f"{folder_base} - {channel_name}"
                
                # Convert time scale to minutes using the frame_time parameter
                time_scale_minutes = np.arange(0, len(intensity_values)) * (frame_time_seconds / 60)
                plt.plot(time_scale_minutes, intensity_values, 
                         color=colors[color_idx], 
                         marker=markers[marker_idx], 
                         markersize=2 if len(mean_intensities) > 10 else 4,
                         linestyle='-', 
                         label=label,
                         alpha=0.7)
        else:
            # Handle legacy single-channel data
            time_scale_minutes = np.arange(0, len(mean_intensities)) * (frame_time_seconds / 60)
            plt.plot(time_scale_minutes, mean_intensities, 
                     color=colors[i % len(colors)], 
                     marker=markers[i % len(markers)], 
                     markersize=4,
                     linestyle='-', 
                     label=folder_base)
    
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Standard plot saved: {output_file}")
    plt.show()

def main():
    image_folders = [

        '/media/al/Extreme SSD/20250519_timelapse/ham_aligned_every_7th'
    ]

    base_output_path = '/media/al/Extreme SSD/20250519_timelapse'
    
    fps = 30
    max_channels = 30  # Set to 30 for hyperspectral images
    max_frames = 500  # Limit video to first 100 frames (adjust as needed)
    #frame_time = 3*60 + 21 # 3 minutes and 21 seconds for 20250522, and 202050521
    frame_time = 2*60 + 21 # 2 minutes and 21 seconds for 20250519
    #frame_time = 4*60 + 21 # 2 minutes and 21 seconds for 20250516
    frame_time = 0.2 # 5FPS for 20250410_dasmeet timelapse

    scale = 940 # 940 um across the image
    
    # Define custom channel labels for 30 channels (700-450 nm, correctly ordered)
    channel_labels, wavelengths = generate_wavelength_labels(30, start_wl=700, end_wl=450)
    
    # Create the output directory if it doesn't exist
    os.makedirs(base_output_path, exist_ok=True)
    
    # Process each folder
    mean_intensities_list = []
    wavelengths_list = []
    processed_folder_names = []
    
    for i, image_folder in enumerate(image_folders):
        folder_name = os.path.basename(image_folder)
        output_video_file = os.path.join(base_output_path, f'{folder_name}_processed.mp4')
        
        print(f"Processing folder {i+1}/{len(image_folders)}: {folder_name}")
        result = create_video_from_images(image_folder, output_video_file, fps, max_channels, channel_labels, frame_time, scale, max_frames)
        
        if isinstance(result, tuple):
            mean_intensities, wl = result
        else:
            mean_intensities, wl = result, wavelengths
        
        if mean_intensities:  # Only add if not empty
            mean_intensities_list.append(mean_intensities)
            wavelengths_list.append(wl)
            processed_folder_names.append(image_folder)
            
            # Print some statistics
            if isinstance(mean_intensities, dict):
                for channel, values in mean_intensities.items():
                    if values:  # Check if not empty
                        print(f"Channel {channel} - Max: {max(values):.2f}, Min: {min(values):.2f}")
    
    # Plot results
    if mean_intensities_list:
        plot_output_file = os.path.join(base_output_path, f'{folder_name}_intensity_plot.png')
        
        # Create all three types of plots with time scaling
        plot_intensity_changes_colored(mean_intensities_list, processed_folder_names, wavelengths_list, plot_output_file, frame_time)
        plot_intensity_changes_3d(mean_intensities_list, processed_folder_names, wavelengths_list, plot_output_file, frame_time)
    else:
        print("No data to plot")

if __name__ == "__main__":
    main()
