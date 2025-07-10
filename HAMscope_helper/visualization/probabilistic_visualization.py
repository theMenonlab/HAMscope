import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy import stats
from scipy.interpolate import interp1d
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.cm as cm

# Hardcoded paths to the TIF stacks (each is a complete file path)
ensemble_tifs = [
    "/media/al/Extreme SSD/20250425_results/results/analyzed_20250613/20250425_0gan_single_hs_1/test_latest/images/hs_gen_1.tif",
    "/media/al/Extreme SSD/20250425_results/results/analyzed_20250613/20250425_0gan_single_hs_2/test_latest/images/hs_gen_1.tif",
    "/media/al/Extreme SSD/20250425_results/results/analyzed_20250613/20250425_0gan_single_hs_3/test_latest/images/hs_gen_1.tif",
    "/media/al/Extreme SSD/20250425_results/results/analyzed_20250613/20250425_0gan_single_hs_4/test_latest/images/hs_gen_1.tif",
    "/media/al/Extreme SSD/20250425_results/results/analyzed_20250613/20250425_0gan_single_hs_5/test_latest/images/hs_gen_1.tif"
]

# Path to ground truth TIF stack
ground_truth_tif = '/media/al/Extreme SSD/20250425_results/results/new_layer_norm/results/20250425_0gan_single_hs_1/test_latest/images/hs_raw_1.tif'

calibration = "/media/al/Extreme SSD/20250425_spectra_2/sensor_calibrated_transmission.txt"

output_dir = '/home/al/Documents/hyperspectral_miniscope_paper/probabilistic_fig/new_layer_norm'

# Target pixel coordinates
pixel_x, pixel_y = 256, 256

# Target channel (spectral band)
target_channel = 15

# Colors for each ensemble member and ground truth
colors = ['red', 'blue', 'green', 'purple', 'orange', 'black']
labels = ['Ensemble 1', 'Ensemble 2', 'Ensemble 3', 'Ensemble 4', 'Ensemble 5', 'Ground Truth']

def load_calibration_data(calibration_file):
    """Load sensor calibration data from the calibration file"""
    try:
        # Read calibration data
        calibration_data = pd.read_csv(calibration_file, comment='#', 
                                     names=['Spectrum_Number', 'Filename', 'Peak_Wavelength', 
                                           'Max_Normalized_Transmission', 'Sensor_Response', 'Combined_Response'])
        
        # Create interpolation function for combined response
        calibration_interp = interp1d(calibration_data['Peak_Wavelength'], 
                                    calibration_data['Combined_Response'], 
                                    bounds_error=False, fill_value=1.0, kind='linear')
        
        return calibration_interp, calibration_data
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return None, None

def apply_sensor_calibration(wavelengths, pixel_values, std_values, calibration_interp):
    """Apply sensor calibration to hyperspectral data"""
    if calibration_interp is None:
        return pixel_values, std_values
    
    # Get calibration factors for each wavelength (these represent efficiency/transmission)
    calibration_factors = calibration_interp(wavelengths)
    
    # Avoid division by zero or very small values
    calibration_factors = np.maximum(calibration_factors, 0.01)  # Minimum 1% efficiency
    
    # Apply calibration to pixel values (DIVIDE by efficiency to boost low-efficiency wavelengths)
    calibrated_pixel_values = pixel_values / calibration_factors
    
    # Apply calibration to standard deviations if they exist
    calibrated_std_values = None
    if std_values is not None:
        calibrated_std_values = std_values / calibration_factors
    
    return calibrated_pixel_values, calibrated_std_values

# Function to extract data for a single TIF
def extract_hyperspectral_data(tif_path, pixel_x, pixel_y, wavelengths, prob=True):
    try:
        # Read the entire TIF stack
        with tifffile.TiffFile(tif_path) as tif:
            stack = tif.asarray()
            
            if len(stack.shape) == 3:  # (bands, height, width)
                band_count = stack.shape[0]
                pixel_values = stack[:, pixel_y, pixel_x]
            else:
                raise ValueError(f"Unexpected TIF shape: {stack.shape}")
            
        if prob:
            band_count = band_count // 2
            std_values = pixel_values[band_count:]
            pixel_values = pixel_values[:band_count]
        else:
            std_values = None
        
        # Use the provided wavelengths instead of generating them
        if len(wavelengths) != band_count:
            print(f"Warning: Wavelength count ({len(wavelengths)}) doesn't match band count ({band_count})")
            # Trim or pad as needed
            if len(wavelengths) > band_count:
                wavelengths = wavelengths[:band_count]
            else:
                # If we have fewer wavelengths than bands, just use the available ones
                pixel_values = pixel_values[:len(wavelengths)]
                if std_values is not None:
                    std_values = std_values[:len(wavelengths)]
        
        return wavelengths, np.array(pixel_values), np.array(std_values) if std_values is not None else None
    
    except Exception as e:
        print(f"Error reading {tif_path}: {e}")
        return None, None, None

# Load calibration data
print("Loading sensor calibration data...")
calibration_interp, calibration_data = load_calibration_data(calibration)

if calibration_interp is not None:
    print(f"Calibration data loaded successfully with {len(calibration_data)} points")
    # Extract wavelengths from calibration data (sorted from highest to lowest to match TIF order)
    calibration_wavelengths = sorted(calibration_data['Peak_Wavelength'].values, reverse=True)
    print(f"Wavelength range: {min(calibration_wavelengths):.1f} - {max(calibration_wavelengths):.1f} nm")
else:
    print("Warning: No calibration data loaded. Proceeding without calibration.")
    calibration_wavelengths = np.linspace(700, 450, 30)  # Fallback

#------------------------
# 1. SPECTROGRAM PLOT (CALIBRATED)
#------------------------
plt.figure(figsize=(8, 5))  # Slightly smaller, better aspect ratio for paper

# Store data for both plots
all_wavelengths = []
all_means = []
all_stds = []
all_means_uncalibrated = []  # Store uncalibrated data
all_stds_uncalibrated = []   # Store uncalibrated data
ground_truth_wavelengths = None
ground_truth_values = None
ground_truth_values_uncalibrated = None

# Process each ensemble TIF for the spectrogram
for idx, tif_path in enumerate(ensemble_tifs):
    wavelengths, pixel_values, std_values = extract_hyperspectral_data(tif_path, pixel_x, pixel_y, calibration_wavelengths, prob=True)
    
    if wavelengths is None or len(pixel_values) == 0:
        print(f"Skipping {tif_path} - no valid data")
        continue
    
    # Store uncalibrated data
    all_wavelengths.append(wavelengths)
    all_means_uncalibrated.append(pixel_values)
    all_stds_uncalibrated.append(std_values)
    
    # Apply sensor calibration
    calibrated_pixel_values, calibrated_std_values = apply_sensor_calibration(
        wavelengths, pixel_values, std_values, calibration_interp)
    
    # Store calibrated data for later use in Laplacian plot
    all_means.append(calibrated_pixel_values)
    all_stds.append(calibrated_std_values)
    
    # Plot the calibrated spectrogram for this ensemble member
    plt.plot(wavelengths, calibrated_pixel_values, color=colors[idx], 
             label=labels[idx], linewidth=2.5)  # Increased linewidth

# Process ground truth TIF for the spectrogram
if os.path.exists(ground_truth_tif):
    ground_truth_wavelengths, ground_truth_values_raw, _ = extract_hyperspectral_data(
        ground_truth_tif, pixel_x, pixel_y, calibration_wavelengths, prob=False)
    
    if ground_truth_wavelengths is not None and len(ground_truth_values_raw) > 0:
        # Store uncalibrated ground truth
        ground_truth_values_uncalibrated = ground_truth_values_raw
        
        # Apply sensor calibration to ground truth
        calibrated_ground_truth, _ = apply_sensor_calibration(
            ground_truth_wavelengths, ground_truth_values_raw, None, calibration_interp)
        ground_truth_values = calibrated_ground_truth
        
        plt.plot(ground_truth_wavelengths, ground_truth_values, color=colors[5], 
                 label=labels[5], linewidth=3.5, linestyle='--')  # Increased linewidth
    else:
        print("No valid ground truth data found")
else:
    print(f"Ground truth file not found: {ground_truth_tif}")

# Check if we have any valid data before proceeding
if len(all_wavelengths) == 0:
    print("Error: No valid ensemble data found. Check TIF files and paths.")
    exit()

# Add plot formatting for calibrated spectrogram
plt.xlabel('Wavelength (nm)', fontsize=16)
plt.ylabel('Fluorescence', fontsize=16)  # Updated label
# Use actual wavelength range from calibration data
plt.xlim(min(calibration_wavelengths), max(calibration_wavelengths))
plt.xticks(fontsize=14)  # Larger font for x-ticks
plt.yticks(fontsize=14)  # Larger font for y-ticks
plt.grid(True, linestyle='--', alpha=0.5)  # Less prominent grid

# Create legend with larger font size
#legend = plt.legend(loc='upper left', fontsize=12, framealpha=0.9)  # Increased font, more opaque background

plt.tight_layout()

# Save the calibrated spectrogram figure to output directory
spectrogram_output_path = os.path.join(output_dir, 'ensemble_spectrogram_calibrated.svg')
plt.savefig(spectrogram_output_path, format='svg', bbox_inches='tight', dpi=300)
plt.show()
print(f"Saved calibrated ensemble spectrogram: {spectrogram_output_path}")

#------------------------
# 1B. SPECTROGRAM PLOT (UNCALIBRATED)
#------------------------
plt.figure(figsize=(8, 5))  # Same size as calibrated version

# Plot uncalibrated spectrogram for each ensemble member
for idx in range(len(all_means_uncalibrated)):
    wavelengths = all_wavelengths[idx]
    pixel_values = all_means_uncalibrated[idx]
    
    # Plot the uncalibrated spectrogram for this ensemble member
    plt.plot(wavelengths, pixel_values, color=colors[idx], 
             label=labels[idx], linewidth=2.5)

# Plot uncalibrated ground truth
if ground_truth_values_uncalibrated is not None and ground_truth_wavelengths is not None:
    plt.plot(ground_truth_wavelengths, ground_truth_values_uncalibrated, color=colors[5], 
             label=labels[5], linewidth=3.5, linestyle='--')

# Add plot formatting for uncalibrated spectrogram
plt.xlabel('Wavelength (nm)', fontsize=16)
plt.ylabel('Fluorescence', fontsize=16)  # Different label for uncalibrated
# Use actual wavelength range from calibration data
plt.xlim(min(calibration_wavelengths), max(calibration_wavelengths))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)

# Create legend
#legend = plt.legend(loc='upper center', fontsize=12, framealpha=0.9, bbox_to_anchor=(0.65, 1.0))

plt.tight_layout()

# Save the uncalibrated spectrogram figure
spectrogram_uncalibrated_path = os.path.join(output_dir, 'ensemble_spectrogram_uncalibrated.svg')
plt.savefig(spectrogram_uncalibrated_path, format='svg', bbox_inches='tight', dpi=300)
plt.show()
print(f"Saved uncalibrated ensemble spectrogram: {spectrogram_uncalibrated_path}")

#-----------------------------------
# 2. LAPLACIAN DISTRIBUTION PLOT
#-----------------------------------
if len(all_means) > 0 and target_channel < len(all_means[0]):
    plt.figure(figsize=(7, 6))  # Adjusted size for better paper fit

    # Extract data for target channel
    ensemble_means = [means[target_channel] for means in all_means if len(means) > target_channel]
    ensemble_stds = [stds[target_channel] for stds in all_stds if stds is not None and len(stds) > target_channel]
    ground_truth_value = ground_truth_values[target_channel] if ground_truth_values is not None and len(ground_truth_values) > target_channel else None

    # Check if we have valid data
    if len(ensemble_means) == 0 or len(ensemble_stds) == 0:
        print(f"Error: No valid data for target channel {target_channel}")
    else:
        # Value range for plotting distributions
        min_val = np.min([np.min(np.array(ensemble_means) - 3*np.array(ensemble_stds)), 
                          ground_truth_value if ground_truth_value is not None else np.inf])
        max_val = np.max([np.max(np.array(ensemble_means) + 3*np.array(ensemble_stds)), 
                          ground_truth_value if ground_truth_value is not None else -np.inf])
        value_range = np.linspace(min_val, max_val, 1000)

        # Store all individual PDFs for later averaging
        all_pdfs = []

        # Plot individual Laplacian distributions for each ensemble model
        for i in range(len(ensemble_means)):
            # For Laplace distribution, scale parameter b = std/sqrt(2)
            scale = ensemble_stds[i] / np.sqrt(2)
            # Calculate Laplacian PDF
            laplace_pdf = stats.laplace.pdf(value_range, loc=ensemble_means[i], scale=scale)
            all_pdfs.append(laplace_pdf)
            
            # Plot vertical distribution (Laplacian PDF on x-axis, values on y-axis)
            # Make individual distributions semi-transparent
            plt.plot(laplace_pdf, value_range, color=colors[i], alpha=0.6, linewidth=2.5,
                     label=f"{labels[i]}")

        # Calculate and plot the combined distribution as simple average of all PDFs
        if len(all_pdfs) > 0:
            combined_pdf = np.mean(all_pdfs, axis=0)

            # Create rainbow-colored line for combined distribution
            from matplotlib.colors import LinearSegmentedColormap
            rainbow_cmap = LinearSegmentedColormap.from_list("rainbow", ['red', 'orange', 'yellow', 'green', 'blue', 'purple'])

            # Create points and segments for vertical line collection
            points = np.array([combined_pdf, value_range]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create and add the line collection
            from matplotlib.collections import LineCollection
            lc = LineCollection(segments, cmap=rainbow_cmap, linewidth=10)  # Increased linewidth for visibility
            lc.set_array(np.linspace(0, 1, len(combined_pdf)-1))
            plt.gca().add_collection(lc)

        # Add horizontal line for ground truth
        if ground_truth_value is not None:
            plt.axhline(y=ground_truth_value, color='black', linewidth=3, linestyle='-')  # Thicker line

        # Format the Laplacian plot
        plt.ylabel(f'Value (Channel {target_channel})', fontsize=16)  # Updated label
        plt.xlabel('Probability Density', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(fontsize=14)  # Larger font for x-ticks
        plt.yticks(fontsize=14)  # Larger font for y-ticks

        # Limit the x-axis to focus on the meaningful part of distributions
        if len(all_pdfs) > 0:
            max_pdf = np.max(all_pdfs) * 1.1  # Add 10% margin
            plt.xlim(0, max_pdf)

        # Create legend entries
        from matplotlib.lines import Line2D

        # Get existing handles and labels
        handles, labels_list = plt.gca().get_legend_handles_labels()

        # Add ground truth handle and label if available
        if ground_truth_value is not None:
            handles.append(Line2D([0], [0], color='black', linewidth=3, linestyle='-'))
            labels_list.append(f"Ground Truth ({ground_truth_value:.2f})")

        # Add combined distribution label (placeholder for rainbow line)
        if len(all_pdfs) > 0:
            # Create a placeholder handle for the rainbow legend entry
            rainbow_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
            class RainbowLine(Line2D):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    
                def draw(self, renderer):
                    # This will be handled by the manual drawing below
                    pass

            # Add a placeholder rainbow line to the handles and its label
            rainbow_handle = RainbowLine([0], [0], linewidth=0)
            handles.append(rainbow_handle)
            labels_list.append("Combined")

        # Create legend with all handles and labels
        LEGEND = False
        if LEGEND:
            legend = plt.legend(handles=handles, labels=labels_list, loc='lower right', fontsize=12, framealpha=0.9, ncol=2)

            # Now add colored line segments directly on top of the legend (from old script)
            if len(all_pdfs) > 0:
                fig = plt.gcf()
                legend_box = legend.get_frame()
                legend_entries = legend.get_texts()
                combined_entry = legend_entries[-1]  # Last entry is the combined one

                # Get the position of the last legend entry's text
                fig.canvas.draw()  # Force a draw to get text position
                bbox = combined_entry.get_window_extent()
                text_pos = bbox.transformed(fig.transFigure.inverted())

                # Calculate position for the rainbow line
                x_start = text_pos.x0 + 0.015  # Adjusted start position
                x_end = x_start + 0.045  # Wider for better visibility
                y_pos = text_pos.y0 + 0.024  # Vertical position

                # Add colored line segments for the rainbow
                rainbow_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
                segment_width = (x_end - x_start) / len(rainbow_colors) * 1.2  # Make segments 20% wider for overlap
                for i, color in enumerate(rainbow_colors):
                    start_x = x_start + i * (x_end - x_start) / len(rainbow_colors) - segment_width * 0.1  # Shift start slightly back
                    end_x = start_x + segment_width  # Use the wider segment width
                    line = plt.Line2D([start_x, end_x], [y_pos, y_pos], 
                                    transform=fig.transFigure, figure=fig,
                                    color=color, linewidth=9)  # Thicker line
                    fig.lines.append(line)

        plt.tight_layout()

        # Save the Laplacian figure to output directory
        laplacian_output_path = os.path.join(output_dir, 'laplacian_distribution_calibrated.svg')
        plt.savefig(laplacian_output_path, format='svg', bbox_inches='tight', dpi=300)
        plt.show()
        print(f"Saved Laplacian distribution: {laplacian_output_path}")

    # Calculate wavelength for the target channel
    if len(all_wavelengths) > 0 and target_channel < len(all_wavelengths[0]):
        target_wavelength = all_wavelengths[0][target_channel]
        print(f"Visualizations complete for pixel ({pixel_x}, {pixel_y})")
        print(f"Target channel: {target_channel} (wavelength: {target_wavelength:.2f} nm)")

        # Print calibration info
        if calibration_interp is not None:
            calibration_factor = calibration_interp(target_wavelength)
            print(f"Calibration factor at {target_wavelength:.2f} nm: {calibration_factor:.4f}")
    else:
        print(f"Target channel {target_channel} is out of range")
else:
    print(f"Cannot create Laplacian plot: target_channel {target_channel} is out of range or no valid data")

#------------------------
# 3. SAVE ADDITIONAL IMAGES
#------------------------
print("Generating and saving additional images...")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each ensemble TIF to generate panchromatic images
for idx, tif_path in enumerate(ensemble_tifs):
    try:
        # Read the entire TIF stack
        with tifffile.TiffFile(tif_path) as tif:
            stack = tif.asarray()
            
            if len(stack.shape) == 3:  # (bands, height, width)
                total_bands = stack.shape[0]
                band_count = total_bands // 2  # First half are predictions, second half are std devs
                
                # Extract mean and std channels
                mean_channels = stack[:band_count, :, :]  # First 30 channels
                std_channels = stack[band_count:, :, :]   # Last 30 channels
                
                # Calculate panchromatic image (average across all 30 predicted channels)
                panchromatic = np.mean(mean_channels, axis=0)
                
                # Save panchromatic as colorized PNG only (no markers)
                clip_min = 0.03
                clip_max = 0.4
                clipped_pan = np.clip(panchromatic, clip_min, clip_max)
                normalized_pan = ((clipped_pan - clip_min) / (clip_max - clip_min) * 255).astype(np.uint8)
                colored_pan = cm.viridis(normalized_pan)
                colored_pan_rgb = (colored_pan[:, :, :3] * 255).astype(np.uint8)
                
                # Save panchromatic PNG (no markers added)
                from PIL import Image
                pan_png_path = os.path.join(output_dir, f'panchromatic_ensemble_{idx+1}.png')
                Image.fromarray(colored_pan_rgb).save(pan_png_path)
                print(f"Saved panchromatic PNG: {pan_png_path}")
                
                # For ensemble model 1, also save channel 23 mean and std images
                if idx == 0:  # First ensemble model
                    channel_23_idx = 22  # Channel 23 (0-indexed as 22)
                    
                    if channel_23_idx < band_count:
                        # Process mean image
                        mean_ch23 = mean_channels[channel_23_idx, :, :]
                        
                        # Clip and normalize mean
                        clipped_mean = np.clip(mean_ch23, clip_min, clip_max)
                        normalized_mean = ((clipped_mean - clip_min) / (clip_max - clip_min) * 255).astype(np.uint8)
                        colored_mean = cm.viridis(normalized_mean)
                        colored_mean_rgb = (colored_mean[:, :, :3] * 255).astype(np.uint8)
                        
                        # Process std image
                        std_clip = 0.07  # Different clip for std
                        std_ch23 = std_channels[channel_23_idx, :, :]
                        clipped_std = np.clip(std_ch23, 0, std_clip)  # Different range for std
                        normalized_std = ((clipped_std - 0) / (std_clip/2) * 255).astype(np.uint8)
                        colored_std = cm.viridis(normalized_std)
                        colored_std_rgb = (colored_std[:, :, :3] * 255).astype(np.uint8)
                        
                        # Add markers only to mean and std images
                        marker_points = [(256, 256), (150, 200), (300, 300)]
                        marker_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
                        marker_size = 10
                        
                        for (x, y), color in zip(marker_points, marker_colors):
                            for img in [colored_mean_rgb, colored_std_rgb]:
                                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                                    for dx in range(-marker_size, marker_size + 1):
                                        for dy in range(-marker_size, marker_size + 1):
                                            nx, ny = x + dx, y + dy
                                            if 0 <= ny < img.shape[0] and 0 <= nx < img.shape[1]:
                                                if abs(dx) <= 1 or abs(dy) <= 1:
                                                    img[ny, nx] = color
                        
                        # Save PNG images only
                        mean_png_path = os.path.join(output_dir, 'channel_23_mean_marked.png')
                        std_png_path = os.path.join(output_dir, 'channel_23_std_marked.png')
                        Image.fromarray(colored_mean_rgb).save(mean_png_path)
                        Image.fromarray(colored_std_rgb).save(std_png_path)
                        print(f"Saved marked mean PNG: {mean_png_path}")
                        print(f"Saved marked std PNG: {std_png_path}")
                    else:
                        print(f"Warning: Channel 23 not available (only {band_count} channels found)")
                        
    except Exception as e:
        print(f"Error processing {tif_path}: {e}")

# Save ground truth channel 23 image
if os.path.exists(ground_truth_tif):
    try:
        with tifffile.TiffFile(ground_truth_tif) as tif:
            gt_stack = tif.asarray()
            
            if len(gt_stack.shape) == 3:
                channel_23_idx = 22  # Channel 23 (0-indexed as 22)
                
                if channel_23_idx < gt_stack.shape[0]:
                    # Save colorized PNG version with markers only
                    gt_ch23 = gt_stack[channel_23_idx, :, :]
                    clip_min = 0.03
                    clip_max = 0.4
                    
                    # Clip and normalize ground truth
                    clipped_gt = np.clip(gt_ch23, clip_min, clip_max)
                    normalized_gt = ((clipped_gt - clip_min) / (clip_max - clip_min) * 255).astype(np.uint8)
                    
                    # Apply viridis colormap
                    colored_gt = cm.viridis(normalized_gt)
                    colored_gt_rgb = (colored_gt[:, :, :3] * 255).astype(np.uint8)

                    
                    for (x, y), color in zip(marker_points, marker_colors):
                        if 0 <= y < colored_gt_rgb.shape[0] and 0 <= x < colored_gt_rgb.shape[1]:
                            # Draw a cross marker
                            for dx in range(-marker_size, marker_size + 1):
                                for dy in range(-marker_size, marker_size + 1):
                                    nx, ny = x + dx, y + dy
                                    if 0 <= ny < colored_gt_rgb.shape[0] and 0 <= nx < colored_gt_rgb.shape[1]:
                                        if abs(dx) <= 1 or abs(dy) <= 1:  # Cross shape
                                            colored_gt_rgb[ny, nx] = color
                    
                    # Save as PNG only
                    from PIL import Image
                    gt_png_path = os.path.join(output_dir, 'channel_23_ground_truth_marked.png')
                    Image.fromarray(colored_gt_rgb).save(gt_png_path)
                    print(f"Saved marked ground truth PNG: {gt_png_path}")
                else:
                    print(f"Warning: Channel 23 not available in ground truth (only {gt_stack.shape[0]} channels found)")
                    
    except Exception as e:
        print(f"Error processing ground truth file: {e}")
else:
    print("Ground truth file not found - skipping ground truth channel 23 save")

#------------------------
# 4. SPECTROGRAM FOR THREE MARKED POINTS (BOTH CALIBRATED AND UNCALIBRATED)
#------------------------
print("Generating spectrograms for marked points...")

# Define the three points and their colors
point_colors = ['red', 'green', 'blue']
#point_labels = [f'px {py}, {py}' for px, py in marker_points]
point_labels = ['' ,'', '']

# 4A. CALIBRATED MARKED POINTS SPECTROGRAM
#------------------------
# Create publication-ready figure - more square with larger fonts
plt.figure(figsize=(6, 6))  # More square format for publication

# Extract data for each point from ensemble 1 and ground truth
for point_idx, ((px, py), color, label) in enumerate(zip(marker_points, point_colors, point_labels)):
    
    # Get ensemble 1 data (mean and std)
    if len(all_wavelengths) > 0 and len(all_means) > 0 and len(all_stds) > 0:
        wavelengths = all_wavelengths[0]  # Ensemble 1 wavelengths
        
        # Extract pixel data from ensemble 1
        try:
            with tifffile.TiffFile(ensemble_tifs[0]) as tif:
                stack = tif.asarray()
                if len(stack.shape) == 3:
                    band_count = stack.shape[0] // 2
                    
                    # Extract mean and std values for this pixel
                    mean_values = stack[:band_count, py, px]
                    std_values = stack[band_count:, py, px]
                    
                    # Apply sensor calibration
                    calibrated_means, calibrated_stds = apply_sensor_calibration(
                        wavelengths, mean_values, std_values, calibration_interp)
                    
                    # Plot mean line (dashed)
                    plt.plot(wavelengths, calibrated_means, color=color, linestyle='--', 
                             linewidth=2, label=f'{label} Generated')
                    
                    # Plot ±2σ error region (translucent)
                    plt.fill_between(wavelengths, 
                                   calibrated_means - 2*calibrated_stds,
                                   calibrated_means + 2*calibrated_stds,
                                   color=color, alpha=0.15, label=f'{label} ±2σ')
                    
        except Exception as e:
            print(f"Error extracting ensemble data for point {px},{py}: {e}")
    
    # Get ground truth data for this point
    if ground_truth_values is not None and ground_truth_wavelengths is not None:
        try:
            with tifffile.TiffFile(ground_truth_tif) as tif:
                gt_stack = tif.asarray()
                if len(gt_stack.shape) == 3:
                    # Extract ground truth values for this pixel
                    gt_values = gt_stack[:, py, px]
                    
                    # Apply sensor calibration
                    calibrated_gt, _ = apply_sensor_calibration(
                        ground_truth_wavelengths, gt_values, None, calibration_interp)
                    
                    # Plot ground truth line (solid)
                    plt.plot(ground_truth_wavelengths, calibrated_gt, color=color, 
                             linestyle='-', linewidth=2.5, label=f'{label} Ground Truth')
                    
        except Exception as e:
            print(f"Error extracting ground truth data for point {px},{py}: {e}")

# Format the calibrated plot for publication with larger fonts
plt.xlabel('Wavelength (nm)', fontsize=16)
plt.ylabel('Fluorescence', fontsize=16)
# Use actual wavelength range from calibration data
plt.xlim(min(calibration_wavelengths), max(calibration_wavelengths))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
#plt.legend(loc='upper left', fontsize=12, ncol=1, framealpha=0.8)
plt.tight_layout()

# Save the calibrated spectrogram
spectrogram_calibrated_path = os.path.join(output_dir, 'marked_points_spectrogram_calibrated.svg')
plt.savefig(spectrogram_calibrated_path, format='svg', bbox_inches='tight', dpi=300)
plt.show()
print(f"Saved calibrated marked points spectrogram: {spectrogram_calibrated_path}")

#------------------------
# 4B. UNCALIBRATED MARKED POINTS SPECTROGRAM
#------------------------
# Create publication-ready figure - more square with larger fonts
plt.figure(figsize=(6, 6))  # More square format for publication

# Extract uncalibrated data for each point from ensemble 1 and ground truth
for point_idx, ((px, py), color, label) in enumerate(zip(marker_points, point_colors, point_labels)):
    
    # Get ensemble 1 data (mean and std) - uncalibrated
    if len(all_wavelengths) > 0 and len(all_means_uncalibrated) > 0 and len(all_stds_uncalibrated) > 0:
        wavelengths = all_wavelengths[0]  # Ensemble 1 wavelengths
        
        # Extract pixel data from ensemble 1
        try:
            with tifffile.TiffFile(ensemble_tifs[0]) as tif:
                stack = tif.asarray()
                if len(stack.shape) == 3:
                    band_count = stack.shape[0] // 2
                    
                    # Extract mean and std values for this pixel (uncalibrated)
                    mean_values = stack[:band_count, py, px]
                    std_values = stack[band_count:, py, px]
                    
                    # Plot mean line (dashed) - uncalibrated
                    plt.plot(wavelengths, mean_values, color=color, linestyle='--', 
                             linewidth=2, label=f'{label} Generated')
                    
                    # Plot ±2σ error region (translucent) - uncalibrated
                    plt.fill_between(wavelengths, 
                                   mean_values - 2*std_values,
                                   mean_values + 2*std_values,
                                   color=color, alpha=0.15, label=f'{label} ±2σ')
                    
        except Exception as e:
            print(f"Error extracting ensemble data for point {px},{py}: {e}")
    
    # Get uncalibrated ground truth data for this point
    if ground_truth_values_uncalibrated is not None and ground_truth_wavelengths is not None:
        try:
            with tifffile.TiffFile(ground_truth_tif) as tif:
                gt_stack = tif.asarray()
                if len(gt_stack.shape) == 3:
                    # Extract ground truth values for this pixel (uncalibrated)
                    gt_values = gt_stack[:, py, px]
                    
                    # Plot ground truth line (solid) - uncalibrated
                    plt.plot(ground_truth_wavelengths, gt_values, color=color, 
                             linestyle='-', linewidth=2.5, label=f'{label} Ground Truth')
                    
        except Exception as e:
            print(f"Error extracting ground truth data for point {px},{py}: {e}")

# Format the uncalibrated plot for publication with larger fonts
plt.xlabel('Wavelength (nm)', fontsize=16)
plt.ylabel('Fluorescence', fontsize=16)  # Different label for uncalibrated
# Use actual wavelength range from calibration data
plt.xlim(min(calibration_wavelengths), max(calibration_wavelengths))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
#plt.legend(loc='upper center', fontsize=12, ncol=1, framealpha=0.8, bbox_to_anchor=(0.65, 1.0))
plt.tight_layout()

# Save the uncalibrated spectrogram
spectrogram_uncalibrated_marked_path = os.path.join(output_dir, 'marked_points_spectrogram_uncalibrated.svg')
plt.savefig(spectrogram_uncalibrated_marked_path, format='svg', bbox_inches='tight', dpi=300)
plt.show()
print(f"Saved uncalibrated marked points spectrogram: {spectrogram_uncalibrated_marked_path}")

print(f"All images saved to: {output_dir}")

#------------------------
# 5. GENERATE AND SAVE COLORBARS
#------------------------
print("Generating and saving colorbars...")

# Create a function to save colorbar
def save_colorbar(colormap, vmin, vmax, label, filename):
    """Save a standalone colorbar as an image"""
    fig, ax = plt.subplots(figsize=(6, 1))
    
    # Create a dummy mappable for the colorbar
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    
    # Create horizontal colorbar
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    #cbar.set_label(label, fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Remove extra whitespace
    plt.tight_layout()
    
    # Save colorbar
    colorbar_path = os.path.join(output_dir, filename)
    plt.savefig(colorbar_path, format='svg', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved colorbar: {colorbar_path}")

# 1. Panchromatic images colorbar (viridis, clipped 0.03-0.4)
save_colorbar(cm.viridis, 0.03, 0.4, 'Fluorescence Intensity', 'colorbar_panchromatic.svg')

# 2. Channel 23 mean images colorbar (viridis, clipped 0.03-0.4)
save_colorbar(cm.viridis, 0.03, 0.4, 'Mean Fluorescence', 'colorbar_channel23_mean.svg')

# 3. Channel 23 std images colorbar (viridis, clipped 0-0.035 based on std_clip/2)
save_colorbar(cm.viridis, 0, 0.035, 'Standard Deviation', 'colorbar_channel23_std.svg')

# 4. Ground truth channel 23 colorbar (same as mean)
save_colorbar(cm.viridis, 0.03, 0.4, 'Ground Truth Fluorescence', 'colorbar_ground_truth.svg')

print("All colorbars saved successfully!")