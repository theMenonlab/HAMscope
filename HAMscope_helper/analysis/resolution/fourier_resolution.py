import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
import glob
from collections import defaultdict

def get_wavelength_from_channel(channel):
    """
    Convert channel number to wavelength.
    Channel 0 = 703nm, Channel 29 = 452nm (linear interpolation)
    """
    return 703 - (channel * (703 - 452) / 29)

def analyze_image_resolution(image_path, pixels_per_micron_x, pixels_per_micron_y, channel=None, manual_threshold=None, show_plot=False):
    """
    Analyzes the resolution of a biological image using its Fourier transform and Otsu's method.

    Args:
        image_path (str): The path to the image file.
        pixels_per_micron_x (float): The number of pixels per micron in the x-direction.
        pixels_per_micron_y (float): The number of pixels per micron in the y-direction.
        channel (int, optional): Channel number for multi-channel images (e.g., TIF stacks).
        manual_threshold (float, optional): Manual threshold value (0.0-1.0). If None, uses Otsu's method.
        show_plot (bool): Whether to display the visualization plots.

    Returns:
        tuple: A tuple containing the estimated resolution in microns and the cutoff frequency in cycles/um.
    """
    # Load the image
    if image_path.lower().endswith('.tif') or image_path.lower().endswith('.tiff'):
        # Load TIFF with tifffile
        image_data = tifffile.imread(image_path)
        if channel is not None:
            if len(image_data.shape) == 3:
                image = image_data[channel]
            else:
                raise ValueError(f"Channel {channel} specified but image is not multi-channel")
        else:
            if len(image_data.shape) == 3:
                image = image_data[0]  # Use first channel if none specified
            else:
                image = image_data
    else:
        # Load regular image formats in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

    # Ensure image is 2D
    if len(image.shape) != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    # Get actual image dimensions
    rows, cols = image.shape
    if show_plot:
        print(f"Loaded image dimensions: {cols}px x {rows}px")

    # normalize to 0-1
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # --- 1. Fourier Transform ---
    # Perform 2D FFT
    f_transform = np.fft.fft2(image)
    # Shift the zero frequency component to the center
    f_transform_shifted = np.fft.fftshift(f_transform)
    # Calculate the power spectrum (magnitude squared)
    power_spectrum = np.abs(f_transform_shifted)**2

    # --- 2. Threshold Detection ---
    # Create a grid of radial distances from the center
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    radius = np.sqrt(x*x + y*y)

    # Calculate radial profile by binning
    max_radius = int(np.min([crow, ccol]))
    radial_bins = np.arange(0, max_radius, 1)
    radial_profile = np.zeros(len(radial_bins))
    
    for i, r in enumerate(radial_bins):
        mask = (radius >= r) & (radius < r + 1)
        if np.any(mask):
            radial_profile[i] = np.mean(power_spectrum[mask])
    
    # Apply threshold detection
    log_profile = np.log1p(radial_profile)
    
    if manual_threshold is not None:
        # Use manual threshold
        threshold_value = manual_threshold * 255
        threshold_method = f"Manual ({manual_threshold})"
    else:
        # Use Otsu's method on the log of the radial profile
        # Normalize to 0-255 for Otsu
        normalized = ((log_profile - log_profile.min()) / (log_profile.max() - log_profile.min()) * 255).astype(np.uint8)
        threshold_value, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_method = "Otsu"
    
    # Convert back to original scale
    threshold_power = np.exp((threshold_value / 255.0) * (log_profile.max() - log_profile.min()) + log_profile.min()) - 1
    cutoff_idx = np.where(radial_profile <= threshold_power)[0]
    cutoff_frequency_pixels = radial_bins[cutoff_idx[0]] if len(cutoff_idx) > 0 else max_radius

    # Calculate resolution in microns
    # Resolution is the inverse of the cutoff frequency
    resolution_microns_x = 1 / (cutoff_frequency_pixels / cols) * (1 / pixels_per_micron_x) if cutoff_frequency_pixels != 0 else float('inf')
    resolution_microns_y = 1 / (cutoff_frequency_pixels / rows) * (1 / pixels_per_micron_y) if cutoff_frequency_pixels != 0 else float('inf')
    # Average the resolution in x and y directions
    resolution_microns = (resolution_microns_x + resolution_microns_y) / 2

    # Convert cutoff frequency to cycles/um
    # Normalized frequency = cutoff_frequency_pixels / (image_size/2)
    # Cycles/um = normalized_frequency * Nyquist_frequency
    # Nyquist frequency = pixels_per_micron / 2
    avg_image_size = (rows + cols) / 2
    normalized_frequency = cutoff_frequency_pixels / (avg_image_size / 2)
    avg_pixels_per_micron = (pixels_per_micron_x + pixels_per_micron_y) / 2
    nyquist_frequency = avg_pixels_per_micron / 2
    cutoff_frequency_cycles_per_um = normalized_frequency * nyquist_frequency

    # --- 3. Visualization ---
    if show_plot:
        plt.figure(figsize=(15, 5))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        if channel is not None:
            plt.title(f'Est. Resolution: {resolution_microns:.2f} µm, (ch {channel})')
        else:
            plt.title(f'Est. Resolution: {resolution_microns:.2f} µm')
        plt.axis('off')

        # Power Spectrum with Resolution Threshold
        plt.subplot(1, 3, 2)
        plt.imshow(np.log1p(power_spectrum), cmap='hot')
        circle = plt.Circle((ccol, crow), cutoff_frequency_pixels, color='cyan', fill=False, linewidth=2)
        plt.gca().add_patch(circle)
        plt.title(f'Power Spectrum ({threshold_method})')
        plt.axis('off')

        # Radial Profile in cycles/um
        plt.subplot(1, 3, 3)
        # Convert radial_bins to cycles/um
        radial_bins_cycles_per_um = []
        for r in radial_bins:
            norm_freq = r / (avg_image_size / 2)
            cycles_per_um = norm_freq * nyquist_frequency
            radial_bins_cycles_per_um.append(cycles_per_um)
        
        plt.plot(radial_bins_cycles_per_um, radial_profile, 'b-', linewidth=2)
        plt.axvline(x=cutoff_frequency_cycles_per_um, color='cyan', linestyle='--', linewidth=2)
        plt.axhline(y=threshold_power, color='red', linestyle=':', linewidth=1, alpha=0.7, label=f'{threshold_method} threshold')
        plt.xlabel('Spatial Frequency (cycles/µm)')
        plt.ylabel('Average Power')
        plt.title('Radial Power Profile')
        plt.yscale('log')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return resolution_microns, cutoff_frequency_cycles_per_um

def process_batch_images(base_path, pixels_per_micron, display_channel=20, manual_threshold=0.1):
    """
    Process all images in the directory and calculate statistics.
    For hyperspectral images, process all 30 channels and average results.
    """
    # Dictionary to store results by image type
    results = defaultdict(list)
    
    # Find all image files
    image_patterns = ['mini_raw_*.tif', 'hs_raw_*.tif', 'hs_gen_*.tif']
    
    for pattern in image_patterns:
        image_type = pattern.split('_')[0] + '_' + pattern.split('_')[1]  # e.g., 'mini_raw', 'hs_raw', 'hs_gen'
        files = glob.glob(os.path.join(base_path, pattern))
        files.sort()  # Sort to ensure consistent ordering
        
        print(f"\nProcessing {image_type} images:")
        print("-" * 40)
        
        for file_path in files:
            filename = os.path.basename(file_path)
            image_number = filename.split('_')[-1].split('.')[0]  # Extract number from filename
            
            try:
                # Show plot only for image 6 and display_channel
                show_plot = (image_number == '6')
                
                if image_type == 'mini_raw':
                    # mini_raw images are grayscale, process single channel
                    resolution, cutoff_freq = analyze_image_resolution(
                        file_path,
                        pixels_per_micron,
                        pixels_per_micron,
                        channel=None,
                        manual_threshold=manual_threshold,
                        show_plot=show_plot
                    )
                    
                    results[image_type].append({
                        'filename': filename,
                        'resolution': resolution,
                        'cutoff_frequency': cutoff_freq
                    })
                    
                    print(f"{filename}: Resolution = {resolution:.2f} µm, Cutoff = {cutoff_freq:.2f} cycles/µm")
                    
                else:
                    # hs_raw and hs_gen: process all 30 channels
                    if show_plot:
                        print(f"\nProcessing all 30 channels for {filename} (displaying channel {display_channel})")
                    
                    channel_resolutions = []
                    channel_cutoffs = []
                    
                    for channel in range(30):  # Process all 30 channels
                        # Only show plot for display_channel
                        show_channel_plot = show_plot and (channel == display_channel)
                        
                        try:
                            resolution, cutoff_freq = analyze_image_resolution(
                                file_path,
                                pixels_per_micron,
                                pixels_per_micron,
                                channel=channel,
                                manual_threshold=manual_threshold,
                                show_plot=show_channel_plot
                            )
                            
                            if resolution != float('inf'):
                                channel_resolutions.append(resolution)
                                channel_cutoffs.append(cutoff_freq)
                                
                        except Exception as e:
                            print(f"  Error processing channel {channel}: {e}")
                    
                    # Calculate average across all valid channels
                    if channel_resolutions:
                        avg_resolution = np.mean(channel_resolutions)
                        avg_cutoff = np.mean(channel_cutoffs)
                        std_resolution = np.std(channel_resolutions)
                        std_cutoff = np.std(channel_cutoffs)
                        
                        results[image_type].append({
                            'filename': filename,
                            'resolution': avg_resolution,
                            'cutoff_frequency': avg_cutoff,
                            'resolution_std': std_resolution,
                            'cutoff_std': std_cutoff,
                            'valid_channels': len(channel_resolutions)
                        })
                        
                        print(f"{filename}: Resolution = {avg_resolution:.2f} ± {std_resolution:.2f} µm, "
                              f"Cutoff = {avg_cutoff:.2f} ± {std_cutoff:.2f} cycles/µm "
                              f"(averaged over {len(channel_resolutions)} channels)")
                    else:
                        print(f"{filename}: No valid channels processed")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return results

def process_batch_images_with_wavelength(base_path, pixels_per_micron, display_channel=20, manual_threshold=0.1):
    """
    Process all images in the directory and calculate statistics.
    For hyperspectral images, process all 30 channels and store results per channel.
    """
    # Dictionary to store results by image type and channel
    results = defaultdict(lambda: defaultdict(list))
    
    # Find all image files
    image_patterns = ['mini_raw_*.tif', 'hs_raw_*.tif', 'hs_gen_*.tif']
    
    for pattern in image_patterns:
        image_type = pattern.split('_')[0] + '_' + pattern.split('_')[1]  # e.g., 'mini_raw', 'hs_raw', 'hs_gen'
        files = glob.glob(os.path.join(base_path, pattern))
        files.sort()  # Sort to ensure consistent ordering
        
        print(f"\nProcessing {image_type} images:")
        print("-" * 40)
        
        for file_path in files:
            filename = os.path.basename(file_path)
            image_number = filename.split('_')[-1].split('.')[0]  # Extract number from filename
            
            try:
                # Show plot only for image 6 and display_channel
                show_plot = (image_number == '6')
                
                if image_type == 'mini_raw':
                    # mini_raw images are grayscale, process single channel
                    resolution, cutoff_freq = analyze_image_resolution(
                        file_path,
                        pixels_per_micron,
                        pixels_per_micron,
                        channel=None,
                        manual_threshold=manual_threshold,
                        show_plot=show_plot
                    )
                    
                    results[image_type]['single'].append({
                        'filename': filename,
                        'resolution': resolution,
                        'cutoff_frequency': cutoff_freq
                    })
                    
                    print(f"{filename}: Resolution = {resolution:.2f} µm, Cutoff = {cutoff_freq:.2f} cycles/µm")
                    
                else:
                    # hs_raw and hs_gen: process all 30 channels
                    if show_plot:
                        print(f"\nProcessing all 30 channels for {filename} (displaying channel {display_channel})")
                    
                    for channel in range(30):  # Process all 30 channels
                        # Only show plot for display_channel
                        show_channel_plot = show_plot and (channel == display_channel)
                        
                        try:
                            resolution, cutoff_freq = analyze_image_resolution(
                                file_path,
                                pixels_per_micron,
                                pixels_per_micron,
                                channel=channel,
                                manual_threshold=manual_threshold,
                                show_plot=show_channel_plot
                            )
                            
                            if resolution != float('inf'):
                                wavelength = get_wavelength_from_channel(channel)
                                results[image_type][channel].append({
                                    'filename': filename,
                                    'wavelength': wavelength,
                                    'resolution': resolution,
                                    'cutoff_frequency': cutoff_freq
                                })
                                
                        except Exception as e:
                            print(f"  Error processing channel {channel}: {e}")
                    
                    # Print summary for this image
                    channel_resolutions = []
                    for channel in range(30):
                        if results[image_type][channel]:
                            channel_data = [d for d in results[image_type][channel] if d['filename'] == filename]
                            if channel_data:
                                channel_resolutions.append(channel_data[0]['resolution'])
                    
                    if channel_resolutions:
                        avg_resolution = np.mean(channel_resolutions)
                        std_resolution = np.std(channel_resolutions)
                        print(f"{filename}: Average resolution = {avg_resolution:.2f} ± {std_resolution:.2f} µm "
                              f"(over {len(channel_resolutions)} channels)")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return results

def plot_resolution_vs_wavelength(results):
    """
    Plot resolution as a function of wavelength for hyperspectral images.
    """
    plt.figure(figsize=(6, 4))  # Smaller figure for journal
    
    colors = {'hs_raw': '#1f77b4', 'hs_gen': '#d62728'}  # Blue and red
    labels = {'hs_raw': 'Ground Truth', 'hs_gen': 'Generated'}
    
    for image_type in ['hs_raw', 'hs_gen']:
        if image_type not in results:
            continue
            
        wavelengths = []
        mean_resolutions = []
        
        # Process each channel
        for channel in range(30):
            if channel in results[image_type] and results[image_type][channel]:
                channel_data = results[image_type][channel]
                resolutions = [d['resolution'] for d in channel_data if d['resolution'] != float('inf')]
                
                if resolutions:
                    wavelength = get_wavelength_from_channel(channel)
                    wavelengths.append(wavelength)
                    mean_resolutions.append(np.mean(resolutions))
        
        if wavelengths:
            wavelengths = np.array(wavelengths)
            mean_resolutions = np.array(mean_resolutions)
            
            # Sort by wavelength (ascending order: 450nm to 700nm)
            sort_idx = np.argsort(wavelengths)
            wavelengths = wavelengths[sort_idx]
            mean_resolutions = mean_resolutions[sort_idx]
            
            # Plot without error bars
            plt.plot(wavelengths, mean_resolutions, 
                    color=colors[image_type], label=labels[image_type], 
                    marker='o', markersize=6, linewidth=2.5)
            
            print(f"\n{labels[image_type]} wavelength analysis:")
            for wl, res in zip(wavelengths, mean_resolutions):
                print(f"  {wl:.0f} nm: {res:.2f} µm")
    
    plt.xlabel('Wavelength (nm)', fontsize=14)
    plt.ylabel('Resolution (µm)', fontsize=14)
    plt.title('Spatial Resolution vs Wavelength', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(440, 710)
    plt.ylim(5, 16)
    
    # Increase tick font size
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return wavelengths, mean_resolutions

def calculate_statistics(results):
    """
    Calculate and display statistics for each image type.
    """
    print("\n" + "="*60)
    print("RESOLUTION ANALYSIS SUMMARY")
    print("="*60)
    
    for image_type in results.keys():
        if image_type == 'mini_raw':
            # Handle single channel images
            if 'single' in results[image_type]:
                data = results[image_type]['single']
                resolutions = [d['resolution'] for d in data if d['resolution'] != float('inf')]
                cutoff_freqs = [d['cutoff_frequency'] for d in data if d['resolution'] != float('inf')]
                
                if resolutions:
                    mean_res = np.mean(resolutions)
                    std_res = np.std(resolutions)
                    mean_cutoff = np.mean(cutoff_freqs)
                    std_cutoff = np.std(cutoff_freqs)
                    
                    print(f"\n{image_type.upper()} Images (n={len(resolutions)}):")
                    print(f"  Resolution: {mean_res:.2f} ± {std_res:.2f} µm")
                    print(f"  Cutoff Frequency: {mean_cutoff:.2f} ± {std_cutoff:.2f} cycles/µm")
        else:
            # Handle hyperspectral images - calculate overall statistics
            all_resolutions = []
            all_cutoffs = []
            
            for channel in range(30):
                if channel in results[image_type]:
                    channel_data = results[image_type][channel]
                    for d in channel_data:
                        if d['resolution'] != float('inf'):
                            all_resolutions.append(d['resolution'])
                            all_cutoffs.append(d['cutoff_frequency'])
            
            if all_resolutions:
                mean_res = np.mean(all_resolutions)
                std_res = np.std(all_resolutions)
                mean_cutoff = np.mean(all_cutoffs)
                std_cutoff = np.std(all_cutoffs)
                
                print(f"\n{image_type.upper()} Images (all channels, n={len(all_resolutions)}):")
                print(f"  Resolution: {mean_res:.2f} ± {std_res:.2f} µm")
                print(f"  Cutoff Frequency: {mean_cutoff:.2f} ± {std_cutoff:.2f} cycles/µm")

if __name__ == '__main__':
    # Configuration

    BASE_PATH = '/media/al/Extreme SSD/20250701_usaf/deconvolution_results/batch_reg/img_6'
    DISPLAY_CHANNEL = 23  # Channel to display in visualizations
    IMAGE_WIDTH_MICRONS = 940
    MANUAL_THRESHOLD = 0.1  # Set to None to use Otsu's method

    # Calculate pixels per micron (assuming square images)
    # We'll use a sample image to get dimensions
    sample_files = glob.glob(os.path.join(BASE_PATH, '*.tif'))
    if sample_files:
        sample_image = tifffile.imread(sample_files[0])
        if len(sample_image.shape) == 3:
            IMAGE_WIDTH_PIXELS = sample_image.shape[2]
        else:
            IMAGE_WIDTH_PIXELS = sample_image.shape[1]
        
        PIXELS_PER_MICRON = IMAGE_WIDTH_PIXELS / IMAGE_WIDTH_MICRONS
        print(f"Sample image dimensions: {IMAGE_WIDTH_PIXELS} pixels")
        print(f"Physical width: {IMAGE_WIDTH_MICRONS} µm")
        print(f"Pixels per micron: {PIXELS_PER_MICRON:.2f}")
        
        # Print wavelength mapping
        print(f"\nWavelength mapping:")
        for ch in [0, 5, 10, 15, 20, 25, 29]:
            wl = get_wavelength_from_channel(ch)
            print(f"  Channel {ch}: {wl:.0f} nm")
    else:
        raise FileNotFoundError(f"No TIF files found in {BASE_PATH}")

    # Process all images with wavelength analysis
    results = process_batch_images_with_wavelength(
        BASE_PATH,
        PIXELS_PER_MICRON,
        display_channel=DISPLAY_CHANNEL,
        manual_threshold=MANUAL_THRESHOLD
    )
    
    # Calculate and display statistics
    calculate_statistics(results)
    
    # Plot resolution vs wavelength
    print("\nGenerating resolution vs wavelength plot...")
    plot_resolution_vs_wavelength(results)