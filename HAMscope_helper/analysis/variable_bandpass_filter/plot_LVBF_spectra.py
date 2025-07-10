import matplotlib.pyplot as plt
import os
import re
import numpy as np

def wavelength_to_rgb(wavelength):
    """
    Convert a wavelength in nanometers to an RGB color value.
    Adapted from code by Dan Bruton: http://www.physics.sfasu.edu/astro/color/spectra.html
    
    Args:
        wavelength (float): Wavelength in nanometers (typically 380-750nm for visible light)
        
    Returns:
        tuple: RGB color as (r,g,b) values between 0 and 1
    """
    gamma = 0.8
    
    # Outside visible spectrum
    if wavelength < 380 or wavelength > 750:
        return (0.5, 0.5, 0.5)  # Return gray
    
    # Determine RGB values based on wavelength range
    if 380 <= wavelength < 440:
        r = -(wavelength - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= wavelength < 490:
        r = 0.0
        g = (wavelength - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wavelength < 510:
        r = 0.0
        g = 1.0
        b = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:
        r = (wavelength - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= wavelength < 645:
        r = 1.0
        g = -(wavelength - 645) / (645 - 580)
        b = 0.0
    elif 645 <= wavelength <= 750:
        r = 1.0
        g = 0.0
        b = 0.0
    
    # Attenuate intensity at extremes of the visible spectrum
    if 380 <= wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 700 < wavelength <= 750:
        factor = 0.3 + 0.7 * (750 - wavelength) / (750 - 700)
    else:
        factor = 1.0
    
    # Apply gamma correction and intensity factor
    r = ((r * factor) ** gamma) if r > 0 else 0
    g = ((g * factor) ** gamma) if g > 0 else 0
    b = ((b * factor) ** gamma) if b > 0 else 0
    
    return (r, g, b)

def read_spectra_data(file_path):
    """
    Reads spectral data from a .txt file and returns the wavelengths and intensities.
    
    Args:
        file_path (str): The path to the SpectraSuite data file.
        
    Returns:
        tuple: (wavelengths, intensities) lists or (None, None) if reading fails
    """
    wavelengths = []
    intensities = []
    data_section_started = False

    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None, None

        # Open and read the file
        with open(file_path, 'r') as f:
            for line in f:
                # Remove leading/trailing whitespace
                line = line.strip()

                # Check for the start of the data section
                if line == ">>>>>Begin Processed Spectral Data<<<<<":
                    data_section_started = True
                    continue  # Skip this line

                # If we are in the data section and the line is not empty
                if data_section_started and line:
                    try:
                        # Split the line into wavelength and intensity
                        parts = line.split()
                        if len(parts) == 2:
                            wavelength = float(parts[0])
                            intensity = float(parts[1])
                            wavelengths.append(wavelength)
                            intensities.append(intensity)
                        else:
                            # Handle lines that might not have 2 parts if needed
                            print(f"Skipping malformed data line: {line}")
                    except ValueError:
                        # Handle potential errors during float conversion
                        print(f"Skipping line due to conversion error: {line}")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None, None

    # Check if any data was read
    if not wavelengths or not intensities:
        print(f"No spectral data found in the file {os.path.basename(file_path)}.")
        return None, None

    return wavelengths, intensities

def plot_multiple_spectra(folder_path, baseline_path=None, generate_calibration=False, calibration_output_path=None):
    """
    Reads all .txt files in a folder and plots all spectra on the same chart.
    Each spectrum is colored according to its peak wavelength and normalized to a baseline if provided.
    
    Args:
        folder_path (str): Path to the folder containing spectral data files.
        baseline_path (str, optional): Path to the baseline spectrum file for normalization.
        generate_calibration (bool): Whether to generate calibration data
        calibration_output_path (str): Path to save calibration data
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return
    
    # Get all .txt files in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    # Sort files numerically (0.txt, 1.txt, 2.txt, etc.)
    txt_files.sort(key=lambda x: int(re.match(r'(\d+)\.txt', x).group(1)) if re.match(r'(\d+)\.txt', x) else float('inf'))
    
    if not txt_files:
        print(f"No .txt files found in {folder_path}")
        return
    
    # Load baseline spectrum if provided
    baseline_wavelengths = None
    baseline_intensities = None
    if baseline_path and os.path.exists(baseline_path):
        print(f"Loading baseline spectrum from {baseline_path}")
        baseline_wavelengths, baseline_intensities = read_spectra_data(baseline_path)
    else:
        print(f"No baseline spectrum provided or file not found")
    
    # Load all spectra first to find the maximum intensity for normalization
    spectra_data = []
    max_intensity = 0
    min_intensity = float('inf')  # Track the minimum intensity across all spectra
    
    # Load all spectra and find the maximum intensity
    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)
        wavelengths, intensities = read_spectra_data(file_path)
        
        if wavelengths and intensities:
            spectra_data.append((file_name, wavelengths, intensities))
            current_max = max(intensities)
            current_min = min(intensities)
            if current_max > max_intensity:
                max_intensity = current_max
            if current_min < min_intensity:
                min_intensity = current_min
    
    # Check if baseline has a higher peak
    baseline_min = 0
    if baseline_intensities:
        baseline_max = max(baseline_intensities)
        baseline_min = min(baseline_intensities)
        if baseline_max > max_intensity:
            max_intensity = baseline_max
        
    plt.figure(figsize=(12, 8))
    
    # Normalize baseline if it exists - subtract min and then normalize to its own max
    normalized_baseline = None
    if baseline_wavelengths and baseline_intensities:
        # Subtract minimum and then normalize to its own max
        corrected_baseline = [intensity - baseline_min for intensity in baseline_intensities]
        baseline_max_corrected = max(corrected_baseline)
        normalized_baseline = [intensity / baseline_max_corrected for intensity in corrected_baseline]
    
    # First pass: calculate all plot intensities to find the maximum
    all_processed_spectra = []
    max_plot_intensity = 0  # Track the maximum plot intensity in the 400-800nm range
    calibration_data = []  # Store calibration data
    
    for file_name, wavelengths, intensities in spectra_data:
        # Correct for the noise floor by subtracting the minimum
        corrected_intensities = [intensity - min_intensity for intensity in intensities]
        
        # Normalize spectrum (divide by the max intensity)
        normalized_intensities = [intensity / max_intensity for intensity in corrected_intensities]
        
        # If baseline exists and has data, normalize against it
        if normalized_baseline:
            # Interpolate baseline to match the wavelengths of this spectrum
            interpolated_baseline = np.interp(
                wavelengths, 
                baseline_wavelengths, 
                normalized_baseline,
                left=normalized_baseline[0], 
                right=normalized_baseline[-1]
            )
            
            # Divide by baseline to get transmission
            transmission = []
            for j in range(len(normalized_intensities)):
                # Avoid division by zero
                if interpolated_baseline[j] > 0.001:  # Small threshold to avoid division by very small values
                    transmission.append(normalized_intensities[j] / interpolated_baseline[j])
                else:
                    transmission.append(0)
            
            plot_intensities = transmission
        else:
            # Just use normalized values if no baseline
            plot_intensities = normalized_intensities
        
        # Find the peak wavelength (from original intensities)
        peak_idx = np.argmax(intensities)
        peak_wavelength = wavelengths[peak_idx]
        
        # Calculate mean wavelength and mean intensity for calibration
        if generate_calibration:
            # Step 1: Calculate mean wavelength within +/- 40 nm of peak
            wavelength_range_mask = [abs(wl - peak_wavelength) <= 40 for wl in wavelengths]
            wavelengths_in_range = [wavelengths[i] for i in range(len(wavelengths)) if wavelength_range_mask[i]]
            intensities_in_range = [intensities[i] for i in range(len(intensities)) if wavelength_range_mask[i]]
            
            if wavelengths_in_range:
                # Calculate weighted mean wavelength (weighted by intensity)
                total_weighted_wavelength = sum(wl * intensity for wl, intensity in zip(wavelengths_in_range, intensities_in_range))
                total_intensity = sum(intensities_in_range)
                mean_wavelength = total_weighted_wavelength / total_intensity if total_intensity > 0 else peak_wavelength
            else:
                mean_wavelength = peak_wavelength
            
            # Step 2: Calculate mean intensity within 5 nm of the mean wavelength
            intensity_range_mask = [abs(wl - mean_wavelength) <= 5 for wl in wavelengths]
            intensities_near_mean = [plot_intensities[i] for i in range(len(plot_intensities)) if intensity_range_mask[i]]
            
            if intensities_near_mean:
                mean_intensity = np.mean(intensities_near_mean)
            else:
                mean_intensity = plot_intensities[peak_idx]
            
            # Store calibration data
            spectrum_number = int(re.match(r'(\d+)\.txt', file_name).group(1)) if re.match(r'(\d+)\.txt', file_name) else 0
            calibration_data.append({
                'spectrum_number': spectrum_number,
                'filename': file_name,
                'mean_wavelength': mean_wavelength,
                'mean_intensity': mean_intensity
            })
        
        # Find max intensity in 400-800nm range for this spectrum
        for idx, wl in enumerate(wavelengths):
            if 400 <= wl <= 800:
                if plot_intensities[idx] > max_plot_intensity:
                    max_plot_intensity = plot_intensities[idx]
        
        # Store calculated values for later use (use mean_wavelength for coloring if available)
        display_wavelength = mean_wavelength if generate_calibration and 'mean_wavelength' in locals() else peak_wavelength
        all_processed_spectra.append((file_name, wavelengths, plot_intensities, display_wavelength))
    
    # Save calibration data if requested
    if generate_calibration and calibration_output_path and calibration_data:
        with open(calibration_output_path, 'w') as f:
            f.write("# Transmission Calibration Data\n")
            f.write("# Mean wavelength calculated within +/- 40nm of peak\n")
            f.write("# Mean intensity calculated within +/- 5nm of mean wavelength\n")
            f.write("# Format: Spectrum_Number, Filename, Mean_Wavelength(nm), Mean_Intensity\n")
            f.write("#" + "="*80 + "\n")
            
            for data in sorted(calibration_data, key=lambda x: x['spectrum_number']):
                f.write(f"{data['spectrum_number']}, {data['filename']}, "
                       f"{data['mean_wavelength']:.2f}, {data['mean_intensity']:.6f}\n")
        
        print(f"Calibration data saved to: {calibration_output_path}")
        
        # Print summary statistics
        print(f"\nCalibration Summary:")
        print(f"Number of spectra: {len(calibration_data)}")
        mean_wavelengths = [d['mean_wavelength'] for d in calibration_data]
        mean_intensities = [d['mean_intensity'] for d in calibration_data]
        print(f"Mean wavelength range: {min(mean_wavelengths):.1f} - {max(mean_wavelengths):.1f} nm")
        print(f"Mean intensity range: {min(mean_intensities):.4f} - {max(mean_intensities):.4f}")
    
    # Set a reasonable minimum value for max_plot_intensity to avoid division by very small numbers
    if max_plot_intensity < 0.001:
        max_plot_intensity = 1.0
    
    # Second pass: plot with final normalization
    i = 0
    for file_name, wavelengths, plot_intensities, display_wavelength in all_processed_spectra:
        i += 1
        
        # Final normalization
        final_intensities = [intensity / max_plot_intensity for intensity in plot_intensities]
        
        # Get color based on display wavelength (mean wavelength if calibration is being generated)
        line_color = wavelength_to_rgb(display_wavelength)
        
        # Plot the spectrum
        plt.plot(wavelengths, final_intensities, 
                 label=f'{i} ({display_wavelength:.1f}nm)', 
                 color=line_color)
    
    # Set appropriate axis labels based on normalization
    plt.xlabel('Wavelength (nm)')
    if normalized_baseline:
        plt.ylabel('Transmission (0-1)')
        plt.title('Normalized Transmission Spectra')
        plt.ylim(0, 1.1)  # Set y-axis limits for transmission
    else:
        plt.ylabel('Normalized Intensity')
        plt.title('Normalized Intensity Spectra')
    
    # Set x-axis limits to show only 400-800 nm
    plt.xlim(400, 800)
    
    # Add more x-axis tick marks
    plt.xticks(np.arange(400, 801, 50))
    
    plt.grid(True)
    plt.legend(loc='center right', fontsize='small')  # Fixed legend position to right side
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # Folder containing the spectrum files
    spectra_folder = '/media/al/Extreme SSD/20250425_spectra_2'
    # Baseline spectrum (halogen lamp)
    baseline_path = '/media/al/Extreme SSD/20250424_spectra/halogen_lamp.txt'
    
    # Output path for calibration data
    calibration_output = '/media/al/Extreme SSD/20250425_spectra_2/transmission_calibration.txt'
    
    # Plot all spectra in the folder, normalized to baseline, and generate calibration
    plot_multiple_spectra(spectra_folder, baseline_path, 
                         generate_calibration=True, 
                         calibration_output_path=calibration_output)
    
    baseline_wavelengths, baseline_intensities = read_spectra_data(baseline_path)
    if baseline_wavelengths and baseline_intensities:
        plt.figure(figsize=(10, 6))
        plt.plot(baseline_wavelengths, baseline_intensities, color='orange', label='Baseline (halogen lamp)')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title('Baseline Spectrum')
        plt.xlim(400, 800)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Could not load baseline spectrum.")

