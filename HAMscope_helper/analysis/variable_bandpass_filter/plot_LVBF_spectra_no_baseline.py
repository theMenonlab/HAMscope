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

def plot_multiple_spectra(folder_path):
    """
    Reads all .txt files in a folder and plots all spectra on the same chart.
    Each spectrum is colored according to its peak wavelength.
    
    Args:
        folder_path (str): Path to the folder containing spectral data files.
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
    
    plt.figure(figsize=(12, 8))
    
    # Process each file and plot its data
    i = 0
    for file_name in txt_files:
        i += 1
        file_path = os.path.join(folder_path, file_name)
        wavelengths, intensities = read_spectra_data(file_path)
        
        if wavelengths and intensities:
            # Find the peak wavelength (wavelength with maximum intensity)
            peak_idx = np.argmax(intensities)
            peak_wavelength = wavelengths[peak_idx]
            
            # Get color based on peak wavelength
            line_color = wavelength_to_rgb(peak_wavelength)
            
            # Plot the spectrum with the file name as the label using the color corresponding to peak wavelength
            plt.plot(wavelengths, intensities, label=f'{i} ({peak_wavelength:.1f}nm)', 
                     color=line_color)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Multiple Spectra Comparison')
    plt.grid(True)
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # Folder containing the spectrum files
    spectra_folder = '/media/al/Extreme SSD/20250424_spectra/spectra/'
    
    # Plot all spectra in the folder
    plot_multiple_spectra(spectra_folder)

