import numpy as np
import matplotlib.pyplot as plt

# File paths
spectrum1_file = "/home/al/Downloads/Fiji.app/plugins/SpectraLibrary/spectrum1_roi.emn"
spectrum2_file = "/home/al/Downloads/Fiji.app/plugins/SpectraLibrary/spectrum2_roi.emn"
spectrum3_file = "/home/al/Downloads/Fiji.app/plugins/SpectraLibrary/spectrum3_roi.emn"
spectrum4_file = "/home/al/Downloads/Fiji.app/plugins/SpectraLibrary/spectrum4_roi.emn"

# Function to load data from file
def load_spectrum(file_path):
    wavelengths = []
    intensities = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                wavelengths.append(float(parts[0]))
                intensities.append(float(parts[1]))
    return np.array(wavelengths), np.array(intensities)

# Load data from files
wavelengths1, spectrum1_intensity = load_spectrum(spectrum1_file)
wavelengths2, spectrum2_intensity = load_spectrum(spectrum2_file)
wavelengths3, spectrum3_intensity = load_spectrum(spectrum3_file)
wavelengths4, spectrum4_intensity = load_spectrum(spectrum4_file)

wavelengths = np.linspace(700, 450, 30)

# Normalize each spectrum individually
spectrum1_normalized = spectrum1_intensity / np.max(spectrum1_intensity)
spectrum2_normalized = spectrum2_intensity / np.max(spectrum2_intensity)
spectrum3_normalized = spectrum3_intensity / np.max(spectrum3_intensity)
spectrum4_normalized = spectrum4_intensity / np.max(spectrum4_intensity)

# Create the plot with smaller figure size for subplot use
plt.figure(figsize=(6, 4))

# Plot with thicker lines and distinct line styles for better visibility
plt.plot(wavelengths, spectrum4_intensity, 'b', linewidth=3, label='Xylem')
plt.plot(wavelengths, spectrum3_intensity, 'g', linewidth=3, label='Phloem')
plt.plot(wavelengths, spectrum2_intensity, 'y', linewidth=3, label='Other')
plt.plot(wavelengths, spectrum1_intensity, 'r', linewidth=3, label='Epi/cort')


# Customize with larger fonts and reduced clutter
plt.xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
plt.ylabel('Intensity', fontsize=14, fontweight='bold')

# Optimize legend for small size
plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, ncol=2, loc='upper left')

# Remove grid to reduce visual clutter
# plt.grid(True, alpha=0.3)

# Set axis limits and reduce tick density
plt.xlim(450, 700)
plt.ylim(0, max(max(spectrum1_intensity), max(spectrum2_intensity), max(spectrum3_intensity), max(spectrum4_intensity)) * 1.1)

# Reduce number of ticks for cleaner look
plt.xticks(np.arange(450, 700, 50), fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# Make tick marks more prominent
plt.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)

# Adjust layout to prevent clipping
plt.tight_layout()

# Display the plot
plt.show()