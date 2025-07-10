import matplotlib.pyplot as plt
import numpy as np
# Load and clean data from file
def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Skip to data section
    data_start = False
    wavelengths = []
    intensities = []
    for line in lines:
        if ">>>>>Begin Processed Spectral Data<<<<<" in line:
            data_start = True
            continue
        if data_start:
            try:
                wavelength, intensity = map(float, line.strip().split())
                wavelengths.append(wavelength)
                intensities.append(intensity)
            except:
                continue  # Skip non-numeric lines
    return np.array(wavelengths), np.array(intensities)
# Load both datasets
wavelength_diff, intensity_diff = load_data("/media/al/Extreme SSD/20250618_diffuser_analysis/transmission_spectra/diffuser.txt")
wavelength_no_diff, intensity_no_diff = load_data("/media/al/Extreme SSD/20250618_diffuser_analysis/transmission_spectra/no_diffuser.txt")
# Check that wavelengths match
if not np.allclose(wavelength_diff, wavelength_no_diff):
    raise ValueError("Wavelengths in the two files do not match!")
# Calculate Transmittance
transmittance = intensity_diff / intensity_no_diff
# Plot and save only the transmittance plot
import matplotlib.pyplot as plt
import numpy as np
# ... your data loading and calculations here ...
# Start clean
plt.close('all')
# Plot and save only the transmittance plot
plt.figure(figsize=(10, 5))
plt.plot(wavelength_diff, transmittance, label='Transmittance = I_with / I_without', color='green')
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Transmittance', fontsize=12)
plt.title('Transmittance of Diffuser', fontsize=16)
plt.grid(True)
plt.xlim(400, 750)
plt.tight_layout()
plt.savefig("transmittance_plot.png", dpi=300)
plt.savefig("transmittance_plot.svg")
plt.show()