import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd

def extract_blue_curve_from_image(image_path):
    """Extract the blue curve from the spectral response graph"""
    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Define blue color range in HSV for better detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Blue color range - adjust these values if needed
    lower_blue = np.array([22, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Create mask for blue pixels
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find contours of the blue curve
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        raise ValueError("No blue curve found in the image")
    
    # Get the largest contour (should be our curve)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Extract points from the contour
    points = largest_contour.reshape(-1, 2)
    
    return points, img_rgb, mask

def convert_pixels_to_wavelength_efficiency(points, img_shape):
    """Convert pixel coordinates to wavelength and efficiency values"""
    height, width = img_shape[:2]
    
    # Find the actual bounds of the detected curve
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    # Graph bounds - map curve endpoints to 400-1000nm range
    wavelength_min, wavelength_max = 400, 1000  # nm
    efficiency_min, efficiency_max = 0, 100     # %
    
    # Convert x-coordinates (pixels) to wavelength using curve bounds
    wavelengths = wavelength_min + ((points[:, 0] - x_min) / (x_max - x_min)) * (wavelength_max - wavelength_min)
    
    # Convert y-coordinates (pixels) to efficiency using curve bounds (note: y-axis is flipped in images)
    efficiencies = efficiency_max - ((points[:, 1] - y_min) / (y_max - y_min)) * (efficiency_max - efficiency_min)
    
    return wavelengths, efficiencies

def process_sensor_response(image_path):
    """Extract and process the sensor response curve"""
    # Extract blue curve points
    points, img_rgb, mask = extract_blue_curve_from_image(image_path)
    
    # Read image to get dimensions
    img = cv2.imread(image_path)
    
    # Convert to wavelength and efficiency
    wavelengths, efficiencies = convert_pixels_to_wavelength_efficiency(points, img.shape)
    
    # Sort by wavelength and remove duplicates
    sorted_indices = np.argsort(wavelengths)
    wavelengths_sorted = wavelengths[sorted_indices]
    efficiencies_sorted = efficiencies[sorted_indices]
    
    # Remove duplicate wavelengths by averaging efficiencies
    unique_wavelengths = []
    unique_efficiencies = []
    
    current_wl = wavelengths_sorted[0]
    current_eff_sum = efficiencies_sorted[0]
    count = 1
    
    for i in range(1, len(wavelengths_sorted)):
        if abs(wavelengths_sorted[i] - current_wl) < 1:  # Group wavelengths within 1nm
            current_eff_sum += efficiencies_sorted[i]
            count += 1
        else:
            unique_wavelengths.append(current_wl)
            unique_efficiencies.append(current_eff_sum / count)
            current_wl = wavelengths_sorted[i]
            current_eff_sum = efficiencies_sorted[i]
            count = 1
    
    # Add the last point
    unique_wavelengths.append(current_wl)
    unique_efficiencies.append(current_eff_sum / count)
    
    return np.array(unique_wavelengths), np.array(unique_efficiencies), points, img_rgb, mask

def combine_transmission_and_sensor_response(transmission_file, sensor_wavelengths, sensor_efficiencies):
    """Combine transmission data with sensor response"""
    # Read transmission data
    transmission_data = pd.read_csv(transmission_file, comment='#', 
                                  names=['Spectrum_Number', 'Filename', 'Peak_Wavelength', 'Max_Normalized_Transmission'])
    
    # Create interpolation function for sensor response
    sensor_interp = interp1d(sensor_wavelengths, sensor_efficiencies, 
                           bounds_error=False, fill_value=0, kind='linear')
    
    # Calculate sensor response at transmission wavelengths
    sensor_response_at_peaks = sensor_interp(transmission_data['Peak_Wavelength'])
    
    # Calculate combined response (transmission * sensor_response / 100)
    combined_response = transmission_data['Max_Normalized_Transmission'] * (sensor_response_at_peaks / 100)
    
    # Create new dataframe with combined data
    combined_data = transmission_data.copy()
    combined_data['Sensor_Response'] = sensor_response_at_peaks
    combined_data['Combined_Response'] = combined_response
    
    return combined_data

def main():
    # File paths
    image_path = "/media/al/Extreme SSD/20250425_spectra_2/hamamatsu_spectral_response.png"
    transmission_file = "/media/al/Extreme SSD/20250425_spectra_2/transmission_calibration.txt"
    output_file = "/media/al/Extreme SSD/20250425_spectra_2/sensor_calibrated_transmission.txt"
    
    try:
        print("Extracting sensor response curve from image...")
        sensor_wavelengths, sensor_efficiencies, extracted_points, original_img, mask = process_sensor_response(image_path)
        
        print("Combining with transmission data...")
        combined_data = combine_transmission_and_sensor_response(
            transmission_file, sensor_wavelengths, sensor_efficiencies)
        
        # Save the combined data
        with open(output_file, 'w') as f:
            f.write("# Sensor-Calibrated Transmission Data\n")
            f.write("# Combined transmission and Hamamatsu sensor response\n")
            f.write("# Source transmission: transmission_calibration.txt\n")
            f.write("# Source sensor response: hamamatsu_spectral_response.png\n")
            f.write("# Format: Spectrum_Number, Filename, Peak_Wavelength(nm), Max_Normalized_Transmission, Sensor_Response(%), Combined_Response\n")
            f.write("#" + "="*100 + "\n")
            
            for _, row in combined_data.iterrows():
                f.write(f"{int(row['Spectrum_Number'])}, {row['Filename']}, {row['Peak_Wavelength']:.2f}, "
                       f"{row['Max_Normalized_Transmission']:.6f}, {row['Sensor_Response']:.2f}, "
                       f"{row['Combined_Response']:.6f}\n")
        
        print(f"Sensor-calibrated data saved to: {output_file}")
        
        # Create debugging visualization with 6 subplots
        plt.figure(figsize=(16, 12))
        
        # 1. Original image
        plt.subplot(3, 2, 1)
        plt.imshow(original_img)
        plt.title('Original Spectral Response Image')
        plt.axis('off')
        
        # 2. Blue mask
        plt.subplot(3, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Blue Color Detection Mask')
        plt.axis('off')
        
        # 3. Original image with extracted points overlaid
        plt.subplot(3, 2, 3)
        plt.imshow(original_img)
        plt.plot(extracted_points[:, 0], extracted_points[:, 1], 'r.', markersize=1, alpha=0.7)
        plt.title('Original Image + Extracted Points')
        plt.axis('off')
        
        # 4. Extracted sensor response curve
        plt.subplot(3, 2, 4)
        plt.plot(sensor_wavelengths, sensor_efficiencies, 'b-', linewidth=2)
        plt.title('Extracted Sensor Response')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Quantum Efficiency (%)')
        plt.grid(True)
        plt.xlim(400, 1000)
        plt.ylim(0, 100)
        
        # 5. Original transmission data vs sensor response
        plt.subplot(3, 2, 5)
        plt.plot(combined_data['Peak_Wavelength'], combined_data['Max_Normalized_Transmission'], 'g-o', label='Transmission')
        plt.plot(combined_data['Peak_Wavelength'], combined_data['Sensor_Response']/100, 'r-s', label='Sensor Response (normalized)')
        plt.title('Transmission vs Sensor Response')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Response')
        plt.legend()
        plt.grid(True)
        
        # 6. Combined response
        plt.subplot(3, 2, 6)
        plt.plot(combined_data['Peak_Wavelength'], combined_data['Combined_Response'], 'm-o')
        plt.title('Combined Response (Transmission × Sensor)')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Combined Response')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('/media/al/Extreme SSD/20250425_spectra_2/calibration_debug_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print some debug info
        print(f"Extracted {len(extracted_points)} points from the curve")
        print(f"Wavelength range: {sensor_wavelengths.min():.1f} - {sensor_wavelengths.max():.1f} nm")
        print(f"Efficiency range: {sensor_efficiencies.min():.1f} - {sensor_efficiencies.max():.1f} %")
        
    except Exception as e:
        print(f"Error: {e}")
        print("You may need to adjust the blue color detection parameters or graph bounds.")

if __name__ == "__main__":
    main()