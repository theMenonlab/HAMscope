import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Constants
PIXEL_SIZE_MICRONS = 1.67  # Your camera's pixel size (µm/px)
DISTANCE_TO_SCREEN_MM = 175.0  # Updated to your 175mm distance

def find_fwhm(x, y):
    """
    Calculate Full Width at Half Maximum (FWHM) of a peak
    """
    # Find the maximum value and its position
    max_value = np.max(y)
    max_index = np.argmax(y)
    
    # Calculate half maximum
    half_max = max_value / 2
    
    # Find indices where the signal crosses half maximum
    # Look for crossings on both sides of the peak
    left_indices = np.where(y[:max_index] <= half_max)[0]
    right_indices = np.where(y[max_index:] <= half_max)[0]
    
    if len(left_indices) == 0 or len(right_indices) == 0:
        return None, None, None
    
    # Get the crossing points (interpolate for sub-pixel accuracy)
    left_idx = left_indices[-1] if len(left_indices) > 0 else 0
    right_idx = right_indices[0] + max_index if len(right_indices) > 0 else len(y) - 1
    
    # Linear interpolation for more accurate crossing points
    if left_idx < len(y) - 1:
        left_cross = left_idx + (half_max - y[left_idx]) / (y[left_idx + 1] - y[left_idx])
    else:
        left_cross = left_idx
        
    if right_idx > 0:
        right_cross = right_idx - (half_max - y[right_idx]) / (y[right_idx - 1] - y[right_idx])
    else:
        right_cross = right_idx
    
    # Calculate FWHM
    fwhm = abs(right_cross - left_cross)
    
    return fwhm, left_cross, right_cross

def analyze_single_image(image_path):
    """
    Analyze a single image and return FWHM in pixels
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Convert to grayscale for intensity analysis
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    height, width = gray_image.shape

    # Subtract background
    background = np.mean(np.mean(gray_image[:, -200:], axis=0))
    gray_image = gray_image - background
    #print(f"Background intensity subtracted: {background:.2f}")
    
    # Calculate average intensity across all rows
    average_line_intensity = np.mean(gray_image, axis=0)
    
    # Calculate FWHM
    fwhm, left_cross, right_cross = find_fwhm(np.arange(width), average_line_intensity)
    
    return fwhm

def analyze_grayscale_image(gray_image):
    """
    Analyze a grayscale image array and return both horizontal and vertical FWHM in pixels
    """
    # Get image dimensions
    height, width = gray_image.shape
    
    # Subtract background
    background = np.mean(np.mean(gray_image[:, -200:], axis=0))
    gray_image = gray_image - background
    
    # Average across rows to get horizontal profile
    horizontal_profile = np.mean(gray_image, axis=0)
    # Average across columns to get vertical profile  
    vertical_profile = np.mean(gray_image, axis=1)
    
    # Calculate FWHM for both directions
    fwhm_horizontal, _, _ = find_fwhm(np.arange(width), horizontal_profile)
    fwhm_vertical, _, _ = find_fwhm(np.arange(height), vertical_profile)
    
    # Print both values
    print(f"    Horizontal FWHM: {fwhm_horizontal:.2f} pixels")
    print(f"    Vertical FWHM: {fwhm_vertical:.2f} pixels")
    
    # Return both values as a tuple
    return fwhm_horizontal, fwhm_vertical

def create_average_images():
    """
    Create grayscale and color average images from all narrow band images
    """
    # Hard-coded directory paths
    base_dir = "/media/al/Extreme SSD/20250618_diffuser_analysis"
    diffuser_dir = os.path.join(base_dir, "diffuser")
    no_diffuser_dir = os.path.join(base_dir, "no_diffuser")
    
    # Generate wavelengths from 450 to 700 with interval of 10
    wavelengths = list(range(450, 701, 10))
    
    # Storage for all images
    diffuser_images = []
    no_diffuser_images = []
    valid_wavelengths = []
    
    print("Loading images for averaging...")
    
    for wavelength in wavelengths:
        filename = f"{wavelength}.png"
        
        diffuser_path = os.path.join(diffuser_dir, filename)
        no_diffuser_path = os.path.join(no_diffuser_dir, filename)
        
        if os.path.exists(diffuser_path) and os.path.exists(no_diffuser_path):
            # Load diffuser image
            diffuser_img = cv2.imread(diffuser_path)
            no_diffuser_img = cv2.imread(no_diffuser_path)
            
            if diffuser_img is not None and no_diffuser_img is not None:
                diffuser_images.append(diffuser_img)
                no_diffuser_images.append(no_diffuser_img)
                valid_wavelengths.append(wavelength)
                print(f"Loaded {wavelength}nm images")
    
    if len(diffuser_images) == 0:
        print("No valid image pairs found!")
        return
    
    # Create average images
    print("Creating average images...")
    
    # Color averages
    diffuser_avg_color = np.mean(diffuser_images, axis=0).astype(np.uint8)
    no_diffuser_avg_color = np.mean(no_diffuser_images, axis=0).astype(np.uint8)
    
    # Grayscale averages
    diffuser_avg_gray = cv2.cvtColor(diffuser_avg_color, cv2.COLOR_BGR2GRAY)
    no_diffuser_avg_gray = cv2.cvtColor(no_diffuser_avg_color, cv2.COLOR_BGR2GRAY)
    
    # Analyze grayscale averages
    print("Analyzing average images...")
    print("Diffuser average:")
    diffuser_avg_fwhm_h, diffuser_avg_fwhm_v = analyze_grayscale_image(diffuser_avg_gray.copy())
    print("No diffuser average:")
    no_diffuser_avg_fwhm_h, no_diffuser_avg_fwhm_v = analyze_grayscale_image(no_diffuser_avg_gray.copy())
    
    # Calculate diffusion angle from averages (using both horizontal and vertical FWHM)
    if diffuser_avg_fwhm_h is not None and no_diffuser_avg_fwhm_h is not None and diffuser_avg_fwhm_v is not None and no_diffuser_avg_fwhm_v is not None:
        # Convert to mm - Horizontal
        diffuser_avg_fwhm_h_mm = diffuser_avg_fwhm_h * PIXEL_SIZE_MICRONS / 1000
        no_diffuser_avg_fwhm_h_mm = no_diffuser_avg_fwhm_h * PIXEL_SIZE_MICRONS / 1000
        
        # Convert to mm - Vertical
        diffuser_avg_fwhm_v_mm = diffuser_avg_fwhm_v * PIXEL_SIZE_MICRONS / 1000
        no_diffuser_avg_fwhm_v_mm = no_diffuser_avg_fwhm_v * PIXEL_SIZE_MICRONS / 1000
        
        # Calculate diffusion angles - Horizontal
        # Standard approach: half-angle divergence
        diffuser_half_angle_h = np.arctan((diffuser_avg_fwhm_h_mm / 2) / DISTANCE_TO_SCREEN_MM)
        no_diffuser_half_angle_h = np.arctan((no_diffuser_avg_fwhm_h_mm / 2) / DISTANCE_TO_SCREEN_MM)
        diffusion_angle_h_rad = diffuser_half_angle_h - no_diffuser_half_angle_h
        diffusion_angle_h_deg = np.degrees(diffusion_angle_h_rad)
        
        # Calculate diffusion angles - Vertical
        diffuser_half_angle_v = np.arctan((diffuser_avg_fwhm_v_mm / 2) / DISTANCE_TO_SCREEN_MM)
        no_diffuser_half_angle_v = np.arctan((no_diffuser_avg_fwhm_v_mm / 2) / DISTANCE_TO_SCREEN_MM)
        diffusion_angle_v_rad = diffuser_half_angle_v - no_diffuser_half_angle_v
        diffusion_angle_v_deg = np.degrees(diffusion_angle_v_rad)
        
        print("\n" + "="*60)
        print("AVERAGE IMAGES ANALYSIS")
        print("="*60)
        print(f"Number of images averaged: {len(diffuser_images)}")
        print(f"Wavelength range: {min(valid_wavelengths)}-{max(valid_wavelengths)} nm")
        print(f"\nFWHM Results from Average Images:")
        print(f"  Diffuser average (H): {diffuser_avg_fwhm_h:.2f} pixels ({diffuser_avg_fwhm_h_mm:.3f} mm)")
        print(f"  Diffuser average (V): {diffuser_avg_fwhm_v:.2f} pixels ({diffuser_avg_fwhm_v_mm:.3f} mm)")
        print(f"  No diffuser average (H): {no_diffuser_avg_fwhm_h:.2f} pixels ({no_diffuser_avg_fwhm_h_mm:.3f} mm)")
        print(f"  No diffuser average (V): {no_diffuser_avg_fwhm_v:.2f} pixels ({no_diffuser_avg_fwhm_v_mm:.3f} mm)")
        print(f"\nDiffusion Angles:")
        print(f"  Horizontal: {diffusion_angle_h_deg:.5f}°")
        print(f"  Vertical: {diffusion_angle_v_deg:.5f}°")
        
        # Update return statement to include both angles
        return (diffuser_avg_fwhm_h, no_diffuser_avg_fwhm_h, diffusion_angle_h_deg, diffusion_angle_v_deg)
    
    # Display the images
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Convert BGR to RGB for matplotlib
    diffuser_avg_rgb = cv2.cvtColor(diffuser_avg_color, cv2.COLOR_BGR2RGB)
    no_diffuser_avg_rgb = cv2.cvtColor(no_diffuser_avg_color, cv2.COLOR_BGR2RGB)
    
    # Plot color averages
    axes[0, 0].imshow(diffuser_avg_rgb)
    axes[0, 0].set_title('Diffuser - Color Average')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(no_diffuser_avg_rgb)
    axes[0, 1].set_title('No Diffuser - Color Average')
    axes[0, 1].axis('off')
    
    # Plot grayscale averages
    axes[1, 0].imshow(diffuser_avg_gray, cmap='gray')
    axes[1, 0].set_title(f'Diffuser - Grayscale Average\nFWHM: {diffuser_avg_fwhm_h:.1f} px')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(no_diffuser_avg_gray, cmap='gray')
    axes[1, 1].set_title(f'No Diffuser - Grayscale Average\nFWHM: {no_diffuser_avg_fwhm_h:.1f} px')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save average images
    cv2.imwrite('diffuser_average_color.png', diffuser_avg_color)
    cv2.imwrite('no_diffuser_average_color.png', no_diffuser_avg_color)
    cv2.imwrite('diffuser_average_gray.png', diffuser_avg_gray)
    cv2.imwrite('no_diffuser_average_gray.png', no_diffuser_avg_gray)
    
    print(f"\nAverage images saved as:")
    print(f"  diffuser_average_color.png")
    print(f"  no_diffuser_average_color.png")
    print(f"  diffuser_average_gray.png")
    print(f"  no_diffuser_average_gray.png")
    
    return (diffuser_avg_fwhm_h, no_diffuser_avg_fwhm_h, diffusion_angle_deg if 'diffusion_angle_deg' in locals() else None)

def batch_analyze_directories():
    """
    Analyze all images in both diffuser and no_diffuser directories
    """
    # Hard-coded directory paths
    base_dir = "/media/al/Extreme SSD/20250618_diffuser_analysis"
    diffuser_dir = os.path.join(base_dir, "diffuser")
    no_diffuser_dir = os.path.join(base_dir, "no_diffuser")
    
    # Generate wavelengths from 450 to 700 with interval of 10
    wavelengths = list(range(450, 701, 10))
    
    # Storage for results
    diffuser_fwhm = []
    no_diffuser_fwhm = []
    valid_wavelengths = []
    
    print("Analyzing images...")
    
    for wavelength in wavelengths:
        filename = f"{wavelength}.png"
        
        # Analyze diffuser image
        diffuser_path = os.path.join(diffuser_dir, filename)
        no_diffuser_path = os.path.join(no_diffuser_dir, filename)
        
        diffuser_result = None
        no_diffuser_result = None
        
        if os.path.exists(diffuser_path):
            diffuser_result = analyze_single_image(diffuser_path)
            if diffuser_result is not None:
                print(f"Wavelength {wavelength}nm (diffuser): FWHM = {diffuser_result:.2f} pixels")
        
        if os.path.exists(no_diffuser_path):
            no_diffuser_result = analyze_single_image(no_diffuser_path)
            if no_diffuser_result is not None:
                print(f"Wavelength {wavelength}nm (no diffuser): FWHM = {no_diffuser_result:.2f} pixels")
        
        # Only include wavelengths where both measurements are successful
        if diffuser_result is not None and no_diffuser_result is not None:
            valid_wavelengths.append(wavelength)
            diffuser_fwhm.append(diffuser_result)
            no_diffuser_fwhm.append(no_diffuser_result)
    
    # Convert to numpy arrays for easier manipulation
    valid_wavelengths = np.array(valid_wavelengths)
    diffuser_fwhm = np.array(diffuser_fwhm)
    no_diffuser_fwhm = np.array(no_diffuser_fwhm)
    
    # Convert FWHM from pixels to mm
    diffuser_fwhm_mm = diffuser_fwhm * PIXEL_SIZE_MICRONS / 1000
    no_diffuser_fwhm_mm = no_diffuser_fwhm * PIXEL_SIZE_MICRONS / 1000
    
    # Calculate diffusion angles in radians then convert to degrees
    diffusion_angle_rad = np.arctan(diffuser_fwhm_mm / DISTANCE_TO_SCREEN_MM) - \
                         np.arctan(no_diffuser_fwhm_mm / DISTANCE_TO_SCREEN_MM)
    diffusion_angle_deg = np.degrees(diffusion_angle_rad)
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: FWHM in pixels vs wavelength
    ax1.plot(valid_wavelengths, diffuser_fwhm, 'b-o', label='With Diffuser', markersize=4)
    ax1.plot(valid_wavelengths, no_diffuser_fwhm, 'r-o', label='No Diffuser', markersize=4)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('FWHM (pixels)')
    ax1.set_title('FWHM vs Wavelength (pixels)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: FWHM in mm vs wavelength
    ax2.plot(valid_wavelengths, diffuser_fwhm_mm, 'b-o', label='With Diffuser', markersize=4)
    ax2.plot(valid_wavelengths, no_diffuser_fwhm_mm, 'r-o', label='No Diffuser', markersize=4)
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('FWHM (mm)')
    ax2.set_title('FWHM vs Wavelength (calibrated)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Diffusion angle in radians vs wavelength
    ax3.plot(valid_wavelengths, diffusion_angle_rad, 'g-o', markersize=4)
    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel('Diffusion Angle (radians)')
    ax3.set_title('Diffusion Angle vs Wavelength')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Diffusion angle in degrees vs wavelength
    ax4.plot(valid_wavelengths, diffusion_angle_deg, 'purple', marker='o', markersize=4)
    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_ylabel('Diffusion Angle (degrees)')
    ax4.set_title('Diffusion Angle vs Wavelength (degrees)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("INDIVIDUAL IMAGES ANALYSIS")
    print("="*50)
    print(f"Number of wavelengths analyzed: {len(valid_wavelengths)}")
    print(f"Wavelength range: {valid_wavelengths.min()}-{valid_wavelengths.max()} nm")
    print(f"Pixel size: {PIXEL_SIZE_MICRONS} µm")
    print(f"Distance to screen: {DISTANCE_TO_SCREEN_MM} mm")
    print("\nFWHM Statistics (mm):")
    print(f"  With diffuser - Mean: {diffuser_fwhm_mm.mean():.3f}, Std: {diffuser_fwhm_mm.std():.3f}")
    print(f"  No diffuser - Mean: {no_diffuser_fwhm_mm.mean():.3f}, Std: {no_diffuser_fwhm_mm.std():.3f}")
    print(f"\nDiffusion angle statistics:")
    print(f"  Mean: {diffusion_angle_deg.mean():.3f}°, Std: {diffusion_angle_deg.std():.3f}°")
    print(f"  Range: {diffusion_angle_deg.min():.3f}° to {diffusion_angle_deg.max():.3f}°")
    
    # Save results to file
    results_file = "diffusion_analysis_results.txt"
    with open(results_file, 'w') as f:
        f.write("Wavelength(nm)\tDiffuser_FWHM(px)\tNo_Diffuser_FWHM(px)\tDiffuser_FWHM(mm)\tNo_Diffuser_FWHM(mm)\tDiffusion_Angle(deg)\n")
        for i in range(len(valid_wavelengths)):
            f.write(f"{valid_wavelengths[i]}\t{diffuser_fwhm[i]:.3f}\t{no_diffuser_fwhm[i]:.3f}\t{diffuser_fwhm_mm[i]:.3f}\t{no_diffuser_fwhm_mm[i]:.3f}\t{diffusion_angle_deg[i]:.3f}\n")
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    # First analyze individual images
    batch_analyze_directories()
    
    # Then create and analyze average images
    print("\n" + "="*60)
    print("CREATING AVERAGE IMAGES")
    print("="*60)
    create_average_images()