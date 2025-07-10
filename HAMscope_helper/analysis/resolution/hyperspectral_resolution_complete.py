import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage.measure import profile_line

# Image paths
img_paths = [
    '/media/al/Extreme SSD/20250701_usaf/20250708_results/20250425_0gan_single_reg_hs_usaf/test_latest/images/hs_gen_0.tif',
    '/media/al/Extreme SSD/20250701_usaf/20250708_deconvolution_results/hs_gen_0_deconvolved.tif',
    '/media/al/Extreme SSD/20250701_usaf/20250708_results/20250425_0gan_single_reg_hs_usaf/test_latest/images/hs_gen_5.tif',
    '/media/al/Extreme SSD/20250701_usaf/20250708_deconvolution_results/hs_gen_5_deconvolved.tif',
    '/media/al/Extreme SSD/20250701_usaf/20250708_results/20250425_0gan_single_reg_hs_usaf/test_latest/images/hs_gen_7.tif',
    '/media/al/Extreme SSD/20250701_usaf/20250708_deconvolution_results/hs_gen_7_deconvolved.tif',
]

# Combined coordinate sets for comprehensive resolution analysis
# Group 6 elements (better resolution, tested first)
group6_coords = {
    'vertical': [
        [215, 203, 242, 203],  # G6E1 Vertical bars, 7.81 micron
        [252, 290, 276, 290],  # G6E2 Vertical bars, 6.91 micron
        [258, 263, 280, 263],  # G6E3 Vertical bars, 6.20 micron
        [262, 238, 282, 238],  # G6E4 Vertical bars, 5.52 micron
        [265, 218, 284, 218],  # G6E5 Vertical bars, 4.92 micron
    ],
    'horizontal': [
        [196, 194, 196, 222],  # G6E1 horizontal bars, 7.81 micron
        [285, 283, 285, 305],  # G6E2 horizontal bars, 6.91 micron
        [286, 256, 286, 281],  # G6E3 horizontal bars, 6.20 micron
        [286, 234, 286, 254],  # G6E4 horizontal bars, 5.52 micron
        [287, 212, 287, 231],  # G6E5 horizontal bars, 4.92 micron
    ]
}

# Group 5 elements (fallback when Group 6 not resolved)
group5_coords = {
    'horizontal': [
        [228, 144, 228, 183],  # G5E6 horizontal bars, 8.77 micron
        [227, 183, 227, 217],  # G5E5 horizontal bars, 9.84 micron
        [226, 223, 226, 263],  # G5E4 horizontal bars, 11.05 micron
        [226, 263, 228, 313],  # G5E3 horizontal bars, 12.4 micron
    ],
    'vertical': [
        [247, 156, 277, 156],  # G5E6 vertical bars, 8.77 micron
        [251, 194, 285, 194],  # G5E5 vertical bars, 9.84 micron
        [247, 235, 292, 235],  # G5E4 vertical bars, 11.05 micron
        [252, 283, 296, 283],  # G5E3 vertical bars, 12.4 micron
    ]
}

# Group 5 E1 and E2 elements from img7 (larger features for better visibility)
img7_coords = {
    'horizontal': [
        [230, 230, 230, 280],  # G5E2 horizontal bars, 13.92 micron
        [230, 290, 230, 345],  # G5E1 horizontal bars, 15.63 micron
    ],
    'vertical': [
        [260, 250, 310, 250],  # G5E2 vertical bars, 13.92 micron
        [260, 310, 320, 310],  # G5E1 vertical bars, 15.63 micron
    ]
}

# Resolution values (microns) for each coordinate set
group6_resolutions = {
    'vertical': [7.81, 6.91, 6.20, 5.52, 4.92],
    'horizontal': [7.81, 6.91, 6.20, 5.52, 4.92]
}

group5_resolutions = {
    'horizontal': [8.77, 9.84, 11.05, 12.4, 13.92, 15.63],
    'vertical': [8.77, 9.84, 11.05, 12.4, 13.92, 15.63]
}

# Combined coordinates and resolutions for comprehensive analysis
# Use img7_coords for G5E1 and G5E2 (the largest features) and group5_coords for G5E3-G5E6
combined_coords = {
    'horizontal': group6_coords['horizontal'] + group5_coords['horizontal'][:4] + img7_coords['horizontal'],
    'vertical': group6_coords['vertical'] + group5_coords['vertical'][:4] + img7_coords['vertical']
}

combined_resolutions = {
    'horizontal': group6_resolutions['horizontal'] + group5_resolutions['horizontal'][:4] + [13.92, 15.63],
    'vertical': group6_resolutions['vertical'] + group5_resolutions['vertical'][:4] + [13.92, 15.63]
}

# Resolution threshold for resolved elements
RESOLUTION_THRESHOLD = 0.2

def calculate_contrast_from_profile(intensity_profile):
    """Calculate contrast from an intensity profile using the same method as auto_line_intensity_chart_2.py"""
    try:
        # Normalize the profile the same way as the original script
        avg_intensity_profiles = intensity_profile / np.max(intensity_profile) if np.max(intensity_profile) > 0 else intensity_profile
        
        # Find local maxima and minima using the exact same logic as the original
        values = []
        values_indices = []

        for i in range(len(avg_intensity_profiles)-2):
            if avg_intensity_profiles[i] < avg_intensity_profiles[i + 1] and avg_intensity_profiles[i + 1] > avg_intensity_profiles[i + 2]:
                values.append(avg_intensity_profiles[i+1])
                values_indices.append(i+1)
            if avg_intensity_profiles[i] > avg_intensity_profiles[i + 1] and avg_intensity_profiles[i + 1] < avg_intensity_profiles[i + 2]:
                values.append(avg_intensity_profiles[i+1])
                values_indices.append(i+1)
        
        if len(values) < 3:
            return 0.0
        
        # Use the exact same method as the original script to find the 3 largest values
        max_values_indices = np.argpartition(values, -3)[-3:]
        max_values_indices = np.sort(max_values_indices)

        max_3 = [values[max_values_indices[0]], values[max_values_indices[1]], values[max_values_indices[2]]]
        max_3_indices = [values_indices[max_values_indices[0]], values_indices[max_values_indices[1]], values_indices[max_values_indices[2]]]
        
        # Check if the max values are too close to each other (same as original)
        diff_1 = max_3_indices[1] - max_3_indices[0]
        diff_2 = max_3_indices[2] - max_3_indices[1]
        relative_diff = diff_1 / diff_2 if diff_2 != 0 else float('inf')
        
        if np.isnan(relative_diff) or relative_diff < 0.8 or relative_diff > 1.4:
            # If spacing is not regular, still calculate but note it
            pass
        
        # Find minima between maxima using the exact same method
        min_2 = [
            np.min(avg_intensity_profiles[max_3_indices[0]:max_3_indices[1]]),
            np.min(avg_intensity_profiles[max_3_indices[1]:max_3_indices[2]])
        ]
        
        # Calculate contrast using the exact same method as the original
        contrast = []
        contrast.append((max_3[0] - min_2[0]) / (max_3[0] + min_2[0]))
        contrast.append((max_3[1] - min_2[0]) / (max_3[1] + min_2[0]))
        contrast.append((max_3[1] - min_2[1]) / (max_3[1] + min_2[1]))
        contrast.append((max_3[2] - min_2[1]) / (max_3[2] + min_2[1]))
        
        return np.mean(contrast)
    
    except Exception as e:
        print(f"Error calculating contrast: {e}")
        return 0.0

def analyze_channel_resolution(img_array, line_coords, orientation='horizontal'):
    """Analyze resolution for a single channel using given line coordinates."""
    sum_intensity_profiles = None
    num_lines = 10  # Use 10 lines like the original script
    
    # Drawing lines and accumulating intensity profiles
    for i in range(num_lines):
        if orientation == 'horizontal':
            # For horizontal bars, use vertical lines (scanning across bars) - same as original when horizontal=True
            current_line_coords = (line_coords[0] + i, line_coords[1], line_coords[2] + i, line_coords[3])
        else:
            # For vertical bars, use horizontal lines (scanning across bars) - same as original when horizontal=False
            current_line_coords = (line_coords[0], line_coords[1] + i, line_coords[2], line_coords[3] + i)
        
        try:
            intensity_profile = profile_line(img_array, 
                                           (current_line_coords[1], current_line_coords[0]), 
                                           (current_line_coords[3], current_line_coords[2]))
            
            if sum_intensity_profiles is None:
                sum_intensity_profiles = intensity_profile
            else:
                sum_intensity_profiles += intensity_profile
        except Exception as e:
            print(f"Error in profile_line: {e}")
            continue
    
    if sum_intensity_profiles is None:
        return 0.0, np.array([])
    
    # Calculate average intensity profile - normalize like the original script
    # The original script divides by num_lines and then by 255 for normalization
    avg_intensity_profiles = sum_intensity_profiles / num_lines
    
    # Normalize to 0-1 range like the original script (which divides by 255)
    if avg_intensity_profiles.max() > 0:
        avg_intensity_profiles = avg_intensity_profiles / avg_intensity_profiles.max()
    
    # Calculate contrast
    contrast = calculate_contrast_from_profile(avg_intensity_profiles)
    
    return contrast, avg_intensity_profiles

def load_and_analyze_tiff_stack(img_path):
    """Load TIFF stack and analyze resolution for all channels using combined Group 5 and Group 6 elements."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {os.path.basename(img_path)}")
    
    # Determine image type
    is_deconvolved = 'deconvolved' in img_path.lower()
    img_type = 'Deconvolved' if is_deconvolved else 'Generated'
    
    print(f"Image type: {img_type}")
    print(f"Analysis: Combined Group 5 + Group 6 elements")
    
    # Load TIFF stack
    try:
        img_stack = tifffile.imread(img_path)
        print(f"TIFF stack shape: {img_stack.shape}")
    except Exception as e:
        print(f"Error loading TIFF: {e}")
        return None, None, None, None
    
    # Use combined coordinate sets for comprehensive analysis
    coords_dict = combined_coords
    resolutions_dict = combined_resolutions
    
    # Analyze first 30 channels
    num_channels = min(30, img_stack.shape[0])
    print(f"Analyzing first {num_channels} channels")
    
    results = {
        'horizontal': {'contrasts': [], 'profiles': [], 'resolutions': [], 'resolved_groups': []},
        'vertical': {'contrasts': [], 'profiles': [], 'resolutions': [], 'resolved_groups': []}
    }
    
    for channel in range(num_channels):
        print(f"Processing channel {channel}...", end=' ')
        
        channel_img = img_stack[channel]
        
        # Convert to float64 but maintain the original scale like the original script
        if channel_img.dtype != np.float64:
            channel_img = channel_img.astype(np.float64)
        
        # Analyze for both orientations
        for orientation in ['horizontal', 'vertical']:
            if orientation in coords_dict:
                coord_set = coords_dict[orientation]
                resolution_set = resolutions_dict[orientation]
                
                channel_contrasts = []
                channel_profiles = []
                
                # Test all coordinate sets for this orientation (Group 6 first, then Group 5)
                for coord_idx, line_coords in enumerate(coord_set):
                    try:
                        contrast, profile = analyze_channel_resolution(channel_img, line_coords, orientation)
                        channel_contrasts.append(contrast)
                        channel_profiles.append(profile)
                    except Exception as e:
                        print(f"Error with {orientation} coords {coord_idx}: {e}")
                        channel_contrasts.append(0.0)
                        channel_profiles.append(np.array([]))
                
                # Store results
                results[orientation]['contrasts'].append(channel_contrasts)
                results[orientation]['profiles'].append(channel_profiles)
                
                # Determine resolution based on threshold with group preference
                # First check Group 6 elements (indices 0-4), then Group 5 elements (indices 5-10)
                resolved_elements = []
                resolved_group = None
                
                # Check Group 6 first (better resolution)
                group6_count = len(group6_resolutions[orientation])
                for i in range(group6_count):
                    if i < len(channel_contrasts) and channel_contrasts[i] > RESOLUTION_THRESHOLD:
                        resolved_elements.append((resolution_set[i], i, channel_contrasts[i], 'Group 6'))
                        if resolved_group is None:
                            resolved_group = 'Group 6'
                
                # If no Group 6 elements resolved, check Group 5
                if not resolved_elements:
                    for i in range(group6_count, len(channel_contrasts)):
                        if channel_contrasts[i] > RESOLUTION_THRESHOLD:
                            resolved_elements.append((resolution_set[i], i, channel_contrasts[i], 'Group 5'))
                            if resolved_group is None:
                                resolved_group = 'Group 5'
                
                if resolved_elements:
                    # Best resolution is the smallest value among resolved elements
                    best_resolution = min(resolved_elements, key=lambda x: x[0])[0]
                    # Find which group provided the best resolution
                    best_element = min(resolved_elements, key=lambda x: x[0])
                    resolved_group = best_element[3]
                else:
                    # No elements resolved
                    best_resolution = float('inf')
                    resolved_group = 'None'
                
                results[orientation]['resolutions'].append(best_resolution)
                results[orientation]['resolved_groups'].append(resolved_group)
        
        print("✓")
    
    return results, img_stack, is_deconvolved, img_type

def create_resolution_plot(all_results, all_types):
    """Create resolution plot showing 4 lines: vertical/horizontal × deconvolved/generated combining both images."""
    
    if not all_results:
        print("No results to plot")
        return
    
    # Create single plot for resolution vs channel number
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Hyperspectral USAF Resolution Analysis (Combined Group 5 + Group 6)', fontsize=16, fontweight='bold')
    
    # Organize results by processing type and orientation
    combined_data = {
        'Generated': {'horizontal': [], 'vertical': []},
        'Deconvolved': {'horizontal': [], 'vertical': []}
    }
    
    # Collect all results by type and orientation
    for img_name, results in all_results.items():
        img_type = all_types[img_name]
        
        for orientation in ['horizontal', 'vertical']:
            if orientation in results and results[orientation]['resolutions']:
                resolutions = results[orientation]['resolutions']
                combined_data[img_type][orientation].append(resolutions)
    
    # Plot 4 lines: combine results across both images for each orientation/type
    colors = ['blue', 'red', 'green', 'orange']
    linestyles = ['-', '--']  # solid for horizontal, dashed for vertical
    
    line_idx = 0
    for proc_type in ['Generated', 'Deconvolved']:
        for orientation_idx, orientation in enumerate(['horizontal', 'vertical']):
            if combined_data[proc_type][orientation]:
                # Get all resolution arrays for this type/orientation combination
                all_resolutions = combined_data[proc_type][orientation]
                
                if all_resolutions:
                    # Take the best (minimum) resolution across both images for each channel
                    num_channels = len(all_resolutions[0])
                    channels = np.arange(num_channels)
                    best_resolutions = []
                    
                    for ch in range(num_channels):
                        channel_resolutions = []
                        for res_array in all_resolutions:
                            if ch < len(res_array):
                                channel_resolutions.append(res_array[ch])
                        
                        if channel_resolutions:
                            # Take the best (smallest) resolution across both images
                            finite_resolutions = [r for r in channel_resolutions if r != float('inf')]
                            if finite_resolutions:
                                best_resolutions.append(min(finite_resolutions))
                            else:
                                best_resolutions.append(float('inf'))
                        else:
                            best_resolutions.append(float('inf'))
                    
                    # Replace inf values with a large number for plotting
                    resolutions_plot = [r if r != float('inf') else 20 for r in best_resolutions]
                    
                    label = f"{orientation.capitalize()} {proc_type}"
                    linestyle = linestyles[orientation_idx]
                    
                    ax.plot(channels, resolutions_plot, marker='o', linewidth=2, 
                            markersize=4, label=label, color=colors[line_idx % len(colors)],
                            linestyle=linestyle)
                    
                    line_idx += 1
    
    ax.set_xlabel('Channel Number')
    ax.set_ylabel('Resolution (μm)')
    ax.set_title('Best Resolution Across Both Images (img_0 + img_5)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_publication_plot(all_results, all_types):
    """Create publication-ready plot showing 2 lines: Generated vs Deconvolved (averaged across orientations)."""
    
    if not all_results:
        print("No results to plot")
        return
    
    # Organize results by processing type
    combined_data = {
        'Generated': [],
        'Deconvolved': []
    }
    
    # Collect all results by type
    for img_name, results in all_results.items():
        img_type = all_types[img_name]
        
        # Average horizontal and vertical resolutions for each channel
        if 'horizontal' in results and 'vertical' in results:
            h_resolutions = results['horizontal']['resolutions']
            v_resolutions = results['vertical']['resolutions']
            
            if h_resolutions and v_resolutions:
                averaged_resolutions = []
                for h_res, v_res in zip(h_resolutions, v_resolutions):
                    finite_vals = [r for r in [h_res, v_res] if r != float('inf')]
                    if finite_vals:
                        averaged_resolutions.append(np.mean(finite_vals))
                    else:
                        averaged_resolutions.append(float('inf'))
                
                combined_data[img_type].append(averaged_resolutions)
    
    # Create publication-ready plot
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
    
    for idx, proc_type in enumerate(['Generated', 'Deconvolved']):
        if combined_data[proc_type]:
            all_resolutions = combined_data[proc_type]
            
            if all_resolutions:
                # Take the best (minimum) resolution across all images for each channel
                num_channels = len(all_resolutions[0])
                
                # Convert channel numbers to wavelengths (700 nm to 450 nm)
                # Channel 0 = 700 nm, Channel 29 = 450 nm
                wavelengths = np.linspace(700, 450, num_channels)
                
                best_resolutions = []
                
                for ch in range(num_channels):
                    channel_resolutions = []
                    for res_array in all_resolutions:
                        if ch < len(res_array):
                            channel_resolutions.append(res_array[ch])
                    
                    if channel_resolutions:
                        finite_resolutions = [r for r in channel_resolutions if r != float('inf')]
                        if finite_resolutions:
                            best_resolutions.append(min(finite_resolutions))
                        else:
                            best_resolutions.append(float('inf'))
                    else:
                        best_resolutions.append(float('inf'))
                
                # Replace inf values with a large number for plotting
                resolutions_plot = [r if r != float('inf') else 20 for r in best_resolutions]
                
                # Reverse the order so shorter wavelengths are on the left
                wavelengths_reversed = wavelengths[::-1]
                resolutions_plot_reversed = resolutions_plot[::-1]
                
                ax.plot(wavelengths_reversed, resolutions_plot_reversed, 'o-', linewidth=3, 
                        markersize=6, label=proc_type, color=colors[idx])
    
    ax.set_xlabel('Wavelength (nm)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Resolution (μm)', fontsize=16, fontweight='bold')
    ax.set_title('USAF Resolution Analysis', fontsize=18, fontweight='bold')
    ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Set y-axis to start from 0 for better visualization
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()
    
    # Reset matplotlib parameters
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Print average resolutions for publication summary
    print("\n" + "="*60)
    print("PUBLICATION FIGURE SUMMARY")
    print("="*60)
    
    for idx, proc_type in enumerate(['Generated', 'Deconvolved']):
        if combined_data[proc_type]:
            all_resolutions = combined_data[proc_type]
            
            if all_resolutions:
                # Calculate the same best resolutions as shown in the plot
                num_channels = len(all_resolutions[0])
                best_resolutions = []
                
                for ch in range(num_channels):
                    channel_resolutions = []
                    for res_array in all_resolutions:
                        if ch < len(res_array):
                            channel_resolutions.append(res_array[ch])
                    
                    if channel_resolutions:
                        finite_resolutions = [r for r in channel_resolutions if r != float('inf')]
                        if finite_resolutions:
                            best_resolutions.append(min(finite_resolutions))
                        else:
                            best_resolutions.append(float('inf'))
                    else:
                        best_resolutions.append(float('inf'))
                
                # Calculate average resolution (excluding unresolved channels)
                resolved_resolutions = [r for r in best_resolutions if r != float('inf')]
                if resolved_resolutions:
                    avg_resolution = np.mean(resolved_resolutions)
                    resolved_count = len(resolved_resolutions)
                    best_resolution = min(resolved_resolutions)
                    
                    print(f"\n{proc_type.upper()}:")
                    print(f"  Channels with resolved features: {resolved_count}/30")
                    print(f"  Best resolution achieved: {best_resolution:.2f} μm")
                    print(f"  Average resolution: {avg_resolution:.2f} μm")
                else:
                    print(f"\n{proc_type.upper()}:")
                    print(f"  No features resolved across all channels")
    
    print("="*60)

def print_simple_summary(all_results, all_types):
    """Print a simple summary of resolution analysis."""
    
    print("\n" + "="*60)
    print("RESOLUTION SUMMARY")
    print("="*60)
    
    for img_name, results in all_results.items():
        img_short = img_name.split('_')[0] + '_' + img_name.split('_')[1]
        img_type = all_types[img_name]
        
        print(f"\n{img_short.upper()} {img_type.upper()}:")
        
        # Get combined best resolutions
        if 'horizontal' in results and 'vertical' in results:
            h_resolutions = results['horizontal']['resolutions']
            v_resolutions = results['vertical']['resolutions']
            
            if h_resolutions and v_resolutions:
                combined_resolutions = []
                for h_res, v_res in zip(h_resolutions, v_resolutions):
                    if h_res == float('inf') and v_res == float('inf'):
                        combined_resolutions.append(float('inf'))
                    elif h_res == float('inf'):
                        combined_resolutions.append(v_res)
                    elif v_res == float('inf'):
                        combined_resolutions.append(h_res)
                    else:
                        combined_resolutions.append(min(h_res, v_res))
                
                resolved_count = len([r for r in combined_resolutions if r != float('inf')])
                best_res = min([r for r in combined_resolutions if r != float('inf')]) if resolved_count > 0 else float('inf')
                avg_res = np.mean([r for r in combined_resolutions if r != float('inf')]) if resolved_count > 0 else float('inf')
                
                print(f"  Resolved channels: {resolved_count}/30")
                print(f"  Best resolution: {best_res:.2f} μm" if best_res != float('inf') else "  Best resolution: No features resolved")
                print(f"  Average resolution: {avg_res:.2f} μm" if avg_res != float('inf') else "  Average resolution: No features resolved")

# Main execution
def main():
    print("="*80)
    print("HYPERSPECTRAL USAF RESOLUTION ANALYSIS")
    print("Combined Group 5 + Group 6 Elements")
    print("="*80)
    
    all_results = {}
    all_types = {}
    
    for img_path in img_paths:
        if not os.path.exists(img_path):
            print(f"Warning: File not found: {img_path}")
            continue
        
        results, img_stack, is_deconvolved, img_type = load_and_analyze_tiff_stack(img_path)
        
        if results is not None:
            img_name = os.path.basename(img_path)
            all_results[img_name] = results
            all_types[img_name] = img_type
    
    # Print simple summary
    print_simple_summary(all_results, all_types)
    
    # Create resolution plot
    if all_results:
        print("\nGenerating resolution plot...")
        create_resolution_plot(all_results, all_types)
        print("Generating publication plot...")
        create_publication_plot(all_results, all_types)
    else:
        print("No results to visualize.")

if __name__ == "__main__":
    main()
