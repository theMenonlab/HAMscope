import os
import csv
import numpy as np
import tifffile
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Hard-coded input and output directories
INPUT_DIRS = [
    '/media/al/Extreme SSD/20250603_suberin/ham_aligned'
]
OUTPUT_DIR = "/media/al/Extreme SSD/20250603_suberin"
PROCESSED_LOG = os.path.join(OUTPUT_DIR, "processed_directories.json")
CSV_FILE = "/media/al/Extreme SSD/20250603_suberin/ham_aligned_intensity_report.csv"

# Wavelength configuration (700nm to 450nm, 30 channels)
START_WAVELENGTH = 700
END_WAVELENGTH = 450
NUM_CHANNELS = 30

def calculate_average_intensity_tiff_stack(image_path):
    """Opens a TIFF stack and calculates the average pixel intensity for each channel/slice."""
    try:
        tiff_stack = tifffile.imread(image_path)
        
        if tiff_stack.ndim == 2:  # Single slice, single channel
            return [float(np.mean(tiff_stack))]
            
        elif tiff_stack.ndim == 3:  # Multiple slices or channels
            if tiff_stack.shape[0] == 60:  # Single slice, multiple channels
                tiff_stack = tiff_stack[:30, :, :]  # Keep only the first 30 slices
            return [float(np.mean(tiff_stack[i])) for i in range(tiff_stack.shape[0])]
            
        elif tiff_stack.ndim == 4:  # Multiple slices and channels
            return [float(np.mean(tiff_stack[:, c, :, :])) for c in range(tiff_stack.shape[1])]
        
        print(f"Warning: Image '{os.path.basename(image_path)}' has unsupported dimensions: {tiff_stack.shape}")
        return None

    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {e}")
        return None

def load_processed_dirs():
    """Load the record of processed directories."""
    if os.path.exists(PROCESSED_LOG):
        try:
            with open(PROCESSED_LOG, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading processed log: {e}")
    return {}

def save_processed_dirs(processed_dirs):
    """Save the updated record of processed directories."""
    os.makedirs(os.path.dirname(PROCESSED_LOG), exist_ok=True)
    try:
        with open(PROCESSED_LOG, 'w') as f:
            json.dump(processed_dirs, f, indent=2)
    except Exception as e:
        print(f"Error saving processed log: {e}")

def process_directory(input_dir, output_csv_file):
    """Process all TIFF files in a directory and save results to CSV."""
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return False

    results_dict = {}
    max_channels = 0

    print(f"Scanning directory: {input_dir}")
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.tif', '.tiff')):
                input_image_path = os.path.join(root, filename)
                relative_path = os.path.relpath(input_image_path, input_dir).replace(os.path.sep, '/')

                print(f"Processing: {relative_path}")
                avg_intensities = calculate_average_intensity_tiff_stack(input_image_path)

                if avg_intensities:
                    results_dict[relative_path] = avg_intensities
                    max_channels = max(max_channels, len(avg_intensities))
                else:
                    print(f"Skipped: {relative_path}")

    if not results_dict:
        print("No TIFF images processed successfully.")
        return False

    try:
        os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
        
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header row
            header = ['RelativePath'] + [f'Channel{i+1}_Intensity' for i in range(max_channels)]
            writer.writerow(header)
            
            # Data rows
            for path, intensities in results_dict.items():
                row = [path] + intensities + [''] * (max_channels - len(intensities))
                writer.writerow(row)

        print(f"Results saved to: {output_csv_file}")
        return True

    except Exception as e:
        print(f"Error writing CSV file: {e}")
        return False

def create_wavelength_array():
    """Create wavelength array from 700nm to 450nm"""
    return np.linspace(START_WAVELENGTH, END_WAVELENGTH, NUM_CHANNELS)

def load_and_process_data(csv_file):
    """Load CSV data and process for spectrogram"""
    df = pd.read_csv(csv_file)
    
    # Extract intensity columns (skip RelativePath column)
    intensity_cols = [col for col in df.columns if 'Channel' in col and 'Intensity' in col]
    
    # Get only the first 30 channels to match wavelength range
    intensity_cols = intensity_cols[:NUM_CHANNELS]
    
    # Extract intensity data
    intensity_data = df[intensity_cols].values
    
    # Remove rows with NaN values
    intensity_data = intensity_data[~np.isnan(intensity_data).any(axis=1)]
    
    return intensity_data, df['RelativePath'].values

def create_publication_spectrogram(intensity_data, wavelengths, output_path):
    """Create a publication-ready spectrum plot"""
    
    # Set up the plot with publication styling - smaller figure size like the reference
    plt.figure(figsize=(6, 4))
    
    # Average spectrum plot - divide by 2^16 for better viewing
    mean_spectrum = np.mean(intensity_data, axis=0) / (2**16)
    std_spectrum = np.std(intensity_data, axis=0) / (2**16)
    
    # Plot with thick line to match reference style
    plt.plot(wavelengths, mean_spectrum, 'darkorange', linewidth=3, label='Mean Spectrum')
    plt.fill_between(wavelengths, 
                    mean_spectrum - std_spectrum, 
                    mean_spectrum + std_spectrum, 
                    alpha=0.3, color='darkorange', label='±1 STD')
    
    # Match font styling from reference plot
    plt.xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
    plt.ylabel('Intensity (AU)', fontsize=14, fontweight='bold')
    
    # Optimize legend to match reference style
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='upper right')
    
    # Set axis limits and customize ticks to match reference
    plt.xlim(450, 700)
    plt.ylim(0, np.max(mean_spectrum + std_spectrum) * 1.1)
    
    # Reduce number of ticks for cleaner look, match reference spacing
    plt.xticks(np.arange(450, 701, 50), fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    
    # Make tick marks more prominent to match reference
    plt.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)
    
    # Remove grid to reduce visual clutter (commented out like in reference)
    # plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Save in multiple formats for publication
    plt.savefig(f"{output_path}_spectrum.pdf", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return mean_spectrum, std_spectrum

def create_summary_statistics(intensity_data, wavelengths, output_path):
    """Create summary statistics table"""
    stats_data = []
    
    for i, wl in enumerate(wavelengths):
        channel_data = intensity_data[:, i]
        stats_data.append({
            'Wavelength_nm': f"{wl:.1f}",
            'Mean_Intensity': f"{np.mean(channel_data):.3f}",
            'Std_Intensity': f"{np.std(channel_data):.3f}",
            'Min_Intensity': f"{np.min(channel_data):.3f}",
            'Max_Intensity': f"{np.max(channel_data):.3f}",
            'Samples': len(channel_data)
        })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(f"{output_path}_spectral_statistics.csv", index=False)
    
    print(f"Summary statistics saved to: {output_path}_spectral_statistics.csv")
    return stats_df

def main():
    processed_dirs = load_processed_dirs()
    
    for input_dir in INPUT_DIRS:
        # Generate output filename based on input directory name
        dir_name = os.path.basename(os.path.normpath(input_dir))
        output_csv_file = os.path.join(OUTPUT_DIR, f"{dir_name}_intensity_report.csv")
        
        print(f"\n{'='*50}\nProcessing: {input_dir}")
        
        # Check if directory was already processed
        if input_dir in processed_dirs:
            print(f"Already processed on {processed_dirs[input_dir]['date']}")
            print(f"Output file: {processed_dirs[input_dir]['output_file']}")
            continue
        
        # Process directory
        if process_directory(input_dir, output_csv_file):
            # Record successful processing
            processed_dirs[input_dir] = {
                "output_file": output_csv_file,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            save_processed_dirs(processed_dirs)
    
    print("\nAll directories processed.")
    
    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file not found: {CSV_FILE}")
        return
    
    print("Loading and processing data...")
    intensity_data, sample_names = load_and_process_data(CSV_FILE)
    
    print(f"Loaded {intensity_data.shape[0]} samples with {intensity_data.shape[1]} channels")
    
    # Create wavelength array
    wavelengths = create_wavelength_array()
    
    # Create output path
    output_base = os.path.join(OUTPUT_DIR, "spectral_analysis")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Creating spectrum plot...")
    mean_spectrum, std_spectrum = create_publication_spectrogram(
        intensity_data, wavelengths, output_base
    )
    
    print("Creating summary statistics...")
    stats_df = create_summary_statistics(intensity_data, wavelengths, output_base)
    
    print(f"\nAnalysis complete!")
    print(f"- Spectrum plot saved as: {output_base}_spectrum.png/pdf/svg")
    print(f"- Statistics saved as: {output_base}_spectral_statistics.csv")
    print(f"- Total samples analyzed: {intensity_data.shape[0]}")
    print(f"- Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")

if __name__ == "__main__":
    main()