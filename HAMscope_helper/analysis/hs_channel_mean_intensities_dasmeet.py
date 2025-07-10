import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Define file paths
files = {
    'Generated WT (30 channels)': '/media/al/Extreme SSD/20250425_results/dasmeet/wt/hs_gen_intensity_report.csv',
    'Generated PDC (30 channels)': '/media/al/Extreme SSD/20250425_results/dasmeet/mut/hs_gen_intensity_report.csv',
    'Real WT (6 channels)': '/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/datasets/20250410_dasmeet/wt/intensity_report.csv',
    'Real PDC (6 channels)': '/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/datasets/20250410_dasmeet/mut/intensity_report.csv'
}

# Define wavelengths
wavelengths_30 = np.linspace(700, 450, 30)  # 30 channels from 700nm to 450nm (backwards)
wavelengths_6 = np.array([400, 450, 500, 550, 600, 650])  # 6 specific wavelengths

def read_and_process_data(filepath, num_channels):
    """Read CSV file and extract intensity data"""
    try:
        # Try reading with different parameters to handle malformed files
        df = pd.read_csv(filepath, on_bad_lines='skip')
        print(f"Original shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Remove any empty rows
        df = df.dropna(how='all')
        
        # Extract intensity columns
        intensity_cols = [col for col in df.columns if 'Channel' in col and 'Intensity' in col]
        print(f"Found intensity columns: {intensity_cols}")
        
        if len(intensity_cols) == 0:
            print(f"No intensity columns found. All columns: {df.columns.tolist()}")
            return None
            
        if len(intensity_cols) != num_channels:
            print(f"Warning: Expected {num_channels} channels, found {len(intensity_cols)} in {filepath}")
        
        # Get intensity data (excluding any summary rows)
        intensity_data = df[intensity_cols].copy()
        
        # Convert to numeric, coercing errors to NaN
        for col in intensity_cols:
            intensity_data[col] = pd.to_numeric(intensity_data[col], errors='coerce')
        
        # Drop any rows with NaN values
        intensity_data = intensity_data.dropna()
        
        print(f"Final processed shape: {intensity_data.shape}")
        print(f"Sample of data:\n{intensity_data.head()}")
        
        return intensity_data
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        # Try alternative reading method
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            print(f"First few lines of file:")
            for i, line in enumerate(lines[:5]):
                print(f"Line {i}: {line.strip()}")
        except:
            pass
        return None

def normalize_to_450_650_region(means, wavelengths):
    """Normalize using only the 450-650nm region for better alignment"""
    # Convert to numpy arrays if not already
    wavelengths = np.array(wavelengths)
    means = np.array(means)
    
    # Find indices corresponding to 450-650nm range
    mask = (wavelengths >= 450) & (wavelengths <= 650)
    
    if np.any(mask):
        # Calculate mean only for the 450-650nm region
        region_mean = means[mask].mean()
        if region_mean > 0:
            return means / region_mean
    
    # Fallback to overall mean if region not found
    overall_mean = means.mean()
    return means / overall_mean if overall_mean > 0 else means

# Read all data
data_dict = {}
for name, filepath in files.items():
    print(f"\n--- Processing {name} ---")
    try:
        if '30 channels' in name:
            data_dict[name] = read_and_process_data(filepath, 30)
        else:
            data_dict[name] = read_and_process_data(filepath, 6)
        
        if data_dict[name] is not None:
            print(f"Successfully loaded {name}: {data_dict[name].shape}")
        else:
            print(f"Failed to load {name}")
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except Exception as e:
        print(f"Error loading {name}: {e}")

# Only proceed with plotting if we have valid data
valid_data = {k: v for k, v in data_dict.items() if v is not None and not v.empty}

if not valid_data:
    print("No valid data found to plot!")
else:
    # Create the plot matching the line scan style exactly
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Color scheme: same color for same mutation type
    colors = {
        'PDC': 'r',  # Red for mutant
        'WT': 'b'    # Blue for wild type
    }
    
    # Line styles: differentiate real vs generated
    line_styles = {
        'Real': '-',      # Solid line for real data
        'Generated': '--' # Dashed line for generated data
    }
    
    # Line widths to match your reference
    line_widths = {
        'Real': 3,        # Thicker for real data
        'Generated': 2    # Thinner for generated data
    }

    for name, data in valid_data.items():
        # Calculate mean for each channel
        means = data.mean()
        
        # Determine wavelengths based on dataset
        if '30 channels' in name:
            wavelengths = wavelengths_30
        else:
            wavelengths = wavelengths_6
        
        # Ensure we have the right number of wavelengths
        wavelengths = wavelengths[:len(means)]
        
        # Normalize using 450-650nm region
        normalized_means = normalize_to_450_650_region(means, wavelengths)
        
        # Determine color and line style based on name
        if 'PDC' in name:
            color = colors['PDC']
        else:
            color = colors['WT']
            
        if 'Real' in name:
            line_style = line_styles['Real']
            line_width = line_widths['Real']
        else:
            line_style = line_styles['Generated']
            line_width = line_widths['Generated']
        
        # Plot normalized mean
        ax.plot(wavelengths, normalized_means, 
                color=color, linestyle=line_style, linewidth=line_width,
                label=name)

    # Match the exact styling from your line scan plot
    ax.set_xlabel('Wavelength (nm)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Normalized Intensity', fontsize=18, fontweight='bold')
    ax.set_title('Hyperspectral Channel Intensities', fontsize=20, fontweight='bold', pad=15)
    ax.legend(fontsize=16, frameon=True, fancybox=True, shadow=True,
              loc='upper center', framealpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
    ax.grid(True, alpha=0.4, linewidth=1.5)
    for sp in ax.spines.values(): 
        sp.set_linewidth(2)
    
    plt.tight_layout()
    fig.savefig('hyperspectral_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Show statistics
    print("\nDataset Statistics:")
    for name, data in valid_data.items():
        print(f"\n{name}:")
        print(f"  Number of samples: {len(data)}")
        print(f"  Number of channels: {len(data.columns)}")
        print(f"  Overall mean intensity: {data.values.mean():.2f}")
        print(f"  Overall std intensity: {data.values.std():.2f}")

    print("Saved hyperspectral_comparison.png")