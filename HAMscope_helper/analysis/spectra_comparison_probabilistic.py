import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider
from skimage.metrics import structural_similarity as ssim
from RGB.HSI2RGB import HSI2RGB  # Assuming HSI2RGB is properly installed
import re

def natural_sort_key(s):
    """
    Key function for natural sorting.
    This will sort strings with numbers in a human-expected way.
    For example: ['file1.tif', 'file2.tif', 'file10.tif'] instead of
    ['file1.tif', 'file10.tif', 'file2.tif']
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

class AverageMetricsCalculator:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        
        # Get all .tif files and sort them naturally
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]
        self.image_files.sort(key=natural_sort_key)
        
        # Extract unique indices from the filenames
        indices = set()
        for filename in self.image_files:
            # Extract number from filenames like "cb_raw_X.tif" or "tl_gen_X.tif"
            match = re.search(r'(\d+)\.tif$', filename)
            if match:
                indices.add(int(match.group(1)))
        
        # Sort the indices naturally
        self.valid_indices = sorted(indices)
        
        self.ssim_3d_list = []
        self.ssim_2d_list = []
        self.mse_list = []
        self.mae_list = []
        self.threshold_mae_list = []
        self.sre_list = []
        self.sigma_mean_list = []  # Add new list for sigma means

        # Calculate and display the metrics
        self.calculate_metrics()

    def calculate_mse(self, image1, image2):
        """Calculate the Mean Squared Error (MSE) between two images."""
        return np.mean((image1 - image2) ** 2)

    def calculate_mae(self, image1, image2):
        """Calculate the Mean Absolute Error (MAE) between two images. This is MAE3D"""
        return np.mean(np.abs(image1 - image2))
    
    '''
    I(x,y,l) - G(x,y,l) = E(x,y,l).
    MAE3D = avg(E(x,y,l))
    MAE1D (x,y) = avg(E(x,y,l))_l
    SRE = avg(MAE1D)_(x,y)_all-images above threshold. (edited) 
    MAE2D(l) = avg(E(x,y,l))_(x,y)_all-images-above-threshold.
    Avg_MAE1D(x,y) = avg(MAE1D)_all-images above threshold.
    '''
        
    def calculate_MAE1D(self, image1, image2):
        """Calculate the Mean Absolute Error (MAE) between two images."""
        return np.mean(np.abs(image1 - image2), axis=0)
    
    def calculate_threshold_MAE(self, image1, image2, threshold=0.1):
        """
        Calculates mean absolute error of all spectral channels, for pixels
        whose intensities in image1 are >= threshold.

        image1, image2: np.ndarray of shape (spectral_channels, height, width)
        threshold: float, any pixel below this intensity in image1 is excluded
        Returns:
            A float representing mean absolute error over the masked pixels.
        """
        mask = image1 >= threshold  # Boolean mask of valid pixels
        E = image1 - image2  # Calculate error
        return np.mean(np.abs(E[mask]))

    def calculate_SRE(self, image1, image2, threshold=0.1):
        """
        Calculate Spectral Reconstruction Error (SRE).
        Excludes pixels in image1 whose intensity is below threshold.
        Returns:
            A single float with the SRE.
        """
        # Compute MAE1D(x,y) for all pixels
        mae1d = self.calculate_MAE1D(image1, image2)  # shape (height, width)

        # Decide how to define "above threshold" – for example, use the mean across spectral channels:
        avg_intensity_image1 = np.mean(image1, axis=0)  # shape (height, width)
        mask = avg_intensity_image1 >= threshold

        # Average mae1d only over valid pixels
        return np.mean(mae1d[mask])

    def calculate_SRE_new(self, gt, gen, threshold=0.01):
        """
         (prediction − ground-truth)/ground-truth × 100
         only calculate for pixels where ground-truth intensity is above threshold.
         """
        E = (gen - gt) / (gt) * 100 
        mask = gt >= threshold
        return np.mean(np.abs(E[mask]))

        
    def load_images(self, index):
        """Load ground truth and reconstructed images by index."""
        bad_gt = False
        prefixes = {
            "gt": ["cb_raw_", "hs_raw_"],
            "gen": ["tl_gen_", "hs_gen_"]
        }
        
        # Find files matching current index with any prefix pattern
        gt_file = None
        gen_file = None
        
        for prefix in prefixes["gt"]:
            candidate = f"{prefix}{index}.tif"
            if candidate in self.image_files:
                gt_file = candidate
                break
                
        for prefix in prefixes["gen"]:
            candidate = f"{prefix}{index}.tif"
            if candidate in self.image_files:
                gen_file = candidate
                break
                
        if not gt_file or not gen_file:
            raise FileNotFoundError(f"Files for index {index} not found: {gt_file} or {gen_file}")

        # Load images with detected file patterns
        cb_path = os.path.join(self.image_dir, gt_file)
        tl_path = os.path.join(self.image_dir, gen_file)
        if bad_gt:
            ground_truth = tiff.imread(tl_path)
        else:
            ground_truth = tiff.imread(cb_path)
        #reconstructed = tiff.imread('/media/al/Extreme SSD/20250425_dataset/dark_img.tif')
        reconstructed = tiff.imread(tl_path)
        #print(f'ground truth min: {ground_truth.min()}, max: {ground_truth.max()}')
        #print(f'reconstructed min: {reconstructed.min()}, max: {reconstructed.max()}')

        gt_min = ground_truth.min()
        gt_max = ground_truth.max()
        #print(f'reconstructed mean: {reconstructed.mean()}, std: {reconstructed.std()}')

        #normlaize reconstructed image to match ground truth range
        reconstructed = (reconstructed - gt_min) / (gt_max - gt_min)
        ground_truth = (ground_truth - gt_min) / (gt_max - gt_min)

        # clip 0 to 1
        ground_truth = np.clip(ground_truth, 0, 1)
        reconstructed = np.clip(reconstructed, 0, 1)
        #print(f'reconstructed norm mean: {reconstructed.mean()}, std: {reconstructed.std()}')
        # non probabilistic case
        if ground_truth.shape[0] == reconstructed.shape[0]:
            mu = reconstructed
            # sigma is zeros in shape or reconstructed
            sigma = np.zeros_like(reconstructed)
        # probabilistic case
        if reconstructed.shape[0] == 2 * ground_truth.shape[0]:
            mu = reconstructed[:reconstructed.shape[0]//2, :, :]
            sigma = reconstructed[reconstructed.shape[0]//2:, :, :]

        
        if bad_gt:
            # If ground truth is bad, use reconstructed as ground truth
            mu = reconstructed[:reconstructed.shape[0]//2, :, :]
            sigma = reconstructed[reconstructed.shape[0]//2:, :, :]
            ground_truth = mu.copy()
        
        return ground_truth, mu, sigma

    def compute_metrics(self, ground_truth, reconstructed):
        """Compute metrics for a single pair of images."""
        data_range = ground_truth.max() - ground_truth.min()

        # Compute metrics
        #ssim_3d = ssim(ground_truth, reconstructed, , multichannel=True)
        ssim_3d = 0
        ssim_2d = ssim(ground_truth[0], reconstructed[0], data_range=data_range)
        mse = self.calculate_mse(ground_truth, reconstructed)
        mae = self.calculate_mae(ground_truth, reconstructed)
        threshold_mae = self.calculate_threshold_MAE(ground_truth, reconstructed)
        sre = self.calculate_SRE_new(ground_truth, reconstructed)

        return ssim_3d, ssim_2d, mse, mae, threshold_mae, sre
    '''
    def compute_sigma_mean(self, sigma):
        # sigma floor of 0.001 used during training
        mask = sigma > 0.001
        sigma_mean = np.mean(sigma[mask])
        # if sigma mean is nan or inf, return 0.001
        if np.isnan(sigma_mean) or np.isinf(sigma_mean):
            sigma_mean = 0.001
        return sigma_mean
    def compute_sigma_mean(self, sigma):
        sigma_mean = np.mean(sigma)
        return sigma_mean

    def compute_sigma_mean(self, mu, sigma):
        sigma_mean = np.mean(sigma / (mu + 1e-8))
        return sigma_mean
    '''
    def compute_sigma_mean(self, sigma):
        sigma = np.maximum(sigma, 0.001)
        sigma_mean = np.mean(sigma)
        return sigma_mean
    
    def calculate_metrics(self):
        """Calculate metrics for all test images and display averages."""
        for index in self.valid_indices:
            try:
                ground_truth, reconstructed, sigma = self.load_images(index)
                ssim_3d, ssim_2d, mse, mae, threshold_mae, sre = self.compute_metrics(ground_truth, reconstructed)

                # Store metrics
                self.ssim_3d_list.append(ssim_3d)
                self.ssim_2d_list.append(ssim_2d)
                self.mse_list.append(mse)
                self.mae_list.append(mae)
                self.threshold_mae_list.append(threshold_mae)
                self.sre_list.append(sre)
                
                # Calculate and store sigma mean if sigma exists (not all zeros)
                if np.any(sigma):  # Check if sigma contains non-zero values
                    sigma_mean = self.compute_sigma_mean(sigma)
                    self.sigma_mean_list.append(sigma_mean)
                    print(f"Index {index} - SRE: {sre:.4f}, MAE: {mae:.4f}, Sigma Mean: {sigma_mean:.6f}")
                else:
                    print(f"Index {index} - SRE: {sre:.4f}, MAE: {mae:.4f}")
            except FileNotFoundError as e:
                print(e)
                continue  # Skip missing files

        # Compute averages
        avg_ssim_3d = np.mean(self.ssim_3d_list) if self.ssim_3d_list else 0
        avg_ssim_2d = np.mean(self.ssim_2d_list) if self.ssim_2d_list else 0
        avg_mse = np.mean(self.mse_list) if self.mse_list else 0
        avg_mae = np.mean(self.mae_list) if self.mae_list else 0
        avg_threshold_mae = np.mean(self.threshold_mae_list) if self.threshold_mae_list else 0
        avg_sre = np.mean(self.sre_list) if self.sre_list else 0
        avg_sigma_mean = np.mean(self.sigma_mean_list) if self.sigma_mean_list else 0

        # Display the averages
        print(f"Average Metrics Over {len(self.ssim_3d_list)} Valid Image Pairs:")
        print(f"  Average 3D SSIM: {avg_ssim_3d:.4f}")
        print(f"  Average 2D SSIM: {avg_ssim_2d:.4f}")
        print(f"  Average MSE: {avg_mse:.8f}")
        print(f"  Average MAE: {avg_mae:.6f}")
        print(f"  Average Threshold MAE: {avg_threshold_mae:.4f}")
        print(f"  Average SRE: {avg_sre:.4f}")
        if self.sigma_mean_list:  # Only print if we have sigma values
            print(f"  Average Sigma Mean: {avg_sigma_mean:.9f}")
        else:
            print(f"  Average Sigma Mean: N/A (no uncertainty data)")

class InteractiveHSIPlot:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]
        self.image_files.sort(key=natural_sort_key)
        self.current_index = 0
        self.selected_points = []
        self.lines = []
        self.colors = ['red', 'blue', 'green']
        self.color_index = 0
        self.ssim_3d = 0
        self.ssim_2d = 0
        self.mse_3d = 0
        self.mae_3d = 0
        self.view_mode = "Single Wavelength"
        self.current_wavelength_index = 0
        self.load_images()
        self.setup_plot()

    def calculate_mse(self, image1, image2):
        """Calculate the Mean Squared Error (MSE) between two images."""
        return np.mean((image1 - image2) ** 2)

    def calculate_mae(self, image1, image2):
        """Calculate the Mean Absolute Error (MAE) between two images."""
        return np.mean(np.abs(image1 - image2))
    
    def calculate_MAE1D(self, image1, image2):
        """Calculate the Mean Absolute Error (MAE) between two images."""
        return np.mean(np.abs(image1 - image2), axis=0)
    
    def calculate_MAE2D(self, image1, image2):
        """Calculate the Mean Absolute Error (MAE) between ave intensitys all spectral channels. Function of wavelength"""
        return np.mean(np.abs(image1 - image2), axis=(1, 2))

    def calculate_MAE2D_threshold(self, image1, image2, threshold=0.1):
        """
        MAE2D(l) = avg(|E(x,y,l)| over x,y) for pixels whose intensity in image1
        is >= threshold. This results in a 1D array with length == spectral_channels.

        image1, image2: np.ndarray of shape (spectral_channels, height, width)
        threshold: float
        Returns:
            A 1D np.ndarray of length == image1.shape[0], where each element is
            the mean absolute error at that spectral channel over valid pixels.
        """
        E = np.abs(self.calculate_E(image1, image2))  # shape (spectral, H, W)
        mask = image1 >= threshold                    # shape (spectral, H, W)

        mae_per_channel = []
        for s in range(E.shape[0]):
            valid_errors = E[s][mask[s]]
            mae_per_channel.append(np.mean(valid_errors))
        return np.array(mae_per_channel)
    
    def calculate_E(self, image1, image2):
        """Calculate the error between two images."""
        return image1 - image2
    

    def load_images(self):
        # Check for different naming patterns
        prefixes = {
            "gt": ["cb_raw_", "hs_raw_"],
            "gen": ["tl_gen_", "hs_gen_"]
        }
        
        # Find files matching current index with any prefix pattern
        gt_file = None
        gen_file = None
        
        for prefix in prefixes["gt"]:
            candidate = f"{prefix}{self.current_index}.tif"
            if candidate in self.image_files:
                gt_file = candidate
                break
                
        for prefix in prefixes["gen"]:
            candidate = f"{prefix}{self.current_index}.tif"
            if candidate in self.image_files:
                gen_file = candidate
                break
                
        if not gt_file or not gen_file:
            print(f"Warning: Files for index {self.current_index} not found")
            return

        # Load images with detected file patterns
        self.cb_path = os.path.join(self.image_dir, gt_file)
        self.tl_path = os.path.join(self.image_dir, gen_file)
        self.ground_truth = tiff.imread(self.cb_path)
        self.reconstructed = tiff.imread(self.tl_path)

        # non probabilistic case
        if self.ground_truth.shape[0] == self.reconstructed.shape[0]:
            self.mu = self.reconstructed
            # sigma is zeros in shape or reconstructed
            self.sigma = np.zeros_like(self.reconstructed)
        # probabilistic case
        if self.reconstructed.shape[0] == 2 * self.ground_truth.shape[0]:
            self.mu = self.reconstructed[:self.reconstructed.shape[0]//2, :, :]
            self.sigma = self.reconstructed[self.reconstructed.shape[0]//2:, :, :]
        
        self.reconstructed = self.mu  # Default to mu for reconstruction view

        gt_min = self.ground_truth.min()
        gt_max = self.ground_truth.max()

        #normlaize reconstructed image to match ground truth range
        self.mu = (self.mu - gt_min) / (gt_max - gt_min)
        self.ground_truth = (self.ground_truth - gt_min) / (gt_max - gt_min)

        # clip 0 to 1
        self.ground_truth = np.clip(self.ground_truth, 0, 1)
        self.mu = np.clip(self.mu, 0, 1)
        

        # Convert to RGB for both ground truth and reconstruction
        if self.ground_truth.shape[0] == 30:
            self.gt_rgb = self.convert_to_rgb(self.ground_truth)
            self.rc_rgb = self.convert_to_rgb(self.mu)
            self.sigma_rgb = self.convert_to_rgb(self.sigma)

        # Calculate SSIM values
        self.data_range = self.ground_truth.max() - self.ground_truth.min()
        #self.ssim_3d = ssim(self.ground_truth, self.reconstructed, data_range=self.data_range, multichannel=True)
        self.ssim_3d = 0
        self.ssim_2d = ssim(self.ground_truth[0], self.reconstructed[0], data_range=self.data_range)

        # Calculate MSE and MAE
        self.mse_3d = self.calculate_mse(self.ground_truth, self.reconstructed)
        self.mae_3d = self.calculate_mae(self.ground_truth, self.reconstructed)

        # Print shapes, SSIM, MSE, and MAE values for verification
        print(f"Ground Truth (hs_raw_{self.current_index}) shape: {self.ground_truth.shape}")
        print(f"Reconstructed (hs_gen_{self.current_index}) shape: {self.reconstructed.shape}")
        print(f"Mu shape: {self.mu.shape}, Sigma shape: {self.sigma.shape}")
        print(f"3D SSIM: {self.ssim_3d:.4f}, 2D SSIM (450nm): {self.ssim_2d:.4f}")
        print(f"3D MSE: {self.mse_3d:.4f}, 3D MAE: {self.mae_3d:.4f}")

        
    def convert_to_rgb(self, hcube):
        """
        1) Loads reference data for red, green, blue squares.
        2) Resamples them from Cubert to Miniscope wavelength range.
        3) Integrates each to get raw XYZ.
        4) Builds a 3x3 calibration matrix.
        5) Converts new hyperspectral data to sRGB using that matrix.
        """

        from colour import MSDS_CMFS
        from colour.colorimetry import SpectralShape
        from scipy.interpolate import interp1d

        red_data = np.load("/home/al/hyperspectral_pix2pix/RGB/red_spectrogram.npz")
        green_data = np.load("/home/al/hyperspectral_pix2pix/RGB/green_spectrogram.npz")
        blue_data = np.load("/home/al/hyperspectral_pix2pix/RGB/blue_spectrogram.npz")
        red_spec = red_data["spectrogram"]
        green_spec = green_data["spectrogram"]
        blue_spec = blue_data["spectrogram"]

        # Original Cubert wavelength range
        cubert_long = 850
        cubert_short = 450
        cubert_bands = red_spec.shape[0]  # Use actual number of bands from data
        
        # Target Miniscope wavelength range
        miniscope_long = 700
        miniscope_short = 450
        num_bands = hcube.shape[0]  # Number of bands in your current data

        # long bands are first
        #hcube = np.flip(hcube, axis=0)

        # Create original Cubert wavelengths
        cubert_wavelengths = np.linspace(cubert_short, cubert_long, cubert_bands)
        
        # Create target Miniscope wavelengths based on your data
        end_wl = num_bands / num_bands * (miniscope_long - miniscope_short) + miniscope_short
        miniscope_wavelengths = np.linspace(miniscope_short, end_wl, num_bands)
        
        # If each is (C, H, W), flatten them to (C,) by averaging over the patch
        if red_spec.ndim == 3:
            red_spec = red_spec.mean(axis=(1,2))
            green_spec = green_spec.mean(axis=(1,2))
            blue_spec = blue_spec.mean(axis=(1,2))

        
        # Create interpolation functions
        red_interp = interp1d(cubert_wavelengths, red_spec, kind='linear', 
                            bounds_error=False, fill_value=0.0)
        green_interp = interp1d(cubert_wavelengths, green_spec, kind='linear', 
                            bounds_error=False, fill_value=0.0)
        blue_interp = interp1d(cubert_wavelengths, blue_spec, kind='linear', 
                            bounds_error=False, fill_value=0.0)
        
        # Resample to Miniscope wavelengths
        red_spec_resampled = red_interp(miniscope_wavelengths)
        green_spec_resampled = green_interp(miniscope_wavelengths)
        blue_spec_resampled = blue_interp(miniscope_wavelengths)
        
        # Create spectral shape for Miniscope
        interval = (end_wl - miniscope_short) / (num_bands - 1)
        shape = SpectralShape(miniscope_short, end_wl, interval)

        cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].copy().align(shape)
        xbar = cmfs.values[:, 0]
        ybar = cmfs.values[:, 1]
        zbar = cmfs.values[:, 2]
        d_lambda = interval

        # A small helper for integration against CMFs:
        def integrate_xyz(spectrum_1d):
            X = (xbar * spectrum_1d).sum() * d_lambda
            Y = (ybar * spectrum_1d).sum() * d_lambda
            Z = (zbar * spectrum_1d).sum() * d_lambda
            return np.array([X, Y, Z])

        # Use resampled spectra
        raw_xyz_red = integrate_xyz(red_spec_resampled)
        raw_xyz_green = integrate_xyz(green_spec_resampled)
        raw_xyz_blue = integrate_xyz(blue_spec_resampled)

        raw_xyz = np.stack([
            raw_xyz_red,
            raw_xyz_green,
            raw_xyz_blue
        ])
        # The target is linear R, G, B as rows:
        target_rgb = np.array([
            [1, 0, 0],   # linear R
            [0, 1, 0],   # linear G
            [0, 0, 1]    # linear B
        ])

        M_calib = target_rgb.T @ np.linalg.inv(raw_xyz.T)

        H, W = hcube.shape[1], hcube.shape[2]
        flat_data = hcube.reshape(num_bands, -1)

        X = (xbar[:, None] * flat_data).sum(axis=0) * d_lambda
        Y = (ybar[:, None] * flat_data).sum(axis=0) * d_lambda
        Z = (zbar[:, None] * flat_data).sum(axis=0) * d_lambda
        raw_xyz_pixels = np.stack([X, Y, Z], axis=0)  # shape (3, H*W)

        rgb_linear = M_calib @ raw_xyz_pixels  # shape (3, H*W)
        rgb_linear[rgb_linear < 0] = 0.0       # clamp negative

        mask = (rgb_linear <= 0.0031308)
        rgb_linear[mask] *= 12.92
        rgb_linear[~mask] = 1.055 * (rgb_linear[~mask] ** (1/2.4)) - 0.055

        rgb_image = rgb_linear.T.reshape(H, W, 3)
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

        # switch rgb to bgr
        rgb_image = rgb_image[:, :, ::-1]
        # normalize channels separately to [0, 1]
        for i in range(3):
            channel = rgb_image[:, :, i]
            rgb_image[:, :, i] = (channel - channel.min()) / (channel.max() - channel.min())
        return rgb_image


    def setup_plot(self):
        # Setup figure and axes with extra space at bottom for slider
        self.fig = plt.figure(figsize=(18, 7))
        
        # Create a gridspec layout
        gs = self.fig.add_gridspec(2, 4, height_ratios=[1, 0.1])
        
        # Create main axes
        self.ax = [
            self.fig.add_subplot(gs[0, 0]),
            self.fig.add_subplot(gs[0, 1]),
            self.fig.add_subplot(gs[0, 2]),
            self.fig.add_subplot(gs[0, 3])
        ]

        # Add wavelength slider
        self.wavelengths = np.linspace(700, 450, self.ground_truth.shape[0])  # Generate wavelength array

        self.slider_ax = self.fig.add_subplot(gs[1, 1:3])  # Span two columns
        self.wavelength_slider = Slider(
            self.slider_ax,
            'Wavelength (nm)',
            0,
            len(self.wavelengths) - 1,
            valinit=0,
            valstep=1
        )
        self.wavelength_slider.on_changed(self.update_wavelength)

        # Add view mode selection
        ax_radio = plt.axes([0, 0.10, 0.12, 0.15])  # [left, bottom, width, height]
        self.radio = RadioButtons(ax_radio, ['RGB Reconstruction', 'Single Wavelength', 'Sigma View'])
        self.radio.on_clicked(self.change_view_mode)

        # Connect click events
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        # Add control buttons
        button_ax_next = plt.axes([0.4, 0.02, 0.15, 0.05])
        self.btn_next = Button(button_ax_next, "Next")
        self.btn_next.on_clicked(self.load_next_image)

        button_ax_clear = plt.axes([0.6, 0.02, 0.15, 0.05])
        self.btn_clear = Button(button_ax_clear, "Clear Spectra")
        self.btn_clear.on_clicked(self.clear_spectra)

        # Add MAE button
        button_ax_mae = plt.axes([0.8, 0.02, 0.15, 0.05])
        self.btn_mae = Button(button_ax_mae, "Show MAE Analysis")
        self.btn_mae.on_clicked(lambda event: self.show_MAE1D_and_MAE2D(self.ground_truth, self.reconstructed))

        # Display initial plot
        self.update_plot()
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()

    def update_wavelength(self, val):
        """Handle wavelength slider changes"""
        self.current_wavelength_index = int(val)
        if self.view_mode in ["Single Wavelength", "Sigma View"]:
            self.update_plot()

    def change_view_mode(self, label):
        """Handle changes in the view mode."""
        self.view_mode = label
        self.update_plot()
    
    def update_plot(self):
        # Clear all axes
        for a in self.ax[:3]:  # Clear image axes
            a.cla()
        
        wavelength = self.wavelengths[self.current_wavelength_index]

        # Display appropriate images based on view mode
        if self.view_mode == "RGB Reconstruction":
            self.ax[0].imshow(self.gt_rgb)
            self.ax[0].set_title("Ground Truth (RGB Reconstruction)")
            self.ax[1].imshow(self.rc_rgb)
            self.ax[1].set_title("Reconstructed (RGB Reconstruction)")
            self.ax[2].imshow(self.sigma_rgb)
            self.ax[2].set_title("Uncertainty (Sigma RGB Visualization)")
        
        elif self.view_mode == "Single Wavelength":
            self.ax[0].imshow(self.ground_truth[self.current_wavelength_index], cmap='viridis')
            self.ax[0].set_title(f"Ground Truth ({wavelength:.1f}nm)")
            self.ax[1].imshow(self.mu[self.current_wavelength_index], cmap='viridis')
            self.ax[1].set_title(f"Reconstructed (μ, {wavelength:.1f}nm)")
            self.ax[2].imshow(self.sigma[self.current_wavelength_index], cmap='plasma')
            self.ax[2].set_title(f"Uncertainty (σ, {wavelength:.1f}nm)")
        
        elif self.view_mode == "Sigma View":
            # Calculate error image
            error_img = np.abs(self.ground_truth[self.current_wavelength_index] - 
                            self.mu[self.current_wavelength_index])
            
            self.ax[0].imshow(error_img, cmap='hot')
            self.ax[0].set_title(f"Absolute Error ({wavelength:.1f}nm)")
            
            self.ax[1].imshow(self.mu[self.current_wavelength_index], cmap='viridis')
            self.ax[1].set_title(f"Reconstructed μ ({wavelength:.1f}nm)")
            
            self.ax[2].imshow(self.sigma[self.current_wavelength_index], cmap='plasma')
            self.ax[2].set_title(f"Uncertainty σ ({wavelength:.1f}nm)")
            
            # Add a text showing min/max sigma values for reference
            sigma_img = self.sigma[self.current_wavelength_index]
            self.ax[2].text(0.02, 0.98, f"Min σ: {sigma_img.min():.4f}\nMax σ: {sigma_img.max():.4f}", 
                        transform=self.ax[2].transAxes, fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Turn off axes for image plots
        for a in self.ax[:3]:
            a.axis('off')

        # Update spectral plot (now in position 3)
        self.ax[3].cla()
        self.ax[3].set_title("Spectral Comparison")
        self.ax[3].set_xlabel("Wavelength (nm)")
        self.ax[3].set_ylabel("Intensity")

        # Redraw any existing spectral lines
        for point in self.selected_points:
            x, y, color = point
            gt_spectrum = self.ground_truth[:, y, x]
            mu_spectrum = self.mu[:, y, x]
            sigma_spectrum = self.sigma[:, y, x]
            
            self.ax[3].plot(self.wavelengths, gt_spectrum, color=color, label="Ground Truth")
            self.ax[3].plot(self.wavelengths, mu_spectrum, "--", color=color, label="Reconstructed (μ)")
            
            # Add shaded confidence interval for sigma (showing uncertainty)
            upper_bound = mu_spectrum + sigma_spectrum
            lower_bound = mu_spectrum - sigma_spectrum
            self.ax[3].fill_between(self.wavelengths, lower_bound, upper_bound, color=color, alpha=0.2)

        # Update figure title
        self.fig.suptitle(
            f"Image Analysis: Ground Truth vs Reconstruction (Index {self.current_index}) | "
            f"3D SSIM: {self.ssim_3d:.4f} | 2D SSIM: {self.ssim_2d:.4f} | "
            f"3D MSE: {self.mse_3d:.4f} | 3D MAE: {self.mae_3d:.4f}",
            fontsize=16
        )

        if len(self.selected_points) > 0:
            self.ax[3].legend(loc='upper left')

        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes == self.ax[0]:  # Ensure click is on the ground truth plot
            x, y = int(event.xdata), int(event.ydata)
            color = self.colors[self.color_index % len(self.colors)]
            self.color_index += 1

            self.selected_points.append((x, y, color))

            # Mark the selected point
            self.ax[0].plot(x, y, "o", color=color, markersize=8)

            # Extract and plot spectra
            gt_spectrum = self.ground_truth[:, y, x]
            mu_spectrum = self.mu[:, y, x]
            sigma_spectrum = self.sigma[:, y, x]
            
            # Plot on self.ax[3] instead of self.ax[2]
            line_gt, = self.ax[3].plot(self.wavelengths, gt_spectrum, color=color, label="Ground Truth")
            line_mu, = self.ax[3].plot(self.wavelengths, mu_spectrum, "--", color=color, label="Reconstructed (μ)")
            
            # Create shaded uncertainty region
            upper_bound = mu_spectrum + sigma_spectrum
            lower_bound = mu_spectrum - sigma_spectrum
            uncertainty = self.ax[3].fill_between(self.wavelengths, lower_bound, upper_bound, color=color, alpha=0.2)
            
            self.lines.append((line_gt, line_mu, uncertainty))

            # Update legend
            self.ax[3].legend()
            self.fig.canvas.draw()

    def clear_spectra(self, event):
        for line_info in self.lines:
            # Each line_info is a tuple of (line_gt, line_mu, uncertainty)
            for item in line_info:
                if hasattr(item, 'remove'):
                    item.remove()
                else:
                    # For fill_between, need to remove the collection
                    try:
                        item.remove()
                    except:
                        pass
        self.lines = []
        self.selected_points = []
        self.update_plot()

    def load_next_image(self, event):
        self.current_index += 1
        if self.current_index >= len(self.image_files) // 2:
            print("No more images.")
            return

        self.load_images()
        self.update_plot()

    def show_MAE1D_and_MAE2D(self, image1, image2):
        """
        Show the MAE1D image and MAE2D plot.
        
        image1, image2: np.ndarray of shape (spectral_channels, height, width)
        """
        # Calculate MAE1D and MAE2D
        mae1d = self.calculate_MAE1D(image1, image2)  # shape (height, width)
        mae2d = self.calculate_MAE2D(image1, image2)  # shape (spectral_channels,)

        # Create a new window for the analysis
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Show MAE1D image
        img = axs[0].imshow(mae1d, cmap='viridis')
        axs[0].set_title('MAE1D Image (Spatial Error)')
        plt.colorbar(img, ax=axs[0])
        axs[0].set_axis_off()

        # Show MAE2D plot
        axs[1].plot(self.wavelengths, mae2d)
        axs[1].set_title('MAE2D Plot (Spectral Error)')
        axs[1].set_xlabel('Wavelength (nm)')
        axs[1].set_ylabel('MAE')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        
        # Show Sigma mean image (uncertainty per wavelength)
        sigma_mean = np.mean(self.sigma, axis=(1, 2))  # Average sigma across spatial dimensions
        axs[2].plot(self.wavelengths, sigma_mean, 'r-')
        axs[2].set_title('Average Uncertainty (σ) per Wavelength')
        axs[2].set_xlabel('Wavelength (nm)')
        axs[2].set_ylabel('Mean σ')
        axs[2].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()


# Usage:
# Make sure to define natural_sort_key function if not already defined
def natural_sort_key(s):
    """Sort strings that contain numbers in natural order."""
    import re
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


image_directory = '/media/al/20250701_AI/hs_results/20250707_results/results/20250425_0gan_single_reg_hs_gradloss1e2/test_latest/images'


# Uncomment this if you have an AverageMetricsCalculator class
AverageMetricsCalculator(image_directory)

InteractiveHSIPlot(image_directory)