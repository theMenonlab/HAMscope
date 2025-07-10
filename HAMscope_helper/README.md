# HAMscope Helper Tools

This directory contains utility scripts and analysis tools for hyperspectral microscopy data processing. The tools are organized into functional modules for different aspects of the hyperspectral imaging pipeline.

## Directory Structure

```
HAMscope_helper/
├── analysis/                 # Data analysis and intensity measurement tools
├── deconvolution/           # Ring deconvolution microscopy implementation
├── movie/                   # Time-lapse movie generation tools
├── nmf/                     # Non-negative matrix factorization tools
├── preprocessing/           # Data preprocessing and dataset preparation
├── util/                    # General utility scripts
└── visualization/           # Data visualization and rendering tools
```

## Module Overview

### 📊 Analysis (`analysis/`)

Tools for quantitative analysis of hyperspectral data:
- **`spectra_comparison_probabilistic.py`**: Analyze metrics and visualize accuracy of test datasets (Table S1).
- **`hs_channel_mean_intensities.py`**: Calculates mean intensity values across all spectral channels for batch processing, useful for finding average scale values (Table S2)
- **`hs_channel_mean_intensities_dasmeet.py`**: Analyze the mean intensities of transgenic poplar hyperspectral data (Fig. S15)
- **`batch_disagreement.py`**: Analyzes model prediction disagreement across ensemble (Fig S7)
- **`laplacian_iterative_scale_multiplier_hs.py`**: Compares scale to calculate standard deviation.

#### � Loss Plotting (`analysis/loss_plotter/`)
- **`loss_log_plotter.py`**: Visualizes training loss curves from AI model training 
- **`loss_log_plotter_testloss.py`**: Specialized version for test loss visualization (Fig. S5)

#### 🔍 Resolution Analysis (`analysis/resolution/`)
Spatial and spectral resolution characterization tools:
- **`fourier_resolution.py`**: Fourier-based resolution analysis (Fig. 5, S10)
- **`hyperspectral_resolution_complete.py`**: USAF chart resolution assessment (Fig. S11)
- **`hyperspectral_resolution_find_elements.py`**: Element detection for resolution testing (Fig. S11)

#### 📡 Diffuser Analysis (`analysis/diffuser/`)
Tools for characterizing diffuser properties:
- **`analyze_diffusion_angle.py`**: Analyzes light diffusion angles and patterns (Fig. S1 A)
- **`diffuser_transmission.py`**: Measures transmission properties of diffuser elements (Fig. S1 D)

#### 🔬 Variable Bandpass Filter (`analysis/variable_bandpass_filter/`)
Liquid Variable Bandpass Filter (LVBF) characterization and calibration:
- **`plot_LVBF_spectra.py`**: Plots LVBF transmission spectra with baseline (Fig. S4)
- **`plot_LVBF_spectra_no_baseline.py`**: LVBF spectra without baseline correction (Fig. S4)
- **`sensor_sensitivity_calibration.py`**: Calibrates sensor spectral sensitivity (Fig. S4)
- **`hamamatsu_spectral_response.png`**: Reference spectral response curve

#### 🌈 RGB Conversion (`analysis/RGB/`)
Tools for converting hyperspectral data to RGB images within spectra_comparison_probabilistic.py:
- **`HSI2RGB.py`**: Converts hyperspectral images to RGB representation
- **`mean_spectrogram.py`**: Calculates mean spectral responses for RGB channels
- **`D_illuminants.mat`**: Standard illuminant data for color calibration
- **`*_spectrogram.npz`**: Precomputed spectrograms for red, green, blue channels
- **`color_calibration/`**: Reference images and calibration data


### 🔬 Deconvolution (`deconvolution/`)

Ring deconvolution microscopy (RDM) implementation works with https://github.com/apsk14/rdmpy:

- **`demo.ipynb`**: Interactive Jupyter notebook with added batch wiener deconvolution and visualization (Fig. S11, S16)
- **`gen_100_points.py`**: Generates synthetic point source data for deconvolution testing


### 🎬 Movie (`movie/`)

Time-lapse movie generation tools:

- **`hyperspectral_folder_movie.py`**: Turn hyperspectral and 3-channel timelapse folders into movies (supplementary videos), plot channel intensities over time (Fig. S12)
- **`movies_to_folder.py`**: Turn movies back into folders of images (Fig. 6B)


### 🧮 NMF (`nmf/`)

Non-negative matrix factorization tools for spectral unmixing:

- **`auto_nmf_fixed.ijm`**: ImageJ macro for automated NMF processing using PoissonNMF (all nmf)
- **`auto_nmf_fixed_4ch.ijm`**: 4-channel version of the NMF macro (Fig. S9)
- **`auto_nmf_fixed_from_pix2pix.ijm`**: NMF processing for pix2pix outputs (first predicting hyperspectral data followed by biomolecule unmixing)
- **`plot_3ch_spectra.py`**: Visualizes 3-channel spectral signatures (Fig. 3, S8)
- **`plot_4ch_spectra.py`**: Visualizes 4-channel spectral signatures (Fig. S9)
- **`PoissonNMF_.jar`**: ImageJ plugin implementing Poisson NMF algorithm (Neher et al., 2009) (all nmf)

### 🔄 Preprocessing (`preprocessing/`)

Data preprocessing and dataset preparation tools:

- **`split_test_train.py`**: Splits datasets into training and testing sets randomly
- **`copy_test_train.py`**: Splits datasets into training and testing sets following existing test dataset for consistency between dataset processing methods.
- **`manual_mat_cropper.py`**: Interactive tool for manual image cropping raw and converting hyperspectral .mat to tif stacks.

### 🛠️ Utilities (`util/`)

- **`svg2pdf.py`**: Converts SVG graphics to PDF format, streamlining figure production


### 📈 Visualization (`visualization/`)

Data visualization and rendering tools:

- **`visualize_models.py`**: Extract colormapped png from test results (Fig. 4, 9, S6)
- **`visualize_models_misc.py`**: Extract colormapped png from test results (Fig. 6, 7, 8, S13, S14, S16)
- **`visualize_single.py`**: Single image visualization
- **`3ch_tif_2_png.py`**: Converts test results with nmf TIFF to PNG
- **`probabilistic_visualization.py`**: Visualizes ensemble probabilistic model (Fig. 2)
- **`stich_imgs_fade.py`**: Creates fade transitions between images (Fig. 3, S8)
- **`stich_imgs_fade_4channel.py`**: 4-channel version of fade stitching (Fig. S9)


## Acknowledgments

### Third-Party Components

- **PoissonNMF Algorithm**: The NMF processing uses the PoissonNMF ImageJ plugin developed by Neher et al. (2009), which implements non-negative matrix factorization optimized for fluorescence microscopy data with Poisson noise characteristics.
  
  **Citation**: Neher, R.A., et al. "Blind source separation techniques for the decomposition of multiply labeled fluorescence images." *Biophysical Journal* 96.9 (2009): 3791-3800.

- **ImageJ Platform**: Macros and plugins built on the ImageJ/Fiji platform for biological image analysis.

- **Scientific Python Ecosystem**: Extensive use of NumPy, SciPy, Matplotlib, and scikit-image for data processing and visualization.

### License Information

- **PoissonNMF**: Used for academic research purposes (original authors: Richard A. Neher, Fabian J. Theis, André Zeug)
- **ImageJ**: Public domain software
- **Python tools**: Custom implementations using standard scientific libraries

---

**Note**: Many scripts contain hard-coded paths that need to be modified for your specific system and dataset locations.
