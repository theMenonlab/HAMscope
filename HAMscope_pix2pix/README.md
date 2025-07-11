# HAMscope pix2pix - Hyperspectral images from grayscale diffusive inputs

## Overview

HAMscope pix2pix is an image to image neural network for converting diffusive images from a miniscope into hyperspectral image stacks. It is based upon the pix2pix framework (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Additions are outlined in the HAMscope specific options below.


- **Input**: Single-channel grayscale images from diffusive HAMscope
- **Output**: 30-channel hyperspectral images (450-700nm)
- **Key Features**: Probabilistic uncertainty estimation, multi-pass U-Net architecture, registration networks, transformer attention
- **Applications**: Plant biology, material characterization, biomedical imaging

## Installation

### Prerequisites
- NVIDIA GPU with CUDA support
- Anaconda or Miniconda installed
- Git

### Step-by-Step Installation

0. **Install NVIDIA Driver and CUDA Toolkit (Linux)**
   ```bash
   # Install NVIDIA driver (tested on Linux Mint 21.3 Cinnamon)
   sudo apt install nvidia-driver-520
   
   # Reboot system after driver installation
   sudo reboot
   
   # Install CUDA toolkit 11.8 via conda
   conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
   ```
   
   **Note**: For other Linux distributions or driver versions, check `ubuntu-drivers devices` or use your distribution's package manager.

1. **Create Conda Environment**
   ```bash
   conda create --name pix2pix python=3.10
   ```

2. **Initialize Conda (if needed)**
   ```bash
   # For bash/zsh shells
   conda init bash
   
   # For tcsh/csh shells (some systems)
   conda init tcsh
   
   # Open a new terminal window after initialization
   ```

3. **Activate Environment**
   ```bash
   conda activate pix2pix
   ```

4. **Check CUDA Installation**
   ```bash
   # Check NVIDIA driver and CUDA version
   nvidia-smi
   
   # Check CUDA toolkit version
   nvcc --version
   ```
   
   **Note**: Ensure your CUDA toolkit version is compatible with PyTorch. Common combinations:
   - CUDA 11.8 with PyTorch 2.0.0
   - CUDA 12.2 with PyTorch 2.1.0+

5. **Install PyTorch**
   ```bash
   # For CUDA 11.8
   conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # For CUDA 12.1+ (adjust version as needed)
   conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

6. **Install Additional Dependencies**
   ```bash
   pip install dominate visdom kornia opencv-python scikit-image tqdm colour-science tifffile pillow scipy
   ```

7. **Fix NumPy Compatibility**
   ```bash
   pip uninstall numpy
   pip install numpy==1.23.5
   ```

8. **Clone Repository**
   ```bash
   # Clone HAMscope (includes modified pix2pix)
   git clone https://github.com/theMenonlab/HAMscope.git
   cd HAMscope/HAMscope_pix2pix
   
   # OR clone original pix2pix for comparison
   git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
   ```

9. **Verify Installation**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   python -c "import torch; print(torch.__version__)"
   ```

### Troubleshooting Installation

**CUDA Version Mismatch:**
```bash
# Check available PyTorch versions for your CUDA
conda search pytorch -c pytorch

# Install specific version
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Data Format
- **Input**: 1-channel grayscale TIFF files (8-bit miniscope images)
- **Output**: 30-channel hyperspectral TIFF files (450-700nm spectral range)
- **Resolution**: 512x512 pixels (default, supports 1024x1024)

## HAMscope-Specific Options

```bash
--netD_mult     # Discriminator loss multiplier (default: 1.0)
--netG_reps     # Number of U-Net passes (default: 1)
--use_reg       # Enable registration network (0/1, default: 0)
--use_comp      # Dataset selection (2=biomolecule, 3=hyperspectral, default: 0)
--use_nll       # Enable probabilistic loss (0/1, default: 1)
--use_trans     # Transformer type (0=none, 1=linear, 2=quadratic, default: 0)
--grad_clip     # Gradient clipping value (default: 0.0)
--sigma_min     # Minimum sigma for probabilistic loss (default: 0.001)
--lambda_grad   # Gradient loss weight (default: 0.0)
--norm          # Normalization type [instance|batch|layer] (default: layer)
--netG          # Generator architecture [unet_512|unet_1024] (default: resnet_9blocks)
```
### Multi-U-Net Architecture
Number of passes of the u-net generator (--netG_reps n), where n is the number of passes

We also introduced a multi-U-Net architecture designed to mimic a larger model’sdepth and expressive power while preserving memory efficiency. Feature maps from the decoder of the previous pass are sent into the encoder of the next pass at every downsampling level of the U-Net via shared skip connections, effectively multiplying the number of convolutional layers without increasing the parameter count. The num ber of U-Net passes can be modified with an input parameter, but the effects diminish after four or five passes. This architecture matched or exceeded the performance of a U-Net with twice the number of filters per layer (from 64 to 128) and fourfold the parameter count without significantly increasing VRAM requirements and remaining trainable on an NVIDIA RTX 3060. In contrast, the higher-capacity control model
required an RTX 3090.

### Probabilistic Architecture
Probabilistic loss function activation (--use_nll 1). The default state is on, set to 0 to turn off the probabilistic loss and return to L1/MAE loss. If probabilistic loss is on, set (--output_nc) to be 2x the channel count of the target hyperspectral image to allow for generation of mean and scale images. For training stability and to prevent division by zero errors the scale value must be clipped to a minimum value before nll loss calculation. Adjust the minimum scale value with (--sigma_min).

Our probabilistic reconstruction model outputs each spectral channel’s predicted
mean and standard deviation. Training is guided by a Laplacian negative log-likelihood
(NLL) loss, which promotes accurate estimation of both the mean signal and associated uncertainty [Weigert, 2018]. These uncertainty estimates are especially critical when imaging directly with the HAMscope in field settings where no ground truth is available.

### Discriminator GAN framework
The weight of the discriminatory loss can be altered with (--netD_mult n) where n is a multiplier applied to the loss of the discriminator. In our experiments discriminator loss was scaled by factor of 0.005, ensuring it remained subordinate to the L1 term. Decrease the multiplier for higher accuracy datasets. The discriminator loss term is set to zero, deactivating the discriminator in all experiments unless specified.

The GAN framework comprises two convolutional neural networks: a generator that predicts the hyperspectral image from the monochrome input and a discriminator that learns to distinguish predicted outputs from real measurements. A composite loss function guides training: the discriminator’s adversarial loss encourages realistic image generation, while an L1 loss enforces pixel-level fidelity. To prevent overfitting or artifact hallucination, 

### Registration Network (`--use_reg 1`)
Separate registration U-Net corrects residual misalignments between predicted and ground-truth images. Estimates deformable vector field to shift predictions before loss calculation. Based on Kong et al. (2021).

### Transformer Attention (`--use_trans n`)
- `n=0`: No transformer
- `n=1`: Linear complexity transformer (spectral attention model)
- `n=2`: Quadratic complexity transformer

Applied to deeper U-Net layers after downsampling to 64×64 to manage VRAM. Quadratic transformer outperforms linear for probabilistic hyperspectral reconstruction. 

### Gradient Loss Function (`--lambda_grad n`)
Augments L1/MAE loss with gradient loss to improve sharpness of generated images. Calculates MAE between gradients of generated and target images. Minor sharpness improvement observed with multiplier of 10. Not used in published experiments.

### Gradient Clipping (`--grad_clip n`)
Prevents exploding gradients during training. Function disabled when `n=0`. Not used in published experiments.

## Data Processing

### U-Net Image Sizes (`--netG unet_512` or `--netG unet_1024`)
Two U-Net sizes support processing of larger images. Input images are interpolated to the U-Net size in `aligned_dataset.py`. Miniscope images are downsampled from 608×608 to 512×512 pixels to standardize input dimensions. Hyperspectral ground-truth images are automatically cropped, spatially registered, and interpolated to the same resolution.

### Normalization (`--norm layer`)
Custom layer normalization preserves relative intensity between channels and feature maps. Images are normalized to [0,1] range by dividing by bit depth (256 for 8-bit miniscope, 65,536 for 16-bit hyperspectral). This fixed scaling preserves relative intensity across channels without distortions from per-image normalization.

### Dataset Selection (`--use_comp n`)
Utility for selecting between datasets in `aligned_dataset.py`:
- `n=2`: Biomolecule maps  
- `n=3`: Hyperspectral data

**Directory Structure:**
```
dataset_root/
├── split_1/
│   ├── test/
│   │   ├── ham/          # 30-channel hyperspectral ground truth (16-bit)
│   │   ├── ham_nmf/      # 3-channel biomolecule maps
│   │   └── miniscope/    # Grayscale input images (8-bit)
│   └── train/
│       ├── ham/
│       ├── ham_nmf/
│       └── miniscope/
├── split_2/
│   └── [same structure]
├── split_3/
│   └── [same structure]
└── split_4/
    └── [same structure]
```

Split 1-4 represent different experimental conditions (e.g., poplar branches). Contact a.ingold@utah.edu for dataset access.

### Model Weights, Test Dataset, Results
Download larger files from this google drive link. 
https://drive.google.com/file/d/1PrtiMcKFvxqeKYma4nB8_XEZJPCGBmZ8/view?usp=sharing

Move the checkpoints, dataset, and results files to your HAMscope_pix2pix directory


## Usage Examples

### Training - Included Model is Pretrained, skip this
```bash
python train.py \
  --dataroot /path/to/dataset \
  --name experiment_name \
  --model pix2pix \
  --input_nc 1 \
  --output_nc 60 \
  --netG unet_512 \
  --use_nll 1 \
  --use_comp 3 \
  --use_reg 1 \
  --use_trans 0 \
  --netD_mult 0.005 \
  --norm layer \
  --n_epochs 15 \
  --n_epochs_decay 15
```

### Testing Included Model
```bash
# Navigate to the dir
cd HAMscope/HAMscope_pix2pix

# activate the environment
conda activate pix2pix

# Test the model 
# Test results are also included, skip this
python test.py \
  --dataroot ./dataset/test_dataset \
  --name 20250425_0gan_single_reg_hs \
  --model pix2pix \
  --input_nc 1 \
  --output_nc 60 \
  --netG unet_512 \
  --use_nll 1 \
  --use_comp 3 \
  --use_reg 1 \
  --use_trans 0 \
  --norm layer \
  --num_test 100

# Analyze the results
python spectra_comparison_probabilistic.py


# For biomolecule mapping model, use this test command
python test.py \
  --dataroot ./dataset/test_dataset \
  --name 20250425_0gan_single_reg_hs_nmf \
  --model pix2pix \
  --input_nc 1 \
  --output_nc 6 \
  --netG unet_512 \
  --use_nll 1 \
  --use_comp 2 \
  --use_reg 1 \
  --use_trans 0 \
  --norm layer \
  --num_test 100
```

## License

This project builds upon the original pix2pix implementation and maintains BSD license terms. See [THIRD_PARTY_ATTRIBUTION.md](../THIRD_PARTY_ATTRIBUTION.md) for complete attribution details.
```