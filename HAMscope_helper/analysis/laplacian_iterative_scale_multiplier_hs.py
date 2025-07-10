import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pathlib import Path
import math
import tifffile

def load_hyperspectral_images(real_path, fake_path):
    """Load hyperspectral images from TIF files"""
    # Load real image (30 channels)
    real_image = tifffile.imread(real_path)  # Shape: (30, H, W)
    
    # Load fake image (60 channels: first 30 are fake, last 30 are scale)
    fake_full = tifffile.imread(fake_path)  # Shape: (60, H, W)
    fake_image = fake_full[:30]  # First 30 channels
    scale_image = fake_full[30:]  # Last 30 channels
    
    # Convert to torch tensors and normalize to [0, 1] if needed
    real_tensor = torch.from_numpy(real_image.astype(np.float32))
    fake_tensor = torch.from_numpy(fake_image.astype(np.float32))
    scale_tensor = torch.from_numpy(scale_image.astype(np.float32))
    
    # Normalize if values are not in [0, 1] range
    if real_tensor.max() > 1.0:
        print('normalizing real tensor')
        real_tensor = real_tensor / real_tensor.max()
    if fake_tensor.max() > 1.0:
        print('normalizing fake tensor')
        fake_tensor = fake_tensor / fake_tensor.max()
    if scale_tensor.max() > 1.0:
        print('normalizing scale tensor')
        scale_tensor = scale_tensor / scale_tensor.max()
    
    return real_tensor, fake_tensor, scale_tensor

def load_images_from_folders(folders, filename):
    means = []
    scales = []
    for folder in folders:
        mean_path = os.path.join(folder, f'{filename}_fake_mean.png')
        scale_path = os.path.join(folder, f'{filename}_fake_scale.png')
        mean_image = Image.open(mean_path)
        print(f'max{np.max(mean_image)}')
        print(f'min{np.min(mean_image)}')
        print(f'filename{filename}')
        scale_image = Image.open(scale_path)
        means.append(transforms.ToTensor()(mean_image))
        scales.append(transforms.ToTensor()(scale_image))
    return means, scales

def load_ground_truth_image(folder, filename):
    ground_truth_path = os.path.join(folder, f'{filename}_real_B.png')
    ground_truth_image = Image.open(ground_truth_path)
    return transforms.ToTensor()(ground_truth_image)

def ensemble_predictions(means, scales):
    mean_ensemble = torch.stack(means).mean(dim=0)
    scale_ensemble = torch.stack(scales).mean(dim=0)
    return mean_ensemble, scale_ensemble

def laplace_pdf(z, mu, sigma):
    sigma += 1e-7  # Avoid division by zero
    return (1 / (2 * sigma)) * torch.exp(-torch.abs(z - mu) / sigma)

def gaussian_pdf(z, mu, sigma):
    sigma += 1e-7  # Avoid division by zero
    return (1 / (sigma * torch.sqrt(torch.tensor(2 * math.pi)))) * torch.exp(-0.5 * ((z - mu) / sigma) ** 2)


def evaluate_confidence_interval(real_images, fake_images, scale_images, multiplier, target_confidence, num_samples=20, ):
    z_values = torch.linspace(0, 1, 200)
    total_within_interval = 0
    total_pixels = 0
    differences = []

    for real_img_path, fake_img_path, scale_img_path in zip(real_images, fake_images, scale_images):
        real_image = transforms.ToTensor()(Image.open(real_img_path))
        fake_image = transforms.ToTensor()(Image.open(fake_img_path))
        scale_image = transforms.ToTensor()(Image.open(scale_img_path))

        height, width = fake_image.shape[1:]
        indices = torch.randperm(height * width)[:num_samples]
        sampled_pixels = [(idx // width, idx % width) for idx in indices]

        for i, j in sampled_pixels:
            if fake_image[0, i, j].item() < 0:# or fake_image[0, i, j].item() > 0.95:
                continue
            mean_value = fake_image[0, i, j].item()
            #ave_laplacian = laplace_pdf(z_values, fake_image[0, i, j], scale_image[0, i, j] * multiplier)
            ave_laplacian = laplace_pdf(z_values, fake_image[0, i, j], scale_image[0, i, j] * multiplier)

            cumulative_sum = ave_laplacian.cumsum(dim=0)
            cumulative_sum_normalized = cumulative_sum / cumulative_sum[-1]  # Normalize to get CDF
            lower_idx = torch.searchsorted(cumulative_sum_normalized, 0.5 - target_confidence / 2).item()
            upper_idx = torch.searchsorted(cumulative_sum_normalized, 0.5 + target_confidence / 2).item()
            lower_bound = z_values[lower_idx-1].item()
            upper_bound = z_values[upper_idx-1].item()

            in_interval = False
            if lower_bound <= real_image[0, i, j] <= upper_bound:
                total_within_interval += 1
                in_interval = True
            total_pixels += 1
            differences.append(mean_value - real_image[0, i, j].item())

            plot_figs = False
            if plot_figs==True:
                plt.figure(figsize=(10, 6))
                plt.plot(z_values.numpy(), ave_laplacian.numpy(), label='Laplacian PDF')
                plt.axvline(mean_value, color='r', linestyle='--', label=f'Mean Value: {mean_value:.2f}')
                plt.axvline(lower_bound, color='g', linestyle='--', label=f'Lower Bound: {lower_bound:.2f}')
                plt.axvline(upper_bound, color='b', linestyle='--', label=f'Upper Bound: {upper_bound:.2f}')
                plt.axvline(real_image[0, i, j].item(), color='k', linestyle='--', label=f'Real Value: {real_image[0, i, j].item():.2f}')
                plt.text(0.1, 0.9, f'In: {in_interval}', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
                plt.title(f'Laplacian Distribution at ({i}, {j})')
                plt.xlabel('Intensity')
                plt.ylabel('Probability Density')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
                plt.tight_layout()
                plt.show()
        plot_distribution = False
        if plot_distribution == True:
            plt.figure(figsize=(10, 6))
            plt.hist(differences, bins=50, alpha=0.75, edgecolor='black')
            plt.title('Distribution of Differences Between Fake Mean and Real Pixel Values')
            plt.xlabel('Difference')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()

    return total_within_interval / total_pixels

def evaluate_confidence_interval_hyperspectral(real_images, fake_images, multiplier, target_confidence, num_samples=20, channel_idx=0):
    """Evaluate confidence interval for hyperspectral images"""
    z_values = torch.linspace(0, 1, 1000)
    total_within_interval = 0
    total_pixels = 0
    differences = []

    for real_img_path, fake_img_path in zip(real_images, fake_images):
        real_image, fake_image, scale_image = load_hyperspectral_images(real_img_path, fake_img_path)
        
        # Select specific channel or average across channels
        if channel_idx == -1:  # Use all channels
            real_ch = real_image.mean(dim=0)  # Average across channels
            fake_ch = fake_image.mean(dim=0)
            scale_ch = scale_image.mean(dim=0)
        else:
            real_ch = real_image[channel_idx]
            fake_ch = fake_image[channel_idx]
            scale_ch = scale_image[channel_idx]

        height, width = fake_ch.shape
        indices = torch.randperm(height * width)[:num_samples]
        sampled_pixels = [(idx // width, idx % width) for idx in indices]

        for i, j in sampled_pixels:
            if fake_ch[i, j].item() < 0:
                continue
                
            mean_value = fake_ch[i, j].item()
            ave_laplacian = laplace_pdf(z_values, fake_ch[i, j], scale_ch[i, j] * multiplier)

            cumulative_sum = ave_laplacian.cumsum(dim=0)
            cumulative_sum_normalized = cumulative_sum / cumulative_sum[-1]
            lower_idx = torch.searchsorted(cumulative_sum_normalized, 0.5 - target_confidence / 2).item()
            upper_idx = torch.searchsorted(cumulative_sum_normalized, 0.5 + target_confidence / 2).item()
            lower_bound = z_values[max(0, lower_idx-1)].item()
            upper_bound = z_values[min(len(z_values)-1, upper_idx-1)].item()

            if lower_bound <= real_ch[i, j] <= upper_bound:
                total_within_interval += 1
            total_pixels += 1
            differences.append(mean_value - real_ch[i, j].item())

    return total_within_interval / total_pixels

def optimize_multiplier(real_images, fake_images, scale_images, target_confidence=0.90, learning_rate=0.1, epochs=100, num_samples=20):
    multiplier = 1

    for epoch in range(epochs):
        #learning_rate = learning_rate / (epoch/2 + 1)
        current_confidence = evaluate_confidence_interval(real_images, fake_images, scale_images, multiplier, target_confidence, num_samples)
        if epoch == 0:
            print(f'Initial Confidence: {current_confidence}')
        loss = (current_confidence - target_confidence) ** 2
        gradient = 2 * (current_confidence - target_confidence)
        multiplier -= learning_rate * gradient
        #print(f'Epoch {epoch+1}/{epochs}, Multiplier: {multiplier}, Confidence: {current_confidence}, Loss: {loss}')
        if loss < 1e-5:
            break

    return multiplier

def optimize_multiplier_hyperspectral(real_images, fake_images, target_confidence=0.90, learning_rate=0.1, epochs=100, num_samples=20, channel_idx=0):
    """Optimize multiplier for hyperspectral images"""
    multiplier = 1

    for epoch in range(epochs):
        current_confidence = evaluate_confidence_interval_hyperspectral(
            real_images, fake_images, multiplier, target_confidence, num_samples, channel_idx
        )
        if epoch == 0:
            print(f'Initial Confidence: {current_confidence}')
        loss = (current_confidence - target_confidence) ** 2
        gradient = 2 * (current_confidence - target_confidence)
        multiplier -= learning_rate * gradient
        if loss < 1e-5:
            break

    return multiplier

# Update the image loading section
img_dir = Path('/media/al/Extreme SSD/20250425_results/results_norm/20250425_0gan_single_1/test_latest/images')
img_dir = Path('/media/al/Extreme SSD/20250425_results/results/analyzed_20250613/20250425_0gan_single_hs_3/test_latest/images')

# Load hyperspectral images
real_images = list(img_dir.glob('hs_raw_*.tif'))  # Real images: hs_raw_n.tif
fake_images = []

# Match fake images to real images
for real_img in real_images:
    # Extract the number from hs_raw_n.tif
    img_number = real_img.stem.split('_')[-1]  # Get 'n' from 'hs_raw_n'
    # Look for corresponding fake image (you'll need to adjust the naming pattern)
    fake_img = img_dir / f"hs_gen_{img_number}.tif"  # Adjust this pattern as needed
    if fake_img.exists():
        fake_images.append(fake_img)
    else:
        print(f"Warning: No corresponding fake image found for {real_img}")

# Filter real_images to only include those with corresponding fake images
paired_real = []
paired_fake = []
for real_img, fake_img in zip(real_images, fake_images):
    if fake_img.exists():
        paired_real.append(real_img)
        paired_fake.append(fake_img)

# Limit to first 10 image pairs for faster testing
n_images = 10
paired_real = paired_real[:n_images]
paired_fake = paired_fake[:n_images]

print(f"Using {len(paired_real)} image pairs for testing")

#intervals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
intervals = [0.9]

# Test with different channels
for channel_idx in [0, 15, 29, -1]:  # First, middle, last channel, and average
    print(f"\n--- Channel {channel_idx if channel_idx != -1 else 'Average'} ---")
    for interval in intervals:
        optimized_multiplier = optimize_multiplier_hyperspectral(
            paired_real, paired_fake, target_confidence=interval, channel_idx=channel_idx
        )
        print(f'Confidence Interval: {interval}, Optimized Multiplier: {optimized_multiplier:.4f}')
