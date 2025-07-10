import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pathlib import Path
import math
import tifffile as tiff

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


def evaluate_confidence_interval(real_images, fake_images, multiplier, target_confidence, num_samples=20, ):
    z_values = torch.linspace(0, 1, 1000)
    total_within_interval = 0
    total_pixels = 0
    differences = []

    for real_img_path, fake_img_path in zip(real_images, fake_images):
        real_image = transforms.ToTensor()(tiff.imread(real_img_path))
        fake_image = transforms.ToTensor()(tiff.imread(fake_img_path))


        #scale_image = abs(real_image - fake_image)
        scale_image = fake_image[:, fake_image.shape[1]//2:, :]
        fake_image = fake_image[:, :fake_image.shape[1]//2, :]

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
            '''
            ave_laplacian = laplace_pdf(z_values, fake_image[0, i, j], scale_image[0, i, j] * multiplier)
            cumulative_sum = ave_laplacian.cumsum(dim=0)
            cumulative_sum_normalized = cumulative_sum / cumulative_sum[-1]

            
            #print(f'mean_value: {mean_value}')
            insert_idx = torch.searchsorted(z_values, mean_value).item()   sigma += 1e-7  # Avoid division by zero
            if insert_idx >= len(z_values):
                insert_idx = len(z_values) - 1


            mean_percentile = cumulative_sum_normalized[insert_idx].item()


            if math.isnan(mean_percentile):
                #print(f'NAN at ({i}, {j})')
                mean_percentile = 0
                
            lower_percentile_value = max(mean_percentile - target_confidence/2, 0)
            upper_percentile_value = min(mean_percentile + target_confidence/2, 1)
            #print(f'lower_percentile_value: {lower_percentile_value}')
            #print(f'upper_percentile_value: {upper_percentile_value}')
            #print(f'mean_percentile: {mean_percentile}')
            lower_idx = torch.searchsorted(cumulative_sum_normalized, lower_percentile_value).item()
            upper_idx = torch.searchsorted(cumulative_sum_normalized, upper_percentile_value).item()

            lower_idx = max(0, lower_idx)
            lower_idx = min(len(z_values) - 1, lower_idx)
            upper_idx = min(len(z_values) - 1, upper_idx)

            #print(lower_idx, upper_idx)

            lower_bound = z_values[lower_idx].item()
            upper_bound = z_values[upper_idx].item()

            lower_bound = max(0, lower_bound)
            lower_bound = min(mean_value, lower_bound)
            upper_bound = max(mean_value, upper_bound)
            upper_bound = min(1, upper_bound)
            '''
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

def optimize_multiplier(real_images, fake_images, target_confidence=0.90, learning_rate=0.1, epochs=10000, num_samples=20):
    multiplier = 1

    for epoch in range(epochs):
        #learning_rate = learning_rate / (epoch/2 + 1)
        current_confidence = evaluate_confidence_interval(real_images, fake_images, multiplier, target_confidence, num_samples)
        if epoch == 0:
            print(f'Initial Confidence: {current_confidence}')
        loss = (current_confidence - target_confidence) ** 2
        gradient = 2 * (current_confidence - target_confidence)
        multiplier -= learning_rate * gradient
        #print(f'Epoch {epoch+1}/{epochs}, Multiplier: {multiplier}, Confidence: {current_confidence}, Loss: {loss}')
        if loss < 1e-5:
            break

    return multiplier
#img_dir = Path('/home/alingold/probabilistic_pix2pix/results/combined_plant_0bp_noBlurryRemoval_n_vis_norm/test_latest/images')
#img_dir = Path('/home/alingold/probabilistic_pix2pix/results/checkpoints_chpc_oldtest/combined_plant_0bp_noBlurryRemoval_probabilistic_new_1_layer_nogan_0/test_latest/images')
#img_dir = Path('/home/alingold/CSBDeep-main/examples/denoising2D_probabilistic/results')
#img_dir = Path('/home/alingold/CSBDeep-main/examples/denoising2D_probabilistic/results_example')
#img_dir = Path('/home/alingold/probabilistic_pix2pix/results/combined_plant_0bp_noBlurryRemoval_pdf_loss/test_latest/images')
#img_dir = Path('/home/alingold/probabilistic_pix2pix/results/combined_plant_0bp_noBlurryRemoval_nll_no_norm/test_latest/images')
#img_dir = Path('/home/alingold/probabilistic_pix2pix/results/combined_plant_0bp_noBlurryRemoval_pdf_loss_over_100/test_latest/images')
#img_dir1 = Path('/home/alingold/pix2pix/results/combined_plant_0bp_noBlurryRemoval_oldtest_ForEdison/test_latest/images')
#img_dir2 = Path('/home/alingold/pix2pix/results/difference_pix2pix/test_latest/images')
#img_dir1 = Path('/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/results/main_test_dataset/20250222_512_large_gan_double_log_sigma_compounds/test_latest/images')
img_dir1 = Path('/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/results/20250222_512_large_gan_double_log_sigma_compounds_test_from_train/test_latest/images')


#real_images = list(img_dir.glob('*real_B.png'))
#fake_images = [img_dir / f"{img.stem.replace('_real_B', '_fake_mean')}.png" for img in real_images]
#scale_images = [img_dir / f"{img.stem.replace('_real_B', '_fake_scale')}.png" for img in real_images]

#real_images = list(img_dir1.glob('*real_B.png'))
#fake_images = [img_dir1 / f"{img.stem.replace('_real_B', '_fake_B')}.png" for img in real_images]
#scale_images = [img_dir2 / f"{img.stem.replace('_real_B', '_fake_B')}.png" for img in real_images]

real_images = list(img_dir1.glob('cb_raw_*.tif'))
fake_images = [img_dir1 / f"tl_gen_{img.stem.split('_')[-1]}.tif" for img in real_images]


intervals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
#intervals = [0.7]
for interval in intervals:
    optimized_multiplier = optimize_multiplier(real_images, fake_images, target_confidence=interval)
    print(f'Confidence Interval: {interval}, Optimized Multiplier: {optimized_multiplier}')
    #print(f'Optimized Multiplier: {optimized_multiplier}')
