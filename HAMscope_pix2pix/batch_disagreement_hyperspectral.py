import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
from tifffile import imwrite, imread


def normalize_mean_img(img):
    #img = img / 255.0
    percentile_5 = np.percentile(img, 2.5)
    percentile_95 = np.percentile(img, 97.5)
    mean = np.mean(img)
    img = (img - percentile_5) / (percentile_95 - percentile_5 + 0.0001)
    img = np.clip(img, 0, 1)
    return img, percentile_5, percentile_95, mean

def save_scalebar(image, min_val, max_val, path):
    fig, ax = plt.subplots()
    image_display = ax.imshow(image, cmap='gray')

    # Remove the axes
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Create a colorbar with specified ticks across the normalized range
    num_ticks = 5  # For example, 5 ticks
    tick_values = np.linspace(0, 1, num_ticks)
    #tick_labels = [f"{v:.2f}" for v in np.linspace(min_val, max_val, num_ticks)]
    tick_labels = [f"{v:.2f}" for v in np.linspace(0, 0.5, num_ticks)] # change for fixed scale between 0 and 0.5


    # Create the colorbar
    cbar = fig.colorbar(image_display, cax=cax, ticks=tick_values)
    cbar.ax.set_aspect(18)  # Adjust for desired height
    default_font_size = plt.rcParams.get('font.size')  # Get the default font size
    cbar.ax.set_yticklabels(tick_labels, fontsize=default_font_size * 2)  # Set text labels on the colorbar

    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def load_images_from_folders(folders, filename):
    means = []
    scales = []
    folder_num = 0
    for folder in folders:
        file_path = os.path.join(folder, f'{filename}.tif')
        tif_stack = imread(file_path)  # Read the .tif stack

        # on the second folder add 0.01 to all images, and on the third folder add 0.02
        if folder_num == 1:
            tif_stack = tif_stack + 0.01
        elif folder_num == 2:
            tif_stack = tif_stack + 0.02
        elif folder_num == 3:
            tif_stack = tif_stack + 0.03
        elif folder_num == 4:
            tif_stack = tif_stack + 0.04

        folder_num += 1

        print(f'tif_stack max: {tif_stack.max()}')
        print(f'tif_stack min: {tif_stack.min()}')

        # clip 0 to 1
        tif_stack = np.clip(tif_stack, 0, 1)

        # Extract channels 1-3 for mean and 4-6 for scale
        mean_image = torch.tensor(tif_stack[0:3, :, :])
        scale_image = torch.tensor(tif_stack[3:6, :, :])

        #plt.imshow(mean_image[0], cmap='gray')
        #lt.show()

        means.append(mean_image)
        scales.append(scale_image)

    return means, scales

def ensemble_predictions(means, scales):
    M = len(means)
    mean_ensemble = torch.stack(means).mean(dim=0)
    scale_ensemble = torch.stack(scales).mean(dim=0)
    return mean_ensemble, scale_ensemble

def kl_divergence(p_mean, p_scale, q_mean, q_scale):

    return torch.log(q_scale / (p_scale + 0.0000001)) + (p_scale + (p_mean - q_mean).pow(2)) / (2 * q_scale.pow(2) + 0.0000001) - 0.5

def compute_disagreement(means, scales, ensemble_mean, ensemble_scale):
    M = len(means)
    disagreements = []

    for mean, scale in zip(means, scales):
        # Compute KL divergence for each channel independently
        kl_per_channel = []
        for c in range(mean.shape[0]):  # Iterate over channels (e.g., 3 for RGB)
            kl = kl_divergence(mean[c], scale[c], ensemble_mean[c], ensemble_scale[c])
            #print(f'kl shape: {kl.shape}')
            #print(f'kl max: {kl.max()}')
            kl_per_channel.append(kl)
        disagreements.append(torch.stack(kl_per_channel))  # Stack per-channel KL divergences

    # Average disagreement across all channels and normalize
    disagreement_score = sum(disagreements) / M / np.log(M)
    #print(f'disagreement score shape: {disagreement_score.shape}')
    return disagreement_score

def calculate_disagreement_for_all_images(folders, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Get the list of filenames (without extensions) from the first folder
    filenames = sorted([os.path.splitext(f)[0] for f in os.listdir(folders[0]) if f.startswith('tl_gen_') and f.endswith('.tif')])
    
    print(np.shape(filenames))

    count = 0
    disagreement_scores = []
    
    for filename in filenames:
        means, scales = load_images_from_folders(folders, filename)
        ensemble_mean, ensemble_scale = ensemble_predictions(means, scales)
        disagreement = compute_disagreement(means, scales, ensemble_mean, ensemble_scale)
        mean_disagreement = disagreement.mean().item()

        print(f'disagreement max: {disagreement.max()}')
        print(f'ensemble_mean max: {ensemble_mean.max()}')

        # Normalize the disagreement image
        disagreement_np = disagreement.numpy()  # Convert to numpy array and scale to 255
        normalized_disagreement = disagreement_np



        # Save the normalized disagreement image
        output_path = os.path.join(output_folder, f'disagreement_{filename}_D_{mean_disagreement:.4g}.tif')

        #print(f'normalized_disagreement max: {normalized_disagreement.max()}')
        #print(f'normalized_disagreement min: {normalized_disagreement.min()}')

        imwrite(output_path, normalized_disagreement)

        # Save the mean image
        mean_image_np = ensemble_mean.numpy()  # Keep as 3-channel RGB
        normalized_mean_image, mean_min_val, mean_max_val, _ = normalize_mean_img(mean_image_np)


        mean_output_path = os.path.join(output_folder, f'normalized_mean_{filename}.tif')
        imwrite(mean_output_path, normalized_mean_image)
        
        # Accumulate disagreement score
        disagreement_scores.append(mean_disagreement)
        count += 1
    
    # Calculate and print the average disagreement score
    average_disagreement_score = np.mean(disagreement_scores)
    std_disagreement_score = np.std(disagreement_scores)
    print(output_folder)
    print(f"Average Disagreement Score: {average_disagreement_score}")
    print(f"Standard Deviation of Disagreement Scores: {std_disagreement_score}")


input_output_pairs = [
#    ([
#    "/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/results/20250222_512_large_gan_double_log_sigma_compounds_1/test_latest/images",
#    "/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/results/20250222_512_large_gan_double_log_sigma_compounds_2/test_latest/images",
#    "/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/results/20250222_512_large_gan_double_log_sigma_compounds_3/test_latest/images",
#    "/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/results/20250222_512_large_gan_double_log_sigma_compounds_4/test_latest/images",
#    "/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/results/20250222_512_large_gan_double_log_sigma_compounds_5/test_latest/images"],
#    '/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/results/20250222_512_large_gan_double_log_sigma_compounds_disagreement'),

    ([
    "/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/results/main_test_dataset/20250222_512_large_gan_double_log_sigma_compounds/test_latest/images",
    "/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/results/main_test_dataset/20250222_512_large_gan_double_log_sigma_compounds/test_latest/images",
    "/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/results/main_test_dataset/20250222_512_large_gan_double_log_sigma_compounds/test_latest/images",
    "/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/results/main_test_dataset/20250222_512_large_gan_double_log_sigma_compounds/test_latest/images",
    "/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/results/main_test_dataset/20250222_512_large_gan_double_log_sigma_compounds/test_latest/images"],
    '/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/results/20250222_512_large_gan_double_log_sigma_compounds_disagreement_same'),
]

#calculate_disagreement_for_all_images(folders, output_folder)
for input_folders, output_folder in input_output_pairs:
    calculate_disagreement_for_all_images(input_folders, output_folder)