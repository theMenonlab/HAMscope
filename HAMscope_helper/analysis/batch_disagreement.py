import os
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
import tifffile as tiff
import cv2


def normalize_mean_img(img):
    # max of all channels
    max_val = np.max(img)
    min_val = np.min(img)
    # Normalize to [0, 1]
    img = (img - min_val) / (max_val - min_val + 1e-8)
    img = img * 255.0
    img = img.astype(np.uint8)
    return img

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

def save_image(image, path):
    """
    Save an image with colorbar similar to compound_tif_to_png.py
    Handles both regular RGB images and hyperspectral data
    """
    # Check shape and ensure we're dealing with a properly formatted image
    print(f"image shape: {image.shape}")
    
    # For saving .tif files directly without conversion
    if path.endswith('.tif') or path.endswith('.tiff'):
        tiff.imwrite(path, image)
        return
    
    # Determine if this is a hyperspectral image (>3 channels)
    is_hyperspectral = False
    
    if image.shape[0] > 3:  # More than 3 channels in CHW format
        is_hyperspectral = True
    rgb_image = image  # Keep in CHW format for hyperspectral handler

    
    if path.endswith('_disagreement.png') or 'disagreement_' in path:
        # Add colorbar for disagreement images
        if is_hyperspectral:
            rgb_image = add_disagreement_colorbar_hyperspectral(rgb_image)
        else:
            rgb_image = add_disagreement_colorbar_rgb(rgb_image)
    
    # Save the image using PIL to avoid matplotlib colormap issues
    from PIL import Image
    print(f' rgb image min and max: {np.min(rgb_image)} {np.max(rgb_image)}')
    print(f' rgb image shape: {rgb_image.shape}')
    
    # Make sure image is in proper format for PIL
    if rgb_image.ndim == 3 and rgb_image.shape[2] == 3:
        # Standard RGB image
        Image.fromarray(rgb_image.astype(np.uint8)).save(path)
    elif rgb_image.ndim == 2:
        # Grayscale image
        Image.fromarray(rgb_image.astype(np.uint8), mode='L').save(path)
    else:
        # For any other unusual format, try to convert to standard format or use tiff
        print(f"Warning: Unusual image shape {rgb_image.shape}. Saving as TIFF.")
        tiff_path = path.replace('.png', '.tif')
        tiff.imwrite(tiff_path, rgb_image)

def add_disagreement_colorbar_rgb(image):
    """
    Add RGB colorbars to the disagreement image similar to compound_tif_to_png.py
    """    
    # Work with a copy to avoid modifying the original
    image_copy = image.copy()

    # print image copy shape
    print(f' image copy shape: {image_copy.shape}')
    
    # Access channels for min/max calculation
    channel_min = np.zeros(image_copy.shape[0])
    channel_max = np.zeros(image_copy.shape[0])
    for i in range(3):
        channel_min[i] = np.min(image_copy[i])
        channel_max[i] = np.max(image_copy[i])
        # Normalize each channel
        image_copy[i] = (image_copy[i] - channel_min[i]) / (channel_max[i] - channel_min[i] + 0.0001) * 255
    
    # Convert back to HWC after normalization
    image_hwc = np.transpose(image_copy.astype(np.uint8), (1, 2, 0))
    
    print(f'normalized image min: {np.min(image_hwc[:,:,0])} {np.min(image_hwc[:,:,1])} {np.min(image_hwc[:,:,2])}')
    print(f'normalized image max: {np.max(image_hwc[:,:,0])} {np.max(image_hwc[:,:,1])} {np.max(image_hwc[:,:,2])}')

    # Get image dimensions
    height, width = image_hwc.shape[:2]
    
    # Create color bar parameters
    bar_height = 25  # Height of each color bar
    margin = 20  # Margin between bars
    text_height = 30  # Height for text
    
    # Total height: 3 bars + 2 margins + text at top and bottom
    colorbar_height = (3 * bar_height) + (2 * margin) + (2 * text_height)
    
    # Create a black background image for the color bars
    colorbar_img = np.zeros((colorbar_height, width, 3), dtype=np.uint8)
    
    # Font properties
    font_scale = 0.5
    font_thickness = 1
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    
    # Colors for each channel (BGR in OpenCV)
    channel_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR format for R,G,B
    channel_names = ['Red', 'Green', 'Blue']
    
    # Create color bars for each channel
    for i in range(image_copy.shape[0]):
        # Calculate y-position for this bar
        y_start = i * (bar_height + margin) + text_height
        y_end = y_start + bar_height
        
        # Create gradient for this channel
        for x in range(width):
            normalized_x = x / width
            value = int(normalized_x * 255)
            
            # Create a color array with the current channel's color at the appropriate intensity
            color = [0, 0, 0]  # Initialize with black
            color[2-i] = value  # Set the appropriate channel (convert RGB index to BGR for OpenCV)
            
            # Fill the column with this color
            colorbar_img[y_start:y_end, x] = color
        
        # Add border around the color bar
        print(f' y_start: {y_start} y_end: {y_end}')
        print(f'width: {width}, height: {height}')
        colorbar_img[y_start-1, 0:width] = [255, 255, 255]
        colorbar_img[y_end, 0:width] = [255, 255, 255]
        colorbar_img[y_start-1:y_end+1, 0] = [255, 255, 255]
        colorbar_img[y_start-1:y_end+1, width-1] = [255, 255, 255]
        
        # Add min/max value labels
        min_text = f"Min: {channel_min[i]:.4f}"
        max_text = f"Max: {channel_max[i]:.4f}"
        
        # Calculate text positions
        min_text_size = cv2.getTextSize(min_text, font_face, font_scale, font_thickness)[0]
        max_text_size = cv2.getTextSize(max_text, font_face, font_scale, font_thickness)[0]
        
        min_text_x = 5
        min_text_y = y_start - 5
        max_text_x = width - max_text_size[0] - 5
        max_text_y = y_start - 5
        
        # Add channel name
        channel_text = channel_names[i]
        channel_text_size = cv2.getTextSize(channel_text, font_face, font_scale, font_thickness)[0]
        channel_text_x = (width - channel_text_size[0]) // 2
        channel_text_y = min_text_y
        
        # Draw text in white
        cv2.putText(colorbar_img, f'Min: {channel_min[i]:.4f}', (min_text_x, min_text_y), 
                    font_face, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(colorbar_img, f'Max: {channel_max[i]:.4f}', (max_text_x, max_text_y), 
                    font_face, font_scale, (255, 255, 255), font_thickness)
        cv2.putText(colorbar_img, channel_text, (channel_text_x, channel_text_y), 
                    font_face, font_scale, (255, 255, 255), font_thickness)
    
    # Combine the image with the color bars
    combined_height = height + colorbar_height
    combined_img = np.zeros((combined_height, width, image_copy.shape[0]), dtype=np.uint8)
    
    # Copy the image to the top portion
    combined_img[:height, :, :] = image_hwc
    
    # Copy the color bars to the bottom portion
    combined_img[height:, :, :] = colorbar_img
    
    return combined_img


def add_disagreement_colorbar_hyperspectral(image):
    """
    Add grayscale colorbars for each channel of a hyperspectral disagreement image
    Works with >3 channel images, creating a separate grayscale gradient for each channel
    """    
    # Work with a copy to avoid modifying the original
    image_copy = image.copy()
    num_channels = image_copy.shape[0]
    
    print(f' Creating hyperspectral colorbar for {num_channels} channels')
    
    # Access channels for min/max calculation
    channel_min = np.zeros(num_channels)
    channel_max = np.zeros(num_channels)
    
    # Normalize each channel and convert to 8-bit
    normalized_channels = []
    for i in range(num_channels):
        channel_min[i] = np.min(image_copy[i])
        channel_max[i] = np.max(image_copy[i])
        # Normalize the channel
        normalized = (image_copy[i] - channel_min[i]) / (channel_max[i] - channel_min[i]) * 255
        normalized_channels.append(normalized.astype(np.uint8))
    
    # Get image dimensions from the first channel
    height, width = normalized_channels[0].shape

    image_copy = np.stack(normalized_channels, axis=0)

    
    # Create color bar parameters
    bar_height = 25  # Height of each color bar
    margin = 10  # Margin between bars
    text_height = 30  # Height for text
    
    # Calculate total height needed for all channel bars
    colorbar_height = (bar_height) + margin + 2 * text_height
    

    
    # Font properties
    font_scale = 0.5
    font_thickness = 1
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    
    # Create channel names based on number of channels
    if num_channels == 6:  # Special case for 6 channels
        channel_names = ['400', '450', '500', '550', '600', '650']
    else:
        channel_names = [f'Band {i+1}' for i in range(num_channels)]

    hyperspectral_img = np.zeros((num_channels, colorbar_height+image_copy.shape[1], width), dtype=np.uint8)
    
    # Create grayscale bars for each channel
    for i in range(num_channels):

        # Create a black background image for the color bars
        colorbar_img = np.zeros((colorbar_height, width), dtype=np.uint8)

        # Calculate y-position for this bar
        y_start = (bar_height + margin) + text_height
        y_end = y_start + bar_height
        
        # Create gradient for this channel
        for x in range(width):
            normalized_x = x / width
            value = int(normalized_x * 255)
            colorbar_img[y_start:y_end, x] = value
        
        # Add border around the color bar (white)
        cv2.line(colorbar_img, (0, y_start-1), (width-1, y_start-1), 255, 1)
        cv2.line(colorbar_img, (0, y_end), (width-1, y_end), 255, 1)
        cv2.line(colorbar_img, (0, y_start-1), (0, y_end), 255, 1)
        cv2.line(colorbar_img, (width-1, y_start-1), (width-1, y_end), 255, 1)
        
        # Add min/max value labels
        min_text = f"Min: {channel_min[i]:.4f}"
        max_text = f"Max: {channel_max[i]:.4f}"
        
        # Calculate text positions
        min_text_size = cv2.getTextSize(min_text, font_face, font_scale, font_thickness)[0]
        max_text_size = cv2.getTextSize(max_text, font_face, font_scale, font_thickness)[0]
        
        min_text_x = 5
        min_text_y = y_start - 5
        max_text_x = width - max_text_size[0] - 5
        max_text_y = y_start - 5
        
        # Add channel name
        channel_text = channel_names[i]
        channel_text_size = cv2.getTextSize(channel_text, font_face, font_scale, font_thickness)[0]
        channel_text_x = (width - channel_text_size[0]) // 2
        channel_text_y = min_text_y
        
        # Draw text in white
        cv2.putText(colorbar_img, min_text, (min_text_x, min_text_y), 
                    font_face, font_scale, 255, font_thickness)
        cv2.putText(colorbar_img, max_text, (max_text_x, max_text_y), 
                    font_face, font_scale, 255, font_thickness)
        cv2.putText(colorbar_img, channel_text, (channel_text_x, channel_text_y), 
                    font_face, font_scale, 255, font_thickness)
    
        # Instead of displaying all channels side by side, just use the first channel
        # with a colorbar that explains all channels
        selected_channel = normalized_channels[0]  # Use first channel for display
        
        # Combine the image with the colorbar
        combined_height = image_copy.shape[1] + colorbar_height
        combined_width = image_copy.shape[2]
        combined_img = np.zeros((combined_height, combined_width), dtype=np.uint8)
        
        # Copy the image to the top portion
        combined_img[:image_copy.shape[1], :] = image_copy[i]
        
        # Copy the color bar to the bottom portion
        combined_img[image_copy.shape[1]:, :] = colorbar_img

        # append
        hyperspectral_img[i] = combined_img

    
    
    return hyperspectral_img

def load_images_from_folders(folders, filename):
    print('loading images from folders')
    means = []
    scales = []
    for folder in folders:
        path = os.path.join(folder, f'hs_gen_{filename}.tif')
        image = tiff.imread(path)

        # put the smalest dim first
        # find the index of the smallest dim
        smallest_dim_index = np.argmin([image.shape[0], image.shape[1], image.shape[2]])
        # permute the image to put the smallest dim first
        if smallest_dim_index == 0:
            image = np.transpose(image, (0, 1, 2))
        elif smallest_dim_index == 1:
            image = np.transpose(image, (1, 0, 2))
        elif smallest_dim_index == 2:
            image = np.transpose(image, (2, 0, 1))
        
        # Check if normalization is needed (max > 2)
        raw_norm = 1
        if raw_norm:
            print(f'Max value {np.max(image)} > 2, normalizing using hs_raw_{filename}.tif')
            
            # Load the corresponding hs_raw image
            raw_path = os.path.join(folder, f'hs_raw_{filename}.tif')
            if os.path.exists(raw_path):
                raw_image = tiff.imread(raw_path)
                
                # Apply same dimension permutation to raw image
                if smallest_dim_index == 0:
                    raw_image = np.transpose(raw_image, (0, 1, 2))
                elif smallest_dim_index == 1:
                    raw_image = np.transpose(raw_image, (1, 0, 2))
                elif smallest_dim_index == 2:
                    raw_image = np.transpose(raw_image, (2, 0, 1))
                
                # Get min and max from raw image
                raw_min = np.min(raw_image)
                raw_max = np.max(raw_image)
                
                print(f'Raw image min: {raw_min}, max: {raw_max}')
                
                # Normalize hs_gen image using raw image min/max
                image = (image - raw_min) / (raw_max - raw_min + 1e-8)
                image = np.clip(image, 0.00000001, 1)
                
                print(f'After normalization - min: {np.min(image)}, max: {np.max(image)}')
            else:
                print(f'Warning: hs_raw_{filename}.tif not found in {folder}, skipping normalization')
                # clip 0 2^16
                image = np.clip(image, 0.00000001, 1)
        else:
            print(f'Max value {np.max(image)} <= 2, no normalization needed')
            # clip 0 2^16
            image = np.clip(image, 0.00000001, 1)
        
        # Split the image into mean and scale parts
        mu = image[:image.shape[0]//2, :, :]
        sigma = image[image.shape[0]//2:, :, :]
        
        # Convert to torch tensor
        mu_tensor = torch.from_numpy(mu).float()
        sigma_tensor = torch.from_numpy(sigma).float()
            
        means.append(mu_tensor)
        scales.append(sigma_tensor)
        
    
    return means, scales

def ensemble_predictions(means, scales):
    M = len(means)
    mean_ensemble = torch.stack(means).mean(dim=0)
    scale_ensemble = torch.stack(scales).mean(dim=0)
    return mean_ensemble, scale_ensemble

def kl_divergence(p_mean, p_scale, q_mean, q_scale, eps=1e-3):

    # Add small epsilon to prevent division by zero
    p_scale = torch.clamp(p_scale, min=eps)
    q_scale = torch.clamp(q_scale, min=eps)
    return torch.log(q_scale / p_scale) + (p_scale.pow(2) + (p_mean - q_mean).pow(2)) / (2 * q_scale.pow(2)) - 0.5
    #return torch.log(q_scale / p_scale) + (p_scale + (p_mean - q_mean).pow(2)) / (2 * q_scale.pow(2)) - 0.5

def compute_disagreement(means, scales, ensemble_mean, ensemble_scale):
    M = len(means)
    disagreements = []
    print(f' mean min: {torch.min(ensemble_mean)}')
    print(f' mean max: {torch.max(ensemble_mean)}')
    print(f' scale min: {torch.min(ensemble_scale)}')
    print(f' scale max: {torch.max(ensemble_scale)}')
    print(f' mean shape: {ensemble_mean.shape}')
    print(f' scale shape: {ensemble_scale.shape}')
    for mean, scale in zip(means, scales):
        kl = kl_divergence(mean, scale, ensemble_mean, ensemble_scale)
        disagreements.append(kl)
    #disagreement_score = sum(disagreements) / M
    disagreement_score = sum(disagreements) / M / np.log(M)
    return disagreement_score

def calculate_disagreement_for_all_images(folders, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    filenames = sorted([os.path.splitext(f)[0].replace('hs_gen_', '') 
                    for f in os.listdir(folders[0]) 
                    if f.startswith('hs_gen_') and f.endswith('.tif')])
    
    print(np.shape(filenames))

    count = 0
    disagreement_scores = []
    
    for filename in filenames:
        means, scales = load_images_from_folders(folders, filename)
        ensemble_mean, ensemble_scale = ensemble_predictions(means, scales)

        show = False
        if show:
            print(f'ensemble mean shape: {ensemble_mean[0].numpy().shape}')
            print(f'ensemble scale shape: {ensemble_mean[0].numpy().shape}')
            print(f'mean shape: {means[0][0].numpy().shape}')
            print(f'scale shape: {scales[0][0].numpy().shape}')
            # convert to numpy array
            plt.subplot(2, 2, 1)
            plt.imshow(ensemble_mean[1].numpy(), cmap='gray')
            plt.title('ensemble mean')
            plt.subplot(2, 2, 2)
            plt.imshow(ensemble_scale[1].numpy(), cmap='gray')
            plt.title('ensemble scale')
            plt.subplot(2, 2, 3)
            plt.imshow(means[0][1].numpy(), cmap='gray')
            plt.title('mean 1')
            plt.subplot(2, 2, 4)
            plt.imshow(scales[0][1].numpy(), cmap='gray')
            plt.title('scale 1')
            plt.show()
        show_2 = False
        if show_2 == True:
            for i in range(ensemble_mean.shape[0]):
                plt.subplot(2, 2, 1)
                plt.imshow(ensemble_mean[i].numpy(), cmap='gray')
                plt.title('ensemble mean')
                plt.subplot(2, 2, 2)
                plt.imshow(ensemble_scale[i].numpy(), cmap='gray')
                plt.title('ensemble scale')
                plt.subplot(2, 2, 3)
                plt.imshow(means[0][i].numpy(), cmap='gray')
                plt.title('mean 1')
                plt.subplot(2, 2, 4)
                plt.imshow(scales[0][i].numpy(), cmap='gray')
                plt.title('scale 1')
                plt.show()

        disagreement = compute_disagreement(means, scales, ensemble_mean, ensemble_scale)
        print(f'disagreement shape: {disagreement.shape}')
        
        # Calculate mean disagreement per channel
        mean_disagreement_per_channel = torch.mean(disagreement, dim=(1, 2)).numpy()
        print(f'mean disagreement shape: {mean_disagreement_per_channel.shape}')
        print(f'mean disagreement: {mean_disagreement_per_channel}')
        
        # Also calculate overall mean for filename
        overall_mean_disagreement = np.mean(mean_disagreement_per_channel)

        # Normalize the disagreement image
        disagreement_np = disagreement.numpy()# * 255.0  # Convert to numpy array and scale to 255
        
        # Format the per-channel disagreement values for the filename
        if len(disagreement_np.shape) == 3:
            disagreement_str = f"r{mean_disagreement_per_channel[0]:.3f}_g{mean_disagreement_per_channel[1]:.3f}_b{mean_disagreement_per_channel[2]:.4f}"
        if len(disagreement_np.shape) == 6:
            disagreement_str = f"ch1_{mean_disagreement_per_channel[0]:.2f}_ch2_{mean_disagreement_per_channel[1]:.2f}_ch3_{mean_disagreement_per_channel[2]:.2f}_ch4_{mean_disagreement_per_channel[3]:.2f}_ch5_{mean_disagreement_per_channel[4]:.2f}_ch6_{mean_disagreement_per_channel[5]:.2f}"
        if len(disagreement_np.shape) >= 7:
            disagreement_score_mean = np.mean(mean_disagreement_per_channel)
            disagreement_str = f"mean {disagreement_score_mean:.3f}"
        output_path = os.path.join(output_folder, f'disagreement_{filename}_D_{disagreement_str}.png')
        save_image(disagreement_np, output_path)

        # save log disagreement
        log_disagreement_np = np.log(disagreement_np + 1e-8)
        save_image(log_disagreement_np, output_path.replace('.png', '_log.png'))


        # Save the mean image
        mean_image_np = ensemble_mean.numpy().squeeze()
        normalized_mean_image = normalize_mean_img(mean_image_np)
        mean_output_path = os.path.join(output_folder, f'mean_{filename}.png')
        save_image(normalized_mean_image, mean_output_path)
        
        # Accumulate disagreement score (store per-channel scores)
        disagreement_scores.append(mean_disagreement_per_channel)
        count += 1
    
    # Calculate and print the average disagreement score
    disagreement_scores = np.array(disagreement_scores)
    average_disagreement_score = np.mean(disagreement_scores, axis=0)
    std_disagreement_score = np.std(disagreement_scores, axis=0)
    overall_avg = np.mean(average_disagreement_score)
    overall_std = np.mean(std_disagreement_score)
    
    print(output_folder)
    print(f"Per-channel Average Disagreement Scores: {average_disagreement_score}")
    print(f"Per-channel Standard Deviation of Disagreement Scores: {std_disagreement_score}")
    print(f"Overall Average Disagreement Score: {overall_avg}")
    print(f"Overall Standard Deviation of Disagreement Scores: {overall_std}")


input_output_pairs = [
    ([
    '/media/al/Extreme SSD/20250425_results/results/new_layer_norm/results/20250425_0gan_single_hs_1/test_latest/images',
    '/media/al/Extreme SSD/20250425_results/results/new_layer_norm/results/20250425_0gan_single_hs_2/test_latest/images',
    '/media/al/Extreme SSD/20250425_results/results/new_layer_norm/results/20250425_0gan_single_hs_3/test_latest/images',
    '/media/al/Extreme SSD/20250425_results/results/new_layer_norm/results/20250425_0gan_single_hs_4/test_latest/images',
    '/media/al/Extreme SSD/20250425_results/results/new_layer_norm/results/20250425_0gan_single_hs_5/test_latest/images'],
    '/media/al/Extreme SSD/20250425_results/results/new_layer_norm/results/20250425_0gan_single_hs_disagreement'
    )
]

#calculate_disagreement_for_all_images(folders, output_folder)
for input_folders, output_folder in input_output_pairs:
    calculate_disagreement_for_all_images(input_folders, output_folder)