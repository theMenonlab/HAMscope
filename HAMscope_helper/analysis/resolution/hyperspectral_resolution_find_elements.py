import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from skimage.measure import profile_line
from scipy.interpolate import make_interp_spline
from skimage import color, img_as_ubyte

from matplotlib.patches import Rectangle


#line_coords = [0, 480, 570, 415] # menon's line 510 pixels long, 788 um
img_paths = [ #20250701 USAF hyperspectral
    #'/media/al/Extreme SSD/20250701_usaf/deconvolution_results/batch_usaf_low_res/hs_gen_0_deconvolved.tif',
    #'/media/al/Extreme SSD/20250701_usaf/deconvolution_20250706/wiener_100.png',
    #'/media/al/Extreme SSD/20250701_usaf/lower_res/results/20250425_0gan_single_reg_hs_usaf/20250425_0gan_single_reg_hs/test_latest/images/hs_gen_2.tif',
    #'/media/al/Extreme SSD/20250701_usaf/20250708_deconvolution_results/hs_gen_5_deconvolved.tif'
    '/media/al/Extreme SSD/20250701_usaf/20250708_deconvolution_results/hs_gen_7_deconvolved.tif'
]
# load with tifffile and check resolution for al 30 channels


# for deconvolved image
line_coords = [215, 203, 242,203] # G6E1 Vertical bars USAF 20250701, Contrast: , 7.81 micron
line_coords = [252, 290, 276, 290] # G6E2 Vertical bars USAF 20250701,Contrast: 0.2872, 6.91 micron
line_coords = [258, 263, 280, 263] # G6E3 Vertical bars USAF 20250701, Contrast:, 0.384, 6.20 micron
line_coords = [262, 238, 282, 238] # G6E4 Vertical bars USAF 20250701, Contrast: 0.195, 5.52 micron
line_coords = [265, 218, 284, 218] # G6E5 Vertical bars USAF 20250701, Contrast: 0.106, 4.92 micron

line_coords = [196, 194, 196, 222] # G6E1 horizontal bars USAF 20250701, Contrast: , 7.81 micron
line_coords = [285, 283, 285,305] # G6E2 horizontal bars USAF 20250701, Contrast: 0.251, 6.91 micron
#line_coords = [286, 256, 286, 281] # G6E3 horizontal bars USAF 20250701, Contrast: 0.218, 6.20 micron
#line_coords = [286, 234, 286, 254] # G6E4 horizontal bars USAF 20250701, Contrast: 0.164, 5.52 micron
#line_coords = [287, 212, 287, 231] # G6E5 horizontal bars USAF 20250701, Contrast: 0.0207, 4.92 micron

# for not deconvolved image
# low resolution USAF 20250425
#line_coords = [216, 101, 216, 140] # G5E6 horizontal bars USAF 20250707,  8.77 micron, not resolved
#line_coords = [215, 140, 215, 174] # G5E5 horizontal bars USAF 20250707, 9.84 micron
#line_coords = [214, 180, 214, 220] # G5E4 horizontal bars USAF 20250707, 11.05 micron
#line_coords = [214, 220, 216, 270] # G5E3 horizontal bars USAF 20250707, 12.4 micron
#line_coords = [214, 270, 214, 330] # G5E2 horizontal bars USAF 20250707, 13.92 micron
#line_coords = [214, 330, 214, 390] # G5E1 horizontal bars USAF 20250707, 15.63 micron


#line_coords = [235, 113, 265, 113] # G5E6 vertical bars USAF 20250707, 8.77 micron
#line_coords = [239, 151, 273, 151] # G5E5 vertical bars USAF 20250707, 9.84 micron
#line_coords = [235, 192, 280, 192] # G5E4 vertical bars USAF 20250707, 11.05 micron
#line_coords = [240, 240, 284, 240] # G5E3 vertical bars USAF 20250707, 12.4 micron
#line_coords = [244, 297, 287, 297] # G5E2 vertical bars USAF 20250707, 13.92 micron
#line_coords = [246, 353, 297, 353] # G5E1 vertical bars USAF 20250707, 15.63 micron

# better focused USAF 20250708
line_coords = [228, 144, 228, 183] # G5E6 horizontal bars USAF 20250707,  8.77 micron, not resolved
line_coords = [228, 183, 228, 217] # G5E5 horizontal bars USAF 20250707, 9.84 micron
line_coords = [230, 223, 230, 263] # G5E4 horizontal bars USAF 20250707, 11.05 micron
line_coords = [230, 263, 230, 313] # G5E3 horizontal bars USAF 20250707, 12.4 micron
line_coords = [230, 313, 230, 373] # G5E2 horizontal bars USAF 20250707, 13.92 micron
line_coords = [230, 377, 230, 440] # G5E1 horizontal bars USAF 20250707, 15.63 micron


line_coords = [252, 156, 279, 156] # G5E6 vertical bars USAF 20250707, 8.77 micron
line_coords = [256, 194, 288, 194] # G5E5 vertical bars USAF 20250707, 9.84 micron
line_coords = [255, 239, 295, 239] # G5E4 vertical bars USAF 20250707, 11.05 micron
line_coords = [260, 286, 305, 286] # G5E3 vertical bars USAF 20250707, 12.4 micron
line_coords = [264, 342, 310, 342] # G5E2 vertical bars USAF 20250707, 13.92 micron
line_coords = [265, 404, 330, 404] # G5E1 vertical bars USAF 20250707, 15.63 micron


# img 7
line_coords = [230, 230, 230, 280] # G5E2 horizontal bars USAF 20250707, 13.92 micron
line_coords = [230, 290, 230, 345] # G5E1 horizontal bars USAF 20250707, 15.63 micron
line_coords = [260, 250, 310, 250] # G5E2 vertical bars USAF 20250707, 13.92 micron
line_coords = [260, 310, 320, 310] # G5E1 vertical bars USAF 20250707, 15.63 micron

horizontal = False # if true, scan horizontal bars with vertical lines

# Custom legend labels
legend_labels = [
    'deconvolved',
    'generated',
]

# Initialize Figure 1 for plotting all the intensity profiles later
plt.figure(0)
plt.xlabel('Distance along the line (μm)')
plt.ylabel('Normalized Mean Pixel Intensity')

# For storing smoothed lines and their labels for later plotting
y_smooth_lines = []
line_labels = []
original_profiles = []
x_values = []

contrast_avg = []
max_indices = []
min_indices = []
# Loop through each image file
for img_path, label in zip(img_paths, legend_labels):
    # Read and display the image with lines
    img = Image.open(img_path)
    img_array = np.array(img)

    # Check if the image has more than one channel (grayscale should have only one)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # Assuming the image is RGBA, convert it to RGB first, then to grayscale
        img_array = color.rgb2gray(color.rgba2rgb(img_array))
        print('Converting RGBA to grayscale')
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Convert RGB to grayscale
        img_array = color.rgb2gray(img_array)
        print('Converting RGB to grayscale')

    if img_array.dtype != np.uint8:  # Check if not already 8-bit
        img_array = img_as_ubyte(img_array)  # Convert to 8-bit unsigned integers

    print(f"Image shape: {img_array.shape}")

    # Initialize a new figure for each image to draw lines on it
    plt.figure(0)

    sum_intensity_profiles = None

    # Drawing lines and accumulating intensity profiles
    for i in range(10):
        if horizontal:
            current_line_coords = (line_coords[0] + i, line_coords[1], line_coords[2] + i, line_coords[3])
            intensity_profile = profile_line(img_array, (current_line_coords[1], current_line_coords[0]), (current_line_coords[3], current_line_coords[2]))
        else:
            current_line_coords = (line_coords[0], line_coords[1] + i, line_coords[2], line_coords[3] + i)
            intensity_profile = profile_line(img_array, (current_line_coords[1], current_line_coords[0]), (current_line_coords[3], current_line_coords[2]))
        if sum_intensity_profiles is None:
            sum_intensity_profiles = intensity_profile
        else:
            sum_intensity_profiles += intensity_profile

        # Draw line on the image
        plt.plot([current_line_coords[0], current_line_coords[2]], [current_line_coords[1], current_line_coords[3]], 'r-')
    
    

    # Calculate and smooth the average intensity profile
    avg_intensity_profiles = sum_intensity_profiles / 10 / 255  # Normalize

    print(f"Average pixel values for {label}: {avg_intensity_profiles}")

    max_values = []
    min_values = []
    values = []
    values_indices = []

    for i in range(len(avg_intensity_profiles)-2):
        #print(f"Comparing indices {i}, {i + 1}, and {i + 2}")
        #print(f"Values: {avg_intensity_profiles[i]}, {avg_intensity_profiles[i + 1]}, {avg_intensity_profiles[i + 2]}")
        if avg_intensity_profiles[i] < avg_intensity_profiles[i + 1] and avg_intensity_profiles[i + 1] > avg_intensity_profiles[i + 2]:
            print(f'local max at {i+1}: {avg_intensity_profiles[i+1]}')
            #max_values.append(avg_intensity_profiles[i+1])
            #max_values_indices.append(i+1)
            values.append(avg_intensity_profiles[i+1])
            values_indices.append(i+1)
        if avg_intensity_profiles[i] > avg_intensity_profiles[i + 1] and avg_intensity_profiles[i + 1] < avg_intensity_profiles[i + 2]:
            print(f'local min at {i+1}: {avg_intensity_profiles[i+1]}')
            #min_values.append(avg_intensity_profiles[i+1])
            values.append(avg_intensity_profiles[i+1])
            values_indices.append(i+1)
    
    try:
        if len(values) < 3:
            contrast_avg.append(0)
            raise ValueError("Not enough data points. Array must contain at least three elements.")
        
        else:
            max_values_indices = np.argpartition(values, -3)[-3:]# this does not work if there are more than 3 max values. Try to use first last and middle
            # what if we found the max vales closest to the maxes of a very smooth curve
            max_values_indices = np.sort(max_values_indices)

            max_3 = [values[max_values_indices[0]], values[max_values_indices[1]], values[max_values_indices[2]]]
            max_3_indices = [values_indices[max_values_indices[0]], values_indices[max_values_indices[1]], values_indices[max_values_indices[2]]]
            max_3_check = [avg_intensity_profiles[max_3_indices[0]], avg_intensity_profiles[max_3_indices[1]], avg_intensity_profiles[max_3_indices[2]]]    
            print("Max 3 values:", max_3)
            print("Max 3 indices:", max_3_indices)
            #print("Max 3 check:", max_3_check)

            # check if the max values are too close to each other
            diff_1 = max_3_indices[1] - max_3_indices[0]
            diff_2 = max_3_indices[2] - max_3_indices[1]
            relative_diff = diff_1 / diff_2
            if np.isnan(relative_diff):
                print("Relative difference NaN.")
                print(f"Diff 1: {diff_1}, Diff 2: {diff_2}")
            elif relative_diff < 0.8 or relative_diff > 1.4:
                print("XXXXXXXXXXXXXXXXXXXXXXXX  Relative difference between max values is too large or small!!! XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                print(f"Relative difference between max values: {relative_diff}")



            min_2 = [
            np.min(avg_intensity_profiles[max_3_indices[0]:max_3_indices[1]]),
            np.min(avg_intensity_profiles[max_3_indices[1]:max_3_indices[2]])
            ]
            print("Min 2 values:", min_2)
            print(avg_intensity_profiles[max_3_indices[1]:max_3_indices[2]])
            min_2_arg = [
            np.argmin(avg_intensity_profiles[max_3_indices[0]:max_3_indices[1]]) + max_3_indices[0],
            np.argmin(avg_intensity_profiles[max_3_indices[1]:max_3_indices[2]]) + max_3_indices[1]
            ]
            #print("Min 2 arg:", min_2_arg)
           
            

        contrast = []
        contrast.append((max_3[0] - min_2[0]) / (max_3[0] + min_2[0]))
        contrast.append((max_3[1] - min_2[0]) / (max_3[1] + min_2[0]))
        contrast.append((max_3[1] - min_2[1]) / (max_3[1] + min_2[1]))
        contrast.append((max_3[2] - min_2[1]) / (max_3[2] + min_2[1]))
        contrast_avg.append(np.mean(contrast))
        print(f'Contrast: {np.mean(contrast)}')

    except ValueError as error:
        print(error)

   


    x = np.arange(len(avg_intensity_profiles))
    spline = make_interp_spline(x, avg_intensity_profiles, k=3)  # Cubic spline
    x_smooth = np.linspace(x.min(), x.max(), 300)  # For smoother curve
    y_smooth = spline(x_smooth)

    y_smooth_lines.append(y_smooth)
    line_labels.append(label)
    x_values.append(x)
    original_profiles.append(avg_intensity_profiles)


    if 'max_3_indices' in locals():
        max_indices.append(np.array(max_3_indices))

    if 'min_2_arg' in locals():
        min_indices.append(np.array(min_2_arg))

print(f'contrast_avg: {contrast_avg}')

# tab separated output for easy copying
rounded_values = [round(float(val), 4) for val in contrast_avg]
output_line = '\t'.join(map(str, rounded_values))
print(output_line)


img_show = Image.open(img_paths[0])
img_array_show = np.array(img_show)
plt.imshow(img_array_show, cmap='gray')
plt.axis('off')  # Hide axes
plt.title(f"Lines on measurement")
plt.show()  # Display the image with lines


# Plot all smoothed lines on the same chart
plt.figure(figsize=(11, 11))
for y_smooth, x, original, label, max_index, min_index in zip(y_smooth_lines, x_values, original_profiles, legend_labels, max_indices, min_indices):
    line, = plt.plot(y_smooth, label=label, linewidth=3)  # Smoothed line
    plt.plot(x * 300 / (len(original) - 1), original, 'x', color='black', markersize=15, markeredgewidth=3)  # Original data points
    
    # Get the color of the current line
    line_color = line.get_color()
    
    # Plot max_3 and min_2 values with markers matching the line color
    max_3 = [original[max_index[0]], original[max_index[1]], original[max_index[2]]]
    min_2 = [original[min_index[0]], original[min_index[1]]]
    #plt.plot(max_index * 300 / (len(original) 