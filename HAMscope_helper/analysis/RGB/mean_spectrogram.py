import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# Global variables to store selected rectangle boundaries
rect_coords = [72, 122, 65, 101]  # [min_x, max_x, min_y, max_y] RED
#rect_coords = [141, 194, 67, 103]  # [min_x, max_x, min_y, max_y] GREEN
rect_coords = [210, 260, 65, 103]  # [min_x, max_x, min_y, max_y] BLUE

def line_select_callback(eclick, erelease):
    # Get the rectangle coordinates
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    min_x, max_x = int(min(x1, x2)), int(max(x1, x2))
    min_y, max_y = int(min(y1, y2)), int(max(y1, y2))
    rect_coords[0], rect_coords[1], rect_coords[2], rect_coords[3] = min_x, max_x, min_y, max_y

    print(f"Selected region: x=({min_x}, {max_x}), y=({min_y}, {max_y})")

def on_key_press(event):
    # When pressing a key, close the region selector
    if event.key == 'enter':
        plt.close()

def main():
    # Load TIF stack shaped [color_channel, x, y]
    path_to_tif = "/home/al/hyperspectral_pix2pix/RGB/color_calibration/cubert/image_1_cubert.tif"
    data = tifffile.imread(path_to_tif)
    print(f"TIF shape: {data.shape}")

    # Display one channel or a sum across channels to pick region
    display_image = data.sum(axis=0)
    fig, ax = plt.subplots()
    ax.set_title("Select region, press Enter when done")
    image_handle = ax.imshow(display_image, cmap='gray')

    # Define RectangleSelector for region picking
    toggle_selector = RectangleSelector(
    ax, line_select_callback,
    useblit=True,
    button=[1],  # left-click only
    minspanx=5, minspany=5,
    spancoords='pixels',
    interactive=True
    )

    # Connect the key press event to close the selection
    plt.connect('key_press_event', on_key_press)
    plt.show()

    # Extract the 1D spectrogram (average of the selected region across channels)
    min_x, max_x, min_y, max_y = rect_coords
    # data shape is [C, X, Y], so slice [C, min_y:max_y, min_x:max_x]
    roi = data[:, min_y:max_y, min_x:max_x]
    spectrogram = roi.mean(axis=(1,2))

    # Save the 1D spectrogram as .npz
    np.savez("/home/al/hyperspectral_pix2pix/RGB/blue_spectrogram.npz", spectrogram=spectrogram)

    # Optional: plot the spectrogram for quick check
    plt.figure()
    plt.plot(spectrogram, marker='o')
    plt.title("Extracted Spectrogram (mean over selected ROI)")
    plt.xlabel("Channel Index")
    plt.ylabel("Intensity (mean)")
    plt.show()

if __name__ == "__main__":
    main()
