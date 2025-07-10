import os
import shutil
import random
from pathlib import Path
import re

def natural_sort_key(s):
    """
    Sort strings with numbers in natural order.
    So pos1 comes before pos2 comes before pos10.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def extract_number(filename):
    """
    Extract the position number from different filename formats.
    """
    if filename.startswith('pos'):
        # Extract number from posX_aligned_stack.tif
        match = re.search(r'pos(\d+)_', filename)
        if match:
            return int(match.group(1))
    else:
        # Extract timestamp from miniscope filename
        match = re.search(r'(\d{2}_\d{2}_\d{2})', filename)
        if match:
            return match.group(1)
    return filename

def split_dataset(A_dir, B_dir, A_test_dir, B_test_dir, A_train_dir, B_train_dir, num_test=100):
    """
    Split paired images from A_dir and B_dir into test and train directories.
    Pairs are matched by their numerical position in the sorted file list.
    
    Args:
        A_dir (str): Source directory for A images
        B_dir (str): Source directory for B images
        A_test_dir (str): Destination directory for A test images
        B_test_dir (str): Destination directory for B test images
        A_train_dir (str): Destination directory for A train images
        B_train_dir (str): Destination directory for B train images
        num_test (int): Number of test images to select
    """
    # Create destination directories if they don't exist
    for dir_path in [A_test_dir, B_test_dir, A_train_dir, B_train_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Get files from both directories and sort them properly
    a_files = [f for f in os.listdir(A_dir) if f.endswith(('.tif', '.png', '.jpg', '.jpeg'))]
    b_files = [f for f in os.listdir(B_dir) if f.endswith(('.tif', '.png', '.jpg', '.jpeg'))]
    
    # Sort files using natural sort
    a_files.sort(key=natural_sort_key)
    b_files.sort(key=natural_sort_key)
    
    # Verify we have the same number of files
    if len(a_files) != len(b_files):
        raise ValueError(f"Different number of images in directories: A_dir has {len(a_files)}, B_dir has {len(b_files)}")
    
    total_pairs = len(a_files)
    if total_pairs < num_test:
        raise ValueError(f"Found only {total_pairs} image pairs, but num_test={num_test}")
    
    # Randomly select indices for test set
    test_indices = set(random.sample(range(total_pairs), num_test))
    
    # Copy files to respective directories
    for idx, (a_file, b_file) in enumerate(zip(a_files, b_files)):
        if idx in test_indices:
            shutil.copy2(os.path.join(A_dir, a_file), os.path.join(A_test_dir, a_file))
            shutil.copy2(os.path.join(B_dir, b_file), os.path.join(B_test_dir, b_file))
        else:
            shutil.copy2(os.path.join(A_dir, a_file), os.path.join(A_train_dir, a_file))
            shutil.copy2(os.path.join(B_dir, b_file), os.path.join(B_train_dir, b_file))
    
    # Print summary
    print(f"\nDataset split complete:")
    print(f"Total image pairs: {total_pairs}")
    print(f"Test set: {num_test} pairs")
    print(f"Train set: {total_pairs - num_test} pairs")
    print("\nExample of first few pairs:")
    for i, (a, b) in enumerate(zip(a_files, b_files)):
        if i < 3:  # Show first 3 pairs
            print(f"Pair {i+1}: \nA: {a}\nB: {b}\n")

# Example usage
if __name__ == "__main__":
    #A_dir = "/media/al/Extreme SSD/20250222_hs_dataset/focused_images/miniscope"
    #B_dir = "/media/al/Extreme SSD/20250222_hs_dataset/focused_images/hamamatsu"
    #A_test_dir = "/media/al/Extreme SSD/20250222_hs_dataset/focused_split_1/test/miniscope"
    #B_test_dir = "/media/al/Extreme SSD/20250222_hs_dataset/focused_split_1/test/hamamatsu"
    #A_train_dir = "/media/al/Extreme SSD/20250222_hs_dataset/focused_split_1/train/miniscope"
    #B_train_dir = "/media/al/Extreme SSD/20250222_hs_dataset/focused_split_1/train/hamamatsu"

    #A_dir = "/media/al/Extreme SSD/20250222_hs_dataset/focused_images_4/miniscope"
    #B_dir = "/media/al/Extreme SSD/20250222_hs_dataset/focused_images_4/hamamatsu"
    #A_test_dir = "/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/datasets/20250222_hs_dataset/test_from_train_set/focused_split_4/test/miniscope"
    #B_test_dir = "/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/datasets/20250222_hs_dataset/test_from_train_set/focused_split_4/test/hamamatsu"
    #A_train_dir = "/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/datasets/20250222_hs_dataset/test_from_train_set/focused_split_4/train/miniscope"
    #B_train_dir = "/home/al/hyperspectral_pix2pix_chpc_20250306/probabilistic_hyperspectral_pix2pix/datasets/20250222_hs_dataset/test_from_train_set/focused_split_4/train/hamamatsu"

    A_dir = "/scratch/general/nfs1/u0573922/20250425_dataset_simple/miniscope_4/AL/Experiment0/Poplar/customEntValHere/imageCaptures"
    B_dir = "/scratch/general/nfs1/u0573922/20250425_dataset_simple/ham_aligned_4"
    A_test_dir = "/scratch/general/nfs1/u0573922/20250425_dataset_simple/split_4/test/miniscope"
    B_test_dir = "/scratch/general/nfs1/u0573922/20250425_dataset_simple/split_4/test/hamamatsu"
    A_train_dir = "/scratch/general/nfs1/u0573922/20250425_dataset_simple/split_4/train/miniscope"
    B_train_dir = "/scratch/general/nfs1/u0573922/20250425_dataset_simple/split_4/train/hamamatsu"

    
    split_dataset(
        A_dir=A_dir,
        B_dir=B_dir,
        A_test_dir=A_test_dir,
        B_test_dir=B_test_dir,
        A_train_dir=A_train_dir,
        B_train_dir=B_train_dir,
        num_test=25
    )
