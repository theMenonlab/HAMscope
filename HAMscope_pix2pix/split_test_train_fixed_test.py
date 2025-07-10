import os
import shutil
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

def create_prefix_mapping(source_files, existing_files, prefix_length=8):
    """
    Create mapping between source files and existing files based on filename prefix.
    
    Args:
        source_files (list): List of source filenames
        existing_files (list): List of existing test filenames
        prefix_length (int): Number of characters to match from the beginning
    
    Returns:
        set: Set of source filenames that should go to test set
    """
    # Create set of prefixes from existing test files
    existing_prefixes = {f[:prefix_length] for f in existing_files}
    
    # Find source files with matching prefixes
    matching_source_files = set()
    for filename in source_files:
        if filename[:prefix_length] in existing_prefixes:
            matching_source_files.add(filename)
    
    return matching_source_files

def get_existing_filenames(directory):
    """
    Get list of filenames from an existing directory.
    """
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    
    files = [f for f in os.listdir(directory) if f.endswith(('.tif', '.png', '.jpg', '.jpeg'))]
    return sorted(files, key=natural_sort_key)

def replicate_split(source_A_dir=None, source_B_dir=None, 
                   existing_A_test_dir=None, existing_B_test_dir=None,
                   dest_A_train_dir=None, dest_A_test_dir=None,
                   dest_B_train_dir=None, dest_B_test_dir=None,
                   process_A=True, process_B=True):
    """
    Replicate an existing train/test split on a newly processed dataset.
    Files with matching first 8 characters to existing test set go to test, all others go to train.
    
    Args:
        source_A_dir (str): Source directory for new A images
        source_B_dir (str): Source directory for new B images
        existing_A_test_dir (str): Existing A test directory to get split pattern
        existing_B_test_dir (str): Existing B test directory to get split pattern
        dest_A_train_dir (str): Destination directory for new A train images
        dest_A_test_dir (str): Destination directory for new A test images
        dest_B_train_dir (str): Destination directory for new B train images
        dest_B_test_dir (str): Destination directory for new B test images
        process_A (bool): Whether to process A directory
        process_B (bool): Whether to process B directory
    """
    
    if process_A:
        if not all([source_A_dir, existing_A_test_dir, dest_A_train_dir, dest_A_test_dir]):
            raise ValueError("All A directory parameters must be provided when process_A=True")
        
        # Create destination directories
        Path(dest_A_train_dir).mkdir(parents=True, exist_ok=True)
        Path(dest_A_test_dir).mkdir(parents=True, exist_ok=True)
        
        # Get existing test files to determine split pattern
        existing_test_files = get_existing_filenames(existing_A_test_dir)
        print(f"Found {len(existing_test_files)} files in existing A test set")
        
        # Get new source files
        source_A_files = [f for f in os.listdir(source_A_dir) if f.endswith(('.tif', '.png', '.jpg', '.jpeg'))]
        source_A_files.sort(key=natural_sort_key)
        print(f"Found {len(source_A_files)} files in new A source directory")
        
        # Create prefix-based mapping
        test_files_set = create_prefix_mapping(source_A_files, existing_test_files)
        print(f"Matched {len(test_files_set)} source files to test set based on first 8 characters")
        
        # Copy files based on prefix mapping
        train_copied = 0
        test_copied = 0
        
        for filename in source_A_files:
            if filename in test_files_set:
                # File should go to test set
                shutil.copy2(os.path.join(source_A_dir, filename), os.path.join(dest_A_test_dir, filename))
                test_copied += 1
            else:
                # File should go to train set
                shutil.copy2(os.path.join(source_A_dir, filename), os.path.join(dest_A_train_dir, filename))
                train_copied += 1
        
        # Check for missing prefix matches
        existing_prefixes = {f[:8] for f in existing_test_files}
        source_prefixes = {f[:8] for f in source_A_files}
        missing_prefixes = existing_prefixes - source_prefixes
        
        if missing_prefixes:
            print(f"Warning: {len(missing_prefixes)} test prefixes not found in new source directory:")
            for prefix in sorted(missing_prefixes):
                print(f"  Missing prefix: {prefix}")
        
        print(f"A directory processing complete:")
        print(f"  Copied {train_copied} files to train set")
        print(f"  Copied {test_copied} files to test set")
    
    if process_B:
        if not all([source_B_dir, existing_B_test_dir, dest_B_train_dir, dest_B_test_dir]):
            raise ValueError("All B directory parameters must be provided when process_B=True")
        
        # Create destination directories
        Path(dest_B_train_dir).mkdir(parents=True, exist_ok=True)
        Path(dest_B_test_dir).mkdir(parents=True, exist_ok=True)
        
        # Get existing test files to determine split pattern
        existing_test_files = get_existing_filenames(existing_B_test_dir)
        print(f"Found {len(existing_test_files)} files in existing B test set")
        
        # Get new source files
        source_B_files = [f for f in os.listdir(source_B_dir) if f.endswith(('.tif', '.png', '.jpg', '.jpeg'))]
        source_B_files.sort(key=natural_sort_key)
        print(f"Found {len(source_B_files)} files in new B source directory")
        
        # Create prefix-based mapping
        test_files_set = create_prefix_mapping(source_B_files, existing_test_files)
        print(f"Matched {len(test_files_set)} source files to test set based on first 8 characters")
        
        # Copy files based on prefix mapping
        train_copied = 0
        test_copied = 0
        
        for filename in source_B_files:
            if filename in test_files_set:
                # File should go to test set
                shutil.copy2(os.path.join(source_B_dir, filename), os.path.join(dest_B_test_dir, filename))
                test_copied += 1
            else:
                # File should go to train set
                shutil.copy2(os.path.join(source_B_dir, filename), os.path.join(dest_B_train_dir, filename))
                train_copied += 1
        
        # Check for missing prefix matches
        existing_prefixes = {f[:8] for f in existing_test_files}
        source_prefixes = {f[:8] for f in source_B_files}
        missing_prefixes = existing_prefixes - source_prefixes
        
        if missing_prefixes:
            print(f"Warning: {len(missing_prefixes)} test prefixes not found in new source directory:")
            for prefix in sorted(missing_prefixes):
                print(f"  Missing prefix: {prefix}")
        
        print(f"B directory processing complete:")
        print(f"  Copied {train_copied} files to train set")
        print(f"  Copied {test_copied} files to test set")

# Example usage
if __name__ == "__main__":
    # Define paths
    source_A_dir = ""
    source_B_dir = "/scratch/general/nfs1/u0573922/20250425_dataset_simple/ham_nonorm/ham_nonorm_4"
    
    # Existing test directories (to get the pattern from)
    existing_A_test_dir = ""
    existing_B_test_dir = "/scratch/general/nfs1/u0573922/20250425_dataset_simple/split_4/test/hamamatsu"
    
    # New destination directories
    dest_A_train_dir = ""
    dest_A_test_dir = ""
    dest_B_train_dir = "/scratch/general/nfs1/u0573922/20250425_dataset_simple/split_4/train/ham_nonorm"
    dest_B_test_dir = "/scratch/general/nfs1/u0573922/20250425_dataset_simple/split_4/test/ham_nonorm"
    
    # Replicate the split - process only B directory in this example
    replicate_split(
        source_A_dir=source_A_dir,
        source_B_dir=source_B_dir,
        existing_A_test_dir=existing_A_test_dir,
        existing_B_test_dir=existing_B_test_dir,
        dest_A_train_dir=dest_A_train_dir,
        dest_A_test_dir=dest_A_test_dir,
        dest_B_train_dir=dest_B_train_dir,
        dest_B_test_dir=dest_B_test_dir,
        process_A=False,   # Set to False to skip A processing
        process_B=True     # Set to True to process B directory
    )
