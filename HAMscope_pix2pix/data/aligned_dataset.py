import os
import re
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import torch.nn.functional as F

def natural_sort_key(s):
    """
    Sort strings with numbers in natural order.
    So pos1 comes before pos2 comes before pos10.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset that merges multiple folders."""
    
    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
        opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.opt = opt
        self.A_paths, self.B_paths = [], []

        use_comp = getattr(self.opt, 'use_comp', 0)
        
        # Define the main subdirectories
        #dataset_folders = ['focused_split_1', 'focused_split_2', 'focused_split_3', 'focused_split_4']
        #dataset_folders = ['branch_split_1', 'branch_split_2', 'branch_split_3', 'branch_split_4']
        dataset_folders = ['split_1', 'split_2', 'split_3', 'split_4']

        for folder in dataset_folders:
            dir_A = os.path.join(opt.dataroot, folder, opt.phase, 'miniscope')
            dir_B = os.path.join(opt.dataroot, folder, opt.phase, 'hamamatsu')
            #dir_B = os.path.join(opt.dataroot, folder, opt.phase, 'components')
            #dir_B = os.path.join(opt.dataroot, folder, opt.phase, 'hamamatsu_aligned')
            #dir_B = os.path.join(opt.dataroot, folder, opt.phase, 'aligned_manual')
            #dir_B = os.path.join(opt.dataroot, folder, opt.phase, 'nmf')
            if use_comp == 1:
                dir_B = os.path.join(opt.dataroot, folder, opt.phase, 'nmf_4ch')
            if use_comp == 2:
                dir_B = os.path.join(opt.dataroot, folder, opt.phase, 'ham_nonorm_nmf')
            if use_comp == 3:
                dir_B = os.path.join(opt.dataroot, folder, opt.phase, 'ham_nonorm')
            if use_comp == 4:
                dir_B = os.path.join(opt.dataroot, folder, opt.phase, 'ham')
            if use_comp == 5:
                dir_B = os.path.join(opt.dataroot, folder, opt.phase, 'nmf')
            # Check if A directory exists
            if os.path.exists(dir_A):
                self.A_paths.extend(make_dataset(dir_A, opt.max_dataset_size))
                
                # Check if B directory exists, if not, create dummy B paths
                if os.path.exists(dir_B):
                    self.B_paths.extend(make_dataset(dir_B, opt.max_dataset_size))
                else:
                    # Create dummy B paths pointing to A paths (for inference mode)
                    a_paths_for_folder = make_dataset(dir_A, opt.max_dataset_size)
                    self.B_paths.extend(a_paths_for_folder)

        # Apply natural sorting
        self.A_paths = sorted(self.A_paths, key=natural_sort_key)
        self.B_paths = sorted(self.B_paths, key=natural_sort_key)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        #A_path = self.A_paths[index % self.A_size]
        #index_B = index % self.B_size
        #B_path = self.B_paths[index_B]
        A_idx = index % min(self.A_size, self.B_size)
        B_idx = A_idx  # Use the same index for both
    
        A_path = self.A_paths[A_idx]
        B_path = self.B_paths[B_idx]
        #print(f'A path: {A_path}')
        #print(f'B path: {B_path}')
        
        A = io.imread(A_path)  # Shape: (608, 608)
        B = io.imread(B_path)  # Shape: (6, 608, 608)

        # put the smallest dimention first
        if B.shape[-1] < B.shape[0]:
            # 0 to 1, 1 to 2, 2 to 0
            B = np.moveaxis(B, [0, 1, 2], [1, 2, 0])

        #B = B[1:4, :, :]

        
        #A = (A - A.min()) / (A.max() - A.min())
        #B = (B - B.min()) / (B.max() - B.min())
        A = A / 255
        use_comp = getattr(self.opt, 'use_comp', 0)
        if use_comp == 3:
            B = B / 65535
        elif use_comp == 2:
            B = B / 300000
        else:
            B = B / 65535

        #print(f'A shape: {A.shape}')
        #print(f'B shape: {B.shape}')
        
        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()
        
        # Ensure both A and B are 4D tensors
        while A.dim() < 4:
            A = A.unsqueeze(0)

        while B.dim() < 4:
            B = B.unsqueeze(0)

        #print(f'A shape after: {A.shape}')
        #print(f'B shape after: {B.shape}')

        netG = self.opt.netG
        if netG == 'unet_128':
            A = F.interpolate(A, size=(128, 128), mode='bilinear', align_corners=False)
            B = F.interpolate(B, size=(128, 128), mode='bilinear', align_corners=False)
        elif netG == 'unet_256':
            A = F.interpolate(A, size=(256, 256), mode='bilinear', align_corners=False)
            B = F.interpolate(B, size=(256, 256), mode='bilinear', align_corners=False)
        elif netG == 'unet_512':
            A = F.interpolate(A, size=(512, 512), mode='bilinear', align_corners=False)
            B = F.interpolate(B, size=(512, 512), mode='bilinear', align_corners=False)
        elif netG == 'unet_1024':
            A = F.interpolate(A, size=(1024, 1024), mode='bilinear', align_corners=False)
            B = F.interpolate(B, size=(1024, 1024), mode='bilinear', align_corners=False)
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
        
        # Remove batch dimension but keep channel dimension
        A = A.squeeze(0)  # Shape becomes (1, H, W)
        B = B.squeeze(0)  # Shape becomes (6, H, W)
        
        #print(f'A shape after: {A.shape}')
        #print(f'B shape after: {B.shape}')
        
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)


