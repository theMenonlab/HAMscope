"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy = (image_numpy + 1) * 128

        #image_numpy = image_numpy * 0.5 + 0.5
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
        print('numpy, doing nothing')
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_tif(image_numpy, image_dir, idx, name, label, aspect_ratio=1.0):
    from tifffile import imwrite
    print(name)
    print(label)
    if 'real_B' in label:
        out_name = 'hs_raw_%s.tif' % (idx)
    elif 'fake_B' in label:
        out_name = 'hs_gen_%s.tif' % (idx)
    elif 'real_A' in label:
        out_name = 'mini_raw_%s.tif' %(idx)
    else:
        out_name = '%s_%s.tif' % (name, label)

    image_path = os.path.join(image_dir, out_name)

    print(np.shape(image_numpy))
    print(image_path)
    print(out_name)
    print(np.max(image_numpy))
    print(np.min(image_numpy))
    imwrite(image_path, image_numpy)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    if len(image_numpy.shape) == 4:
        image_numpy = image_numpy[:, 0, :, :]
    if len(image_numpy.shape) == 5:
        image_numpy = image_numpy[:, 0, :, :, 0]
    image_numpy = image_numpy.transpose([1, 2, 0])

    if image_numpy.shape[2] <= 3:
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(image_path)
    else:
        for i in range(image_numpy.shape[2]):
            img = image_numpy[:, :, i]
            image_pil = Image.fromarray(img)

            image_pil.save(f'{image_path}{i}.png')



def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
