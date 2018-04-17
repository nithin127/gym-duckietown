import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def make_var(arr):
    arr = np.ascontiguousarray(arr)
    arr = torch.from_numpy(arr).float()
    arr = Variable(arr)
    if torch.cuda.is_available():
        arr = arr.cuda()
    return arr

def save_img(file_name, img):
    from skimage import io

    img = img.squeeze(0)
    img = img.clamp(0, 1)
    img = img.data
    img = img.transpose(0, 2).transpose(0, 1)
    img = np.flip(img, 0)
    img = img * 255
    img = img.astype(np.uint8)

    io.imsave(file_name, img)

def load_img(file_name):
    from skimage import io

    # Drop the alpha channel
    img = io.imread(file_name)
    img = img[:,:,0:3] / 255

    # Flip the image vertically
    img = np.flip(img, 0)

    # Transpose the rows and columns
    img = img.transpose(2, 0, 1)

    # Make it a batch of size 1
    var = make_var(img)
    var = var.unsqueeze(0)

    return var

def gen_batch(gen_data_fn, batch_size=2):
    """
    Returns a tuple of PyTorch Variable objects
    gen_data is expected to produce a tuple
    """

    assert batch_size > 0

    data = []
    for i in range(0, batch_size):
        data.append(gen_data_fn())

    # Create arrays of data elements for each variable
    num_vars = len(data[0])
    arrays = []
    for idx in range(0, num_vars):
        vals = []
        for datum in data:
            vals.append(datum[idx])
        arrays.append(vals)

    # Make a variable out of each element array
    vars = []
    for array in arrays:
        var = make_var(np.stack(array))
        vars.append(var)

    return tuple(vars)
