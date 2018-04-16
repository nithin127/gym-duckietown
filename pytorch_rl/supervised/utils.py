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

# TODO: gen_batch with gen_data_fn
