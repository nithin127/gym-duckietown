import os
import argparse
import re
import glob
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from representation_analysis.models import VAE

# Monkey-patch because I trained with a newer version.
# This can be removed once PyTorch 0.4.x is out.
# See https://discuss.pytorch.org/t/question-about-rebuild-tensor-v2/14560
import torch._utils

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor


    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


def kernel(z, ktype='gaussian'):
	d = z.shape[0]
	K = np.zeros((d,d))
	for i in range(d):
		for j in range(i+1):
			if ktype == 'gaussian':
				K[i,j] = K[j,i] = np.exp(-0.5*(np.linalg.norm(z[i]-z[j])**2))
			elif ktype == 'laplacian':
				K[i,j] = K[j,i] = np.exp(-1.0*np.linalg.norm(z[i]-z[j]))
			else:
				raise('Kernel type unknown, please check the code')
	return K

def get_HSIC(z_array, ktype='gaussian'):
	# Assuming that the z_array.shape = [no.of examples, latent space dimension]
	m, d = z_array.shape
	HSIC_array = np.zeros((d,d))
	H = np.eye(m) - 1.0/m
	for i in range(d):
		for j in range(i+1):
			K = kernel(z_array[:,i], ktype=ktype)
			L = kernel(z_array[:,j], ktype=ktype)
			HSIC_array[i,j] = HSIC_array[j,i] = np.trace\
				(np.matmul(np.matmul(np.matmul(K, H),L),H))/((m-1)**2)
	return HSIC_array


parser = argparse.ArgumentParser(description='Disentanglement')
parser.add_argument('--save-dir', type=str,
                    help='Path to the save folder (subfolder of `.saves` by default)')
parser.add_argument('--kernel', type=str,
                    help='Type of kernel: Choose from Gaussian/Laplacian')
parser.add_argument('--save-file', type=str, default=None, help='Checkpoint file')
parser.add_argument('--num-images', type=int, default=100, metavar='N',
                    help='Number of images (default: 10)')
parser.add_argument('--num-samples', type=int, default=11, metavar='N',
                    help='Number of samples (default: 11)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# taken from: https://github.com/nithin127/nest-vae/blob/master/kernel.py
data_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.ImageFolder(root=os.path.join(os.getcwd(), 'representation_analysis/data/'),
                                     transform=data_transform)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.num_images, shuffle=True)

final_results = []

for model in os.listdir('representation_analysis/saves/beta-vae/'):
    args.saved_model = os.path.join('representation_analysis/saves/beta-vae/', model)
    if args.saved_model:
        try:
            loaded_state = torch.load(args.saved_model)
            step = loaded_state['step']
            model = loaded_state['model']
            args.state_size = model['encoder_mean.weight'].shape[0]
            vae = VAE(z_dim=args.state_size, use_cuda=args.cuda)
            vae.load_state_dict(model)
            beta = loaded_state['beta']


            # save_curve(total_losses, TC_losses)
            parameters = list(vae.parameters())
            if args.cuda:
                vae.cuda()
            print('model found and loaded successfully...')
        except:
            print('problem loading model! Check model file!')
            exit(1)

    print('proceeding to train with beta={0} h={1}. Wish me luck!'.format(beta, args.state_size))
    fixed_x, _ = next(iter(data_loader))

    if args.cuda:
        fixed_x = Variable(fixed_x.cuda())
    else:
        fixed_x = Variable(fixed_x)

    # Get the latent representation for each example
    _, _, _, fixed_z = vae.forward(fixed_x)

    HSIC_array = get_HSIC(fixed_z.data.cpu().numpy(), ktype='gaussian')
    print('The mean value of the HSIC_array is {}'.format(np.mean(HSIC_array)))

    # result_path = os.path.join('.logs', 'hsic_dependency', args.save_dir, '{}.txt'.format(args.dataset))

    # with open(result_path, 'w') as f:
    print('The HSIC array for {} is:\n'.format(HSIC_array))
    print('The mean value of the HSIC_array is {}'.format(np.mean(HSIC_array)))
    final_results.append([beta, args.state_size, np.mean(HSIC_array)])
    np.save('representation_analysis/hsic_results', np.array(final_results))