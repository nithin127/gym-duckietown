import argparse
import numpy as np
import tqdm
from PIL import Image
import os

import torch
from torch.autograd import Variable

from gym_duckietown.envs import SimpleSimEnv
from representation_analysis.models import VAE

env = SimpleSimEnv(draw_curve=False)
parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--num_samples_train', type=int, default=100,
                    help='num samples for training')
parser.add_argument('--num_samples_test', type=int, default=50,
                    help='num samples per testing')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')
parser.add_argument('--saved_model', type=str, help='Save file to use')
parser.add_argument('--state_size', type=int, default=50,
                    help='Size of latent code (default: 100)')
parser.add_argument('--seed', type=int, default=7691, metavar='S',
                    help='Random seed (default: 7691)')
parser.add_argument('--num_per_sample', type=int, default=100,
                    help='L in Disentangling paper (default: 100)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#  Load model from before
try:
    loaded_state = torch.load(args.saved_model)
    step = loaded_state['step']
    model = loaded_state['model']
    vae = VAE(z_dim=args.state_size, use_cuda=args.cuda)
    vae.load_state_dict(model)
    optimizer_states = loaded_state['optimizer']
    fixed_x = loaded_state['fixed_x']
    # save_curve(total_losses, TC_losses)
    parameters = list(vae.parameters())
    if args.cuda:
        vae.cuda()

    print('model found and loaded successfully...')
except:
    print('problem loading model! Check model file!')
    exit(1)

factors = ['posx', 'posy', 'angle']

train_targets = np.zeros(shape=[args.num_samples_train])
# generate training data
for i in tqdm.tqdm(range(args.num_samples_train)):
    #  chose factor to fix randomly
    group_to_use = np.random.randint(0, len(factors))

    #  simulate pairs with fixed factor size
    output = env.higgens_sample(factor=factors[group_to_use], num_sample=2*args.num_per_sample)
    for ind, out in enumerate(output):
        img = Image.fromarray(np.flipud((out*255)).astype('uint8'))
        if not os.path.exists('representation_analysis/train_data/{}/'.format(i, ind)):
            os.makedirs('representation_analysis/train_data/{}/'.format(i, ind))
        img.save(fp='representation_analysis/train_data/{}/{}.jpg'.format(i, ind))
    train_targets[i] = group_to_use

np.save('representation_analysis/train_targets', train_targets)

test_targets = np.zeros(shape=[args.num_samples_test])
# generate training data
for i in tqdm.tqdm(range(args.num_samples_test)):
    #  chose factor to fix randomly
    group_to_use = np.random.randint(0, len(factors))

    #  simulate pairs with fixed factor size
    output = env.higgens_sample(factor=factors[group_to_use], num_sample=2*args.num_per_sample)
    for ind, out in enumerate(output):
        img = Image.fromarray(np.flipud((out*255)).astype('uint8'))
        if not os.path.exists('representation_analysis/test_data/{}/'.format(i, ind)):
            os.makedirs('representation_analysis/test_data/{}/'.format(i, ind))
        img.save(fp='representation_analysis/test_data/{}/{}.jpg'.format(i, ind))
    test_targets[i] = group_to_use

np.save('representation_analysis/test_targets', test_targets)
