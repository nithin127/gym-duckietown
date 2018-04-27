from gym_duckietown.envs import SimpleSimEnv
from PIL import Image
import numpy as np
import argparse
import os

env = SimpleSimEnv(draw_curve=False)
parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--num_samples_train', type=int, default=10000,
                    help='num samples per factor')
parser.add_argument('--num_samples_test', type=int, default=1000,
                    help='num samples per factor')
args = parser.parse_args()
factors = ['ground_color', 'road_color', 'wheelDist', 'camHeight', 'camAngle', 'camFovY']

for factor in factors:
    if not os.path.exists('representation_analysis/train_data/{}/trajectories/'.format(factor)):
        os.makedirs('representation_analysis/train_data/{}/trajectories/'.format(factor))
    if not os.path.exists('representation_analysis/test_data/{}/trajectories/'.format(factor)):
        os.makedirs('representation_analysis/test_data/{}/trajectories/'.format(factor))
    for i in range(args.num_samples_train):
        obs = env.reset_keep_same(keep_same=[factor])
        img = Image.fromarray(np.flipud((obs*255)).astype('uint8'))
        img.save(fp='representation_analysis/train_data/{}/trajectories/'.format(factor)+'{}.jpg'.format(i))
    for i in range(args.num_samples_test):
        obs = env.reset_keep_same(keep_same=[factor])
        img = Image.fromarray(np.flipud((obs*255)).astype('uint8'))
        img.save(fp='representation_analysis/test_data/{}/trajectories/'.format(factor)+'{}.jpg'.format(i))