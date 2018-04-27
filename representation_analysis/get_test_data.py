from gym_duckietown.envs import SimpleSimEnv
from PIL import Image
import numpy as np
import argparse
import os

env = SimpleSimEnv(draw_curve=False)
parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--num_samples', type=int, default=1000,
                    help='num samples per factor')
args = parser.parse_args()

if not os.path.exists('representation_analysis/test_data/ground_color/trajectories/'.format(args.output_folder)):
    os.makedirs('representation_analysis/test_data/ground_color/trajectories/'.format(args.output_folder))
for i in range(args.num_samples):
    obs = env.reset_keep_same(keep_same=['groundColor'])
    img = Image.fromarray(np.flipud((obs*255)).astype('uint8'))
    img.save(fp='representation_analysis/test_data/ground_color/trajectories/{}.jpg'.format(i))

if not os.path.exists('representation_analysis/test_data/road_color/trajectories/'.format(args.output_folder)):
    os.makedirs('representation_analysis/test_data/road_color/trajectories/'.format(args.output_folder))
for i in range(args.num_samples):
    obs = env.reset_keep_same(keep_same=['roadColor'])
    img = Image.fromarray(np.flipud((obs*255)).astype('uint8'))
    img.save(fp='representation_analysis/test_data/road_color/trajectories/{}.jpg'.format(i))

if not os.path.exists('representation_analysis/test_data/wheelDist/trajectories/'.format(args.output_folder)):
    os.makedirs('representation_analysis/test_data/wheelDist/trajectories/'.format(args.output_folder))
for i in range(args.num_samples):
    obs = env.reset_keep_same(keep_same=['wheelDist'])
    img = Image.fromarray(np.flipud((obs*255)).astype('uint8'))
    img.save(fp='representation_analysis/test_data/wheelDist/trajectories/{}.jpg'.format(i))

if not os.path.exists('representation_analysis/test_data/camHeight/trajectories/'.format(args.output_folder)):
    os.makedirs('representation_analysis/test_data/camHeight/trajectories/'.format(args.output_folder))
for i in range(args.num_samples):
    obs = env.reset_keep_same(keep_same=['camHeight'])
    img = Image.fromarray(np.flipud((obs*255)).astype('uint8'))
    img.save(fp='representation_analysis/test_data/camHeight/trajectories/{}.jpg'.format(i))

if not os.path.exists('representation_analysis/test_data/camAngle/trajectories/'.format(args.output_folder)):
    os.makedirs('representation_analysis/test_data/camAngle/trajectories/'.format(args.output_folder))
for i in range(args.num_samples):
    obs = env.reset_keep_same(keep_same=['camAngle'])
    img = Image.fromarray(np.flipud((obs*255)).astype('uint8'))
    img.save(fp='representation_analysis/test_data/camAngle/trajectories/{}.jpg'.format(i))

if not os.path.exists('representation_analysis/test_data/camFovY/trajectories/'.format(args.output_folder)):
    os.makedirs('representation_analysis/test_data/camFovY/trajectories/'.format(args.output_folder))
for i in range(args.num_samples):
    obs = env.reset_keep_same(keep_same=['camFovY'])
    img = Image.fromarray(np.flipud((obs*255)).astype('uint8'))
    img.save(fp='representation_analysis/test_data/camFovY/trajectories/{}.jpg'.format(i))
