from gym_duckietown.envs import SimpleSimEnv
from PIL import Image
import numpy as np

env = SimpleSimEnv(draw_curve=False)
num_samples = 100

for i in range(num_samples):
    obs = env.reset_keep_same(keep_same=['groundColor'])
    img = Image.fromarray(np.flipud((obs*255)).astype('uint8'))
    img.save(fp='representation_analysis/test_data/ground_color/trajectories/{}.jpg'.format(i))

for i in range(num_samples):
    obs = env.reset_keep_same(keep_same=['roadColor'])
    img = Image.fromarray(np.flipud((obs*255)).astype('uint8'))
    img.save(fp='representation_analysis/test_data/road_color/trajectories/{}.jpg'.format(i))

for i in range(num_samples):
    obs = env.reset_keep_same(keep_same=['wheelDist'])
    img = Image.fromarray(np.flipud((obs*255)).astype('uint8'))
    img.save(fp='representation_analysis/test_data/wheelDist/trajectories/{}.jpg'.format(i))

for i in range(num_samples):
    obs = env.reset_keep_same(keep_same=['camHeight'])
    img = Image.fromarray(np.flipud((obs*255)).astype('uint8'))
    img.save(fp='representation_analysis/test_data/camHeight/trajectories/{}.jpg'.format(i))

for i in range(num_samples):
    obs = env.reset_keep_same(keep_same=['camAngle'])
    img = Image.fromarray(np.flipud((obs*255)).astype('uint8'))
    img.save(fp='representation_analysis/test_data/camAngle/trajectories/{}.jpg'.format(i))

for i in range(num_samples):
    obs = env.reset_keep_same(keep_same=['camFovY'])
    img = Image.fromarray(np.flipud((obs*255)).astype('uint8'))
    img.save(fp='representation_analysis/test_data/camFovY/trajectories/{}.jpg'.format(i))
