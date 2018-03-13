#!/usr/bin/env python3

import time
import sys
import argparse

import numpy as np
import gym
import gym_duckietown

from supervised.autoenc_angle import Model

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckie-SimpleSim-v0')
args = parser.parse_args()

env = gym.make(args.env_name)
obs = env.reset()
env.render()

model = Model()
model.load('trained_models/angle_model.pt')

# Define the gains
l_gain = 0.0075
r_gain = 0.0075

try:
    while True:
        angle = model.getAngle(obs)
        print('angle=%.2f' % angle)

        """
        if angle > 8:
            vel = np.array([0.7, 0.4])
        elif angle < -8:
            vel = np.array([0.4, 0.7])
        else:
            vel = np.array([0.6, 0.6])
        """

        angle = min(max(angle, -20), 20)
        l_vel = 0.6 + l_gain * angle
        r_vel = 0.6 - r_gain * angle
        vel = np.array([l_vel, r_vel])

        print(vel)

        obs, reward, done, info = env.step(vel)
        #print('stepCount = %s, reward=%.3f' % (env.stepCount, reward))

        env.render()

        if done:
            print('done!')
            env.reset()
            env.render()

        time.sleep(0.1)

except:
    print('closing env')
    env.close()
    time.sleep(0.25)
