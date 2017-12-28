import gym
import numpy as np
import time

env = gym.make("Asteroids-v0")
env.reset()
env.render()

for i in range(10):
    action = env.action_space.sample()
    obs, rew, _, _ = env.step(action)
    print (obs.shape)
    print ("min: {}, max: {}".format(obs.min(), obs.max()))
    env.render()
    time.sleep(.5)