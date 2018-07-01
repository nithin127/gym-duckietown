import argparse
import os
import types
import time

import numpy as np
import torch
from torch.autograd import Variable
from pytorch_rl.vec_env.dummy_vec_env import DummyVecEnv

from envs import make_env
from representation_analysis.models import VAE

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-stack', type=int, default=4,
                    help='number of frames to stack (default: 4)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--start-container', action='store_true', default=False,
                    help='start the Duckietown container image')
parser.add_argument('--saved-encoder-model', type=str, help='Additional string added to save files')
parser.add_argument('--save-tag', type=str, default = "_1", help='Additional string added to save files; Includes the random seed at the end')
parser.add_argument('--latent-space-size', type=int, default=100,
                        help='Size of latent code (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

env = make_env(args.env_name, args.seed, 0, None, args.start_container)
env = DummyVecEnv([env])

if args.saved_encoder_model:
    try:
        loaded_state = torch.load(args.saved_encoder_model)
        model = loaded_state['model']
        
        vae = VAE(z_dim=args.latent_space_size, use_cuda=args.cuda) 
        vae.load_state_dict(model)
        args.save_tag = "_"+args.saved_encoder_model.split("/")[-1].split(".")[0] + "_" + str(args.seed)

        if args.cuda:
            vae.cuda()

        print('encoder model found and loaded successfully')
    except:
        print('problem loading encoder model. Check file!')
        exit(1)

actor_critic, ob_rms = torch.load(os.path.join(args.load_dir, args.env_name + args.save_tag + ".pt"))

render_func = env.envs[0].render

obs_shape = env.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
if args.saved_encoder_model:
    obs_shape = (args.num_stack, args.latent_space_size)

current_obs = torch.zeros(1, *obs_shape)
states = torch.zeros(1, actor_critic.state_size)
masks = torch.zeros(1, 1)

def update_current_obs(obs):
    shape_dim0 = env.observation_space.shape[0]
    if args.saved_encoder_model:
        shape_dim0 = 1
        obs, _ = vae.encode(Variable(torch.cuda.FloatTensor(obs)))
        obs = obs.data.cpu().numpy()    
    obs = torch.from_numpy(obs).float()
    if args.num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs

render_func('human')
obs = env.reset()
update_current_obs(obs)

window = env.envs[0].unwrapped.window
@window.event
def on_key_press(symbol, modifiers):
    from pyglet.window import key
    import sys
    if symbol == key.ESCAPE:
        env.close()
        sys.exit(0)
    return

avg_reward = 0.0

try:
    while True:
        value, action, _, states = actor_critic.act(
            Variable(current_obs),
            Variable(states),
            Variable(masks),
            deterministic=True
        )
        states = states.data
        cpu_actions = action.data.squeeze(1).cpu().numpy()

        
        # Obser reward and next obs
        obs, reward, done, _ = env.step(cpu_actions)
        
        avg_reward = 0.99*avg_reward + 0.01*reward
        print("action: {}, avg_reward{}\n".format(cpu_actions, reward))
        time.sleep(0.08)

        masks.fill_(0.0 if done else 1.0)

        if current_obs.dim() == 4:
            current_obs *= masks.unsqueeze(2).unsqueeze(2)
        else:
            current_obs *= masks
        update_current_obs(obs)

        render_func('human')

except:
    env.envs[0].unwrapped.close()
    time.sleep(0.25)
