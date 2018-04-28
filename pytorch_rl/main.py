import copy
import glob
import os
import time
import operator
from functools import reduce

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from arguments import get_args
from vec_env.dummy_vec_env import DummyVecEnv
from vec_env.subproc_vec_env import SubprocVecEnv
from envs import make_env
from kfac import KFACOptimizer
from model import CNNPolicy, MLPPolicy
from storage import RolloutStorage
from visualize import visdom_plot

from logger import Logger
from utils import MyContainer, create_folder
from representation_analysis.models import VAE

args = get_args()

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

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

save_path = os.path.join(args.save_dir, args.algo)
log_path = os.path.join(args.log_dir, args.algo)

create_folder(save_path)
create_folder(log_path)

logger = Logger(log_path)

tr = MyContainer()
tr.train_reward_avg = [[],[]]
#tr.train_episode_len = [[],[]]
tr.pg_loss = [[],[]]
tr.val_loss = [[],[]]
tr.entropy_loss = [[],[]]
#tr.first_val = [[],[]]
tr.test_reward = [[],[]]
tr.test_episode_len = [[],[]]
tr.iterations_done = 0
tr.global_steps_done = 0
tr.episodes_done = 0

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


def append_to(tlist, tr, val):
        tlist[0].append(val)
        tlist[1].append([tr.episodes_done, tr.global_steps_done, tr.iterations_done])


def main():
    os.environ['OMP_NUM_THREADS'] = '1'

    if args.vis:
        from visdom import Visdom
        viz = Visdom()
        win = None

    envs = [make_env(args.env_name, args.seed, i, args.log_dir, args.start_container)
                for i in range(args.num_processes)]

    test_envs = [make_env(args.env_name, args.seed, i, args.log_dir, args.start_container)
                for i in range(args.num_processes)]
    
    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
        test_envs = SubprocVecEnv(test_envs)
    else:
        envs = DummyVecEnv(envs)
        test_envs = DummyVecEnv(test_envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    if args.saved_encoder_model:
        obs_shape = (args.num_stack, args.latent_space_size)

    obs_numel = reduce(operator.mul, obs_shape, 1)

    if len(obs_shape) == 3 and obs_numel > 1024:
        actor_critic = CNNPolicy(obs_shape[0], envs.action_space, args.recurrent_policy)
    else:
        assert not args.recurrent_policy, \
            "Recurrent policy is not implemented for the MLP controller"
        actor_critic = MLPPolicy(obs_numel, envs.action_space)
        
    modelSize = 0
    for p in actor_critic.parameters():
        pSize = reduce(operator.mul, p.size(), 1)
        modelSize += pSize
    print(str(actor_critic))
    print('Total model size: %d' % modelSize)

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.resume_experiment:
        print("\n############## Loading saved model ##############\n")
        actor_critic, ob_rms = torch.load(os.path.join(save_path, args.env_name + args.save_tag + ".pt"))
        tr.load(os.path.join(log_path, args.env_name + args.save_tag + ".p"))
    
    if args.cuda:
        actor_critic.cuda()

    if args.algo == 'a2c':
        optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
    elif args.algo == 'ppo':
        optimizer = optim.Adam(actor_critic.parameters(), args.lr, eps=args.eps)
    elif args.algo == 'acktr':
        optimizer = KFACOptimizer(actor_critic)

    print(obs_shape)

    
    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    rollouts_test = RolloutStorage(args.num_steps_test, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(args.num_processes, *obs_shape)
    current_obs_test = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs, test = False):
        shape_dim0 = envs.observation_space.shape[0]
        if args.saved_encoder_model:
            shape_dim0 = 1
            obs, _ = vae.encode(Variable(torch.cuda.FloatTensor(obs)))
            obs = obs.data.cpu().numpy()
        obs = torch.from_numpy(obs).float()
        if not test:
            if args.num_stack > 1:
                current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
            current_obs[:, -shape_dim0:] = obs
        else:
            if args.num_stack > 1:
                current_obs_test[:, :-shape_dim0] = current_obs_test[:, shape_dim0:]
            current_obs_test[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)
    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])
    reward_avg = 0

    if args.cuda:
        current_obs = current_obs.cuda()
        current_obs_test = current_obs_test.cuda()
        rollouts.cuda()
        rollouts_test.cuda()

    start = time.time()

    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            value, action, action_log_prob, states = actor_critic.act(Variable(rollouts.observations[step], volatile=True),
                                                                      Variable(rollouts.states[step], volatile=True),
                                                                      Variable(rollouts.masks[step], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Observation, reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)

            # Maxime: clip the reward within [0,1] for more reliable training
            # This code deals poorly with large reward values
            reward = np.clip(reward, a_min=0, a_max=None) / 400

            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks
            
            tr.episodes_done += args.num_processes - masks.sum()


            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs)
            rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, value.data, reward, masks)

        next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True),
                                  Variable(rollouts.states[-1], volatile=True),
                                  Variable(rollouts.masks[-1], volatile=True))[0].data

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        tr.iterations_done += 1

        if args.algo in ['a2c', 'acktr']:
            values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(Variable(rollouts.observations[:-1].view(-1, *obs_shape)),
                                                                                           Variable(rollouts.states[0].view(-1, actor_critic.state_size)),
                                                                                           Variable(rollouts.masks[:-1].view(-1, 1)),
                                                                                           Variable(rollouts.actions.view(-1, action_shape)))

            values = values.view(args.num_steps, args.num_processes, 1)
            action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

            advantages = Variable(rollouts.returns[:-1]) - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(Variable(advantages.data) * action_log_probs).mean()

            if args.algo == 'acktr' and optimizer.steps % optimizer.Ts == 0:
                # Sampled fisher, see Martens 2014
                actor_critic.zero_grad()
                pg_fisher_loss = -action_log_probs.mean()

                value_noise = Variable(torch.randn(values.size()))
                if args.cuda:
                    value_noise = value_noise.cuda()

                sample_values = values + value_noise
                vf_fisher_loss = -(values - Variable(sample_values.data)).pow(2).mean()

                fisher_loss = pg_fisher_loss + vf_fisher_loss
                optimizer.acc_stats = True
                fisher_loss.backward(retain_graph=True)
                optimizer.acc_stats = False

            optimizer.zero_grad()
            (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()

            if args.algo == 'a2c':
                nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

            optimizer.step()

        elif args.algo == 'ppo':
            advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            for e in range(args.ppo_epoch):
                if args.recurrent_policy:
                    data_generator = rollouts.recurrent_generator(advantages,
                                                            args.num_mini_batch)
                else:
                    data_generator = rollouts.feed_forward_generator(advantages,
                                                            args.num_mini_batch)

                for sample in data_generator:
                    observations_batch, states_batch, actions_batch, \
                       return_batch, masks_batch, old_action_log_probs_batch, \
                            adv_targ = sample

                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(Variable(observations_batch),
                                                                                                   Variable(states_batch),
                                                                                                   Variable(masks_batch),
                                                                                                   Variable(actions_batch))

                    adv_targ = Variable(adv_targ)
                    ratio = torch.exp(action_log_probs - Variable(old_action_log_probs_batch))
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ
                    action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

                    value_loss = (Variable(return_batch) - values).pow(2).mean()

                    optimizer.zero_grad()
                    (value_loss + action_loss - dist_entropy * args.entropy_coef).backward()
                    nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)
                    optimizer.step()

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                            hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + args.save_tag + ".pt"))

            total_test_reward_list = []
            step_test_list = []

            for _ in range(args.num_tests):
                test_obs = test_envs.reset()
                update_current_obs(test_obs, test = True)
                rollouts_test.observations[0].copy_(current_obs_test)
                step_test = 0
                total_test_reward = 0

                while step_test < args.num_steps_test:
                    value_test, action_test, action_log_prob_test, states_test = actor_critic.act(Variable(rollouts_test.observations[step_test], volatile=True),
                                                                          Variable(rollouts_test.states[step_test], volatile=True),
                                                                          Variable(rollouts_test.masks[step_test], volatile=True))
                    cpu_actions_test = action_test.data.squeeze(1).cpu().numpy()

                    # Observation, reward and next obs
                    obs_test, reward_test, done_test, info_test = test_envs.step(cpu_actions_test)

                    # masks here doesn't really matter, but still
                    masks_test = torch.FloatTensor([[0.0] if done_test_ else [1.0] for done_test_ in done_test])
                    
                    # Maxime: clip the reward within [0,1] for more reliable training
                    # This code deals poorly with large reward values
                    reward_test = np.clip(reward_test, a_min=0, a_max=None) / 400
                    
                    total_test_reward += reward_test[0]
                    reward_test = torch.from_numpy(np.expand_dims(np.stack(reward_test), 1)).float()

                    update_current_obs(obs_test)
                    rollouts_test.insert(step_test, current_obs_test, states_test.data, action_test.data, action_log_prob_test.data,\
                     value_test.data, reward_test, masks_test)

                    step_test += 1
                    
                    if done_test:
                        break
                
                #rollouts_test.reset() # Need to reinitialise with .cuda(); don't forget
                total_test_reward_list.append(total_test_reward)
                step_test_list.append(step_test)

            append_to(tr.test_reward, tr, sum(total_test_reward_list)/args.num_tests)
            append_to(tr.test_episode_len, tr, sum(step_test_list)/args.num_tests)

            logger.log_scalar_rl("test_reward", tr.test_reward[0], args.sliding_wsize, [tr.episodes_done, tr.global_steps_done, tr.iterations_done])
            logger.log_scalar_rl("test_episode_len", tr.test_episode_len[0], args.sliding_wsize, [tr.episodes_done, tr.global_steps_done, tr.iterations_done])

            # Saving all the MyContainer variables
            tr.save(os.path.join(log_path, args.env_name + args.save_tag + ".p"))

        if j % args.log_interval == 0:
            reward_avg = 0.99 * reward_avg + 0.01 * final_rewards.mean()
            end = time.time()
            tr.global_steps_done = (j + 1) * args.num_processes * args.num_steps

            print(
                "Updates {}, num timesteps {}, FPS {}, running avg reward {:.3f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(
                    j,
                    tr.global_steps_done,
                    int(tr.global_steps_done / (end - start)),
                    reward_avg,
                    dist_entropy.data[0],
                    value_loss.data[0],
                    action_loss.data[0]
                )
            )

            append_to(tr.pg_loss, tr, action_loss.data[0])
            append_to(tr.val_loss, tr, value_loss.data[0])
            append_to(tr.entropy_loss, tr, dist_entropy.data[0])
            append_to(tr.train_reward_avg, tr, reward_avg)

            logger.log_scalar_rl("train_pg_loss", tr.pg_loss[0], args.sliding_wsize, [tr.episodes_done, tr.global_steps_done, tr.iterations_done])
            logger.log_scalar_rl("train_val_loss", tr.val_loss[0], args.sliding_wsize, [tr.episodes_done, tr.global_steps_done, tr.iterations_done])
            logger.log_scalar_rl("train_entropy_loss", tr.entropy_loss[0], args.sliding_wsize, [tr.episodes_done, tr.global_steps_done, tr.iterations_done])
            logger.log_scalar_rl("train_reward_avg", tr.train_reward_avg[0], args.sliding_wsize, [tr.episodes_done, tr.global_steps_done, tr.iterations_done])

            """
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    final_rewards.mean(),
                    final_rewards.median(),
                    final_rewards.min(),
                    final_rewards.max(), dist_entropy.data[0],
                    value_loss.data[0], action_loss.data[0])
                )
            """

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name, args.algo)
            except IOError:
                pass

if __name__ == "__main__":
    main()
