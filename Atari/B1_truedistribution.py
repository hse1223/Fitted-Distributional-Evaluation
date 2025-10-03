##### PACKAGES

from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
import math # math.isnan
import os
import time
import psutil
from torch.utils.tensorboard import SummaryWriter # needed for displaying in tensorboard (need to do pip install tensorboard. needs to be above specific version.)
import argparse

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor # stable_baselines3 requires gym==0.17.3
from wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames

import msgpack
from msgpack_numpy import patch as mspack_numpy_patch
mspack_numpy_patch() # needed to save the trained model.

from A1_dqn_train import nature_cnn, DQN
from torch.distributions.normal import Normal
import pickle
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for OPE Experiment")
    parser.add_argument('--game', type=str, help='Name of the Atari game environment')
    parser.add_argument('--return_samples', type=int, help='Number of return samples.')
    parser.add_argument('--epsilon_target', type=float, help='Epsilon of target policy (Only up to 1 digit below the decimal point.)')
    parser.add_argument('--reward_var', type=float, help='Variance of reward (Only integer values recommended.)')    
    return parser.parse_args()


if __name__ == "__main__": 
    

    args = parse_args()
    game = args.game    
    M_inaccuracy = args.return_samples
    reward_var=args.reward_var
    epsilon_target=args.epsilon_target

    # reward_var=1. ; epsilon_target=0.3      # Stage=8 (Breakout)
    # reward_var=1. ; epsilon_target=0.0      # Stage=8 (non-Breakout)

    GAMMA=0.95; reward_mean = 0.; reward_min=-10.0; reward_max=10.0; nonzero_reward_multiply=20 # Hard-coded tuning parameters.

    NUM_ENVS = 1; use_cuda = True; dummy_or_subproc = "subproc"

    if use_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    ### Setting

    input_envs = [lambda: make_atari_deepmind(game, seed=i, scale_values=True) for i in range(NUM_ENVS)]

    if dummy_or_subproc=="dummy":
        vec_env = DummyVecEnv(input_envs) 
    elif dummy_or_subproc=="subproc":
        vec_env = SubprocVecEnv(input_envs) 
    else:
        raise ValueError("dummy_or_subproc must be either 'dummy' or 'subproc'")    

    env = BatchedPytorchFrameStack(vec_env, k=4) 

    optimal_policy = DQN(env, device)
    optimal_policy = optimal_policy.to(device)
    optimal_policy.load('./optimal/model/' + game + '.pack')


    seedno=123456789
    torch.manual_seed(seedno)
    # env.seed(seed) # common seed. but we are going to use different seeds for each env.
    env.action_space.seed(seedno) 
    random.seed(seedno)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dpi_states_list = []
    dpi_actions_list = []
    return_marginal_list = []
    iteration_return = int(np.emath.logn(GAMMA, 1e-5))  # so that gamma^{max_iter} = 1e-5 (very small)

    obses = env.reset() # 이거 for-loop 바깥으로 빼는 게 좋을 듯. 
    for sample_ind in range(M_inaccuracy):
        
        act_obses = np.stack([o.get_frames() for o in obses]) # 4 (NUM_ENVS), 4 (stacked), 84, 84
        actions = optimal_policy.act(act_obses, epsilon=epsilon_target)


        ### Sample s,a ~ dpi.

        prev_life=0 # can be arbitrary
        new_stage=True
        accept = False
        while not accept:
            u = random.random()
            if u <= 1 - GAMMA:
                accept = True
                dpi_states_list.append(obses[0])
                dpi_actions_list.append(actions[0])
            else:
                if new_stage:
                    actions=[1] # For Breakout.
                new_obses, rewards, dones, infos = env.step(actions)
                # print(actions); print(new_obses[0].get_frames().shape); exit()

                life = infos[0]['ale.lives']
                if life < prev_life:
                    new_stage=True
                    # new_obses=env.reset()
                else:
                    new_stage=False
                prev_life = life

                # if rewards[0]!=0:
                #     print("Step=" + str(t) + " (new) reward : {:.4f}".format(rewards[0]))
                # rewards[0] = max(reward_min, min(rewards[0], reward_max)) * nonzero_reward_multiply  # clip  & multiply
                # rewards[0] += random.gauss(reward_mean, math.sqrt(reward_var))       # add noise

                new_act_obses = np.stack([o.get_frames() for o in new_obses]) # 4 (NUM_ENVS), 4 (stacked), 84, 84
                new_actions = optimal_policy.act(new_act_obses, epsilon=epsilon_target)


                obses = new_obses
                actions = new_actions


        ### Generate a return.

        rewards_list = []
        for t in range(iteration_return):

            # a=env.render()

            if new_stage:
                actions=[1]

            new_obses, rewards, dones, infos = env.step(actions)
            # print(rewards[0])

            life = infos[0]['ale.lives']
            if life < prev_life:
                new_stage=True
                # new_obses=env.reset() # 이상하다. env.reset()이 게임을 리셋하지 않는다. 
            else:
                new_stage=False
            prev_life = life

            if rewards[0]!=0:
                print("Step=" + str(t) + " (new) reward : {:.4f}".format(rewards[0]))

            rewards[0] = max(reward_min, min(rewards[0], reward_max)) * nonzero_reward_multiply  # clip  & add
            rewards[0] += random.gauss(reward_mean, math.sqrt(reward_var))       # add noise

            new_act_obses = np.stack([o.get_frames() for o in new_obses]) # 4 (NUM_ENVS), 4 (stacked), 84, 84
            new_actions = optimal_policy.act(new_act_obses, epsilon=epsilon_target)

            obses = new_obses
            actions = new_actions
            rewards_list.append(rewards[0])            
            discounted_return = sum(r * (GAMMA ** i) for i, r in enumerate(rewards_list))

        # print(rewards_list)
        # exit()

        return_marginal_list.append(discounted_return)

        if (sample_ind+1) % 1 == 0:
            print(str(sample_ind+1)+"-th (marginal) return value : {:.10f}".format(discounted_return))

    data = {
        "states": dpi_states_list,
        "actions": dpi_actions_list,
        "returns": return_marginal_list
    }


    ### Save

    # directory_path = "dpi_values/"
    directory_path = "dpi_values_rewardvar" + str(int(reward_var)) + "/"
    # print(directory_path)

    os.makedirs(os.path.dirname(directory_path), exist_ok=True)    
    filename = directory_path + game + "_Gamma" + str(int(GAMMA*100)) + ".pkl"
    with open(filename, "wb") as f:
        pickle.dump(data, f)

    ### Plot 

    plt.figure(figsize=(8, 5))
    # plt.hist(return_marginal_list, bins=50, edgecolor='black', alpha=0.7)
    plt.hist(return_marginal_list, edgecolor='black', alpha=0.7, density=True)
    plt.title("(Marginal returns) " + game + " : " + "gamma=0."+ str(int(GAMMA*100)))
    plt.xlabel("Return Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    imagename = directory_path + game + "_Gamma" + str(int(GAMMA*100)) + ".png"
    plt.savefig(imagename)

# Multiple modes represent the difference from the games. 
# If there was no effect from game (e.g. reward=0 in all steps), then it should be one gaussian distribution.



