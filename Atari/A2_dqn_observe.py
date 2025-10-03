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
from torch.utils.tensorboard import SummaryWriter # needed for displaying in tensorboard
import argparse

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor # stable_baselines3 requires gym==0.17.3
from wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames

import msgpack
from msgpack_numpy import patch as mspack_numpy_patch
mspack_numpy_patch() # needed to save the trained model.

from A1_dqn_train import DQN

parser = argparse.ArgumentParser()
parser.add_argument("--game", type=str, help="Include the full version (e.g. BreakoutNoFrameskip-v4)")    
parser.add_argument("--epsilon", type=float, help="Epsilon for the target policy (trained DQn)")    
args = parser.parse_args()
game = args.game
epsilon = args.epsilon

device = torch.device('cpu')
input_envs = [lambda: make_atari_deepmind(game, seed=i, scale_values=True) for i in range(1)] # This seed ensures the same play.
vec_env = DummyVecEnv(input_envs) 
env = BatchedPytorchFrameStack(vec_env, k=4)

net = DQN(env, device)
net = net.to(device)
net.load('./optimal/model/' + game + '.pack')

obses=env.reset()
new_stage = True
prev_life = 5

cum_reward = 0
for t in itertools.count():

    a=env.render()
    time.sleep(0.02)

    act_obses = np.stack([o.get_frames() for o in obses])
    actions = net.act(act_obses, epsilon) 

    if new_stage:
        actions=[1] # If action=2 in the new_stage, the game suddenly stops for some reason for Breakout.

    obses, rew, done, info = env.step(actions)

    if rew[0]!=0:
        print("Step=" + str(t) + " reward : {:.4f}".format(rew[0]))

    life = info[0]['ale.lives']
    if life < prev_life:
        new_stage=True
        print("---------------------------------- Current life : {}".format(life))

    else:
        new_stage=False
    
    prev_life = life



