#PYPI modules
import numpy as np
import random
import gym
from gym import spaces

import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, layers
import datetime
from scipy.io import savemat
import tensorflow_probability as tfp
from collections import deque


# local modules
from utilities.utils import ReplayBuffer
from environments.grid_world import Grid_World
from adversary.agent import ActorNetwork, CriticNetwork, EstimatedGlobalReward, train_multi_agent


args = {
    'state_dim' : 5,
    'action_dim' : 5,
    'actor_lr': 0.001,
    'batch_size' : 200,
    'actor_l1' : 30,
    'actor_l2' : 30,
    'critic_lr':0.01,
    'gamma':0.7,
    'critic_l1':30,
    'critic_l2':30,
    'egr_lr':0.1,
    'egr_l1':30,
    'egr_l2':30,
    'max_episodes':500,
    'summary_dir':'./Power-Converters/marl/results',
    'max_episode_len':200,
    'scaling':False,
    'buffer_size':1000000,
    'random_seed':1234,
    'mini_batch_size':500,
    'use_gpu':True
}

if args['use_gpu']:
        physical_devices = tf.config.list_physical_devices('GPU') 
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

env = Grid_World(6,6,5)

replay_buffer1 = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
replay_buffer2 = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
replay_buffer3 = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
replay_buffer4 = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
replay_buffer5 = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

replay_buffers = [replay_buffer1, replay_buffer2, replay_buffer3, replay_buffer4, replay_buffer5]


actor_args = {
    'state_dim' : args['state_dim'],
    'action_dim' : args['action_dim'],
    'learning_rate' : args['actor_lr'],
    'batch_size' : args['batch_size'],
    'params_l1' : args['actor_l1'],
    'params_l2' : args['actor_l2']
}
actors = [ActorNetwork(**actor_args) for _ in range(5)]
#actors[0].actor_model.summary()


critic_args = {
    'state_dim' : args['state_dim'],
    'action_dim' : args['action_dim'],
    'learning_rate' : args['critic_lr'],
    'gamma' : args['gamma'],
    'params_l1' : args['critic_l1'],
    'params_l2' : args['critic_l2']
}
critics = [CriticNetwork(**critic_args) for _ in range(5)]
#critics[0].critic_model.summary()


egr_args = {
    'state_dim' : args['state_dim'],
    'action_dim' : args['action_dim'],
    'learning_rate' : args['egr_lr'],
    'params_l1' : args['egr_l1'],
    'params_l2' : args['egr_l2']
}
egrs = [EstimatedGlobalReward(**egr_args) for _ in range(5)]


reward_result = np.zeros(2500)
paths, reward_result = train_multi_agent(env, args, actors, critics, egrs, reward_result, replay_buffers)