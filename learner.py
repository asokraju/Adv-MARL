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
import os
import argparse
import pprint as pp

# local modules
#from utilities.utils import ReplayBuffer
from environments.grid_world import Grid_World
#from adversary.agent import ActorNetwork, CriticNetwork, EstimatedGlobalReward, train_multi_agent

class Actor(tf.keras.Model):
    """
    creates a two layer nn with a soft max output layer for actor.
    we use 'categorical_crossentropy' as the loss function and Adam optimzer
    Arguments:
    num_actions - (int) 
    lr - (float)
    """
    def __init__(self, num_actions = 5):
        super(Actor, self).__init__()
        self.dense_1 = keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal())
        self.dense_2 = keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal())
        self.output_layer = keras.layers.Dense(num_actions, activation='softmax')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return self.output_layer(x)




class Critic(tf.keras.Model):
    """
    creates a two layer nn with a soft max output layer for critic.
    """
    def __init__(self):
        super(Critic, self).__init__()
        self.dense_1 = keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal())
        self.dense_2 = keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal())
        self.output_layer = keras.layers.Dense(1)
        #self.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam())


    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return self.output_layer(x)



def discount_reward(rewards, gamma=0.95):
    """
    given a numpy array/list and a discount factor, it compute the returns.
    ex: [1,2,3,4,5] with gamma = 1, returns [1+2+3+4+5, 2+3+4+5, 3+4+5, 4+5, 5]
    Arguments
    rewards: numpy array or list of shapes (_, 1) or (_, )
    gamma: float
    Returns:
    a numpy array of shape (_, 1)
    where _ denotes the len of the input array
    """
    reward_sum = 0
    discounted_rewards = []
    for reward in rewards[::-1]:  # reverse buffer r
        reward_sum = reward + gamma * reward_sum
        discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()
    discounted_rewards = np.array(discounted_rewards)
    len = discounted_rewards.shape[0]
    return np.array(discounted_rewards).reshape((len, 1))


def get_action(actor_model, state, eps=0.04):
    """
    given the actor model and the current state, it provide an epsilon greedy action
    Arguments:
    actor_model - actor nn
    state - numpy array
    eps - float
    Returns: int
    """
    p = actor_model.predict(np.reshape(state, (1,-1)))
    n_actions = np.shape(p)[1]
    action_from_policy = np.random.choice(n_actions, p = p[0])
    random_action=np.random.choice(5)
    #np.random.choice([action_from_policy,random_action], p = [1-eps,eps])
    return np.random.choice([action_from_policy,random_action], p = [1-eps,eps])



def train_multi_agent(env, args, actors, critics, reward_result, plot=True):
    """
    A working decentralized multi agent policy gradient algorithm.
    Arguments:
    env - gym enviroment
    args - various hyper parameters
    actors - a list containing the actor nn's
    critic - a list containing critic nn's
    reward_result - saves the episodic returns
    plot -  bool, if True plots the entire simulation

    returns: 
    paths - a dict containing all the trajectories, actors and rewards of the simulation for every episode
    reward_result -  a list containing the episodic returns
    """
    
    writer = tf.summary.create_file_writer(logdir = args['summary_dir'])

    #nodes and agents are equivalent
    nodes = env.n_agents
    
    paths = list()

    x = np.arange(env.total_states)
    if args['scaling']:
        mean, var = x.mean(), x.var()
    else:
        mean, var = 0, 1

    _eps = .5
    for t in range(args['max_episodes']):
        #resetting the environments
        env.reset()

        #conditions on exploration
        if t<10:
            eps = _eps/(t+1)
        else:
            eps = 0.04

        
        ep_reward = 0

        #initializing the lists
        obs, obs_scaled, actions, rewards = [[] for _ in range(nodes)], [[] for _ in range(nodes)], [[] for _ in range(nodes)], [[] for _ in range(nodes)]
        done = False
        #running the episode
        j= 0
        for node in range(nodes):
            s, r, done, _ = env.get_node(node)
            s_scaled = (s-mean)/var
            obs[node].append(s.tolist())
            obs_scaled[node].append(s_scaled.tolist())


        while not done:
            for node in range(nodes):
                if args['scaling']:
                    a = get_action(actors[node], obs_scaled[node][-1], eps = eps)
                else:
                    a = get_action(actors[node], obs[node][-1], eps = eps)
                actions[node].append(a)    
        
            _, new_reward, done, _ = env.step([actions[node][-1] for node in range(nodes)])

            for node in range(nodes):
                s, r, done, _ = env.get_node(node)
                obs[node].append(s.tolist())
                s = (s-mean)/var
                obs_scaled[node].append(s.tolist())
                rewards[node].append(r)
            ep_reward = ep_reward + new_reward
            j= j+1
            if done | (j==args['max_episode_len']):
                act_loss, crit_loss = [], []
                for node in range(nodes):
                    returns = discount_reward(rewards[node], gamma=args['gamma'])
                    returns -= np.mean(returns)
                    returns /= np.std(returns)

                    if args['scaling']:
                        states = np.vstack(obs_scaled[node][:-1])
                        final_state = np.vstack(obs_scaled[node][-1])
                    else:
                        states = np.vstack(obs[node][:-1])
                        final_state = np.vstack(obs[node][-1])
                    
                    targets_actions = np.array([[1 if a==i else 0 for i in range(env.n_actions)]  for j, a in enumerate(actions[node])])
                    V_s0 = critics[node].predict(states)
                    V_s1 = critics[node].predict(np.reshape(final_state,(1, len(final_state))))
                    #print(V_s1)
                    fin_discount = np.array([args['gamma'] ** (i+1) for i in range(j)][::-1])*V_s1
                    td = returns + fin_discount.reshape((j,1)) - V_s0
                    #td = returns - V_s0
                    #print('np.shape(V_s0), np.shape(returns), np.shape(td), np.shape(targets_actions), states.shape')
                    #print(np.shape(V_s0), np.shape(returns), np.shape(td), np.shape(targets_actions), states.shape)
                    # a_loss = actors[node].train_on_batch(states, targets_actions, sample_weight=td)
                    # c_loss = critics[node].train_on_batch(states, returns+fin_discount.reshape((j,1)))
                    
                    for _ in range(100):
                        c_loss = critics[node].train_on_batch(states, returns+fin_discount.reshape((j,1)))
                    for _ in range(10):
                        a_loss = actors[node].train_on_batch(states, targets_actions, sample_weight=td.reshape(-1, ))
                    act_loss.append(a_loss)
                    crit_loss.append(c_loss)

                with writer.as_default():
                    tf.summary.scalar("actor loss", np.mean(act_loss), step = t)
                    tf.summary.scalar("critic loss", np.mean(crit_loss), step = t)
                    writer.flush()
                print('| Reward: {} | Episode: {} | actor loss: {} |critic loss: {} |done: {}'.format(ep_reward, t, np.mean(act_loss), np.mean(crit_loss), done))
                # fig, ax = plt.subplots(nrows=1, ncols=5, figsize = (24,4))
                # for i in range(env.n_agents):
                #     ax[i].plot(range(j), rewards[i])
                # plt.show()
                reward_result[t] = ep_reward.sum()

                path = {
                    "Observation":obs, 
                    "Action":actions,#np.concatenate(actions), 
                    "Reward":rewards#np.asarray(rewards)
                    }
                paths.append(path)
                break
    return [paths, reward_result] 


def main(args, reward_result):
    #tf.config.run_functions_eagerly(True)
    if args['use_gpu']:
        physical_devices = tf.config.list_physical_devices('GPU') 
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    env = Grid_World(6,6,5)

    critics = [Critic() for _ in range(env.n_agents)]
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    for critic in critics:
        critic.compile(loss = huber_loss, optimizer = keras.optimizers.Adam(learning_rate=args['critic_lr']))

    actors = [Actor(num_actions = env.n_actions) for _ in range(env.n_agents)]
    for actor in actors:
        actor.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=args['actor_lr']))

    

    #loading the weights
    if args['load_model']:
        for node in range(env.n_agents):
            if os.path.isfile(args['summary_dir'] + "/actor_"+ str(node) + "_weights.h5"):
                print('loading actor {} weights'.format(node+1))
                actors[node].load_weights(args['summary_dir'] + "/actor_"+ str(node) + "_weights.h5")

            if os.path.isfile(args['summary_dir'] + "/critic_"+ str(node) + "_weights.h5"):
                print('loading critic {} weights'.format(node+1))
                critics[node].load_weights(args['summary_dir'] + "/critic_"+ str(node) + "_weights.h5")
    
    paths, reward_result = train_multi_agent(env, args, actors, critics, reward_result)

    if args['save_model']:
        for node in range(env.n_agents):
            actors[node].save_weights(
            filepath = args['summary_dir'] + "/actor_"+ str(node) + "_weights.h5",
            overwrite=True,
            save_format='h5')
            print('Saving actor {} weights'.format(node+1))

            critics[node].save_weights(
                filepath = args['summary_dir'] + "/critic_"+ str(node) + "_weights.h5",
                overwrite=True,
                save_format='h5')
            print('Saving critic {} weights'.format(node+1))

    #---------------------------------------------------------------------------
    #Plotting an new testing environment
    test_env = Grid_World(6,6,5)
    nodes = test_env.n_agents

    obs = [[] for _ in range(test_env.n_agents)]
    obs_scaled = [[] for _ in range(test_env.n_agents)]
    actions = [[] for _ in range(test_env.n_agents)]
    rewards = [[] for _ in range(test_env.n_agents)]
    x = np.arange(env.total_states)
    if args['scaling']:
        mean, var = x.mean(), x.var()
    else:
        mean, var = 0, 1
    
    for node in range(nodes):
        s, r, done, _ = test_env.get_node(node)
        s_scaled = (s-mean)/var
        obs[node].append(s.tolist())
        obs_scaled[node].append(s_scaled.tolist())
        rewards[node].append(r)

    done = False
    j=0
    while not done:
        for node in range(nodes):
            if args['scaling']:
                a = get_action(actors[node], obs_scaled[node][-1], eps = 0)
            else:
                a = get_action(actors[node], obs[node][-1], eps = 0)
            actions[node].append(a)

        _, new_reward, done, _ = test_env.step([actions[node][-1] for node in range(nodes)])
        for node in range(nodes):
            s, r, done, _ = test_env.get_node(node)
            obs[node].append(s.tolist())
            s = (s-mean)/var
            obs_scaled[node].append(s.tolist())
            rewards[node].append(r)
        j=j+1    
        if done | (j==args['max_episode_len']):
            fig, ax = plt.subplots(nrows=1, ncols=5, figsize = (24,4))
            for i in range(test_env.n_agents):
                ax[i].plot(range(j+1), rewards[i])
            if args['savefig_filename']:
                savefig_filename = os.path.join(args['summary_dir'], 'marl.png')
                plt.savefig(savefig_filename)
            else:
                plt.show()
            break

    savemat(os.path.join(args['summary_dir'], 'marl.mat'), dict(data=paths, reward=reward_result))

    return [actors, critics, paths, reward_result]



if __name__ == '__main__':

    #loading the environment to get it default params
    #--------------------------------------------------------------------
    env = Grid_World(6,6,5)
    #---------------------------------------------------------------------


    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')
    #general params
    parser.add_argument('--summary_dir', help='directory for saving and loading model and other data', default='./Power-Converters/kristools/results')
    parser.add_argument('--use_gpu', help='weather to use gpu or not', type = bool, default=True)
    parser.add_argument('--save_model', help='Saving model from summary_dir', type = bool, default=True)
    parser.add_argument('--load_model', help='Loading model from summary_dir', type = bool, default=False)
    parser.add_argument('--random_seed', help='seeding the random number generator', default=1754)

    #
    parser.add_argument('--actor_lr', help='learning rate for actor model', type = float, default=0.001)
    parser.add_argument('--critic_lr', help='learning rate for critic model', type = float, default=0.01)
    parser.add_argument('--gamma', help='models the long term returns', type = float, default=0.01)
    parser.add_argument('--max_episodes', help='max number of episodes', type = int, default=500)
    parser.add_argument('--max_episode_len', help='Number of steps per epsiode', type = int, default=1000)
    parser.add_argument('--scaling', help='scaling the states', type = bool, default=True)
    parser.add_argument('--savefig_filename', help='Saving the figure of the test simulation', type = bool, default=True)
    parser.add_argument('--state_dim', help='learning rate for actor model', type = int, default=env.n_agents)
    parser.add_argument('--action_dim', help='learning rate for critic model', type = int, default=env.n_actions)
    #
    args = vars(parser.parse_args())
    pp.pprint(args)
    
    reward_result = np.zeros(2500)
    [actors, critics, paths, reward_result] = main(args, reward_result)

    savemat('data4_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat',dict(data=paths, reward=reward_result))




