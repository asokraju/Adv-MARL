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
from collections import deque
import os
import argparse
import pprint as pp




class Grid_World(gym.Env):
    """
    Multi agent Grid-World
    Objective of each agent is to reach their desired positions without colliding and following a shortest path
    nrow, ncol: dimensions of the grid world 
    n_agents: number of agents/players
    rew_scaling: if True scales the reward with some values [p1,p2,p3] (s.t. p1+p2+p3 =1 ), 
                 for example for 3 agents, the rewards [r1,r2,r3] will be transformed to
                  [p1*r1, p2*r2, p3*r3]
    """ 
    metadata = {'render.modes': ['console']}

    def __init__(self, nrow = 5, ncol=5, n_agents = 1, rew_scaling = False):
        super(Grid_World, self).__init__()
        self.nrow = nrow
        self.ncol = ncol
        self.n_agents = n_agents
        self.rew_scaling = rew_scaling

        self.total_states = self.nrow * self.ncol
        self.n_actions = 5
        self.actions = {0:'LEFT', 1:'DOWN', 2:'RIGHT', 3:'UP', 4:'STAY'}
        #self.observation_space = spaces.Discrete(self.total_states*n_agents)
        self.observation_space = gym.spaces.MultiDiscrete([self.total_states for _ in range(self.n_agents)])
        self.action_space = gym.spaces.MultiDiscrete([self.n_actions for _ in range(self.n_agents)])

        self._state_map()
        self._get_state()
        self._get_desired()
        self._reward_scaling()

        self.reward, self.done = np.full_like(self.state, 0.0), False

    def _get_state(self):
        self.state = np.array(random.sample(range(self.total_states), self.n_agents))

    def _get_desired(self):
        self.desired_state = np.array([0, 4, 14, 24, 32])#np.array(random.sample(range(self.total_states), self.n_agents) )

    def _set_state(self, state):
        self.state = state

    def _set_desired(self, desired_state):
        self.desired_state = desired_state
    
    def _reward_scaling(self):
        temp = np.random.uniform(low=0, high=1, size=(self.n_agents,))
        self.reward_scaling = temp/temp.sum()

    def _to_s(self, row, col):
        return row*self.ncol + col
    
    def _state_map(self):
        """
        self.state_transformation: is a dict with 
                                    key - (x,y) co-ordinate
                                    vaue - state (0 - self.total_states)
        self.i_state_transformation: is a dict with 
                                    key - state (0 - self.total_states)
                                    vaue - (x,y) co-ordinate                                   
        """
        self.state_transformation={}
        for row in range(self.nrow):
            for col in range(self.ncol):
                self.state_transformation[(row, col)] = self._to_s(row, col)
        self.i_state_transformation={}
        for key, value in self.state_transformation.items():
            self.i_state_transformation[value]=key

    def _dist(self, s1, s2):
        """
        measures the shortest distance between the state s1 and s2
        arguments: s1, s2 are integers
        returns: integer
        """
        x1, y1 = self.i_state_transformation[s1]
        x2, y2 = self.i_state_transformation[s2]
        return abs(x1-x2)+abs(y1-y2)

    def _inc(self, row, col, a):
        """
        at (row,col) coordinate, taking the action from (0,1,2,3)
        we arrive at (row, col) coordinate
        
        arguments: row, col, a are integers
        a:  0 - LEFT
            1 - DOWN
            2 - RIGHT
            3 - UP
        returns: (row, col) tuple of integers
        """
        if a == 0:
            col = max(col - 1, 0)
        elif a == 1:
            row = min(row + 1, self.nrow - 1)
        elif a == 2:
            col = min(col + 1, self.ncol - 1)
        elif a == 3:
            row = max(row - 1, 0)
        return (row, col)
    
    def reset(self):
        """
        resets the environment
        1 - resets the state and its desired values to some random values
        2- returns the current state numpy array of dim (self.n_agents, )
        """
        self._get_state()
        #self._get_desired()
        return self.state

    def step(self, a):
        """
        --returns the new state, reward, terminal states after taking action a

        -arguments
        a: represent the actions of all agents, a numpy array of shape (self.n_agents, )
        -returns
        self.state: numpy array (self.n_agents, )
        self.reward: numpy array (self.n_agents, )
        done: True (if all agents reached their desired positions) or False (otherwise)
        """
        temp_s = []
        rew =[]
        d = []
        for s,a, s_d in zip(self.state, a, self.desired_state):
            x, y = self.i_state_transformation[s]
            new_x, new_y = self._inc(x, y, a)
            new_s = self.state_transformation[(new_x, new_y)]
            temp_s.append(new_s)
            rew.append(-self._dist(s,s_d))

        for i, s in enumerate(self.state):
            sub_s = np.delete(self.state, [i])
            if s in sub_s:
                rew[i] = rew[i] - 1#(self.ncol+self.nrow)/2
        
        self.state = np.array(temp_s)
        if self.rew_scaling:
            self.reward = np.array(rew)*self.reward_scaling
        else:
            self.reward = np.array(rew, dtype=float)
        self.done = np.array_equal(self.state, self.desired_state)

        return self.state, self.reward, self.done, {}
    
    def get_node(self, node_index, complete_state = True):
        """
        This can be used to control information leak in MARL
        using get_node we get the reward of the specified node (only)
        If complete_state is False, then we can not observe the states of other agents
        always use get_node after using the step method
        -arguments
        node_index: integer in (0, self.n_agents)
        complete_state: bool
        """
        if complete_state:
            return self.state, self.reward[node_index], self.done, {}
        else:
            return self.state[node_index], self.reward[node_index], self.done, {}

    def close(self):
        pass


def discount_reward(rewards, GAMMA=0.95):
    reward_sum = 0
    discounted_rewards = []
    for reward in rewards[::-1]:  # reverse buffer r
        reward_sum = reward + GAMMA * reward_sum
        discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()
    discounted_rewards = np.array(discounted_rewards)
    len = discounted_rewards.shape[0]
    return np.array(discounted_rewards).reshape((len, 1))

def get_action(actor_model, state, eps=0.04):
    p = actor_model.predict(np.reshape(state, (1,-1)))#.numpy()
    n_actions = np.shape(p)[1]
    action_from_policy = np.random.choice(n_actions, p = p[0])
    random_action=np.random.choice(5)
    #np.random.choice([action_from_policy,random_action], p = [1-eps,eps])
    return np.random.choice([action_from_policy,random_action], p = [1-eps,eps])

def train_multi_agent(env, args, actors, critics, rew_approx, reward_result):
    
    # Needs 'ReplayBuffer' class
    writer = tf.summary.create_file_writer(logdir = args['summary_dir'])

    nodes = env.n_agents
    des=np.array([0, 4, 14, 24, 32])
    adv_des=np.array([0, 35, 35, 35, 35])
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
        # else:
        #     eps = 0.04

        #decreasing the learning rates:
        if t>100:
            for node in range(nodes):
                tf.keras.backend.set_value(rew_approx[node].optimizer.learning_rate, args['rew_lr']*100/t)
                tf.keras.backend.set_value(critics[node].optimizer.learning_rate, args['critic_lr']*100/t)
                tf.keras.backend.set_value(actors[node].optimizer.learning_rate, args['actor_lr']*100/t)
        
        ep_reward = 0

        #initializing the lists
        obs, obs_scaled, actions, rewards = [[] for _ in range(nodes)], [[] for _ in range(nodes)], [[] for _ in range(nodes)], [[] for _ in range(nodes)]
        rewards_mean = [[] for _ in range(nodes)]
        #rewards_predicted = [[] for _ in range(nodes)]
        done = False
        #running the episode
        j= 0
        #we first save all the samples in the replay buffer
        for node in range(nodes):
            s, r, done, _ = env.get_node(node)
            s_scaled = (s-mean)/var
            obs[node].append(s.tolist())
            obs_scaled[node].append(s_scaled.tolist())

        average_critic_weights = [[] for _ in range(nodes)]
        if done:
            print(t,j)
        while not done:
            for node in range(nodes):
                if args['scaling']:
                    a = get_action(actors[node], obs_scaled[node][-1], eps = eps)
                else:
                    a = get_action(actors[node], obs[node][-1], eps = eps)
                actions[node].append(a)    
        
            _, new_reward, done, _ = env.step([actions[node][-1] for node in range(nodes)])
            if done:
                print("done={}".formaat(done))
            for node in range(nodes):
                s, r, done, _ = env.get_node(node)
                obs[node].append(s.tolist())
                if node==0:
                    temp = 0
                    for s1, s2 in zip(adv_des,s):
                        x1, y1 = env.i_state_transformation[s1]
                        x2, y2 = env.i_state_transformation[s2]
                        temp = temp + abs(x1-x2)+abs(y1-y2)
                    r =  -3*temp
                s = (s-mean)/var
                obs_scaled[node].append(s.tolist())
                rewards[node].append(r)
                rewards_mean[node].append(new_reward.mean())
            ep_reward = ep_reward + new_reward
            j= j+1
            if done | (j==args['max_episode_len']):
                act_loss, crit_loss, rew_loss = [], [], []
                #training the reward network
                for node in range(nodes):
                    states = np.vstack(obs_scaled[node][:-1])

                    if t<100:
                        for _ in range(100):
                            r_loss = rew_approx[node].train_on_batch(states, np.reshape(rewards[node], (-1,1)))
                    else:
                        r_loss = rew_approx[node].train_on_batch(states, np.reshape(rewards[node], (-1,1)))
                    rew_loss.append(r_loss)

                #Consensus update on the reward network
                for i in range(len(rew_approx[node].trainable_variables)):
                    temp = 0
                    for node in range(nodes):
                        temp = temp + rew_approx[node].trainable_variables[i]
                    for node in range(nodes):
                        rew_approx[node].trainable_variables[i].assign(temp/env.n_agents)

                #training the Actor and Critic networks
                for node in range(nodes):
                    states = np.vstack(obs_scaled[node][:-1])
                    final_state = np.vstack(obs_scaled[node][-1])
                    predicted_rewards = rew_approx[node].predict(states)
                    if node==0:
                        returns = discount_reward(rewards[node], GAMMA=args['gamma'])
                    else:
                        returns = discount_reward(predicted_rewards, GAMMA=args['gamma'])
                    #returns = discount_reward(rewards_mean[node], GAMMA=args['gamma'])
                    returns -= np.mean(returns)
                    returns /= np.std(returns)
                    
                    targets_actions = np.array([[1 if a==i else 0 for i in range(env.n_actions)]  for j, a in enumerate(actions[node])])

                    V_s0 = critics[node].predict(states)
                    V_s1 = critics[node].predict(np.reshape(final_state,(1, len(final_state))))

                    fin_discount = np.array([args['gamma'] ** (i+1) for i in range(j)][::-1])*V_s1
                    td = returns + fin_discount.reshape((j,1)) - V_s0

                    
                    for _ in range(100-np.min([np.int(t/1), 99])):
                        c_loss = critics[node].train_on_batch(states, returns+fin_discount.reshape((j,1)))
                    
                    for _ in range(10-np.min([np.int(t/10),9])):
                        a_loss = actors[node].train_on_batch(states, targets_actions, sample_weight=td)
                    act_loss.append(a_loss)
                    crit_loss.append(c_loss)
                
                #Consensus update on the critic networks
                for i in range(len(critics[node].trainable_variables)):
                    temp = 0
                    for node in range(nodes):
                        temp = temp + critics[node].trainable_variables[i]
                    for node in range(nodes):
                        critics[node].trainable_variables[i].assign(temp/env.n_agents)


                with writer.as_default():
                    tf.summary.scalar("actor loss", np.mean(act_loss), step = t)
                    tf.summary.scalar("critic loss", np.mean(crit_loss), step = t)
                    tf.summary.scalar("critic loss", np.mean(rew_loss), step = t)
                    writer.flush()
                print('| Reward: {} | Episode: {} | actor loss: {} |critic loss: {} | reward loss: {} | done: {}'.format(ep_reward, t, np.mean(act_loss), np.mean(crit_loss),np.mean(rew_loss), done))
                fig, ax = plt.subplots(nrows=1, ncols=5, figsize = (24,4))
                for i in range(5):
                    ax[i].plot(range(j), rewards[i])
                plt.show()
                reward_result[t] = ep_reward.sum()

                path = {
                    "Observation":obs, 
                    "Action":actions,#np.concatenate(actions), 
                    "Reward":rewards#np.asarray(rewards)
                    }
                paths.append(path)
                break
                done = False
    return [paths, reward_result] 


def main(args, reward_result):
    np.random.seed(args['random_seed'])
    random.seed(args['random_seed'])
    tf.random.set_seed(args['random_seed'])

    env = Grid_World(6,6,5)

    num_actions = env.n_actions

    actor_1 = keras.Sequential([
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(num_actions, activation='softmax')
    ])
    actor_1.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    actor_2 = keras.Sequential([
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(num_actions, activation='softmax')
    ])
    actor_2.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    actor_3 = keras.Sequential([
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(num_actions, activation='softmax')
    ])
    actor_3.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    actor_4 = keras.Sequential([
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(num_actions, activation='softmax')
    ])
    actor_4.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    actor_5 = keras.Sequential([
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(num_actions, activation='softmax')
    ])
    actor_5.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    actors = [actor_1, actor_2, actor_3, actor_4, actor_5]


    critic_output = 1

    critic_1 = keras.Sequential([
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(critic_output)
    ])
    critic_1.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    critic_2 = keras.Sequential([
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(critic_output)
    ])
    critic_2.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    critic_3 = keras.Sequential([
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(critic_output)
    ])
    critic_3.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    critic_4 = keras.Sequential([
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(critic_output)
    ])
    critic_4.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    critic_5 = keras.Sequential([
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(critic_output)
    ])
    critic_5.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=0.0001))

    critics = [critic_1, critic_2, critic_3, critic_4, critic_5]


    rew_output = 1

    rew_1 = keras.Sequential([
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(rew_output)
    ])
    rew_1.compile(loss='mse',optimizer=keras.optimizers.Adam())

    rew_2 = keras.Sequential([
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(rew_output)
    ])
    rew_2.compile(loss='mse',optimizer=keras.optimizers.Adam())

    rew_3 = keras.Sequential([
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(rew_output)
    ])
    rew_3.compile(loss='mse',optimizer=keras.optimizers.Adam())

    rew_4 = keras.Sequential([
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(rew_output)
    ])
    rew_4.compile(loss='mse',optimizer=keras.optimizers.Adam())

    rew_5 = keras.Sequential([
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(30, activation='relu', kernel_initializer=keras.initializers.he_normal()),
        keras.layers.Dense(rew_output)
    ])
    rew_5.compile(loss='mse',optimizer=keras.optimizers.Adam())

    rew_approx = [rew_1, rew_2, rew_3, rew_4, rew_5]

    paths, reward_result = train_multi_agent(env, args, actors, critics, rew_approx, reward_result)

    savemat(os.path.join(args['summary_dir'], 'data.mat'), dict(data=paths, reward=reward_result))

    return [paths, reward_result]


#---------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')
    #loading the environment to get it default params
    env = Grid_World(6,6,5)
    #state_dim = env.observation_space.shape[0]
    #action_dim = env.action_space.shape[0]
    #action_bound = env.action_space.high
    #--------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    #general params
    parser.add_argument('--summary_dir', help='directory for saving and loading model and other data', default='./Power-Converters/kristools/results')
    #parser.add_argument('--use_gpu', help='weather to use gpu or not', type = bool, default=True)
    #parser.add_argument('--save_model', help='Saving model from summary_dir', type = bool, default=False)
    #parser.add_argument('--load_model', help='Loading model from summary_dir', type = bool, default=True)
    parser.add_argument('--random_seed', help='seeding the random number generator', default=1754)
    
    #agent params
    #parser.add_argument('--buffer_size', help='replay buffer size', type = int, default=1000000)
    parser.add_argument('--max_episodes', help='max number of episodes', type = int, default=500)
    parser.add_argument('--max_episode_len', help='Number of steps per epsiode', type = int, default=500)
    #parser.add_argument('--mini_batch_size', help='sampling batch size',type =int, default=200)
    parser.add_argument('--actor_lr', help='actor network learning rate',type =float, default=0.001)
    parser.add_argument('--critic_lr', help='critic network learning rate',type =float, default=0.001)
    parser.add_argument('--rew_lr', help='critic network learning rate',type =float, default=0.001)
    parser.add_argument('--gamma', help='models the long term returns', type =float, default=0.999)
    #parser.add_argument('--noise_var', help='Variance of the exploration noise', default=0.0925)
    parser.add_argument('--scaling', help='weather to scale the states before using for training', type = bool, default=True)
    
    #model/env paramerters
    #parser.add_argument('--state_dim', help='state dimension of environment', type = int, default=state_dim)
    #parser.add_argument('--action_dim', help='action space dimension', type = int, default=action_dim)
    #parser.add_argument('--action_bound', help='upper and lower bound of the actions', type = float, default=action_bound)
    #parser.add_argument('--discretization_time', help='discretization time used for the environment ', type = float, default=1e-3)

    #Network parameters
    #parser.add_argument('--time_steps', help='Number of time-steps for rnn (LSTM)', type = int, default=2)
    #parser.add_argument('--actor_rnn', help='actor network rnn paramerters', type = int, default=20)
    #parser.add_argument('--actor_l1', help='actor network layer 1 parameters', type = int, default=400)
    #parser.add_argument('--actor_l2', help='actor network layer 2 parameters', type = int, default=300)
    #parser.add_argument('--critic_rnn', help='critic network rnn parameters', type = int, default=20)
    #parser.add_argument('--critic_l1', help='actor network layer 1 parameters', type = int, default=400)
    #parser.add_argument('--critic_l2', help='actor network layer 2 parameters', type = int, default=300)
    #parser.add_argument('--tau', help='target network learning rate', type = float, default=0.001)
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    reward_result = np.zeros(2500)
    [paths, reward_result] = main(args, reward_result)

    savemat('data4_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat',dict(data=paths, reward=reward_result))