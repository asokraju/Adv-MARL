import numpy as np
import random

import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, layers
import datetime
from scipy.io import savemat



class ActorNetwork(object):
    """
    How to use:
    actor = ActorNetwork(state_dim = 5, action_dim = 5, learning_rate =0.001, batch_size = 1, params_l1 = 10, params_l2=10)
    actor.actor_model.summary()
    
    #using single input:
    #predict returns action propability vector
    a= actor.predict(s_expanded) #s_expanded is numpy array of shape (1,state_dim)
    print("predicted_action", a)
    da = a.numpy()
    print('da.argmax()',da.argmax())
    a_one_hot = [[1 if da.argmax()==i else 0 for i in range(5)]]
    print("greedy_action", a_one_hot)
    #To train use:
    print(actor.train(s_expanded, a_one_hot, np.array([0.1])))

    #to predict train on batches:
    #s_batch, a_batch, td_error are numpy arrays of shape (batch_size, state_dim), (batch_size, action_dim), and (batch_size, 1)
    #a_batch is one hot encoded vectors
    actor.predict(s_batch) # return a tensor of shape=(batch_size, action_dim) #return nothing
    actor.train(s_batch, a_batch, td_error) 
    """
    def __init__(self, state_dim, action_dim, learning_rate, batch_size, params_l1, params_l2):

        """
        params:
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.params_l1 = params_l1
        self.params_l2 = params_l2

        self.cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        #actor network
        self.inputs, self.out= self.create_actor_network()
        self.actor_model = keras.Model(inputs=self.inputs, outputs=self.out, name='actor_network')
        self.network_params = self.actor_model.trainable_variables



    def create_actor_network(self):
        """
        Arguments: (inputs) - state variable (1, state_dim) or
                            - numpy array of shape (batch_size, state_dim)
        outputs: (out)- propabilities (1, action_dim) or 
                      - tensor of shape (batch_size, action_dim)
                 (inputs) same as above
        """
        inputs = Input(shape = (self.state_dim,), batch_size = None, name = "actor_input_state")
        
        w_init = keras.initializers.he_normal()

        net = layers.Dense(self.params_l1, name = 'actor_dense_1', kernel_initializer = w_init, activation='relu')(inputs)

        net = layers.Dense(self.params_l2, name = 'actor_dense_2', kernel_initializer = w_init, activation='tanh')(net)
        
        out = layers.Dense(self.action_dim, activation='softmax', name = 'output_layer', kernel_initializer = w_init)(net)
        return inputs, out

    def train(self, state, action, weights):
        """
        Here we minimize the weighted negative log-likelihood of the 
        selected action multiplied by locally estimated global td error
        with respect to policy parameters.
        Arguments:
        state - state variable 
              - numpy array of shape (batch_size, state_dim)
        action - greedy action performed on the environment (one-hot encoded vector)
               - numpy array of shape (batch_size, action_dim)
        weights - Locally estimated global td error
                - numpy array of shape (batch_size, 1)
        """
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            y_pred = self.actor_model(state)
            #loss = tf.keras.losses.categorical_crossentropy(action*weights, y_pred)
            loss =self.cce(action, y_pred, sample_weight = weights)
        unnormalized_actor_gradients = tape.gradient(loss, self.actor_model.trainable_variables)
        actor_gradients = list(map(lambda x: tf.math.divide(x, self.batch_size), unnormalized_actor_gradients))
        self.optimizer.apply_gradients(zip(actor_gradients, self.actor_model.trainable_variables))
        #print(y_pred)
    
    def train_on_batch(self, state, action, weights):
        self.actor_model.compile(loss=self.cce,optimizer=keras.optimizers.Adam())
        self.actor_model.train_on_batch(state, action, sample_weight=weights)



    def predict(self, inputs):
        return self.actor_model(inputs)
        #tf.keras.activations.softmax(self.actor_model(inputs))



class CriticNetwork(object):
    """
    critic = CriticNetwork(5, 5, 0.001, 0.99, 10, 10)
    critic.critic_model.summary()
    # here we approximate the Q funtion Q(s, a), through which we compute the Value function V(s)
    #to predict train on batches:
    #s_batch, a_batch, targets = [r +\gamma V(s_(k+1))] are numpy arrays of shape 
    # (batch_size, state_dim), (batch_size, action_dim), and (batch_size, 1)
    #a_batch is one hot encoded vectors
    critic.predict(s_batch) # return a tensor of shape=(batch_size, action_dim) #return the Q-values shape=(batch_size, action_dim)
    critic.train(s_batch, a_batch, targets)  #returns the max_a(Q(s,a)) tensor of shape (batch_size,)
    """
    def __init__(self, state_dim, action_dim, learning_rate, gamma, params_l1, params_l2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.params_l1 = params_l1
        self.params_l2 = params_l2

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        #Critic Network and parameters
        self.inputs_state, self.out = self.create_critic_network()
        self.critic_model = keras.Model(inputs=self.inputs_state, outputs=self.out, name='critic_network')
        self.network_params = self.critic_model.trainable_variables
        

        #gradients of Q function with respect to actions
    
    def create_critic_network(self):
        inputs_state = Input(shape = (self.state_dim,), batch_size = None, name = "critic_input_state")
        #inputs_action = Input(shape = (self.action_dim,), batch_size = None, name = "critic_input_action")
        w_init =keras.initializers.he_normal()# tf.random_uniform_initializer(minval=-0.03, maxval=0.03, seed=None)
        
        #first hidden layer
        net = layers.Dense(self.params_l1, name = 'critic_dense_1', kernel_initializer = w_init, activation='relu')(inputs_state)


        # second hidden layer
        net = layers.Dense(self.params_l2, name = 'critic_dense_2', kernel_initializer = w_init, activation='relu')(net)

        #w_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03, seed=None)
        out = layers.Dense(self.action_dim, name = 'Q_val')(net)
        return inputs_state, out

    def train(self, input_state, input_action, target):
        """
        Given the targets = [r +\gamma V(s_(k+1))]
        Here we are minimizing the td error = (V(state, network_params) -target)^2 
        Note that our critic network construct Q function,
        therfore we compute the value function by taking an arg_max
        Arguments:
        input_state - state variable is used to compute prediction
                    - numpy array of shape (batch_size, state_dim)
        input_action - action used in this state (one-hot encoded)
                     - numpy array of shape (batch_size, action_dim)
        target - Q(s,a) = [r +\gamma V(s_(k+1))]
               -  numpy array of shape (batch_size, action_dim)
        returns:
                 the max_a(Q(s,a)) tensor of shape (batch_size, )
        """
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            prediction = self.critic_model(input_state)
            #prediction = tf.math.reduce_sum(prediction, axis = 1) #computes the value function
            #target_reshaped = tf.reshape(target, shape = tf.shape(prediction).numpy())
            #target_reshaped = tf.cast(target_reshaped, dtype = 'float32')
            loss = tf.keras.losses.MSE(prediction, target)
        gradients = tape.gradient(loss, self.critic_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.critic_model.trainable_variables))
        return tf.reduce_max(self.critic_model(input_state), axis=1)
  
    def predict(self, inputs_state):
        return self.critic_model(inputs_state)
  

    def consensus_update(self, weights):
        """
        weights: given weights, it will assign these to the trainables
        The shape of the weights should be same ans the 'weights' file.
        """
        for i in range(len(self.critic_model.trainable_variables)):
            self.self.critic_model.trainable_variables[i].assign(weights[i])



class EstimatedGlobalReward(object):
    """
    Estimates the average global reward.
    erg = EstimatedGlobalReward(5, 5, 0.001,  10, 10)
    erg.egr_model.summary()

    # here we approximate the R_bar = sum_{i=1...N}(r_i)
    #to predict train on batches:
    #s_batch, r^i_{t+1} are numpy arrays of shape 
    # (batch_size, state_dim), and (batch_size, 1)
    erg.predict(s_batch) # return a tensor of shape=(batch_size, 1)
    erg.train(s_batch, targets)  # return a tensor of shape=(batch_size, 1)
    """
    def __init__(self, state_dim, action_dim, learning_rate, params_l1, params_l2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.params_l1 = params_l1
        self.params_l2 = params_l2

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        #Critic Network and parameters
        self.inputs_state, self.out = self.create_reward_network()
        self.egr_model = keras.Model(inputs=self.inputs_state, outputs=self.out, name='egr_network')
        self.network_params = self.egr_model.trainable_variables
            
    def create_reward_network(self):
        inputs_state = Input(shape = (self.state_dim,), batch_size = None, name = "egr_input_state")
        #inputs_action = Input(shape = (self.action_dim,), batch_size = None, name = "critic_input_action")
        w_init = keras.initializers.he_normal()#tf.random_uniform_initializer(minval=-0.03, maxval=0.03, seed=None)
        
        #first hidden layer
        net = layers.Dense(self.params_l1, name = 'EGR_dense_1', kernel_initializer = w_init, activation='relu')(inputs_state)


        # second hidden layer
        net = layers.Dense(self.params_l2, name = 'ER_dense_2', kernel_initializer = w_init, activation='tanh')(net)


        #w_init = tf.random_uniform_initializer(minval=-0.03, maxval=0.03, seed=None)
        out = layers.Dense(1, name = 'EGR_val', kernel_initializer = w_init)(net)
        return inputs_state, out

    def train(self, state, target):
        """
        Given the targets = r^i_{t+1}
        Here we are minimizing the td error = (r(network_params) -target)^2 
        Arguments:
        input_state - state variable is used to compute Estimated Global Reward
                    - numpy array of shape (batch_size, state_dim)
        target - r^i_{t+1}
               -  numpy array of shape (batch_size, 1)
        """
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            prediction = self.egr_model(state)
            loss = tf.keras.losses.MSE(prediction, target)
        gradients = tape.gradient(loss, self.egr_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.egr_model.trainable_variables))
        return self.egr_model(state)
  
    def predict(self, state):
        return self.egr_model(state)
    
    def consensus_update(self, weights):
        """
        weights: given weights, it will assign these to the trainables
        The shape of the weights should be same ans the 'weights' file.
        """
        for i in range(len(self.egr_model.trainable_variables)):
            self.egr_model.trainable_variables[i].assign(weights[i])



def train_multi_agent(env, args, actors, critics, egrs, reward_result, replay_buffers):
    
    # Needs 'ReplayBuffer' class
    writer = tf.summary.create_file_writer(logdir = args['summary_dir'])

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
        if t<50:
            eps = _eps
        else:
            eps = 0.04

        
        ep_reward = 0
        ep_ave_max_q = 0

        #initializing the lists
        obs, obs_scaled, actions, rewards = [[] for _ in range(nodes)], [[] for _ in range(nodes)], [[] for _ in range(nodes)], [[] for _ in range(nodes)]
        done = False
        #running the episode
        j= 0
        while not done:

            #we first save all the samples in the replay buffer
            for node in range(nodes):
                s, r, done, _ = env.get_node(node)
                obs.append(s)
                s = (s-mean)/var
                obs_scaled[node].append(s.tolist())
                s_expanded = np.reshape(s, (1, env.observation_space.shape[0]))
                #greedy action
                a_p = actors[node].predict(s_expanded).numpy()

                a= a_p.argmax()
                #epsilon greedy action
                p = [eps/(env.n_actions-1) for _ in range(env.n_actions)]
                p[a] = 1-eps
                a = np.random.choice(env.n_actions, p = p)
                actions[node].append(a)
            _, new_reward, done, _ = env.step([actions[node][j] for node in range(5)])
            
            for node in range(nodes):
                s, r, done, _ = env.get_node(node)
                obs.append(s)
                s = (s-mean)/var
                obs_scaled[node].append(s.tolist())
                rewards[node].append(r)

            for node in range(nodes):
                s_0 = obs_scaled[node][-2]
                s_1 = obs_scaled[node][-1]
                replay_buffers[node].add(s_0, np.reshape(actions[node][-1], (1,)), rewards[node][-1], done, s_1)

            #we train now
            for node, replay_buffer, critic, actor in zip(range(nodes), replay_buffers, critics, actors):
                if replay_buffer.size() >= int(args['mini_batch_size']):
                    s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(int(args['mini_batch_size']))
                    a_batch_hot = np.array([[1 if a==i else 0 for i in range(env.n_actions)]  for a in a_batch])

                    r_bar = egrs[node].egr_model(s_batch).numpy()
                    Q_s0 = critics[node].predict(s_batch).numpy()
                    V_s0 = [Q_s0[i][a_batch[i][0]] for i in range(args['mini_batch_size']) ]
                    V_s1 = critics[node].predict(s2_batch).numpy().max(axis=1)

                    y, e_td, td = [], [], []
                    #e_td =[]
                    #td = []
                    for k in range(args['mini_batch_size']):
                        if t_batch[k]:
                            y.append(r_batch[k])
                            e_td.append(r_bar[k]- V_s0[k])
                            td.append(r_batch[k]- V_s0[k])
                        else:
                            y.append(r_batch[k] + critic.gamma * V_s1[k])
                            e_td.append(r_bar[k] + critic.gamma * V_s1[k] - V_s0[k])
                            td.append(r_batch[k] + critic.gamma * V_s1[k] - V_s0[k])
                    
                    target = [[Q_s0[i][a] if a != a_batch[i][0] else y[i] for a in range(env.n_actions)] for i in range(args['mini_batch_size'])]
                    #print('Q_s0', Q_s0.shape, Q_s0)
                    #print('target', np.shape(target), target)
                    #[Q_s0[i][a_batch[i][0]]=y[i] for i in range(args['mini_batch_size']) ]
                    critics[node].train(s_batch, a_batch_hot,  np.array(target))
                    actors[node].train(s_batch, a_batch_hot, np.array(td).reshape(args['mini_batch_size'],1))
                    egrs[node].train(s_batch, np.array(r_batch))


            
            #consensus update
            # for i in range(len(critics[0].network_params)):
            #     temp = 0
            #     for node in range(nodes):
            #         temp = temp + critics[node].network_params[i]
            #     for node in range(nodes):
            #         critics[node].network_params[i].assign(temp/env.n_agents) 
            
            for i in range(len(egrs[0].network_params)):
                temp = 0
                for node in range(nodes):
                    temp = temp + egrs[node].network_params[i]
                for node in range(nodes):
                    egrs[node].network_params[i].assign(temp/env.n_agents)
            j=j+1


            ep_reward = ep_reward + new_reward
            if done | (j==args['max_episode_len']):

                with writer.as_default():
                    tf.summary.scalar("Reward_1", ep_reward[0], step = t)
                    tf.summary.scalar("Reward_2", ep_reward[1], step = t)
                    tf.summary.scalar("Reward_3", ep_reward[2], step = t)
                    tf.summary.scalar("Reward_4", ep_reward[3], step = t)
                    tf.summary.scalar("Reward_5", ep_reward[4], step = t)
                    writer.flush()
                print('| Reward: {} | Episode: {} | last action: {}'.format(ep_reward, t, [actions[node][-1] for node in range(5)]))
                reward_result[t] = ep_reward.sum()

                path = {
                    "Observation":obs, 
                    "Action":actions,#np.concatenate(actions), 
                    "Reward":rewards#np.asarray(rewards)
                    }
                paths.append(path)
                break
    return [paths, reward_result] 
