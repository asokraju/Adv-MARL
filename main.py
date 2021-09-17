import numpy as np
import gym
from gym import spaces
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential
import argparse
from environments.grid_world import Grid_World
from agents.consensus_AC import Consensus_AC_agent
import training.train_CAC_agents as training

'''
Consensus actor-critic algorithm for cooperative navigation
- This is a main file, where the user selects learning hyperparameters, environment parameters,
  and neural network architecture for the actor, critic, and team reward estimates.
- The script triggers a training process whose results are passed to folder Simulation_results.
'''

if __name__ == '__main__':

    '''USER-DEFINED PARAMETERS'''
    parser = argparse.ArgumentParser(description='Provide parameters for training consensus AC agents')
    parser.add_argument('--n_agents',help='number of agents',type=int,default=4)
    parser.add_argument('--n_actions',help='size of action space of each agent',type=int,default=5)
    parser.add_argument('--n_states',help='state dimension of each agent',type=int,default=2)
    parser.add_argument('--n_episodes', help='number of episodes', type=int, default=10)
    parser.add_argument('--max_ep_len', help='number of steps per episode', type=int, default=20)
    parser.add_argument('--slow_lr', help='learning rate for actor updates',type=float, default=0.002)
    parser.add_argument('--fast_lr', help='learning rate for critic and team reward updates',type=float, default=0.005)
    parser.add_argument('--gamma', help='discount factor', type=float, default=0.9)
    parser.add_argument('--eps', help='exploration probability',type=float,default=0.05)
    parser.add_argument('--consensus_freq',help='frequency of consensus updates wrt to stochastic updates of critic and team reward',type=int,default=1)
    parser.add_argument('--n_ep_fixed',help='number of episodes under a fixed policy',type=int,default=10)
    parser.add_argument('--n_epochs',help='number of consecutive gradient steps in the critic and team reward updates',type=int,default=10)
    parser.add_argument('--desired_state',help='desired state of the agents',type=int,default=np.array([[3,2],[3,4],[2,3],[1,4]]))
    parser.add_argument('--in_nodes',help='specify a list of neighbors that transmit values to each agent (include the index of the agent as the first element)',type=int,default=[[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3]])
    parser.add_argument('--randomize_state',help='Set to True if the agents start at random initial state in every episode',type=bool,default=True)
    parser.add_argument('--initial_state',help='initial state of the agents',type=int,default=np.random.randint(0,6,size=(4,2)))
    parser.add_argument('--scaling', help='Normalize states for training?', type = bool, default=True)
    parser.add_argument('--summary_dir',help='Create a directory to save simulation results', default='./sim_results/')
    parser.add_argument('--random_seed',help='Set random seed for the random number generator',type=int,default=None)

    args = vars(parser.parse_args())
    np.random.seed(args['random_seed'])
    tf.random.set_seed(args['random_seed'])
    #-------------------------------------------------------------------------------
    '''NEURAL NETWORK ARCHITECTURE'''

    agents = []

    for node in range(args['n_agents']):
        actor = keras.Sequential([
            keras.layers.Dense(20, activation=keras.layers.LeakyReLU(alpha=0.3),input_shape=(args['n_agents']*args['n_states'],)),
            keras.layers.Dense(20, activation=keras.layers.LeakyReLU(alpha=0.3)),
            keras.layers.Dense(20, activation=keras.layers.LeakyReLU(alpha=0.3)),
            keras.layers.Dense(args['n_actions'], activation='softmax')
                                ])

        team_reward = keras.Sequential([
            keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.3),input_shape=(args['n_agents']*args['n_states']+args['n_agents'],)),
            keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.3)),
            keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.3)),
            keras.layers.Dense(1)
                                    ])

        critic = keras.Sequential([
            keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.3),input_shape=(args['n_agents']*args['n_states'],)),
            keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.3)),
            keras.layers.Dense(30, activation=keras.layers.LeakyReLU(alpha=0.3)),
            keras.layers.Dense(1)
                                    ])

        actor.compile(optimizer=keras.optimizers.Adam(learning_rate=args['slow_lr']),loss=keras.losses.SparseCategoricalCrossentropy())
        team_reward.compile(optimizer=keras.optimizers.Adam(learning_rate=args['fast_lr']),loss=keras.losses.MeanSquaredError())
        critic.compile(optimizer=keras.optimizers.Adam(learning_rate=args['fast_lr']),loss=keras.losses.MeanSquaredError())

        agents.append(Consensus_AC_agent(actor,
                                        critic,
                                        team_reward,
                                        gamma=args['gamma'],
                                        )
                      )
#------------------------------------------------------------------------------

if __name__ == '__main__':
    print(args)
    '''TRAIN AGENTS'''
    env = Grid_World(nrow=6,
                     ncol=6,
                     n_agents=args['n_agents'],
                     desired_state=args['desired_state'],
                     initial_state=args['initial_state'],
                     randomize_state=args['randomize_state'],
                     scaling=args['scaling']
                     )
    trained_agents,learning_stats = training.train_batch(env,agents,args)
    #--------------------------------------------------------------------------------
    '''PLOT RESULTS'''
    training.plot_results(learning_stats,args)
