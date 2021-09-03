import numpy as np
import gym
from gym import spaces
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, layers
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 40})

'''
This file contains a function for training consensus AC agents in a gym environment and for plotting training stats.
'''

def train_batch(env,agents,args):
    '''
    FUNCTION train_batch()
    The agents are trained in batches using sampled actions from the actor network. In this realization, the agent accumulates local
    rewards from multiple episodes. At the end of the sequence of episodes, the agent updates its critic and team reward parameters
    using stochastic approximation and consensus updates over multiple epochs. The updated parameters are used in the evaluation
    of the actor gradient that is applied only once.
    ARGUMENTS: MARL environment
               list of consensus AC agents
               user-defined parameters for the simulation
    RETURNS: trained agents
             training statistics
    '''
    paths = []
    n_agents, n_states, n_actions = env.n_agents, args['n_states'], env.n_actions
    gamma,eps = args['gamma'],args['eps']
    in_nodes = args['in_nodes']
    max_ep_len,n_episodes,n_ep_fixed = args['max_ep_len'],args['n_episodes'],args['n_ep_fixed']
    n_epochs = args['n_epochs']
    states = np.zeros((n_ep_fixed,max_ep_len+1,n_agents,n_states))
    actions = np.zeros((n_ep_fixed,max_ep_len,n_agents),dtype=int)
    rewards = np.zeros((n_ep_fixed,max_ep_len,n_agents))
    estimated_ep_rewards=np.zeros(n_agents)
    estimated_ep_returns=np.zeros(n_agents)

    writer = tf.summary.create_file_writer(logdir = args['summary_dir'])

    for t in range(n_episodes):
        #-----------------------------------------------------------------------
        '''BEGINNING OF TRAINING EPISODE'''
        j,ep_rewards,ep_returns=0,0,0
        actor_loss,critic_loss,TR_loss=np.zeros(n_agents),np.zeros(n_agents),np.zeros(n_agents)
        i = t % n_ep_fixed
        env.reset()
        states[i,j], rewards[i,j], done, _ = env.get_data()
        for node in range(n_agents):
            estimated_ep_returns[node]=agents[node].critic(states[i,j].reshape(1,-1))

        while j < max_ep_len:
            for node in range(n_agents):
                  actions[i,j,node] = agents[node].get_action(states[i,j].reshape(1,-1),from_policy=True,mu=eps)
            env.step(actions[i,j])
            states[i,j+1], rewards[i,j], done, _ = env.get_data()
            ep_rewards += rewards[i,j]
            ep_returns += rewards[i,j]*(gamma**j)
            j=j+1
        '''END OF TRAINING EPISODE'''
        #-----------------------------------------------------------------------
        if i == n_ep_fixed-1:
            s = states[:,:-1].reshape(n_ep_fixed*max_ep_len,-1)
            ns = states[:,1:].reshape(n_ep_fixed*max_ep_len,-1)
            local_r = rewards.reshape(n_ep_fixed*max_ep_len,n_agents,1)
            local_a = actions.reshape(n_ep_fixed*max_ep_len,n_agents,1)
            team_a = actions.reshape(n_ep_fixed*max_ep_len,n_agents)
            n,k = 1,1
            while n <= n_epochs:
                '''BATCH STOCHASTIC UPDATES OF CRITIC AND TEAM REWARD'''
                critic_params,TR_params=[],[]
                for node in range(n_agents):
                    x, l1 = agents[node].TR_update(s,team_a,local_r[:,node])
                    y, l2 = agents[node].critic_update(s,ns,local_r[:,node])
                    TR_params.append(x)
                    critic_params.append(y)
                    if n == n_epochs:
                        TR_loss[node] += l1
                        critic_loss[node] += l2
                '''CONSENSUS UPDATES OF CRITIC AND TEAM REWARD PARAMETERS'''
                if k == args['consensus_freq']:
                    for node in range(n_agents):
                        critic_params_innodes=[critic_params[i] for i in in_nodes[node]]
                        TR_params_innodes=[TR_params[i] for i in in_nodes[node]]
                        agents[node].consensus_update(critic_params_innodes,TR_params_innodes)
                    k = 1
                else:
                    k += 1
                n = n+1
            '''BATCH ACTOR UPDATES'''
            for node in range(n_agents):
                actor_loss[node] += agents[node].actor_update(s,ns,team_a,local_a[:,node])
        #-----------------------------------------------------------------------
        '''SUMMARY OF TRAINING EPISODE'''
        critic_mean_loss=np.mean(critic_loss)
        TR_mean_loss=np.mean(TR_loss)
        actor_mean_loss=np.mean(actor_loss)

        with writer.as_default():
            tf.summary.scalar("estimated episode team-average returns", np.mean(estimated_ep_returns),step = t)
            tf.summary.scalar("true episode team-average returns",np.mean(ep_returns), step = t)
            tf.summary.scalar("true episode team-average rewards",np.mean(ep_rewards), step = t)
            writer.flush()

        print('| Episode: {} | Est. returns: {} | Returns: {} | Rewards: {} | Average critic loss: {} | Average TR loss: {} | Average actor loss: {} | Target reached: {} '.format(t,np.mean(estimated_ep_returns),np.mean(ep_returns),np.mean(ep_rewards),critic_mean_loss,TR_mean_loss,actor_mean_loss,done))

        path = {
                "Episode_rewards":np.array(ep_rewards),
                "True_team_returns":np.mean(ep_returns),
                "Estimated_team_returns":np.mean(estimated_ep_returns)
               }
        paths.append(path)

    return agents,paths

def plot_results(sim_data,args):
    '''
    Plots the simulation results - individual accumulated episode rewards and team-average returns.
    '''
    ep_rewards = np.zeros((args['n_episodes'],args['n_agents']))
    estimated_ep_rewards = np.zeros((args['n_episodes'],args['n_agents']))
    ep_returns = np.zeros(args['n_episodes'])
    estimated_ep_returns = np.zeros(args['n_episodes'])
    for i,item in enumerate(sim_data):
        ep_rewards[i] = item["Episode_rewards"]
        ep_returns[i] = item["True_team_returns"]
        estimated_ep_returns[i] = item["Estimated_team_returns"]
    t = np.arange(len(sim_data))
    fig,ax = plt.subplots(1,args['n_agents']+1,sharey=False,figsize=(100,15))
    for i in range(args['n_agents']):
        ax[i].set_xlabel("Episode")
        ax[i].set_ylabel("Rewards")
        ax[i].plot(t,ep_rewards[:,i],label='True')
        ax[i].legend()
    ax[-1].set_xlabel("Episode")
    ax[-1].set_ylabel("Team-average returns")
    ax[-1].plot(t,ep_returns,label='True')
    ax[-1].plot(t,estimated_ep_returns,label='Est')
    ax[-1].legend()
    plt.savefig(args['summary_dir']+'sim_results.png')
