import numpy as np
import tensorflow as tf
from tensorflow import keras

class Consensus_AC_agent():
    '''
    RESILIENT ACTOR-CRITIC AGENT
    This is an implementation of the consensus actor-critic algorithm by Zhang et al.(2018). This is a version
    with discounted rewards that was employed by Figura et al.(2020). The algorithm is a realization of temporal
    difference learning with one-step lookahead, also known as TD(0). It is an instance of decentralized learning,
    where each agent receives its own reward and observes the global state and action. The consensus AC agent seeks
    to maximize a team-average objective function. The agent employs neural networks to approximate the actor,
    critic, and team-average reward function. The agent performs stochastic updates of the actor, critic, and team
    reward through methods actor_update(), critic_update(), and TR_update(). The consensus updates are executed
    through method consensus_update(). The consensus updates involve simple averaging over the received parameter values.
    The consensus AC agent includes method get_action() to sample actions from the policy approximated by the actor network.
    ARGUMENTS: NN models for actor, critic, and team_reward
               slow learning rate (for the actor network)
               fast learning rate (for the critic and team reward networks)
               discount factor gamma
    '''
    def __init__(self,actor,critic,team_reward,gamma=0.95,ep_len=1):
        self.actor = actor
        self.critic = critic
        self.TR = team_reward
        self.gamma=gamma
        self.n_actions=self.actor.output_shape[1]

    def actor_update(self,states,new_states,team_actions,local_actions):
        '''
        Stochastic update of the actor network
        - performs a single batch update of the actor using the estimated team-average TD error
        ARGUMENTS: visited states, team actions, agent's actions
        RETURNS: training loss
        '''
        sa = np.concatenate((states,team_actions),axis=1)
        team_rewards = self.TR(sa).numpy()

        V = self.critic(states).numpy()
        nV = self.critic(new_states).numpy()
        global_TD_error=team_rewards+self.gamma*nV-V
        actor_loss = self.actor.train_on_batch(states,local_actions,sample_weight=global_TD_error)

        return actor_loss

    def critic_update(self,states,new_states,local_rewards):
        '''
        Stochastic update of the critic network
        - performs a single batch update of the critic network
        - evaluates local TD targets with a one-step lookahead
        - applies an MSE gradient with local TD targets as target values
        ARGUMENTS: visited consecutive states, local rewards
        RETURNS: updated critic parameters, training loss
        '''
        nV = self.critic(new_states).numpy()
        local_TD_targets=local_rewards+self.gamma*nV
        critic_loss = self.critic.train_on_batch(states,y=local_TD_targets)
        critic_params = [tf.identity(item) for item in self.critic.trainable_variables]

        return critic_params, critic_loss

    def TR_update(self,states,team_actions,local_rewards):
        '''
        Stochastic update of the team reward network
        - performs a single batch update of the team reward network
        - applies an MSE gradient with local rewards as target values
        ARGUMENTS: visited states, team actions, local rewards
        RETURNS: updated team reward parameters, training loss
        '''
        sa = np.concatenate((states,team_actions),axis=1)
        TR_loss = self.TR.train_on_batch(sa,y=local_rewards)
        TR_params = [tf.identity(item) for item in self.TR.trainable_variables]

        return TR_params, TR_loss

    def consensus_update(self,critic_innodes,TR_innodes):
        '''
        Consensus update of the critic and team reward parameters
        - computes simple average of parameters received from neighbors (including the agent)
        ARGUMENTS: critic and team reward network parameters from neighbors
        '''
        for i,layer in enumerate(self.critic.trainable_variables):
            temp = 0
            for agent in range(len(critic_innodes)):
                temp += critic_innodes[agent][i]
            layer.assign(tf.Variable(temp/len(critic_innodes)))

        for i,layer in enumerate(self.TR.trainable_variables):
            temp = 0
            for agent in range(len(TR_innodes)):
                temp += TR_innodes[agent][i]
            layer.assign(tf.Variable(temp/len(TR_innodes)))

    def get_action(self,state,from_policy=False,mu=0.1):
        '''
        Selects an action at the current state based on the probability distribution
        over actions under the current policy
        - set from_policy to True to sample from the policy
        - set from_policy to False to sample actions with uniform probability
        - set exploration parameter mu to [0,1] to control probability of choosing a random action
        '''
        random_action = np.random.choice(self.n_actions)
        if from_policy==True:
            state = np.array(state).reshape(1,-1)
            action_prob = self.actor.predict(state)
            action_from_policy = np.random.choice(self.n_actions, p = action_prob[0])
            self.action = np.random.choice([action_from_policy,random_action], p = [1-mu,mu])
        else:
            self.action = random_action

        return self.action
