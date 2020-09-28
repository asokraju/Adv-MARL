##%
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import seaborn as sns

import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
import os

from environments.grid_world import Grid_World
sns.set_context("paper")
sns.set_style("whitegrid")
sns.set()
##%
import matplotlib.pylab as pylab
params = {#'legend.fontsize': 'x-large',
          #'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

env = Grid_World(6,6,5)
des=np.array([0, 4, 14, 24, 32])
adv_des=np.array([0, 35, 35, 35, 35])

def dist(s1, s2):
    x1, y1 = env.i_state_transformation[s1]
    x2, y2 = env.i_state_transformation[s2]
    return -abs(x1-x2)-abs(y1-y2)

def net_reward(paths, episodes):
    r_agent = [[] for _ in  range(5)]
    episodes_ = []
    for mat_path in iter(paths):
        r = [[] for _ in range(5)]
        DATA = loadmat(mat_path)
        #episodes_.append(np.shape(DATA['data'][0])[0])
        #episodes = min(episodes_)
        for epi in range(episodes):
            _, _, rew, _ = DATA['data'][0][epi][0][0]
            for i in range(5):
                r[i].append(rew[i].tolist())
        #print(np.array(r).shape)
        r_sum = np.array(r).sum(axis = 2)
        [r_agent[i].append(rew) for i, rew in enumerate(r_sum)]
    return r_agent

def net_reward_2(paths, episodes):
    r_agent = [[] for _ in  range(5)]
    episodes_ = []
    for mat_path in iter(paths):
        r = [[] for _ in range(5)]
        DATA = loadmat(mat_path)
        #episodes_.append(np.shape(DATA['data'][0])[0])
        #episodes = min(episodes_)
        for epi in range(episodes):
            _, _, rew, _, _, _ = DATA['data'][0][epi][0][0]
            for i in range(5):
                r[i].append(rew[i].tolist())
        #print(np.array(r).shape)
        r_sum = np.array(r).sum(axis = 2)
        [r_agent[i].append(rew) for i, rew in enumerate(r_sum)]
    return r_agent

def net_compute_reward(paths, episodes):
    des=np.array([0, 4, 14, 24, 32])
    r_agent = [[] for _ in  range(5)]
    #predicted_rewards_all = [[] for _ in  range(5)]
    for mat_path in iter(paths):
        r = [[] for _ in range(5)]
        #predicted_returns = [[] for _ in range(5)]
        DATA = loadmat(mat_path)
        returns = []
        for epi in range(episodes):
            states, _, _ ,_= DATA['data'][0][epi][0][0]
            #print(np.shape(predicted_rewards))
            #predicted_returns.append(np.array(predicted_rewards).reshape((5,1000)).sum(axis=1).tolist())
            r = [[dist(s1, s2) for s1,s2 in zip(s, des)] for s in states[0]]
            returns.append(np.sum(r, axis=0))
        returns = np.array(returns).T.tolist()
        #print(np.shape(predicted_returns), np.shape(returns))
        [r_agent[i].append(rew) for i, rew in enumerate(returns)]
        #[predicted_rewards_all[i].append(rew) for i, rew in enumerate(returns)]
        #print(np.shape(r_agent))
    return r_agent

def net_compute_reward_2(paths, episodes):
    des=np.array([0,5 , 30, 35, 32])#np.array([0, 4, 14, 24, 32])
    r_agent = [[] for _ in  range(5)]
    #predicted_rewards_all = [[] for _ in  range(5)]
    for mat_path in iter(paths):
        r = [[] for _ in range(5)]
        #predicted_returns = [[] for _ in range(5)]
        DATA = loadmat(mat_path)
        returns = []
        for epi in range(episodes):
            states, _, _ , _, _, _= DATA['data'][0][epi][0][0]
            #print(np.shape(predicted_rewards))
            #predicted_returns.append(np.array(predicted_rewards).reshape((5,1000)).sum(axis=1).tolist())
            r = [[dist(s1, s2) for s1,s2 in zip(s, des)] for s in states[0]]
            returns.append(np.sum(r, axis=0))
        returns = np.array(returns).T.tolist()
        #print(np.shape(predicted_returns), np.shape(returns))
        [r_agent[i].append(rew) for i, rew in enumerate(returns)]
        #[predicted_rewards_all[i].append(rew) for i, rew in enumerate(returns)]
        #print(np.shape(r_agent))
    return r_agent


def net_predicted_returns(paths, episodes = 200):
    exp_returns = []
    for mat_path in iter(paths):
        DATA = loadmat(mat_path)
        epi_returns = []
        for epi in range(episodes):
            _, _, _, computed_rewards = DATA['data'][0][epi][0][0]
            epi_returns.append(np.mean(computed_rewards, axis=0).sum())
        exp_returns.append(epi_returns)
    return exp_returns

def net_predicted_returns_2(paths, episodes = 200):
    exp_returns = []
    for mat_path in iter(paths):
        DATA = loadmat(mat_path)
        epi_returns = []
        for epi in range(episodes):
            _, _, _, computed_rewards, _, _ = DATA['data'][0][epi][0][0]
            epi_returns.append(np.mean(computed_rewards, axis=0).sum())
        exp_returns.append(epi_returns)
    return exp_returns


def plot_team_rewards(paths_1, paths_2, episodes, savefig_filename, set_format = 'pdf', compute = True):
    if compute:
        data_1 = net_compute_reward(paths = paths_1, episodes = episodes)
        data_2 = net_compute_reward_2(paths = paths_2, episodes = episodes)
        pred_rew_1 = net_predicted_returns(paths_1 , episodes = 200) 
        pred_rew_2 = net_predicted_returns_2(paths_2, episodes = 200) 
    else:
        data_1 = net_reward(paths = paths_1, episodes = episodes)
        data_2 = net_reward_2(paths = paths_2, episodes = episodes)
    team_reward_1 = np.mean(data_1, axis=0)#.mean(axis=0)
    team_reward_2 = np.mean(data_2, axis=0)#.mean(axis=0)

    team_reward_1 = np.array(team_reward_1).T[2:,:].T
    team_reward_2 = np.array(team_reward_2).T[2:,:].T
    pred_rew_1 = np.array(pred_rew_1).T[2:,:].T 
    pred_rew_1 = pred_rew_1.tolist()
    pred_rew_2 = np.array(pred_rew_2).T[2:,:].T.tolist()
    #print('pred_rew_1',np.shape(pred_rew_1))
    

    label = ['ad', 'ad-free']
    #DataFrame_1 = [pd.DataFrame(r) for r in iter(team_reward_1)]
    #DataFrame_2 = [pd.DataFrame(r) for r in iter(team_reward_2)]
    
    fig, axes = plt.subplots(1,1,figsize = (5,4))
    steps = np.arange(np.shape(team_reward_1)[1])

    mean_1 = team_reward_1.mean(axis=0)
    mean_2 = team_reward_2.mean(axis=0)
    pred_mean_1 = np.mean(pred_rew_1, axis=0)
    pred_mean_2 = np.mean(pred_rew_2, axis=0)

    std_1 = team_reward_1.std(axis=0)
    std_2 = team_reward_2.std(axis=0)
    
    pred_std_1 = np.std(pred_rew_1, axis=0)
    pred_std_2 = np.std(pred_rew_2, axis=0)

    axes.plot(steps, mean_1, 'r', alpha=1, label='true rewards (ad)')
    axes.plot(steps, mean_2, 'b', alpha=1, label='true rewards (ad-free)')

    axes.fill_between(steps, mean_1 - 0.5*std_1, mean_1 + 0.5*std_1, color='r', alpha=0.25)
    axes.fill_between(steps, mean_2 - 0.5*std_2, mean_2 + 0.5*std_2, color='b', alpha=0.5)

    axes.plot(steps, pred_mean_1, 'orange', alpha=1, label='Estimated rewards (ad)')
    axes.plot(steps, pred_mean_2, 'purple', alpha=0.7, label='Estimated rewards (ad-free)')

    axes.fill_between(steps, pred_mean_1 - 0.5*pred_std_1, pred_mean_1 + 0.5*pred_std_1,color='orange', alpha=0.25)
    axes.fill_between(steps, pred_mean_2 - 0.5*pred_std_2, pred_mean_2 + 0.5*pred_std_2,color='purple', alpha=0.15)
    if compute:
        axes.set_ylim(-5e3,0e3)
        #axes[0].set_ylim(-7e3,0e3)
    #axes.set_title('True returns vs Predicted returns ', fontsize=15)
    axes.set_label('Label via method')
    axes.legend()
    axes.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    if savefig_filename is not None:
        assert isinstance(savefig_filename, str), "filename for saving the figure must be a string"
        plt.savefig(savefig_filename, format = set_format)
    else:
        plt.show()
##%

def plot_agent_rewards(paths_1, paths_2, episodes, savefig_filename, set_format = 'pdf', compute = True):
    if compute:
        data_1 = net_compute_reward(paths = paths_1, episodes = episodes)
        data_2 = net_compute_reward_2(paths = paths_2, episodes = episodes)
    else:
        data_1 = net_reward(paths = paths_1, episodes = episodes)
        data_2 = net_reward_2(paths = paths_2, episodes = episodes)
    #team_reward_1 = np.mean(data_1, axis=0).mean(axis=0)
    #team_reward_2 = np.mean(data_2, axis=0).mean(axis=0)
    #print(np.shape(data_1))

    label = ['ad', 'ad-free']
    DataFrame_1 = [pd.DataFrame(r) for r in iter(data_1)]
    DataFrame_2 = [pd.DataFrame(r) for r in iter(data_2)]
    fig, axes = plt.subplots(1,5,figsize = (25,4))

    for i, df_1 in enumerate(DataFrame_1):
        #print(df.head())
        df_2 = DataFrame_2[i]
        mean_1 = df_1.mean(axis=0)
        mean_2 = df_2.mean(axis=0)

        std_1  = df_1.std(axis=0)
        std_2  = df_2.std(axis=0)
        #print(mean, std)

        steps = np.arange(mean_1.size)

        #print(steps,mean)
        # for run in range(df_1.shape[0]):
        #     axes[i].plot(steps, df_1.iloc[run,:], color='000000', alpha=0.15)
        # for run in range(df_2.shape[0]):
        #     axes[i].plot(steps, df_2.iloc[run,:], color='chocolate', alpha=0.15)

        axes[i].plot(steps, mean_1, color='r', alpha=1, label=label[0])
        axes[i].plot(steps, mean_2, color='b', alpha=1, label=label[1])
        #axes[i].plot(steps, team_reward_1, 'b--', alpha=1, label=label[1])
        #axes[i].plot(steps, team_reward_2, 'r--', alpha=1, label=label[1])
        axes[i].fill_between(steps, mean_1 - 0.5*std_1, mean_1 + 0.5*std_1,color='r', alpha=0.25)
        axes[i].fill_between(steps, mean_2 - 0.5*std_2, mean_2 + 0.5*std_2,color='b', alpha=0.25)
        #axes[i].set_ylim(-6e6,0.1e6)
        if compute:
            axes[i].set_ylim(-5e3,0e3)
        axes[i].set_title('Agent-'+str(i+1), fontsize=17)
        axes[i].set_label('Label via method')
        axes[i].legend(fontsize=20)
        axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axes[i].tick_params(axis='both', which='major', labelsize=20)
        axes[i].tick_params(axis='both', which='minor', labelsize=20)
    if savefig_filename is not None:
        assert isinstance(savefig_filename, str), "filename for saving the figure must be a string"
        plt.savefig(savefig_filename, format = set_format)
        #plt.savefig('savefig_filename', format='eps')

    else:
        plt.show()
##%

path = './Power-Converters/marl/results/matfiles/'
files_1 = os.listdir(path + 'Adversory/')
files_2 = os.listdir(path + 'Adversory_1/')
files_3 = os.listdir(path + 'No_Adversory/')
files_4 = os.listdir(path + 'No_Adversory_new/')

#print(files)
paths_1 = [path + 'Adversory/'  + file for file in files_1]
paths_2 = [path + 'Adversory_1/' + file for file in files_2]
paths_3 = [path + 'No_Adversory/' + file for file in files_3]
paths_4 = [path + 'No_Adversory_new/' + file for file in files_4]

print(paths_1)
print(paths_2)

# rrr_1 = net_predicted_returns(paths_1 + paths_2, episodes = 200)
# rrr_2 = net_predicted_returns_2(paths_4, episodes = 200)
# print(np.shape(rrr_1))
#print(np.mean(rrr_2, axis=0))
#plt.savefig('destination_path.eps', format='eps')

plot_agent_rewards(paths_1 = paths_1 + paths_2,  paths_2 = paths_4, episodes = 200, savefig_filename=path+'plot.pdf', set_format = 'pdf', compute = True)
#plot_team_rewards(paths_1 + paths_2, paths_4, episodes = 200, savefig_filename=path+'team_reward.pdf', set_format = 'pdf', compute = True)
#plot_agent_rewards(paths_ddpg, episodes = 200)

#################################################################