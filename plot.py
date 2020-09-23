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
            _, _, rew = DATA['data'][0][epi][0][0]
            for i in range(5):
                r[i].append(rew[i].tolist())
        #print(np.array(r).shape)
        r_sum = np.array(r).sum(axis = 2)
        [r_agent[i].append(rew) for i, rew in enumerate(r_sum)]
    return r_agent


def net_compute_reward(paths, episodes):
    r_agent = [[] for _ in  range(5)]
    for mat_path in iter(paths):
        r = [[] for _ in range(5)]
        DATA = loadmat(mat_path)
        returns = []
        for epi in range(episodes):
            states, _, _ = DATA['data'][0][epi][0][0]
            r = [[dist(s1, s2) for s1,s2 in zip(s, des)] for s in states[0]]
            returns.append(np.sum(r, axis=0))
        returns = np.array(returns).T.tolist()
        [r_agent[i].append(rew) for i, rew in enumerate(returns)]
    return r_agent



def plot_agent_rewards(paths, episodes, savefig_filename, set_format = 'pdf', compute = True):
    if compute:
        data = net_compute_reward(paths = paths, episodes = episodes)
    else:
        data = net_reward(paths = paths, episodes = episodes)
    label = ['adversary', 'good', 'good', 'good', 'good']
    DataFrame = [pd.DataFrame(r) for r in iter(data)]
    fig, axes = plt.subplots(1,5,figsize = (20,4))

    for i, df in enumerate(DataFrame):
        #print(df.head())
        mean = df.mean(axis=0)
        
        std  = df.std(axis=0)
        #print(mean, std)
        steps = np.arange(mean.size)
        #print(steps,mean)
        # for run in range(df.shape[0]):
        #     axes[i].plot(steps, df.iloc[run,:], color='000000', alpha=0.1)
        axes[i].plot(steps, mean, color='b', alpha=1, label=label[i])
        axes[i].fill_between(steps, mean - std, mean + std,color='b', alpha=0.25)
        #axes[i].set_ylim(-6e6,0.1e6)
        axes[i].set_ylim(-5e3,0e3)
        axes[i].set_title('Agent-'+str(i), fontsize=15)
        axes[i].set_label('Label via method')
        axes[i].legend()
    if savefig_filename is not None:
        assert isinstance(savefig_filename, str), "filename for saving the figure must be a string"
        plt.savefig(savefig_filename, format = set_format)
    else:
        plt.show()
# ##%
# path = './Power-Converters/marl/results/matfiles/'
# files_1 = os.listdir(path + 'method_1/')
# files_2 = os.listdir(path + 'method_2/')

# #print(files)
# paths_1 = [path + 'method_1/' + file for file in files_1]
# paths_2 = [path + 'method_2/' + file for file in files_2]
# print(paths_1, files_2)
# print(paths_2)
# plot_agent_rewards(paths = paths_1 + paths_2, episodes = 100, savefig_filename=path+'true.pdf', set_format = 'pdf', compute = True)
# plot_agent_rewards(paths = paths_1 + paths_2, episodes = 100, savefig_filename=path+'compramized.pdf', set_format = 'pdf', compute = False)

# #plot_agent_rewards(paths_ddpg, episodes = 200)

# ##%
# path = './Power-Converters/marl/results/matfiles-crc/'

# files_1 = os.listdir(path + 'test-1/')
# files_2 = os.listdir(path + 'test-2/')

# #print(files)
# paths_1 = [path + 'test-1/' + file for file in files_1]
# paths_2 = [path + 'test-2/' + file for file in files_2]

# print(paths_1, files_2)
# print(paths_2)
# plot_agent_rewards(paths = paths_1 + paths_2, episodes = 500, savefig_filename=path+'true.pdf', set_format = 'pdf', compute = True)
# plot_agent_rewards(paths = paths_1 + paths_2, episodes = 500, savefig_filename=path+'compramized.pdf', set_format = 'pdf', compute = False)

# # #plot_agent_rewards(paths_ddpg, episodes = 200)


# #plot_agent_rewards(paths_ddpg, episodes = 200)

##%
path = './Power-Converters/marl/results/matfiles-crc/'

files_1 = os.listdir(path + 'test-3/')

#print(files)
paths_3 = [path + 'test-3/' + file for file in files_1]
print(paths_3)
plot_agent_rewards(paths = paths_3, episodes = 500, savefig_filename=path+'true.pdf', set_format = 'pdf', compute = True)
plot_agent_rewards(paths = paths_3, episodes = 500, savefig_filename=path+'compramized.pdf', set_format = 'pdf', compute = False)

# #plot_agent_rewards(paths_ddpg, episodes = 200)