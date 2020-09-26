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
des= np.array([0, 4, 14, 24, 32])
adv_des=np.array([0, 35, 35, 35, 35])
print("state_tranformation")
for (key1, key2) in iter(env.state_transformation):
    print((key1, key2),'<===>',env.state_transformation[(key1, key2)])
print('_'*10)
print("desired states")
for s in des:
    print(s,'<===>', env.i_state_transformation[s])
print("End states of all the agents")
print('_'*10)
for s in [0, 8, 17, 29, 35]:
    print(s,'<===>', env.i_state_transformation[s])
def net_compute_reward(paths, episodes):
    end_states = []# [[] for _ in range(len(paths))]
    #predicted_rewards_all = [[] for _ in  range(5)]
    for mat_path in iter(paths):
        DATA = loadmat(mat_path)
        returns = []
        epi_states = []
        for epi in range(episodes):
            states, _, _ ,_= DATA['data'][0][epi][0][0]
            epi_states.append(states[0][-1])
        end_states.append(epi_states)
    #fig, axes = plt.subplots(1,len(paths),figsize = (20,4))
    DataFrame = [pd.DataFrame(r) for r in iter(end_states)]
    fig, axes = plt.subplots(1,len(paths),figsize = (20,4))

    for i, df_1 in enumerate(DataFrame):
        #print(df.head())
        mean_1 = df_1.mean(axis=0)

        std_1  = df_1.std(axis=0)
        #print(mean, std)

        steps = np.arange(mean_1.size)

        #print(steps,mean)
        # for run in range(df_1.shape[0]):
        #     axes[i].plot(steps, df_1.iloc[run,:], color='000000', alpha=0.1)
        
        axes[i].plot(steps, mean_1, color='b', alpha=1)

        axes[i].fill_between(steps, mean_1 - std_1, mean_1 + std_1,color='b', alpha=0.25)
        #axes[i].set_ylim(-6e6,0.1e6)
        axes[i].set_title('Agent-'+str(i+1), fontsize=15)
        axes[i].set_label('Label via method')
        #axes[i].legend()
    plt.show()
    print(np.shape(end_states))

    return end_states

path = './Power-Converters/marl/results/matfiles/'
files_1 = os.listdir(path + 'Adversory/')
files_2 = os.listdir(path + 'Adversory_1/')
files_3 = os.listdir(path + 'No_Adversory/')
#print(files)
paths_1 = [path + 'Adversory/'  + file for file in files_1]
paths_2 = [path + 'Adversory_1/' + file for file in files_2]
paths_3 = [path + 'No_Adversory/' + file for file in files_3]

print(paths_1)
print(paths_2)
#net_compute_reward(paths = paths_1, episodes = 200)
#plot_agent_rewards(paths_1 = paths_1 + paths_2,  paths_2 = paths_3, episodes = 200, savefig_filename=path+'plot.pdf', set_format = 'pdf', compute = True)