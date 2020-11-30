# Adversarial - Multi-Agent Reinforcement Learning (Adv-MARL)

The aim is test and implement MARL algorithms in the presence adversarial agents.

- Currently modeled a Grid_World of multi agent systems using tools from Gym.

## Multi-Agent Gridworld
We consider a grid-world of dimension (6 x 6). We note that the tuples (0, 0) and (5, 5) correspond to the top-left and bottom-right corners of the grid, respectively. The agents are randomly initialized in the grid-world. The objective of each agent is to reach their desired position in a minimum number of steps while minimizing their collisions. We consider the instatanious reward of each agent is
r<sup>i</sup>(x<sup>i</sup>, y<sup>i</sup>, x<sup>i</sup><sub>des</sub>, y<sup>i</sup><sub>des</sub>, q<sup>i</sup>) = - |x<sup>i</sup> - x<sup>i</sup><sub>des</sub>| - |y<sup>i</sup> - y<sup>i</sup><sub>des</sub>| - q<sup>i</sup>

where  (x<sup>i</sup>, y<sup>i</sup>) denote their current position, (x<sup>i</sup><sub>des</sub>, y<sup>i</sup><sub>des</sub>) denote their desired position, and q<sup>i</sup> denote the current number of collisions.

Ideally, when the adversary is not present, the objective of the agents is to maximize the sum of the cumulative rewards of all the agents. This can be achieved by following Algorithm 2 in [[1]](#1). We consider agent 1 as a malicious or adversary, which aims to maximize its own cumulative reward but not the cumulative sum of rewards of all agents. The estimated reward function converged in both scenarios but the convergence rate was slower in the presence of the adversary. The accumulated rewards per episode of each agent are depicted in figure below
<img src="https://github.com/asokraju/Adv-MARL/blob/master/results/plot-1.png" width="1200">


We can see that all agents learn a near-optimal policy when there is no adversary in the network. The adversary indeed has a negative impact on the network, i.e., it learns a near-optimal policy but the remaining agents in the network perform poorly compared to the adversary-free scenario. Final states of a simulation of the grid-world after training for 200 episodes are shown in figure below
<img src="https://github.com/asokraju/Adv-MARL/blob/master/results/team_reward-1.png" width="400" align="center">

## References
<a id="1">[1]</a> 
Zhang, Z. Yang, H. Liu, T. Zhang, and T. Basar
Fully decentral-ized multi-agent reinforcement learning with networked agents.
arXivpreprint arXiv:1802.08757, 2018.



