# Adversarial - Multi-Agent Reinforcement Learning (Adv-MARL)

The aim is test and implement MARL algorithms in the presence adversarial agents.

- Currently modeled a Grid_World of multi agent systems using tools from Gym.

## Multi-Agent Gridworld
We consider a grid-world of dimension (6 x 6). We note that the tuples (0, 0) and (5, 5) correspond to the top-left and bottom-right corners of the grid, respectively. The agents are randomly initialized in the grid-world. The objective of each agent is to reach their desired position in a minimum number of steps while minimizing their collisions. We consider the instatanious reward of each agent is
r<sup>i</sup>(x<sup>i</sup>, y<sup>i</sup>, x<sup>i</sup><sub>des</sub>, y<sup>i</sup><sub>des</sub>, q<sup>i</sup>) = - |x<sup>i</sup> - x<sup>i</sup><sub>des</sub>| - |y<sup>i</sup> - y<sup>i</sup><sub>des</sub>| - q<sup>i</sup>

where  (x<sup>i</sup>, y<sup>i</sup>) denote their current position, (x<sup>i</sup><sub>des</sub>, y<sup>i</sup><sub>des</sub>) denote their desired position, and q<sup>i</sup> denote the current number of collisions.

When the adversary is not present, the objective of the agents is to maximize the sum of the cumulative rewards of all the agents. 


<img src="https://github.com/asokraju/Adv-MARL/blob/master/results/plot-1.png" width="1200">

<img src="https://github.com/asokraju/Adv-MARL/blob/master/results/team_reward-1.png" width="400" align="center">
