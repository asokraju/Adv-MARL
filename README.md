# Adversarial - Multi-Agent Reinforcement Learning (Adv-MARL)

Our goal is to test training performance of cooperative MARL agents in the presence of adversaries. Specifically, we take under the scope the consensus actor-critic algorithm that was proposed in [[1]](#1) with discounted returns in the objective function. The cooperative MARL problem with an adversary in the network was studied in [[2]](#2) - the results showed that a single adversary can arbitrarily hurt the network performance. The published code aims to validate the theoretical results.

## Multi-Agent Gridworld: Cooperative Navigation
We consider a cooperative navigation task, where a group of agents operate in a grid world environment. The goal of each agent is to approach its desired position without colliding with other agents in the network. We design a grid-world of dimension (6 x 6) and consider a reward function that penalizes the agents for distance from the target and colliding with other agents. We implement the consensus AC algorithm to solve the cooperative navigation task.

We consider two scenarios.
1) The objective of the agents is to maximize the team-average expected returns.
2) One agent seeks to maximize its own expected returns and disregards the rest of the network.

<img src="https://github.com/asokraju/Adv-MARL/blob/batch_training/simulation_results/All_cooperative2/sim_results.png" width="2000" align="right">

<img src="https://github.com/asokraju/Adv-MARL/blob/batch_training/simulation_results/Adversary1/sim_results.png" width="2000" align="right">

The simulation results reveal that while the cooperative agents learn a near-optimal policy in the adversary-free scenario, their learning is hampered in the presence of adversary. It is important to note that the adversary easily achieves its own objective (bottom left figure).

Details of the second Scenario:
## Study of Adversarial Attacks on a Network of Agents with Consensus-based MARL Algorithm

1. In this study, we investigate the impact of adversarial attacks on a network of agents that employs consensus-based Multi-Agent Reinforcement Learning (MARL) algorithm.

2. Our results show that an adversarial agent can manipulate all other agents in the network to pursue the objective that it desires.

3. We consider a grid world with four agents, each of which can take one of five actions: moving up, down, left, right, or staying in place.

4. The objective of the agents is to reach their desired positions in the shortest path while avoiding collisions. The cost at each time step is the sum of the shortest distance from the current position of each agent to its desired position and the number of collisions.

5. In order to optimally avoid collisions, each agent needs to minimize the average cost of all agents. However, due to privacy concerns, agents may not be able to share their costs with others.

6. To overcome this issue, the agents try to learn the average cost function through a neural network, as follows:
  1. First, they update the weights by minimizing the error between their current cost and the neural network.
  2. Second, they share their parameters of the neural network with their neighbors.
  3. Finally, each agent updates the weights of the neural network to the average of its neighbors.

7. An adversarial agent, however, would skip the final step and move to its destination along the shortest path, causing other agents to move out of its way and potentially not reaching their desired positions.


## References
<a id="1">[1]</a> 
Zhang, K., Yang, Z., Liu, H., Zhang, T., Basar, T.
Fully decentralized multi-agent reinforcement learning with networked agents.
arXivpreprint arXiv:1802.08757, 2018.

<a id="2">[2]</a> 
Figura, M., Kosaraju, K. C., and Gupta, V.
Adversarial attacks in consensus-based multi-agent reinforcement learning.
arXiv preprint arXiv:2103.06967, 2021.



