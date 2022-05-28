# Reinforcement-Learning
various implementation of RL problems

This implementation is for the section 2.3 of Richard S. Sutton book on reinforcement learning.
An armed testbed is an example of armed bandit problem where for each bandit random numbers `q(a)`
simulating actions/arms values are generated from a standard normal distribution. When a learning
is applied, a selected action `a` will result in a reward sampled from a normal distribution of mean
`q(a)` and variance 1. A large number (2000) independent bandits are simulated and a learning method
is applied over 1000 time steps each. The results are averaged over these bandits to obtain a measure
of the learning algorithm's average behavior. The applied learning algorithm is ε-greedy and
the action values are sampled according to sample-average method.

The implementation is an improvement on SahanaRamnath implementation (link below), offering more
vectorization and explanation.

SahanaRamnath implementation:
https://github.com/SahanaRamnath/MultiArmedBandit_RL/blob/master/Epsilon_Greedy_Method/eps-greedy.py
