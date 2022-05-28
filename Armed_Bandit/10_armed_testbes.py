'''
---------------------------------
10 ARM TESTBED IMPLEMENTATION
---------------------------------
This is an improved version of SahanaRamnath
implementation offering more vectorization
and explanation along the code.
note: Matplotlib required version is 3.5
'''


import matplotlib.pyplot as plt
import numpy as np

K = 10  # number of arms
N_STEPS = 1000  # each run consists of N_STEPS steps
N_BANDITS = 2000  # we'll average the results over 2000 independent bandits
EPSILON = [0, 0.01, 0.1]
E = len(EPSILON)

# Generate the true distribution of arms' rewards from Standard Normal
# distribution
true_q = np.random.randn(N_BANDITS, K)

# Initialize a matrix containing the optimal actions for each step
optimal_pull = np.argmax(true_q, 1)

# Initialize the subplots
fig, ax = plt.subplots(2)
colors = ['r', 'b', 'g']

for epsilon in EPSILON:

    # initialize Q matrix with a pull of all arms from each bandit to get
    # initial q-estimate to be able to choose first arm with highest reward.
    # Note that we will use Qi rather tham Q in the first time because we
    # need to start with zeroed Q values.
    Q = np.zeros((N_BANDITS, K))
    Qi = np.random.normal(true_q, 1)

    # Store the number of times each arm was pulled
    # all arms are initially pulled once, thus N initialed to one
    N = np.ones((N_BANDITS, K))

    # Initialize matrix R to store rewards of each step to use them later,
    # and store the results of first pull of all arms. Note that we will take
    # the average of all arms' rewards for each bandit (as if we pulled one arm)
    R = np.empty((N_BANDITS, N_STEPS))
    R[:, 0] = np.mean(Qi, 1)

    # Initialize a matrix to save a count of optimal actions taken
    optimal_count = np.zeros((N_BANDITS, N_STEPS))

    # Start pulling for the 2000 bandits using greedy method
    for step in range(1, N_STEPS):
        # Explore-exploit probabilistic decision according to epsilon value
        # Either exploit an arm with highest Q or explore an arm randomly
        r = np.random.random(N_BANDITS)
        arms_to_pull = np.where(r < epsilon, np.random.randint(K, size=N_BANDITS), np.argmax(Q, 1))

        # if the pull is the optimal one increment the optimal pulls count
        optimal_count[:, step] = optimal_count[:, step] + (arms_to_pull == optimal_pull).astype('int')

        # set the corresponding index
        index = np.arange(N_BANDITS), arms_to_pull

        # pull
        reward = np.random.normal(true_q[index], 1)

        # fill R matrix
        R[:, step] = reward

        # update the corresponding arm pulls count
        N[index] = N[index] + 1

        # update Q value of pulled arms to get better estimate with each step
        Q[index] = Q[index] + (reward - Q[index]) / N[index]

    # plot the average reward of the 2000 bandits over the 1000 steps
    ax[0].plot(np.mean(R, 0), colors[EPSILON.index(epsilon)], label="\u03B5-" + str(epsilon))
    ax[1].plot(np.mean(optimal_count * 100, 0), colors[EPSILON.index(epsilon)])

# Customize the plot
ax[0].set_xlabel('Steps')
ax[0].set_ylabel('Average Reward')

ax[1].set_xlabel('Steps')
ax[1].set_ylabel('% Optimal Action')
ax[1].set_yticks([0, 20, 40, 60, 80, 100], labels=['0 %', '20 %', '40 %', '60 %', '80 %', '100 %'])

fig.legend(loc=(0.824, 0.58))

plt.suptitle('Variation of Reward with Experience')
plt.show()
