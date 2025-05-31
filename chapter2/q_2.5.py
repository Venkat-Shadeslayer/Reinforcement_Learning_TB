import numpy as np
import matplotlib.pyplot as plt

#Non-Stationary Bandit Environment

class NonStationaryBandit:
    def __init__(self, k=10, drift_std=0.01):          
        self.k = k                                    #k=number of arms
        self.q_star = np.zeros(k)                     #True action values
        self.drift_std = drift_std
    

    #q∗​(a)=E[Rt​∣At​=a]
    #This means: if you take an action a, the expected reward is q_star but the actual reward is a sample from the random probablity distribution with mean q_star[a] and std deviation 1.0
    def step(self, action):
        reward = np.random.normal(self.q_star[action], 1.0) #self.q_star is an array of size 10 representing the true expected reward values for each of the 10 arms
        return reward
    
    #Say self.q_star = [0.5, 0.1, -0.2, ..., 1.3] and say action = 2 array of size 10
    #then self.q_star[action] = -0.2 which is the same as self.q_star[2]


    def update_q_star(self):                                     #This is what is causing the non-stationarity. As the best move @t=100 may not be the best move @t=1000
        self.q_star += np.random.normal(0, self.drift_std, self.k)#Probablity distribution function generating the rewards changes over time

    def get_optimal_action(self):
        return np.argmax(self.q_star)

#Agent Base Class

class Agent:
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k)
        self.action_counts = np.zeros(k)#An array keeping track of how many times eaach action a(0-9) was taken

    def  select_action(self):  
        #this is the epsilon-greedy action selection method
        #With probability epsilon, select a random action
        #With probability 1-epsilon, select the action with the highest estimated value
        #self.epsilon is the probability of selecting a random action
        #self.Q is an array of size k representing the estimated action values
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)#explore
        else:
            return np.argmax(self.Q)#Exploit
        
        #Note to self:Fun fact,think about what happen if we set the explore condition to liew between any number to any number as long as a width of e is maintained(Not necessarily 0 to e), but (any number) to (any number+e)

#Sample average agent
class SampleAverageAgent(Agent):
    def update(self,action,reward):
        self.action_counts[action] += 1
        #Update the estimated action value using sample average method
        self.Q[action] += (1 / self.action_counts[action]) * (reward - self.Q[action])

#Constant step-size agent

class ConstantStepSizeAgent(Agent):
    def __init__(self, k=10, epsilon=0.1, alpha=0.1):
        super().__init__(k, epsilon)
        self.alpha = alpha

    def update(self, action, reward):
        self.Q[action] += self.alpha * (reward - self.Q[action])
#Q: Why does the constanstepsize agent have a k?

#Run the experiments


def run_experiment(runs=2000, steps=10000):
    avg_rewards_sample = np.zeros(steps)
    avg_rewards_const = np.zeros(steps)

    for run in range(runs):
        bandit = NonStationaryBandit()
        agent_sample = SampleAverageAgent()
        agent_const = ConstantStepSizeAgent()

        for t in range(steps):
            # Sample-average agent
            action_s = agent_sample.select_action()
            reward_s = bandit.step(action_s)
            agent_sample.update(action_s, reward_s)
            avg_rewards_sample[t] += reward_s

            # Constant-alpha agent
            action_c = agent_const.select_action()
            reward_c = bandit.step(action_c)
            agent_const.update(action_c, reward_c)
            avg_rewards_const[t] += reward_c

            # Drift true action values
            bandit.update_q_star()

    # Average over all runs
    avg_rewards_sample /= runs
    avg_rewards_const /= runs

    return avg_rewards_sample, avg_rewards_const

#Plot the results

def plot_results(avg_rewards_sample, avg_rewards_const):
    plt.figure(figsize=(10, 6))
    plt.plot(avg_rewards_sample, label='Sample-Average')
    plt.plot(avg_rewards_const, label='Constant Step-Size (α=0.1)')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Nonstationary Bandit: Sample-Average vs Constant-α')
    plt.legend()
    plt.grid(True)
    plt.show()

#Run everything
if __name__ == "__main__":
    rewards_sample, rewards_const = run_experiment()
    plot_results(rewards_sample, rewards_const)


"""Conclusion:

Learn more
In reinforcement learning scenarios, especially when dealing with non-stationary environments, the constant step-size method generally outperforms the sample average method. This is because constant step-size methods can adapt more readily to changes in rewards over time, while sample averaging can get "stuck" on an outdated estimate. 
1. Adaptability to Changes:

Constant Step-Size:
    . This method allows for a continuous stream of learning, where more recent data is given more weight. The step size parameter controls how quickly the agent adjusts its estimates, allowing it to react to changes in the environment.

Sample Averaging:
.This method averages rewards over a set of trials, but it doesn't inherently prioritize recent data. As a result, it may take longer to adapt to shifts in the environment and may not be as efficient as the constant step-size approach when dealing with non-stationary problems. 

2. Bias and Exploration:

Constant Step-Size:
    . While the initial estimates can introduce a bias, the bias typically diminishes as the agent gathers more experience. This bias can even be leveraged to encourage exploration, as optimistic initial values can make the agent more likely to try different actions.

Sample Averaging:
.       The bias from initial estimates is only removed after all actions have been explored at least once. This can make it less effective in scenarios where exploration is crucial or when the environment changes rapidly
"""



    