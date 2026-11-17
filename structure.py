import numpy as np
from scipy import stats
np.random.seed(42)


class slot_machine:
    def __init__(self, prob):
        self.prob = prob
        self.expected_value = prob.expect(lambda x: x, lb=0)    # E[max(0, X)]
        
    def pull(self):
        return max(0, round(self.prob.rvs(), 2))

    def reset(self, prob):
        self.prob = prob

class env:
    def __init__(self, tab_prob):
        self.slot_machines = [slot_machine(prob) for prob in tab_prob]

    def full_reset(self, tab_prob):
        for sm, p in zip(self.slot_machines, tab_prob): sm.reset(p)

    def one_reset(self, prob, index):
        self.slot_machines[index].reset(prob)

    def play(self, index):
        return self.slot_machines[index].pull()
    
class agent:
    def __init__(self, env):
        self.env = env
        self.total_reward = 0
        self.current_step = 0
        self.history = []

    def act(self, action):
        reward = self.env.play(action)
        self.total_reward += reward
        self.current_step += 1
        self.history.append((action, reward))
        return reward

    def get_history(self):
        return self.history
    
class epsilon_greedy_agent(agent):
    def __init__(self, env, epsilon):
        super().__init__(env)
        self.epsilon = epsilon
        self.q_values = np.zeros(len(env.slot_machines))
        self.action_counts = np.zeros(len(env.slot_machines))

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.env.slot_machines))
        else:
            return np.argmax(self.q_values)

    def update(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]

class ucb_agent(agent):
    def __init__(self, env, c):
        super().__init__(env)
        self.c = c
        self.action_counts = np.zeros(len(env.slot_machines))
        self.q_values = np.zeros(len(env.slot_machines))

    def act(self):
        ucb_values = self.q_values + self.c * np.sqrt(np.log(self.current_step + 1) / (self.action_counts + 1e-5))
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.action_counts[action] += 1
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]

class thompson_sampling_agent(agent):
    def __init__(self, env):
        super().__init__(env)
        self.alpha = np.ones(len(env.slot_machines))
        self.beta = np.ones(len(env.slot_machines))

    def act(self):
        sampled_values = np.random.beta(self.alpha, self.beta)
        return np.argmax(sampled_values)

    def update(self, action, reward):
        r = np.random.rand() < reward      # reward ∈ [0,1]
        self.alpha[action] += r
        self.beta[action] += 1 - r

class EXP3_agent(agent):
    def __init__(self, env, gamma=0.1):
        super().__init__(env)
        self.gamma = gamma
        self.weights = np.ones(len(env.slot_machines))
        self.probs = np.ones(len(env.slot_machines)) / len(env.slot_machines)

    def act(self):
        k = len(self.env.slot_machines)
        return np.random.choice(k, p=self.probs)

    def update(self, action, reward):
        k = len(self.env.slot_machines)
        estimated_reward = reward / self.probs[action]
        self.weights[action] *= np.exp(self.gamma * estimated_reward / k)
        self.weights /= np.max(self.weights)
        self.probs = (1 - self.gamma) * self.weights / np.sum(self.weights) + self.gamma / k

class scenario():
    def __init__(self, env, agent, steps):
        self.env = env
        self.agent = agent
        self.steps = steps

    def run(self):
        for _ in range(self.steps):
            action = self.agent.act()
            reward = self.env.play(action)
            self.agent.update(action, reward)
            self.agent.history.append((action, reward))
            self.agent.total_reward += reward
            self.agent.current_step += 1
        return self.agent.get_history()
    
    def change_env(self, new_env):
        self.env = new_env
        self.agent.env = new_env

    def regret_function(self):
        optimal_reward = max([sm.expected_value for sm in self.env.slot_machines])
        regret = [optimal_reward - reward for _, reward in self.agent.get_history()]
        return np.cumsum(regret)

class results():
    def __init__(self, history, regret):
        self.history = history
        self.regret = regret

    def total_reward(self):
        return sum([reward for _, reward in self.history])

    def average_reward(self):
        return np.mean([reward for _, reward in self.history])

    def action_distribution(self):
        actions = [action for action, _ in self.history]
        return np.bincount(actions) / len(actions)

    def cumulative_regret(self):
        return np.cumsum(self.regret)

scenario1 = scenario(env([stats.norm(loc=0, scale=1), 
                          stats.norm(loc=1, scale=1), 
                          stats.norm(loc=2, scale=1)]), 
                          epsilon_greedy_agent(env([stats.norm(loc=0, scale=1), 
                                                    stats.norm(loc=1, scale=1), 
                                                    stats.norm(loc=2, scale=1)]), 
                                                    epsilon=0.1), steps=1000)

history1 = scenario1.run()
print("Total reward:", results(history1, scenario1.regret_function()).total_reward())
print("Average reward:", results(history1, scenario1.regret_function()).average_reward())
print("Action distribution:", results(history1, scenario1.regret_function()).action_distribution())
print("Expected values of slot machines:", [sm.expected_value for sm in scenario1.env.slot_machines])