import numpy as np
np.random.seed(42)


class slot_machine:
    def __init__(self, prob):
        self.prob = prob
        
    def pull(self, value):
        return max(0, round(self.prob() * value, 2))

    def reset(self, prob):
        self.prob = prob


class env:
    def __init__(self, tab_prob):
        self.slot_machines = [slot_machine(prob) for prob in tab_prob]

    def full_reset(self, tab_prob):
        for sm in self.slot_machines:
            sm.reset(tab_prob)

    def one_reset(self, prob, index):
        self.slot_machines[index].reset(prob)

    def play(self, index, value):
        return self.slot_machines[index].pull(value)
    

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
        self.alpha[action] += reward
        self.beta[action] += 1 - reward


class EXP3_agent(agent):
    def __init__(self, env):
        super().__init__(env)
        pass


    def act(self):
        pass

    def update(self, action, reward):
        pass


# Example usage
slot_machines = [slot_machine(np.random.normal) for _ in range(10)]
print(slot_machines[1].pull(3))
print(slot_machines[1].pull(10))
print(slot_machines[1].pull(22))