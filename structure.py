import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
np.random.seed(42)


class slot_machine:
    def __init__(self, prob):
        self.prob = prob
        self.expected_value = prob.expect(lambda x: x, lb=0)    # E[max(0, X)]
        
    def pull(self):
        return max(0, round(self.prob.rvs(), 2))

    def reset(self, prob):
        self.__init__(prob)

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
        # do sprawdzenia wpływu
        self.max_expected_value = 10

    def new_env(self, env):
        self.env = env
        self.max_expected_value = 10

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
        untried = np.flatnonzero(self.action_counts == 0)
        if untried.size > 0:
            return untried[0]
        t = self.action_counts.sum()
        ucb_values = self.q_values + self.c * np.sqrt(np.log(t) / self.action_counts)
        return np.argmax(ucb_values)

    def update(self, action, reward):
        r_scaled = min(reward, self.max_expected_value) / self.max_expected_value
        self.action_counts[action] += 1
        self.q_values[action] += (r_scaled - self.q_values[action]) / self.action_counts[action]

# class thompson_sampling_agent(agent):
#     def __init__(self, env):
#         super().__init__(env)
#         self.alpha = np.ones(len(env.slot_machines))
#         self.beta = np.ones(len(env.slot_machines))

#     def act(self):
#         sampled_values = np.random.beta(self.alpha, self.beta)
#         return np.argmax(sampled_values)

#     def update(self, action, reward):
#         r_scaled = min(reward, self.max_expected_value) / self.max_expected_value
#         self.alpha[action] += r_scaled
#         self.beta[action] += 1 - r_scaled

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
        r_scaled = min(reward, self.max_expected_value) / self.max_expected_value
        k = len(self.env.slot_machines)
        estimated_reward = r_scaled / self.probs[action]
        self.weights[action] *= np.exp(self.gamma * estimated_reward / k)
        self.weights /= np.max(self.weights)
        self.probs = (1 - self.gamma) * self.weights / np.sum(self.weights) + self.gamma / k

class scenario():
    def __init__(self, agent, steps):
        self.agent = agent
        self.steps = steps

    def run(self):
        for _ in range(self.steps):
            action = self.agent.act()
            reward = self.agent.env.play(action)
            self.agent.update(action, reward)
            self.agent.history.append((action, reward))
            self.agent.total_reward += reward
            self.agent.current_step += 1
        return self.agent.get_history()
    
    def change_env(self, new_env):
        self.env = new_env
        self.agent.env = new_env

    def regret_function(self):
        expected_values = [sm.expected_value for sm in self.agent.env.slot_machines]
        optimal_reward = max(expected_values)
        regret = [optimal_reward - expected_values[action] for action, _ in self.agent.get_history()]
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

    def fregret(self):
        return self.regret
    
def show_results(history, regret):
    print("Total reward:", results(history, regret).total_reward())
    print("Average reward:", results(history, regret).average_reward())
    print("Action distribution:", results(history, regret).action_distribution())

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(results(history, regret).action_distribution())), results(history, regret).action_distribution())
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Proportion")
    plt.subplot(1, 2, 2)
    plt.plot(results(history, regret).fregret())
    plt.title("Regret")
    plt.xlabel("Steps")
    plt.ylabel("Regret")
    plt.show()


env1 = env([stats.norm(loc=0, scale=1), 
            stats.norm(loc=1, scale=1), 
            stats.norm(loc=2, scale=1)])

env2 = env([stats.expon(loc=0, scale=1), 
            stats.norm(loc=1, scale=1),
            stats.binom(n=2, p=0.5)])

# scenario1 = scenario(epsilon_greedy_agent(env1, epsilon=0.1), steps=1000)
# history1 = scenario1.run()
# #show_results(history1, scenario1.regret_function())

# scenario2 = scenario(epsilon_greedy_agent(env2, epsilon=0.1), steps=1000)
# history2 = scenario2.run()
# #how_results(history2, scenario2.regret_function())

# scenario3 = scenario(epsilon_greedy_agent(env2, epsilon=0.05), steps=1000)
# history3 = scenario3.run()
# #show_results(history3, scenario3.regret_function())

# scenario4 = scenario(epsilon_greedy_agent(env2, epsilon=0.01), steps=1000)
# history4 = scenario4.run()
# #show_results(history4, scenario4.regret_function())


def run_experiment(make_agent, env, steps, M, title=None, seed=None):

    all_results = []
    for _ in range(M):
        sc = scenario(make_agent(env), steps)
        history = sc.run()
        all_results.append(results(history, sc.regret_function()))

    show_experiment(all_results, len(env.slot_machines), title)
    return all_results


def show_experiment(all_results, K, title=None):
    M = len(all_results)
    steps = len(all_results[0].history)

    # liczba użyć maszyn: action_distribution() * steps
    usage = np.zeros((M, K))
    for m, r in enumerate(all_results):
        dist = r.action_distribution()
        usage[m, :len(dist)] = dist * steps

    regrets = np.array([r.fregret() for r in all_results])

    print("Średni total reward:", np.mean([r.total_reward() for r in all_results]))
    print("Średni average reward:", np.mean([r.average_reward() for r in all_results]))

    plt.figure(figsize=(12, 6))
    if title:
        plt.suptitle(title)

    plt.subplot(1, 2, 1)
    plt.boxplot([usage[:, k] for k in range(K)])
    plt.xticks(range(1, K + 1), [f"Maszyna {k}" for k in range(K)])
    plt.title(f"Użycie maszyn ({M} powtórzeń)")
    plt.xlabel("Action")
    plt.ylabel("Liczba użyć")

    plt.subplot(1, 2, 2)
    alpha = min(0.5, max(0.02, 5 / M))
    plt.plot(regrets.T, color="tab:blue", alpha=alpha, lw=0.8)
    plt.plot(regrets.mean(axis=0), color="black", lw=1, label="średnia")
    plt.title(f"Regret ({M} powtórzeń)")
    plt.xlabel("Steps")
    plt.ylabel("Regret")
    plt.legend()

    plt.tight_layout()
    plt.show()


#res = run_experiment(lambda e: ucb_agent(e, 2), env2, steps=10000, M=10,
#                     title="UCB, c=2, env2", seed=42)

res = run_experiment(lambda e: EXP3_agent(e, 0.2), env2, steps=10000, M=1000,
                     title="EXP3, gamma=0.2, env2", seed=42)

res = run_experiment(lambda e: epsilon_greedy_agent(e, 0.2), env2, steps=10000, M=1000,
                     title="Epsilon-Greedy, epsilon=0.2, env2", seed=42)