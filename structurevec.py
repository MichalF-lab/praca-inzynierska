import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Skalowanie nagród jak w oryginale (min(r, 10) / 10)
MAX_EXPECTED_VALUE = 10


# ---------------------------------------------------------------------------
# Środowisko: M niezależnych kopii tych samych K maszyn naraz
# ---------------------------------------------------------------------------
class VecEnv:
    def __init__(self, tab_prob, M, rng, chunk=1000):
        self.tab_prob = tab_prob
        self.K = len(tab_prob)
        self.M = M
        self.rng = rng
        self.chunk = chunk
        # E[max(0, X)] dla każdej maszyny, kształt (K,)
        self.expected_values = np.array([p.expect(lambda x: x, lb=0) for p in tab_prob])
        self._buf = None
        self._i = chunk  # wymusza losowanie przy pierwszym play()

    def _refill(self):
        # (chunk, M, K): nagrody dla każdego kroku, przebiegu i maszyny,
        # losowane jednym wywołaniem rvs na maszynę zamiast 1 wywołania na krok
        self._buf = np.stack(
            [np.maximum(0, np.round(p.rvs(size=(self.chunk, self.M), random_state=self.rng), 2))
             for p in self.tab_prob],
            axis=-1,
        )
        self._i = 0

    def play(self, actions):
        # actions: (M,) -> nagrody: (M,)
        if self._i >= self.chunk:
            self._refill()
        r = self._buf[self._i, np.arange(self.M), actions]
        self._i += 1
        return r


# ---------------------------------------------------------------------------
# Agenci: stan ma kształt (M, K) – jeden wiersz = jeden niezależny przebieg
# ---------------------------------------------------------------------------
class VecAgent:
    def __init__(self, M, K, rng):
        self.M, self.K, self.rng = M, K, rng
        self.idx = np.arange(M)


class VecEpsilonGreedy(VecAgent):
    def __init__(self, M, K, rng, epsilon):
        super().__init__(M, K, rng)
        self.epsilon = epsilon
        self.q = np.zeros((M, K))
        self.n = np.zeros((M, K))

    def act(self):
        explore = self.rng.random(self.M) < self.epsilon
        random_a = self.rng.integers(self.K, size=self.M)
        greedy_a = self.q.argmax(axis=1)
        return np.where(explore, random_a, greedy_a)

    def update(self, a, r):
        i = self.idx
        self.n[i, a] += 1
        self.q[i, a] += (r - self.q[i, a]) / self.n[i, a]


class VecUCB(VecAgent):
    def __init__(self, M, K, rng, c):
        super().__init__(M, K, rng)
        self.c = c
        self.q = np.zeros((M, K))
        self.n = np.zeros((M, K))

    def act(self):
        untried = self.n == 0
        has_untried = untried.any(axis=1)
        first_untried = untried.argmax(axis=1)          # pierwsza nieużyta maszyna
        n_safe = np.maximum(self.n, 1)                  # unikamy dzielenia przez 0
        t = np.maximum(self.n.sum(axis=1, keepdims=True), 1)
        ucb = self.q + self.c * np.sqrt(np.log(t) / n_safe)
        return np.where(has_untried, first_untried, ucb.argmax(axis=1))

    def update(self, a, r):
        i = self.idx
        r_scaled = np.minimum(r, MAX_EXPECTED_VALUE) / MAX_EXPECTED_VALUE
        self.n[i, a] += 1
        self.q[i, a] += (r_scaled - self.q[i, a]) / self.n[i, a]


class VecEXP3(VecAgent):
    def __init__(self, M, K, rng, gamma=0.1):
        super().__init__(M, K, rng)
        self.gamma = gamma
        self.w = np.ones((M, K))
        self.probs = np.full((M, K), 1.0 / K)

    def act(self):
        # losowanie z rozkładu dyskretnego osobno w każdym wierszu
        u = self.rng.random((self.M, 1))
        cdf = np.cumsum(self.probs, axis=1)
        a = (u > cdf).sum(axis=1)
        return np.minimum(a, self.K - 1)                # zabezpieczenie na błąd zaokrągleń

    def update(self, a, r):
        i, K, g = self.idx, self.K, self.gamma
        r_scaled = np.minimum(r, MAX_EXPECTED_VALUE) / MAX_EXPECTED_VALUE
        est = r_scaled / self.probs[i, a]
        self.w[i, a] *= np.exp(g * est / K)
        self.w /= self.w.max(axis=1, keepdims=True)
        self.probs = (1 - g) * self.w / self.w.sum(axis=1, keepdims=True) + g / K


# ---------------------------------------------------------------------------
# Eksperyment: jedna pętla po krokach, wszystkie M przebiegów naraz
# ---------------------------------------------------------------------------
def run_experiment_vec(make_agent, tab_prob, steps, M, title=None, seed=None):
    rng = np.random.default_rng(seed)
    env = VecEnv(tab_prob, M, rng)
    agent = make_agent(M, env.K, rng)

    actions = np.empty((steps, M), dtype=np.int8)
    total_reward = np.zeros(M)

    for t in range(steps):
        a = agent.act()
        r = env.play(a)
        agent.update(a, r)
        actions[t] = a
        total_reward += r

    ev = env.expected_values
    regrets = np.cumsum(ev.max() - ev[actions], axis=0).T                 # (M, steps)
    usage = np.stack([(actions == k).sum(axis=0) for k in range(env.K)], axis=1)  # (M, K)

    show_experiment_vec(usage, regrets, total_reward, steps, title)
    return {"actions": actions, "total_reward": total_reward,
            "regrets": regrets, "usage": usage}


def show_experiment_vec(usage, regrets, total_reward, steps, title=None):
    M, K = usage.shape
    print("Średni total reward:", total_reward.mean())
    print("Średni average reward:", (total_reward / steps).mean())

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


# ---------------------------------------------------------------------------
probs_env2 = [stats.expon(loc=0, scale=1),
              stats.norm(loc=1, scale=1),
              stats.binom(n=2, p=0.5)]

# res = run_experiment_vec(lambda M, K, rng: VecEXP3(M, K, rng, gamma=0.2),
#                          probs_env2, steps=10000, M=1000,
#                          title="EXP3, gamma=0.2, env2", seed=42)

# res = run_experiment_vec(lambda M, K, rng: VecEpsilonGreedy(M, K, rng, epsilon=0.2),
#                          probs_env2, steps=10000, M=1000,
#                          title="Epsilon-Greedy, epsilon=0.2, env2", seed=42)

res = run_experiment_vec(lambda M, K, rng: VecUCB(M, K, rng, c=0.2),
                         probs_env2, steps=10000, M=1000,
                         title="UCB, c=0.2, env2", seed=42)