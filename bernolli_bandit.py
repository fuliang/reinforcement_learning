import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import k_means

class BernoulliBandit:

    def __init__(self, K):
        self.probs = np.random.uniform(size=K)
        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]
        self.K = K

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

class Solver:
    def __init__(self, bandit: BernoulliBandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0
        self.actions = []
        self.regrets = []

    def update_regret(self, k):
        # print("select k=%d" % k)
        self.regret += (self.bandit.best_prob - self.bandit.probs[k])
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError

    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

class EpsilonGreedy(Solver):
    def __init__(self, bandit: BernoulliBandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class UCB(Solver):
    def __init__(self, bandit: BernoulliBandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class ThompsonSampling(Solver):
    def __init__(self, bandit: BernoulliBandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)
        self._b = np.ones(self.bandit.K)

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)
        k = np.argmax(samples)
        r = self.bandit.step(k)

        self._a[k] += r
        self._b[k] += (1-r)
        return k

def plot_results(solvers, solvers_names):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets,label=solvers_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title("%d-armed bandit" % solvers[0].bandit.K)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    np.random.seed(0)
    K = 10
    bandit_10_arm = BernoulliBandit(10)
    print("随机生成一个%d臂伯努利老虎机" % K)
    print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" % (bandit_10_arm.best_idx, bandit_10_arm.best_prob))
    # epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
    # epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons]
    # epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
    # for solver in epsilon_greedy_solver_list:
    #     solver.run(500)
    # plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)
    solvers = [EpsilonGreedy(bandit_10_arm, epsilon=0.01), DecayingEpsilonGreedy(bandit_10_arm), 
    UCB(bandit_10_arm, coef=1), ThompsonSampling(bandit_10_arm)]
    
    solver_names = ["epsilon greedy", "decaying epsilong greedy", "UCB", "Thompson"]
    for solver in solvers: 
        solver.run(5000)
    plot_results(solvers, solver_names)