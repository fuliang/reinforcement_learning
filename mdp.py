from math import gamma
from sqlite3 import Timestamp
import numpy as np

S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合
# 状态转移函数
P = {
    "s1-保持s1-s1": 1.0,
    "s1-前往s2-s2": 1.0,
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
}

# 奖励函数
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
}

gamma = 0.5
MDP = [S, A, P, R, gamma]

# 策略1,随机策略
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}
# 策略2
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}

def join(str1, str2):
    return str1 + '-' + str2

def monte_carlo_sample(MDP, Pi, timestep_max, number):
    S, A, P, R, gamma = MDP
    episodes = []
    for i in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]
        while s != 's5' and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))
            s = s_next
        episodes.append(episode)
    return episodes

def MC(episodes, V, N, gamma):
    for episode in episodes:
        G = 0
        for i in range(len(episode) - 1, -1, -1):
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]

def occupancy(episodes, s, a, timestamp_max, gamma):
    rho = 0
    total_times = np.zeros(timestamp_max)
    occur_times = np.zeros(timestamp_max)
    for episode in episodes:
        for i in range(len(episode)):
            (s_opt, a_opt, r, s_next) = episode[i]
            total_times[i] += 1
            if s == s_opt and a == a_opt:
                occur_times[i] += 1
    for i in reversed(range(timestamp_max)):
        if total_times[i]:
            rho += gamma ** i * occur_times[i] / total_times[i]

    return (1 - gamma) * rho

timestamp_max = 1000
episodes = monte_carlo_sample(MDP, Pi_1, 20, 1000)
# for i in range(0, 5):
#     print("第%d条序列\n" % (i + 1),  episodes[i])
gamma = 0.5
V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5" : 0}
N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5" : 0}
MC(episodes, V, N, gamma)
print("使用蒙特卡洛方法计算MDP的状态值为\n", V)

episodes_1 = monte_carlo_sample(MDP, Pi_1, timestamp_max, 1000)
episodes_2 = monte_carlo_sample(MDP, Pi_2, timestamp_max, 1000)
rho_1 = occupancy(episodes_1, "s4", "概率前往", timestamp_max, gamma)
rho_2 = occupancy(episodes_2, "s4", "概率前往", timestamp_max, gamma)
print(rho_1, rho_2)