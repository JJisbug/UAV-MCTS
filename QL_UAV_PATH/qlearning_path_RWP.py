# !usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import tkinter as tk
import time
import pandas as pd
import matplotlib.pyplot as plt
import pyttsx3
import random

UNIT = 30
MAZE_H = 20
MAZE_W = 20
TRAINING_EPISODE = 500
global all_count
all_count = 0
global all_reward
all_reward = 0

AVAILABLE_CHOICES = [[20, 20], [30, 120], [50, 250], [70, 355], [80, 220], [120, 140], [150, 300], [200, 200],
                                     [220, 300], [250, 150], [280, 70], [330, 200], [350, 30], [380, 180], [400, 400],
                                     [200, 450], [450, 60], [470, 160], [475, 300], [490, 400]]
USER_POSITIONS = [[random.randint(40, 61), random.randint(190, 211)], [random.randint(140, 161), random.randint(190, 211)],
                  [random.randint(240, 261), random.randint(190, 211)],
                  [random.randint(340, 361), random.randint(190, 211)], [random.randint(440, 461), random.randint(190, 211)],
                  [random.randint(40, 61), random.randint(390, 411)], [random.randint(140, 161), random.randint(390, 411)],
                  [random.randint(240, 261), random.randint(390, 411)], [random.randint(340, 361), random.randint(390, 411)],
                  [random.randint(440, 461), random.randint(390, 411)]]  # 10user
AVAILABLE_CHOICES_x = [20, 30, 50, 70, 80, 120, 150, 200, 220, 250, 280, 330, 350, 380, 400,
                                     200, 450, 470, 475, 490]
AVAILABLE_CHOICES_y = [20, 120, 250, 355, 220, 140, 300, 200, 300, 150, 70, 200, 30, 180, 400,
                                     450, 60, 160, 300, 400]
USER_POSITIONS_x = [USER_POSITIONS[0][0], USER_POSITIONS[1][0], USER_POSITIONS[2][0], USER_POSITIONS[3][0],
                    USER_POSITIONS[4][0], USER_POSITIONS[5][0], USER_POSITIONS[6][0], USER_POSITIONS[7][0],
                    USER_POSITIONS[8][0], USER_POSITIONS[9][0]]
USER_POSITIONS_y = [USER_POSITIONS[0][1], USER_POSITIONS[1][1], USER_POSITIONS[2][1], USER_POSITIONS[3][1],
                    USER_POSITIONS[4][1], USER_POSITIONS[5][1], USER_POSITIONS[6][1], USER_POSITIONS[7][1],
                    USER_POSITIONS[8][1], USER_POSITIONS[9][1]]

np.random.seed(1)
# tf.set_random_seed(1)


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                             '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
        self.n_actions = len(self.action_space)
        self.n_features = 3
        self.title('UAV测试环境-Q')
        # self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()
        self.user_center = [[50, 200], [150, 200], [250, 200], [350, 200], [450, 200],
                              [50, 400], [150, 400], [250, 400], [350, 400], [450, 400]]  # 10user的初始坐标

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

        self.oval_center = np.array([[20, 20], [30, 120], [50, 250], [70, 355], [80, 220], [120, 140], [150, 300], [200, 200],
                                     [220, 300], [250, 150], [280, 70], [330, 200], [350, 30], [380, 180], [400, 400],
                                     [200, 450], [450, 60], [470, 160], [475, 300], [490, 400]])
        for i in range(20):
            self.canvas.create_oval(self.oval_center[i, 0] - 10, self.oval_center[i, 1] - 10,
                                    self.oval_center[i, 0] + 10, self.oval_center[i, 1] + 10,
                                    fill='blue')

        self.img = tk.PhotoImage(file="UAV.png")
        self.uav = self.canvas.create_image((5, 5), image=self.img)

        # pack all
        self.canvas.pack()

    def reset_uav(self):
        self.update()
        # time.sleep(0.1)
        self.battery = 10
        self.canvas.delete(self.uav)
        self.uav = self.canvas.create_image((5, 5), image=self.img)
        return self.canvas.coords(self.uav)
        # -------------------------------------------------------------------------------------------
        # return np.hstack((np.array([self.canvas.coords(self.uav)[0], self.canvas.coords(self.uav)[1], self.battery])))

    def step(self, action):
        s = np.array(self.canvas.coords(self.uav))
        for i in range(20):
            if action == i:
                self.canvas.delete(self.uav)
                point = self.oval_center[i, :]
                self.uav = self.canvas.create_image((point[0], point[1]), image=self.img)
                break

        next_coords = self.canvas.coords(self.uav)
        # point = next_coords
        # print('point', 'next_coords', point, next_coords)

        u = np.random.normal(loc=10, scale=5, size=10)
        for j in range(10):
            if u[j] < 0:
                u[j] = 0
            if u[j] > 20:
                u[j] = 20
        # print('u', u)
        # --------------------------用户--------------------------------

        for m in range(10):
            self.user_center[m][0] = self.user_center[m][0] + random.randint(-10, 10)
            self.user_center[m][1] = self.user_center[m][1] + random.randint(-10, 10)
            if self.user_center[m][0] < 0:
                self.user_center[m][0] = 0
            if self.user_center[m][1] < 0:
                self.user_center[m][1] = 0
            if self.user_center[m][0] > 500:
                self.user_center[m][0] = 500
            if self.user_center[m][1] > 500:
                self.user_center[m][1] = 500
            # print("aaa", self.user_center[m][0])

        # reward function
        for j in range(20):
            if next_coords == self.oval_center[j, :].tolist():
                p = 1
                pu = 0.1
                ph = 1
                rho = 10e-5
                sigma = 9.999999999999987e-15

                ef1 = 0.5 * 10 * (np.sqrt((s[0] - self.oval_center[j, 0]) ** 2 + (s[1] - self.oval_center[j, 1]) ** 2)) * 20
                ef2 = 0.03 * 8000 + 0.25 * (1 + 225 / 96.04)
                ef = ef1 + ef2
                distance = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

                for k in range(10):
                    distance[k] = (next_coords[0] - self.user_center[k][0])**2 + (next_coords[1] - self.user_center[k][1])**2

                o = distance.tolist().index(distance.min())

                # eh = p * (u[o] / np.emath.log2(1 + ((p*(rho / (100**2 + distance.min())))/sigma))) * 5
                eh = ph * (u[o] * 10e4 / np.emath.log2(1 + ((pu * (rho / (100**2 + distance.min())))/sigma)))

                #ec = 400 * u[o]
                st = u[o] * 10e4
                ec = 10e-27 * 1000 * (2e9)**2 * st

                # utility = 1 - np.exp(-((u[o] ** 2) / (u[o] + 10)))
                utility = u[o] / 20
                # energy_utility = (ef + ec + eh) / 161000  # v=10m/s 158847
                energy_utility = (ef + ec + eh) / 186000  # v=20m/s
                reward = 2*utility - energy_utility

                self.battery -= (ef + ec + eh)
                global all_count
                global all_reward
                all_reward = all_reward + reward
                all_count = all_count + 1
                # print('reward', 'utility', 'ef', 'ec', 'eh', reward, utility, ef, ec, eh)

                if self.battery <= p*(np.sqrt((40-self.oval_center[j, 0])**2+(40-self.oval_center[j, 1])**2))/800:
                    done = True
                else:
                    done = False

                s_ = np.hstack((np.array([next_coords[0] / (MAZE_H * UNIT), next_coords[1] / (MAZE_W * UNIT)]),
                                 self.battery / 10))
                # -------------------------------------------------------------------------------------
                # s_ = np.hstack((np.array([next_coords[0], next_coords[1], self.battery])))
                return s_, reward, done, u[o], o

    def render(self):
        # time.sleep(0.5)
        self.update()
        s = 0


class QTable:
    def __init__(self,
                 actions,
                 learning_rate=0.1,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 ):
        self.actions = actions
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):

        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        q_target = r + self.gamma * self.q_table.loc[s_, :].max() - self.q_table.loc[s, a]
        self.q_table.loc[s, a] = (1 - self.lr) * q_predict + self.lr * q_target
        return self.q_table


start = time.time()

N = TRAINING_EPISODE


def run_maze():

    remark0 = []
    average_remark0 = []
    cumulative_reward0 = 0

    migration0 = []
    average_migration0 = [0]
    cumulative_migration0 = 0
    average_migration_100 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #                  0  1  2  3  4  5  6  7  8  9  10
    average_remark_100 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    t0 = 0  # updates times

    global q_table
    uu = 0
    uu_100 = 0
    cumulative_choices = [[0, 0]]
    u_new = 0

    for episode in range(N):  # episode 0-99

        s0 = []
        a0 = []
        user = []

        observation = env.reset_uav()

        for i in range(10):

            s0.append(observation)

            env.render()

            action = RL1.choose_action(str(observation))

            a0.append(action)

            observation_, reward, done, u, user_index = env.step(action)

            user.append(u)

            q_table = RL1.learn(str(observation), action, reward, str(observation_))

            t0 += 1

            uu += u
            cumulative_reward0 += reward

            remark0.append(cumulative_reward0)
            average_remark0.append(cumulative_reward0 / t0)
            migration0.append(uu)
            average_migration0.append(uu / t0)

            if episode == N - 1:
                u_new = u_new + u
                cumulative_choices.append(AVAILABLE_CHOICES[action])

            observation = observation_

    x0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16}

    print('average_reward', all_reward / all_count)
    print('all_count', all_count)
    # print(average_migration_100)
    print(cumulative_choices)

    plt.plot(x0, average_migration_100, color='purple', linestyle='--', marker='^', label='training_episode=5000')
    plt.ylabel('Average Throughput', font2)
    plt.xlabel('Play Round', font2)
    plt.axis([-0.5, 10.5, -0.8, 13.5])
    plt.legend()
    # plt.savefig('average_migration.eps', bbox_inches='tight')
    # plt.show()

    cumulative_choices1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cumulative_choices2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for j in range(11):
        cumulative_choices1[j] = cumulative_choices[j][0]
        cumulative_choices2[j] = cumulative_choices[j][1]
    plt.scatter(AVAILABLE_CHOICES_x, AVAILABLE_CHOICES_y, marker='*', color='green')
    plt.scatter(USER_POSITIONS_x, USER_POSITIONS_y, marker='^', color='b')
    for i in range(10):
        plt.quiver(cumulative_choices1[i], cumulative_choices2[i],
                   cumulative_choices1[i+1] - cumulative_choices1[i], cumulative_choices2[i+1] - cumulative_choices2[i], color='red', width=0.005)
    plt.plot(cumulative_choices1, cumulative_choices2, color='yellow', marker='*', linestyle=':')
    plt.ylabel('y')
    plt.xlabel('x')
    # plt.show()

    # env.render()
    end = time.time()
    print("game over!")
    print('运行时间:%.4f' % (end - start))
    engine = pyttsx3.init()
    engine.say('程序运行完成')
    engine.runAndWait()
    # env.destory()


if __name__ == "__main__":
    env = Maze()
    RL1 = QTable(actions=list(range(env.n_actions)))

    env.after(100, run_maze)
    env.mainloop()

