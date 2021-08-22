import numpy as np
import tkinter as tk
import time
import random

UNIT = 40
MAZE_H = 15
MAZE_W = 25
global all_count
global all_reward
all_reward = 0
all_count = 0

np.random.seed(1)


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                             '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
        self.n_actions = len(self.action_space)
        self.n_features = 3
        self.title('UAV测试环境')
        # self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()
        self.user_center = np.array([[50, 200], [150, 200], [250, 200], [350, 200], [450, 200],
                                     [50, 400], [150, 400], [250, 400], [350, 400], [450, 400]])

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

        self.oval_center = np.array([[20, 20], [30, 120], [50, 250], [70, 355], [80, 220], [120, 140], [150, 300], [200, 200],
                                     [220, 300], [250, 150], [280, 70], [330, 200], [350, 30], [380, 180], [400, 400],
                                     [200, 450], [450, 60], [470, 160], [475, 300], [490, 400]])

        self.user_center = np.array([[50, 200], [150, 200], [250, 200], [350, 200], [450, 200],
                                     [50, 400], [150, 400], [250, 400], [350, 400], [450, 400]])
        for i in range(20):
            self.canvas.create_oval(self.oval_center[i, 0] - 10, self.oval_center[i, 1] - 10,
                                    self.oval_center[i, 0] + 10, self.oval_center[i, 1] + 10,
                                    fill='blue')

        for i in range(10):
            self.canvas.create_oval(self.user_center[i][0] - 5, self.user_center[i][1] - 5,
                                    self.user_center[i][0] + 5, self.user_center[i][1] + 5,
                                    fill='black')

        self.img = tk.PhotoImage(file="UAV.png")
        self.uav = self.canvas.create_image((40, 40), image=self.img)

        # pack all
        self.canvas.pack()

    def reset_uav(self):
        self.update()
        # time.sleep(0.1)
        self.battery = 10
        self.canvas.delete(self.uav)
        self.uav = self.canvas.create_image((5, 5), image=self.img)
        # return np.array([self.canvas.coords(self.uav)[0] / (MAZE_W * UNIT),
        #  self.canvas.coords(self.uav)[1] / (MAZE_H * UNIT)])
        return np.hstack((np.array([self.canvas.coords(self.uav)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.uav)[1] / (MAZE_H * UNIT)]), self.battery / 10))

    def reset_uav_(self):
        self.update()
        time.sleep(0.1)
        self.battery = 20
        self.canvas.delete(self.uav)
        self.uav = self.canvas.create_image((5, 5), image=self.img)
        # return np.array([self.canvas.coords(self.uav)[0] / (MAZE_W * UNIT),
        #  self.canvas.coords(self.uav)[1] / (MAZE_H * UNIT)])
        return np.hstack((np.array([self.canvas.coords(self.uav)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.uav)[1] / (MAZE_H * UNIT)]), self.battery / 10))

    def reset_uav__(self):
        self.update()
        time.sleep(0.1)
        self.battery = 30
        self.canvas.delete(self.uav)
        self.uav = self.canvas.create_image((5, 5), image=self.img)
        # return np.array([self.canvas.coords(self.uav)[0] / (MAZE_W * UNIT),
        #  self.canvas.coords(self.uav)[1] / (MAZE_H * UNIT)])
        return np.hstack((np.array([self.canvas.coords(self.uav)[0] / (MAZE_W * UNIT),
                                    self.canvas.coords(self.uav)[1] / (MAZE_H * UNIT)]), self.battery / 10))

    def step(self, action):
        s = np.array(self.canvas.coords(self.uav))
        for i in range(20):
            if action == i:
                self.canvas.delete(self.uav)
                point = self.oval_center[i, :]
                self.uav = self.canvas.create_image((point[0], point[1]), image=self.img)
                break

        next_coords = self.canvas.coords(self.uav)

        u = np.random.normal(loc=10, scale=5, size=10)
        for i in range(10):
            if u[i] < 0:
                u[i] = 0
            elif u[i] > 20:
                u[i] = 20

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

        # reward function
        for j in range(20):
            if next_coords == self.oval_center[j, :].tolist():
                pu = 0.1
                ph = 1
                rho = 10e-5
                sigma = 9.999999999999987e-15
                # ef = p * (np.sqrt((s[0] - self.oval_center[j, 0]) ** 2 +(s[1] - self.oval_center[j, 1]) ** 2)) / 800
                ef1 = 0.5 * 10 * (np.sqrt((s[0] - self.oval_center[j, 0]) ** 2 + (s[1] - self.oval_center[j, 1]) ** 2)) * 20
                ef2 = 0.03 * 8000 + 0.25 * (1 + 225 / 96.04)

                ef = ef1 + ef2

                distance = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

                for k in range(10):
                    distance[k] = (next_coords[0] - self.user_center[k, 0])**2 +\
                                  (next_coords[1] - self.user_center[k, 1])**2

                o = distance.tolist().index(distance.min())

                # eh = p * (u[o] / np.emath.log2(1 + ((p*(rho / (100**2 + distance.min())))/sigma))) * 5
                eh = ph * (u[o] * 10e4 / np.emath.log2(1 + ((pu * (rho / (100 ** 2 + distance.min()))) / sigma)))

                #ec = 400 * u[o]
                st = u[o] * 10e4
                ec = 10e-27 * 1000 * (2e9) ** 2 * st

                # utility = 1 - np.exp(-((u[o] ** 2) / (u[o] + 10)))
                utility = u[o] / 20
                energy_utility = (ef + ec + eh) / 186000  # v=20m/s

                reward = 2 * utility - energy_utility

                self.battery -= (ef + ec + eh)
                # reward = utility - ef - ec - eh
                global all_count
                global all_reward
                all_reward = all_reward + reward
                all_count = all_count + 1

                p = 1
                if self.battery <= p*(np.sqrt((40-self.oval_center[j, 0])**2+(40-self.oval_center[j, 1])**2))/800:
                    done = True
                else:
                    done = False
                # s_ = np.array([next_coords[0] / (MAZE_H * UNIT), next_coords[1] / (MAZE_W * UNIT)])
                s_ = np.hstack((np.array([next_coords[0] / (MAZE_H * UNIT), next_coords[1] / (MAZE_W * UNIT)]),
                                self.battery / 10))
                return s_, reward, done

    def render(self):
        # time.sleep(0.5)
        self.update()


if __name__ == "__main__":
    env = Maze()
    env.mainloop()
