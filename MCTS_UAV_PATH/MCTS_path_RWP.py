import sys
import math
import random
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import time

AVAILABLE_CHOICES = [[20, 20], [30, 120], [50, 250], [70, 355], [80, 220], [120, 140], [150, 300], [200, 200],
                                     [220, 300], [250, 150], [280, 70], [330, 200], [350, 30], [380, 180], [400, 400],
                                     [200, 450], [450, 60], [470, 160], [475, 300], [490, 400]]
AVAILABLE_CHOICES_x = [20, 30, 50, 70, 80, 120, 150, 200, 220, 250, 280, 330, 350, 380, 400,
                                     200, 450, 470, 475, 490]
AVAILABLE_CHOICES_y = [20, 120, 250, 355, 220, 140, 300, 200, 300, 150, 70, 200, 30, 180, 400,
                                     450, 60, 160, 300, 400]

TASK = np.arange(100).reshape(10, 10)
TASK2 = np.arange(100).reshape(10, 10)

global COUNTING
COUNTING = 0
global round_reward
round_reward = []
global all_count
all_count = 0
global all_reward
all_reward = 0

canvas_length = 500
canvas_width = 500

for i in range(10):
    TASK[i] = np.random.normal(loc=10, scale=5, size=10)
    print(TASK[i])

AVAILABLE_CHOICE_NUMBER = len(AVAILABLE_CHOICES)
MAX_ROUND_NUMBER = 10
p = 0.1
rho = 10e-5
sigma = 9.999999999999987e-15
np.random.seed(1)
serve_account = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
user_task_number = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


class State(object):

    def __init__(self):
        # self.current_value = 0.0
        # For the first root node, the index is 0 and the game should start from 1
        self.current_round_index = 0
        self.cumulative_choices = [[0, 0]]  # list
        self.battery = 10
        self.user_position_min = [0, 0]
        self.user_position = [[50, 200], [150, 200], [250, 200], [350, 200], [450, 200],
                              [50, 400], [150, 400], [250, 400], [350, 400], [450, 400]]
        self.user_index = 0

    def set_user_position(self):
        for m in range(10):
            self.user_position[m][0] = self.user_position[m][0] + random.randint(-10, 10)
            self.user_position[m][1] = self.user_position[m][1] + random.randint(-10, 10)
            if self.user_position[m][0] < 0:
                self.user_position[m][0] = 0
            if self.user_position[m][1] < 0:
                self.user_position[m][1] = 0
            if self.user_position[m][0] > 500:
                self.user_position[m][0] = 500
            if self.user_position[m][1] > 500:
                self.user_position[m][1] = 500
            # print("aaa", self.user_position[m][0])

    def get_current_round_index(self):
        return self.current_round_index

    def set_current_round_index(self, turn):
        self.current_round_index = turn

    def get_cumulative_choices(self):
        return self.cumulative_choices

    def set_cumulative_choices(self, choices):
        self.cumulative_choices = choices  # list

    def is_terminal(self):
        # The round index starts from 1 to max round number
        if self.current_round_index == MAX_ROUND_NUMBER:
            return True
        else:
            return False


    def compute_reward(self):
        current_value = 0
        for i in range(20):
            if self.cumulative_choices[self.current_round_index - 1] == AVAILABLE_CHOICES[i]:
                current_value = i

        self.set_user_position()
        # print("self.user_position", self.user_position)

        dis = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        for k in range(10):
            dis[k] = np.sqrt((AVAILABLE_CHOICES[current_value][0] - self.user_position[k][0]) ** 2 + \
                             (AVAILABLE_CHOICES[current_value][1] - self.user_position[k][1]) ** 2)
        index_tem = dis.tolist().index(dis.min())

        if serve_account[index_tem] == 0:
            self.user_index = index_tem
            serve_account[index_tem] = serve_account[index_tem] + 1
            user_task_number[index_tem] = user_task_number[index_tem] + TASK[self.current_round_index - 1][index_tem]
        elif serve_account[index_tem] == 1 and TASK[self.current_round_index - 1][index_tem] <= 15:
            dis[index_tem] = 1000
            self.user_index = dis.tolist().index(dis.min())
            serve_account[self.user_index] = serve_account[self.user_index] + 1
        elif serve_account[index_tem] == 1 and TASK[self.current_round_index - 1][index_tem] > 15:
            self.user_index = index_tem
            serve_account[index_tem] = serve_account[index_tem] + 1
            user_task_number[index_tem] = user_task_number[index_tem] + TASK[self.current_round_index - 1][index_tem]
        else:
            dis[index_tem] = 1000
            self.user_index = dis.tolist().index(dis.min())
            serve_account[self.user_index] = serve_account[self.user_index] + 1

        # self.user_index = dis.tolist().index(dis.min())
        self.user_position_min = self.user_position[self.user_index]

        x1 = self.cumulative_choices[self.current_round_index - 1][0]
        y1 = self.cumulative_choices[self.current_round_index - 1][1]
        x2 = self.cumulative_choices[self.current_round_index][0]
        y2 = self.cumulative_choices[self.current_round_index][1]
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        ef = p * distance / 20 / 2.7
        eh = p * (TASK[self.current_round_index - 1][self.user_index] / np.emath.log2(
            1 + ((p * (rho / (100 ** 2 + dis.min()))) / sigma))) * 9

        ec = 0.3 * TASK[self.current_round_index - 1][self.user_index] / 5
        utility = TASK[self.current_round_index - 1][self.user_index] / 20
        reward = 2*utility - 0.4*ef - 0.3*ec - 0.3*eh
        print('reward', 'utility', 'ef', 'ec', 'eh', reward, utility, ef, ec, eh)
        print('self.user_index', self.user_index)
        return reward

    def get_next_state_with_random_choice(self):
        random_choice = random.choice([choice for choice in AVAILABLE_CHOICES])
        next_state = State()
        next_state.set_current_round_index(self.current_round_index + 1)
        # next_state.set_current_value(self.current_value + random_choice)
        next_state.set_cumulative_choices(self.cumulative_choices + [random_choice])

        next_state.set_user_position()

        dis = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        for k in range(10):
            dis[k] = np.sqrt((random_choice[0] - next_state.user_position[k][0]) ** 2 + \
                             (random_choice[1] - next_state.user_position[k][1]) ** 2)
        index_tem = dis.tolist().index(dis.min())
        if serve_account[index_tem] == 0:
            next_state.user_index = index_tem
            # serve_account[index_tem] = serve_account[index_tem] + 1
        elif serve_account[index_tem] == 1 and TASK[next_state.current_round_index - 1][index_tem] <= 15:
            dis[index_tem] = 1000
            next_state.user_index = dis.tolist().index(dis.min())
        elif serve_account[index_tem] == 1 and TASK[next_state.current_round_index - 1][index_tem] > 15:
            next_state.user_index = index_tem
            # serve_account[index_tem] = serve_account[index_tem] + 1
        else:
            dis[index_tem] = 1000
            next_state.user_index = dis.tolist().index(dis.min())
        # next_state.user_index = dis.tolist().index(dis.min())

        next_state.user_position_min = next_state.user_position[next_state.user_index]

        return next_state


class Node(object):

    def __init__(self):
        self.parent = None
        self.children = []

        self.visit_times = 0
        self.quality_value = 0.0
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

    def get_children(self):
        return self.children

    def get_visit_times(self):
        return self.visit_times

    def set_visit_times(self, times):
        self.visit_times = times

    def visit_times_add_one(self):
        self.visit_times += 1

    def get_quality_value(self):
        return self.quality_value

    def set_quality_value(self, value):
        self.quality_value = value

    def quality_value_add_n(self, n):
        self.quality_value += n

    def is_all_expand(self):
        return len(self.children) == AVAILABLE_CHOICE_NUMBER

    def add_child(self, sub_node):
        sub_node.set_parent(self)
        self.children.append(sub_node)


def tree_policy(node):

    # Check if the current node is the leaf node
    while not node.get_state().is_terminal():
        if node.is_all_expand():
            node = best_child(node, True)
        else:
            # Return the new sub node
            sub_node = expand(node)
            return sub_node

    # Return the leaf node
    return node


def default_policy(node):

    # Get the state of the game
    current_state = node.get_state()

    # if node.state.is_terminal():
        # final_state_reward = 0
        # return final_state_reward
    # Run until the game over
    while not current_state.is_terminal():
        # Pick one random action to play and get next state
        current_state = current_state.get_next_state_with_random_choice()

    final_state_reward = current_state.compute_reward()
    return final_state_reward


def expand(node):

    tried_sub_node_states = [sub_node.get_state() for sub_node in node.get_children()]
    new_state = node.get_state().get_next_state_with_random_choice()

    # Check until get the new state which has the different action from others
    while new_state in tried_sub_node_states:
        new_state = node.get_state().get_next_state_with_random_choice()

    sub_node = Node()
    sub_node.set_state(new_state)
    node.add_child(sub_node)

    return sub_node


def best_child(node, is_exploration):

    best_score = -sys.maxsize
    best_sub_node = None

    # Travel all sub nodes to find the best one
    for sub_node in node.get_children():
        if is_exploration:
            C = 1 / math.sqrt(2.0)
        else:
            C = 0.0

        # UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
        left = sub_node.get_quality_value() / sub_node.get_visit_times()
        right = 2.0 * math.log(node.get_visit_times()) / sub_node.get_visit_times()
        score = left + C * math.sqrt(right)

        if score > best_score:
            best_sub_node = sub_node
            best_score = score

    return best_sub_node


def backup(node, reward):

    # Update util the root node
    while node is not None:
        # Update the visit times
        node.visit_times_add_one()

        # Update the quality value
        node.quality_value_add_n(reward)

        # Change the node to the parent node
        node = node.parent


def monte_carlo_tree_search(node):

    training_episode = 500
    migration = []
    total_task_number = 0
    simulation_average_task_list = []
    training_episode_list = []

    global COUNTING
    training_episode_2 = int(training_episode - (training_episode/10) * COUNTING)
    COUNTING = COUNTING + 1
    print('training_episode_2', training_episode)
    # fig, ax = plt.subplots()
    # Run as much as possible under the computation budget
    reward_sum = 0
    for i in range(training_episode):
        # 1. Find the best node to expand
        expand_node = tree_policy(node)  # round1 expand_node=sub_node

        # 2. Random run to add node and get reward
        reward = default_policy(expand_node)
        reward_sum = reward_sum + reward
        global all_reward
        global all_count
        all_reward = all_reward + reward
        all_count = all_count + 1
        backup(expand_node, reward)
        # default policy————current_state.compute_reward; compute_reward
        # node.state.user_index
        # 3. Update all passing nodes with reward

        # print('reward', reward)
        # user_index2 = expand_node.state.user_index
        # current_task_number = TASK[expand_node.state.current_round_index - 1][user_index2]
        # total_task_number = total_task_number + current_task_number
        # average_task = total_task_number / (i + 1)
        # print("simulation_average_task", average_task)
        # simulation_average_task_list.append(average_task)
        # training_episode_list.append(i)

        # if not expand_node.state.is_terminal():

    # print('round_reward', round_reward)
    # N. Get the best next node
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16}

    global round_reward
    round_reward.append(reward_sum / training_episode)
    print('round_reward', round_reward)

    best_next_node = best_child(node, False)
    return best_next_node


def main():
    # Create the initialized state and initialized node
    init_state = State()
    init_node = Node()
    init_node.set_state(init_state)
    current_node = init_node


    tk = Tk()
    canvas = Canvas(tk, height=canvas_length, width=canvas_width, bg='white')

    serve_center = np.array([[20, 20], [30, 120], [50, 250], [70, 355], [80, 220], [120, 140], [150, 300], [200, 200],
                                     [220, 300], [250, 150], [280, 70], [330, 200], [350, 30], [380, 180], [400, 400],
                                     [200, 450], [450, 60], [470, 160], [475, 300], [490, 400]])
    user_center = np.array([[50, 200], [150, 200], [250, 200], [350, 200], [450, 200],
                            [50, 400], [150, 400], [250, 400], [350, 400], [450, 400]])

    for i in range(20):
        canvas.create_oval(serve_center[i, 0] - 8, serve_center[i, 1] - 8,
                           serve_center[i, 0] + 8, serve_center[i, 1] + 8,
                           fill='yellow')

    for i in range(10):
        canvas.create_oval(user_center[i, 0] - 5, user_center[i, 1] - 5,
                           user_center[i, 0] + 5, user_center[i, 1] + 5,
                           fill='black')

    uav_center = [0, 0]
    canvas.create_polygon(uav_center[0], uav_center[1] + 10,
                          uav_center[0] - 10, uav_center[1] - 10,
                          uav_center[0] + 10, uav_center[1] - 10, fill='', outline='black')
    canvas.pack()

    total_task_number = 0

    xx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    yy = [0]
    QN_list = [0]
    cumulative_choices1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cumulative_choices2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Set the rounds to play
    reward_list = []

    for i in range(10):
        print("Play round: {}".format(i + 1))
        current_node = monte_carlo_tree_search(current_node)

        # uav_center1 = format(current_node)
        # list_convert = uav_center1.split()
        # uav_center = np.array(list_convert)
        # print(current_node.state.cumulative_choices)
        uav_center = current_node.state.cumulative_choices[i + 1]
        uav_center_last = current_node.state.cumulative_choices[i]
        print('uav_center', uav_center)

        canvas.create_polygon(uav_center[0], uav_center[1] + 10,
                              uav_center[0] - 10, uav_center[1] - 10,
                              uav_center[0] + 10, uav_center[1] - 10, fill='', outline='black')
        canvas.create_line((uav_center_last[0], uav_center_last[1]), (uav_center[0], uav_center[1]), width=3, fill="red")

        # print("Choose node: {}".format(current_node))
        print(" Q/N: {}/{} ".format(
            current_node.quality_value, current_node.visit_times))
        QN_list.append(current_node.quality_value / current_node.visit_times )
        print("round: {}, user position: {}, choices: {}".format(
            current_node.state.current_round_index,
            current_node.state.user_position,
            current_node.state.cumulative_choices))
        print("user_index", current_node.state.user_index)

        if i == 9:
            for j in range(11):
                cumulative_choices1[j] = current_node.state.cumulative_choices[j][0]
                cumulative_choices2[j] = current_node.state.cumulative_choices[j][1]

        user_index1 = current_node.state.user_index
        current_task_number = TASK[current_node.state.current_round_index - 1][user_index1]
        total_task_number = total_task_number + current_task_number
        average_task = total_task_number / (i + 1)
        yy.append(average_task)

        # user_task_number[user_index1] = user_task_number[user_index1] + current_task_number
    print("cumulative_choices1", cumulative_choices1)
    print("cumulative_choices2", cumulative_choices2)
    plt.scatter(AVAILABLE_CHOICES_x, AVAILABLE_CHOICES_y, marker='*', color='green')
    # plt.scatter(USER_POSITIONS_x, USER_POSITIONS_y, marker='^', color='b')
    for i in range(10):
        plt.quiver(cumulative_choices1[i], cumulative_choices2[i],
                   cumulative_choices1[i+1] - cumulative_choices1[i], cumulative_choices2[i+1] - cumulative_choices2[i], color='red', width=0.005)
    plt.plot(cumulative_choices1, cumulative_choices2, color='yellow', marker='*', linestyle=':')
    # plt.annotate(' ', xy=(cumulative_choices1[i+1], cumulative_choices2[i+1]), xytext=(cumulative_choices1[i], cumulative_choices2[i]), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    plt.ylabel('y')
    plt.xlabel('x')
    # plt.show()

    """
    canvas.pack()
    tk.mainloop()
    """
    print("serve_account", serve_account)
    print("user_task_number", user_task_number)

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16}

    plt.plot(xx, yy, color='red', marker='^', label='training_episode=1000')
    plt.ylabel('Average Throughput', font2)
    plt.xlabel('Play Round', font2)
    plt.axis([-0.5, 10.5, -0.8, 13.5])
    plt.legend()
    # plt.show()


if __name__ == "__main__":
    main()
