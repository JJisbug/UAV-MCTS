from Maze import Maze
from DQN_brain import DeepQNetwork
from DQN_brain import DeepQNetwork2
from DQN_brain import DeepQNetwork3
import matplotlib.pyplot as plt
import numpy as np
import time
import pyttsx3
AVAILABLE_CHOICES = [[20, 20], [30, 120], [50, 250], [70, 355], [80, 220], [120, 140], [150, 300], [200, 200],
                                     [220, 300], [250, 150], [280, 70], [330, 200], [350, 30], [380, 180], [400, 400],
                                     [200, 450], [450, 60], [470, 160], [475, 300], [490, 400]]

start = time.time()
N = 1000

def run_maze():
    cumulative_reward1 = 0
    cumulative_reward1_ = 0
    average_return1 = []
    episode_return1 = []
    step = 0
    count = 0
    cumulative_choices = [[0, 0]]

    for episode in range(N+1):

        observation = env.reset_uav()
        # env.render()
        #time.sleep(1)

        for i in range(10):

            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done = env.step(action)

            cumulative_reward1_ += reward

            RL.store_transition(observation, action, reward, observation_)

            RL.learn()

            observation = observation_

            step += 1

            if episode == N:
                cumulative_choices.append(AVAILABLE_CHOICES[action])
            """
            if done:
                average_return1.append(cumulative_reward1 / step)
                episode_return1.append(cumulative_reward1_)
                cumulative_reward1_ = 0
                break
            """

    print('cumulative_choices', cumulative_choices)
    cumulative_reward2 = 0
    cumulative_reward2_ = 0
    average_return2 = []
    episode_return2 = []
    step = 0


    env.reset_uav()
    env.render()
    end = time.time()
    print("game over!")
    print('运行时间:', end - start)
    engine = pyttsx3.init()
    engine.say('程序运行完成')
    engine.runAndWait()
    # env.destory()


if __name__ == "__main__":
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.1,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=500,
                      output_graph=False
                      )
    RL_ = DeepQNetwork2(env.n_actions, env.n_features,
                        learning_rate=0.05,
                        reward_decay=0.9,
                        e_greedy=0.9,
                        replace_target_iter=200,
                        memory_size=2000,
                        )
    RL__ = DeepQNetwork3(env.n_actions, env.n_features,
                         learning_rate=0.05,
                         reward_decay=0.9,
                         e_greedy=0.9,
                         replace_target_iter=200,
                         memory_size=2000,
                         )

    env.after(100, run_maze)
    env.mainloop()

    # RL.plot_cost()
    # RL_.plot_cost()
    # RL__.plot_cost()
