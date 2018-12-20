import sys
sys.path.append("../")

import torch
import torch.nn.functional as F
from model import *

from random import shuffle
import time
from math import inf

import numpy as np
import gym


class AcrobotModel(BaseModel):

    def __init__(self, state_dict_path=None, **kwargs):
        super(AcrobotModel, self).__init__(**kwargs)

        self.fc1 = torch.nn.Linear(6, 12).double()
        self.fc2 = torch.nn.Linear(12, 3).double()

        if state_dict_path is not None:
            self.load_weights(state_dict_path)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def transform(x):
    return np.array(x)


def random_test(iterations=3):
    env = gym.make('Acrobot-v1')

    for iteration in range(iterations):
        timestep = 0
        observation = env.reset()
        cumulative_reward = 0
        done = False

        while not done:

            env.render()
            time.sleep(0.01)

            action = env.action_space.sample()
            
            observation, reward, done, info = env.step(action)

            timestep += 1
            cumulative_reward += reward

            print(f"\rTimestep:  {timestep}".ljust(20), "  Reward: ", cumulative_reward, end="")

            if done:
                # print(f"Episode finished after {timestep} timesteps.  Reward: {cumulative_reward}")
                break


def test(iterations=100, verbose=False, delay=0.0025):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Testing on:  ", device)

    reward_history = []

    env = gym.make('Acrobot-v1')

    model = AcrobotModel("./best_acrobot_model").to(device)

    num_actions = [0, 0, 0]

    for iteration in range(iterations):
        timestep = 0
        observation = env.reset()
        cumulative_reward = 0
        done = False

        while not done:

            if verbose:
                env.render()
                time.sleep(delay)

            rewards = model.predict(transform(observation), device)
            action = argmax(rewards)
            num_actions[action] += 1
            
            observation, reward, done, info = env.step(action)

            timestep += 1
            cumulative_reward += reward

            if done:
                # print(f"Episode finished after {timestep} timesteps.  Reward: {cumulative_reward}")
                reward_history.append(cumulative_reward)
                break
    
    mean = np.mean(reward_history)
    std = round(np.std(reward_history), 5)
    print(f"{mean}±{std}   ({mean-std}, {mean+std})")
    print(num_actions)



def train(learning_rate=0.01, momentum=0, explore_prob=0.1, discount=0.99, memory_size=10, batch_size=1, mini_batch_size=32, num_episodes=1000):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on:  ", device)

    env = gym.make('Acrobot-v1')

    model = AcrobotModel(max_size=memory_size, batch_size=batch_size, mini_batch_size=mini_batch_size).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.0001)
    criterion = torch.nn.MSELoss()

    with open("./rewards_log.csv", "w") as file:
        file.write("episode,cumulative_reward\n")
    with open("./mean_rewards.csv", "w") as file:
        file.write("episode,mean_reward\n")
    with open("./loss_log.csv", "w") as file:
        file.write("iteration,loss\n")
        
    cumulative_rewards = []
    max_reward = -inf
    iteration = 0
    for episode in range(num_episodes):
        timestep = 0
        state = env.reset()
        cumulative_reward = 0
        done = False

        short_term = Memory(None, buckets=True)

        while not done:
            prev_state = transform(state)
            if np.random.rand(1) < explore_prob*(num_episodes-episode)/num_episodes:
                action = env.action_space.sample()
            else:
                rewards = model.predict(prev_state, device)
                # try:
                action = argmax(rewards)
                # except Exception as e:
                #     print()
                #     print(e)
                #     print(rewards)
                #     exit(1)
            
            state, reward, done, info = env.step(action)

            timestep += 1
            iteration += 1
            cumulative_reward += reward
            short_term.remember([prev_state, action, reward, done, transform(state)])

            loss = model.experience_replay(optimizer, criterion, device)

            if done:

                # discounted_reward = 0
                # for i in range(len(short_term)-1, -1, -1):
                #     discounted_reward = discount * discounted_reward + short_term[i][2]

                #     short_term[i][2] = discounted_reward

                # print(np.array([short_term[i][2] for i in range(len(short_term))]))

                print(f"\rEpisode: {episode}".ljust(20), end="")
                # if cumulative_reward > 195:
                #     print("     Goal Reached!".ljust(20))

                with open("./rewards_log.csv", "a") as file:
                    file.write(f"{episode},{cumulative_reward}\n")

                with open("./loss_log.csv", "a") as file:
                    file.write(f"{iteration},{loss}\n")

                cumulative_rewards.append(cumulative_reward)
                remember = True
                if len(cumulative_rewards) >= 100:
                    mean = np.mean(cumulative_rewards[-100:])
                    print("Mean: ", mean, end="")
                    if mean > max_reward:
                        model.save_weights("./best_acrobot_model")
                        max_reward = mean
                    with open("./mean_rewards.csv", "a") as file:
                        file.write(f"{episode},{mean}\n")
                    if cumulative_reward < (mean-np.std(cumulative_rewards[-100:])):
                        remember = False

                if remember:
                    model.memory.remember(short_term.memory)

    
    # print(model.predict(short_term.memory[0][0], device))


    model.save_weights("./trained_acrobot_model")
    print("\r", end="")

if __name__ == "__main__":
    train(
        learning_rate=0.001, momentum=0.9, 
        explore_prob=0.25, discount=0.95, 
        memory_size=150, batch_size=1, mini_batch_size=64, 
        num_episodes=10000)
    # test(iterations=1, verbose=True, delay=0.0025)
    # random_test()