import sys
sys.path.append("../")

import torch
import torch.nn.functional as F
from model import *
from Trainer import *

from random import shuffle
import time

import numpy as np
import gym


class CartPoleModel(BaseModel):

    def __init__(self, state_dict_path=None, verbose=False):
        super(CartPoleModel, self).__init__()

        self.fc1 = torch.nn.Linear(4, 8).double()
        self.fc2 = torch.nn.Linear(8, 2).double()

        if state_dict_path is not None:
            self.load_weights(state_dict_path)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def transform(x):
    return np.array(x)


def test(iterations=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Testing on:  ", device)

    reward_history = []

    env = gym.make('CartPole-v0')

    model = CartPoleModel("./trained_cartpole_model").to(device)

    num_actions = [0, 0]

    for iteration in range(iterations):
        timestep = 0
        observation = env.reset()
        cumulative_reward = 0
        done = False

        while not done:

            env.render()
            time.sleep(0.05)

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



def train(num_episodes, explore_prob, average=100, **kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training on:  ", device)

    env = gym.make('CartPole-v0')

    model = CartPoleModel().to(device)
    trainer = Trainer(model, device, **kwargs)

    with open("./rewards_log.csv", "w") as file:
        file.write("episode,cumulative_reward\n")
    with open("./mean_rewards.csv", "w") as file:
        file.write("episode,mean_reward\n")
        
    cumulative_rewards = []
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
                action = argmax(rewards)
            
            state, reward, done, info = env.step(action)
            reward = reward if not done else -reward

            timestep += 1
            iteration += 1
            cumulative_reward += reward
            short_term.remember([prev_state, action, reward, done, transform(state)])

            loss = trainer.experience_replay()

            if done:

                # discounted_reward = 0
                # for i in range(len(short_term)-1, -1, -1):
                #     discounted_reward = discount * discounted_reward + short_term[i][2]

                #     short_term[i][2] = discounted_reward

                print(f"\rEpisode: {episode}".ljust(20), end="")

                with open("./rewards_log.csv", "a") as file:
                    file.write(f"{episode},{cumulative_reward}\n")

                cumulative_rewards.append(cumulative_reward)
                remember = True
                if len(cumulative_rewards) >= average:
                    mean = np.mean(cumulative_rewards[-average:])
                    if mean > 195:
                        print("\nGoal Reached!\n")
                        model.save_weights("./trained_cartpole_model")
                        return
                    else:
                        print("Mean: ", mean, end="")
                        with open("./mean_rewards.csv", "a") as file:
                            file.write(f"{episode},{mean}\n")
                        if cumulative_reward < (mean-0.5*np.std(cumulative_rewards[-average:])):
                            remember = False

                if remember:
                    trainer.memory.remember(short_term.memory)


    model.save_weights("./trained_cartpole_model")
    print("\r", end="")

if __name__ == "__main__":
    train(**{
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.00001,
        "explore_prob": 0.25,
        "discount": 0.95,
        "max_memory_size": 150,
        "batch_size": 1,
        "mini_batch_size": 64,
        "num_episodes": 10000
    })
    test(iterations=100)

    '''
    Training on:   cpu
    Episode: 285       Mean:  194.99
    Goal Reached!

    Training on:   cpu
    200.0±0.0   (200.0, 200.0)
    [10047, 9953]
    '''