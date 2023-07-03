from stable_baselines3.common.env_checker import check_env
from env import ImageEnv
from tracking import track
import matplotlib.pyplot as plt
import numpy as np
path = "/home/carlos/Desktop/Decimo/Titulacion_II/Algorithms/PPO/train/" 
env = ImageEnv()
obs = env.reset()
#print(env.observation_space)
#print(obs.shape)
#If the environment do not follow the interface, an error will be thrown
check_env(env, warn = True)
episodes = 20
for episodes in range(1, episodes+1):
    obs = env.reset()
    done = False
    score= 0
    while True:
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward
        # Render the centerer
        env.render()
        if done == True:
            break
    track(env.center, env.getpos, path, episodes)
    print('Score {0:.1f} '. format(score))
    env.close()
