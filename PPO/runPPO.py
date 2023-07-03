from env import ImageEnv
from tracking import track
from stable_baselines3.ppo.ppo import PPO
import matplotlib.pyplot as plt
import numpy as np

path = "/home/carlos/Documents/Decimo/Titulacion_II/Algorithms/PPO/retraining/E/11M3/"#+str(i)+"/"
model = PPO.load(path + "best_model_3900000.zip")
env = ImageEnv()
episodes = 4
for episodes in range(1, episodes+1):
    obs = env.reset()
    done = False
    score= 0
    while True:
        # Take a random action
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
        # Render the centerer
        env.render()
        if done == True:
            break
    # track(env.center, env.getpos, path, episodes)
    print('Score {0:.1f} '. format(score))
    env.close()

