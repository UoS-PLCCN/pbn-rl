import pickle
from stable_baselines3 import PPO

with open("envs/n_28.pkl", "rb") as f:
    env = pickle.load(f)

model = PPO("MlpPolicy", env, verbose=1).learn(50_000)
