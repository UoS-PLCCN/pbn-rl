import pickle
from stable_baselines3 import PPO
from eval import compute_ssd_hist
from stable_baselines3.common.logger import configure

with open("envs/n_28.pkl", "rb") as f:
    env = pickle.load(f)

logger = configure("logs", ["csv"])

# Model
# model = PPO("MlpPolicy", env, verbose=1)
# model.logger(logger)
# model.learn(50_000)

vanilla_ssd = compute_ssd_hist(env, model=None, output="images/n28_vanilla.png")
with open(f"logs/{env.name}_ssd_vanilla.csv", "w") as f:
    vanilla_ssd.to_csv(f)

# model_ssd = ssd_eval(env, model)
# with open("logs/{env.name}_ssd.csv", "w") as f:
#     model_ssd.to_csv(f)
