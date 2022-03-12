import gym
import slimevolleygym
import torch as th
from stable_baselines3 import PPO, DQN, HerReplayBuffer
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
# import highway_env
#%%
n_cpu = 1
batch_size = 256

# env = gym.make("SlimeVolley-v0")
env = make_vec_env("SlimeVolley-v0", n_envs=n_cpu, )#vec_env_cls=SubprocVecEnv
model = PPO("MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
        n_steps=batch_size * 24 // n_cpu,
        batch_size=batch_size,
        n_epochs=5,
        learning_rate=5e-4,
        gamma=0.95,
        gae_lambda=0.95,
        verbose=2,
        tensorboard_log="logs/")
#%%
# Train the agent
model.learn(total_timesteps=int(1e8), tb_log_name="slimevol_gamma95")
# Save the agent
model.save("slime_PPO/model")
#%%
model.learning_rate = 1E-4
model._setup_lr_schedule()
model.learn(total_timesteps=int(5e7), tb_log_name="slimevol_gamma95", reset_num_timesteps=False)
#%%
model.ent_coef = 0.005
model.learning_rate = 1E-4
model._setup_lr_schedule()
model.gamma = 0.998
model.gae_lambda = 0.99
# Train the agent
for i in range(20):
    model.learn(total_timesteps=int(5e6), tb_log_name="slimevol_gamma95_gae95_high_G", reset_num_timesteps=False)
    model.save(f"slime_PPO/model_gamma95_gae95_high_G_{int(model.num_timesteps//1E6):d}M")
#%%
#%%
n_cpu = 1
batch_size = 256

# env = gym.make("SlimeVolley-v0")
env = make_vec_env("SlimeVolley-v0", n_envs=n_cpu, )#vec_env_cls=SubprocVecEnv
model = PPO("MlpPolicy", env,
        policy_kwargs=dict(net_arch=[dict(pi=[32, 16], vf=[32, 16])]),
        n_steps=batch_size * 24 // n_cpu,
        batch_size=batch_size,
        n_epochs=5, learning_rate=5e-4,
        gamma=0.90, gae_lambda=0.90,
        verbose=2, tensorboard_log="logs/")
#%%
for i in range(20):
    model.learn(total_timesteps=int(5e6), tb_log_name="slimevol_gamma90_gae90", reset_num_timesteps=False)
    model.save(f"slime_PPO/model_gamma90_gae90_{int(model.num_timesteps//1E6):d}M")
#%%
for i in range(20):
    model.learn(total_timesteps=int(5e6), tb_log_name="slimevol_gamma998_gae99", reset_num_timesteps=False)
    model.save(f"slime_PPO/model_gamma998_gae99_{int(model.num_timesteps//1E6):d}M")
#%%
n_cpu = 1
batch_size = 256

# env = gym.make("SlimeVolley-v0")
env = make_vec_env("SlimeVolley-v0", n_envs=n_cpu, )#vec_env_cls=SubprocVecEnv
model = PPO("MlpPolicy", env,
        policy_kwargs=dict(net_arch=[dict(pi=[32, 16], vf=[32, 16])]),
        n_steps=batch_size * 24 // n_cpu,
        batch_size=batch_size,
        n_epochs=5, learning_rate=5e-4,
        gamma=0.95, gae_lambda=0.95,
        verbose=2, tensorboard_log="logs/")
#%%
for i in range(20):
    model.learn(total_timesteps=int(5e6), tb_log_name="slimevol_smallNet", reset_num_timesteps=False)
    model.save(f"slime_PPO/slimevol_smallNet_{int(model.num_timesteps//1E6):d}M")

#%%
class MultiBinaryAsDiscreteAction(gym.ActionWrapper):
    """Transforms MultiBinary action space to Discrete.
    If the action space of a given env is `gym.spaces.MultiBinary(n)`, then
    the action space of the wrapped env will be `gym.spaces.Discrete(2**n)`,
    which covers all the combinations of the original action space.
    Args:
        env (gym.Env): Gym env whose action space is `gym.spaces.MultiBinary`.
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        self.orig_action_space = env.action_space
        self.action_space = gym.spaces.Discrete(2 ** env.action_space.n)

    def action(self, action):
        return [(action >> i) % 2 for i in range(self.orig_action_space.n)]


class DictObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_observation_space = env.observation_space
        self.observation_space = gym.spaces.Dict({"observation": env.observation_space,
                                                  "achieved_goal": gym.spaces.MultiBinary([1])})

    def observation(self, obs):
        # modify obs
        return {"observation": obs, "achieved_goal": 1}
#%%
n_cpu = 1
batch_size = 256
max_episode_length = 3000
# online_sampling = True
# goal_selection_strategy = 'future'
env = gym.make("SlimeVolley-v0", )
# env.atari_mode = True
# env.action_space = gym.spaces.Discrete(6)
env = MultiBinaryAsDiscreteAction(env, )
# env = DictObservationWrapper(MultiBinaryAsDiscreteAction(env))
# venv = make_vec_env(env, n_envs=n_cpu, )#vec_env_cls=SubprocVecEnv
model = DQN("MlpPolicy", env, buffer_size=1000000, learning_starts=50000,
        policy_kwargs=dict(net_arch=[32, 16]),
        train_freq=(5, "episode"), batch_size=256,
        learning_rate=5e-4, gamma=0.95,
        exploration_fraction=0.5,
        # replay_buffer_class=HerReplayBuffer,
        # # Parameters for HER
        # replay_buffer_kwargs=dict(
        #     n_sampled_goal=4,
        #     goal_selection_strategy=goal_selection_strategy,
        #     online_sampling=online_sampling,
        #     max_episode_length=max_episode_length,
        # ),
        verbose=2, tensorboard_log="logs/")
#%%
for i in range(20):
    model.learn(total_timesteps=int(5e6), tb_log_name="slimevol_DQN_smallNet", reset_num_timesteps=False)
    model.save(f"slime_DQN/slimevol_smallNet-DQN_{int(model.num_timesteps//1E6):d}M")
#%%
import gym
from stable_baselines3 import PPO
import slimevolleygym

env = gym.make("SlimeVolley-v0")
model = PPO.load("slime_PPO/model_gamma95_gae95_5M.zip")
model.set_env(env)
#%%
obs = env.reset()
done = False
total_reward = 0

while not done:
  action, _ = model.predict(obs, deterministic=True)
  obs, reward, done, info = env.step(action)
  total_reward += reward
  env.render()

print("score:", total_reward)
#%%
obs1 = env.reset()
obs2 = obs1 # both sides always see the same initial observation.

done = False
total_reward = 0

while not done:

  action1, _ = model.predict(obs1)
  action2, _ = model.predict(obs2)

  obs1, reward, done, info = env.step(action1, action2) # extra argument
  obs2 = info['otherObs']

  total_reward += reward
  env.render()

print("policy1's score:", total_reward)
print("policy2's score:", -total_reward)
