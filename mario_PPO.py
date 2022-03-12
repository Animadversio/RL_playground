from stable_baselines3 import PPO
import gym
import gym_super_mario_bros
import torch
#%
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.vec_env import VecNormalize
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# VecNormalize()

# done = True
# for step in range(5000):
#     if done:
#         state = env.reset()
#     state, reward, done, info = env.step(env.action_space.sample())
#     env.render()
#
# env.close()
#%%
model = PPO("CnnPolicy", env, learning_rate=0.0003, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.995, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
    verbose=2, tensorboard_log="logs")
#%%
model.learn(total_timesteps=1E5, tb_log_name="mario_pilot", reset_num_timesteps=False)
#%%
model.save(f"ckpt\\mario_pilot_{model.num_timesteps//1000:d}K")
#%%
model.learning_rate = 5E-5
# model.clip_range = get_schedule_fn(0.10)
model._setup_lr_schedule()
for i in range(40):
    model.learn(total_timesteps=1E5, tb_log_name="mario_pilot", reset_num_timesteps=False)
    model.save(f"ckpt\\mario_pilot_{model.num_timesteps//1000:d}K")
#%%
from stable_baselines3.common.env_checker import check_env
check_env(env)
#%%
model = PPO.load("ckpt/mario_pilot_4003K.zip")

#%%
import numpy as np
done = False
env.render()
# obs = env.reset()
cumrew = 0
for step in range(10000):
    if done:
        obs = env.reset()
        print(cumrew)
        cumrew = 0

    action, _ = model.predict(np.array(obs))
    obs, reward, done, info = env.step(action)
    cumrew += reward
    env.render()
#%%
model = PPO("CnnPolicy", env, learning_rate=0.0003, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.998, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
    verbose=2, tensorboard_log="logs")
#%%
# model.gamma = 0.998
# model.ent_coef = 0.01
# model.learning_rate = 3E-4
# model._setup_lr_schedule()
# model.clip_range = get_schedule_fn(0.10)
for i in range(40):
    model.learn(total_timesteps=1E5, tb_log_name="mario_ent001_gamma998_1", reset_num_timesteps=False)
    model.save(f"ckpt\\mario_ent001_gamma998_{model.num_timesteps//1000:d}K")
# env.close()
#%%
model = PPO("CnnPolicy", env, learning_rate=0.0003, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.95, gae_lambda=0.90, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
    verbose=2, tensorboard_log="logs")
#%%
# model.gamma = 0.998
# model.ent_coef = 0.01
# model.learning_rate = 3E-4
# model._setup_lr_schedule()
# model.clip_range = get_schedule_fn(0.10)
for i in range(40):
    model.learn(total_timesteps=1E5, tb_log_name="mario_ent001_gamma95_gae90_1", reset_num_timesteps=False)
    model.save(f"ckpt\\mario_ent001_gamma95_gae90_{model.num_timesteps//1000:d}K")
# env.close()
#%%
model = PPO("CnnPolicy", env, learning_rate=0.0003, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
    verbose=2, tensorboard_log="logs")
#%%
# model.gamma = 0.998
# model.ent_coef = 0.01
# model.learning_rate = 3E-4
# model._setup_lr_schedule()
# model.clip_range = get_schedule_fn(0.10)
for i in range(40):
    model.learn(total_timesteps=1E5, tb_log_name="mario_ent001_gamma99_gae95_1", reset_num_timesteps=False)
    model.save(f"ckpt\\mario_ent001_gamma99_gae95_{model.num_timesteps//1000:d}K")
#%%
from stable_baselines3 import HerReplayBuffer, SAC, DQN
# from sb3_contrib import TQC
# env = gym.make("parking-v0")
her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future',
                  online_sampling=True, max_episode_length=500,)
# You can replace TQC with SAC agent
# model = SAC('CnnPolicy', env, replay_buffer_class=HerReplayBuffer,
#             replay_buffer_kwargs=her_kwargs, buffer_size=int(1e6),
#             learning_rate=1e-3,
#             gamma=0.95, batch_size=256, tau=0.05,
#             policy_kwargs=dict(net_arch=[512, 512, 512]),
#             verbose=2,
#             tensorboard_log="logs")
model = DQN('CnnPolicy', env, learning_rate=0.0001,
            buffer_size=int(1e5), learning_starts=50000,
            batch_size=128, tau=1.0, gamma=0.99, train_freq=4,
            gradient_steps=1, #replay_buffer_class=None,  replay_buffer_kwargs=her_kwargs,
            # policy_kwargs=dict(net_arch=[512, 512, 512]),
            optimize_memory_usage=False, target_update_interval=10000, exploration_fraction=0.1,
            exploration_initial_eps=1.0, exploration_final_eps=0.05, max_grad_norm=10,
            tensorboard_log="logs",
            verbose=2, seed=None, device='auto',)
#%%
model.batch_size = 128
for i in range(50):
    model.learn(int(5e5), tb_log_name="mario_vanillaDQN", reset_num_timesteps=False)
    model.save(f"ckpt\\mario_vanillaDQN_{model.num_timesteps / 1E3:.0f}K")
#%%
from stable_baselines3 import HerReplayBuffer, SAC, DQN

model = DQN.load(f"ckpt\\mario_vanillaDQN_85K")
model.set_env(env)
