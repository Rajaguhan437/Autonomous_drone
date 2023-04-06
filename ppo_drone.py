#import setup_path
import gym
import airgym
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
#from stable_baselines3.common.policies import MlpPolicy, CnnPolicy
from stable_baselines3.ppo import MlpPolicy,CnnPolicy

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#from airgym.envs.airsim_env import AirSimEnv
#from airgym.envs.car_env import AirSimCarEnv
from airgym.envs.drone_env import AirSimDroneEnv

from gym.envs.registration import register

register(
    id='airsim-drone-sample-v0',
    entry_point='airgym.envs.drone_env:AirSimDroneEnv',
    max_episode_steps=100,
)





# Create a DummyVecEnv for main airsim gym env
model_name = "drone_ppo_1"
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airsim-drone-sample-v0",
                ip_address="127.0.0.1",
                step_length=2,
                image_shape=(84, 84, 1),
                destination=(-5, -10, -10, 10)
            ),"G:/Drone/Model/"+model_name
        )
    ]
)
# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = PPO(
    CnnPolicy,
    env,
    learning_rate=0.0003,
    verbose=1,
    batch_size=64,
    ent_coef=0.0,
    n_epochs=10,
    gamma=0.999,
    n_steps=64,
    device="cuda",
    tensorboard_log="G:/Drone/Model/tb_logs/"+model_name,
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=10000,
)


kwargs = {}
kwargs["callback"] = callbacks
kwargs["progress_bar"] = True
kwargs["log_interval"] = 4
kwargs["total_timesteps"] = 10
# Train for a certain number of timesteps
model.learn(
    tb_log_name="ppo_airsim_drone_run_" + str(time.time()),
    **kwargs
)

# Save policy weights
model.save("G:/Drone/Model"+model_name)



env.reset()
obs_cart_vel=[]
obs_pole_vel=[]
for i in range(10):
    action = env.action_space.sample()
    obs,reward,done,info = env.step(action)
    env.render()
    time.sleep(.01)
    if done == True:
        break
env.close()
