#import setup_path
import gym
import airgym
import time
import os

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
#from stable_baselines3.common.policies import ActorCriticCnnPolicy
#from stable_baselines3.ppo import MlpPolicy,CnnPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure




from datetime import datetime
# timestamp is number of seconds since 1970-01-01 
timestamp = 1545730073
# convert the timestamp to a datetime object in the local timezone
dt_object = datetime.fromtimestamp(timestamp)
# print the datetime object and its type
print("dt_object =", dt_object)



models_dir = "G:/Drone/Drone_Agent/models"
log_dir = "G:/Drone/Drone_Agent/logs"

if not os.path.exists(models_dir):
    os.mkdir(models_dir)
    
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

#configure()
new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])    
    
env = gym.make(
        "airsim-drone-sample-v0",
        ip_address="127.0.0.1",
        step_length=1,
        image_shape=(84, 84, 1),
        destination=(-5, -10, -10, 10)
)# Create a DummyVecEnv for main airsim gym env

check_env(env, warn=True, skip_render_check=True)

env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda:env])
# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)


# Parallel environments
#env = make_vec_env("CartPole-v1", n_envs=4)
import logging

logging.basicConfig(level=logging.INFO)


# Initialize RL algorithm type and parameters
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.00025,
    verbose=2,
    batch_size=32,
    train_freq=4,
    target_update_interval=10000,
    learning_starts=10000,
    buffer_size=200000,
    max_grad_norm=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log=log_dir
)

# Create an evaluation callback with the same env, called every 10000 iterations
#callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=models_dir+"/best_models/",
    log_path=log_dir,
    eval_freq=1000,
)
#callbacks.append(eval_callback)


model.set_logger(new_logger)

timesteps = 10
steps = 0

kwargs = {}
kwargs["callback"] = eval_callback
kwargs["progress_bar"] = True
kwargs["reset_num_timesteps"] = False
kwargs["total_timesteps"] = timesteps
kwargs["tb_log_name"] = "dqn_drone"+"_"+str(steps),
# Train for a certain number of timesteps  
for i in range(1,3):
    
    model.learn(
        log_interval=5,
        **kwargs
    )
    steps = steps + timesteps
    
    model.save(models_dir+"/DQN_Drone_"+str(steps))



model_path = models_dir+"/DQN_Drone_"+str(steps)

def play(path):
    model = DQN.load(path, env=env, device="cuda")
    eps = 1
    for i in range(eps):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
    env.close()
play(model_path)

