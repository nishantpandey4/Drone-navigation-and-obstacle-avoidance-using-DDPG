from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback,EvalCallback
import numpy as np
import torch as th
import time
import os
from drone_env_ddpg import AirSimDroneEnv
test=AirSimDroneEnv()

shape=test.action_space.shape[-1]
noise_sigma = 0.1 * np.ones(shape)
noise = NormalActionNoise(mean=np.zeros(shape),
                                             sigma=noise_sigma)


policy_kwargs = dict(activation_fn=th.nn.Tanh)
policy_kwargs['net_arch'] = [64, 32, 16]

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)
eval_log_dir = "./eval_logs/"
os.makedirs(eval_log_dir, exist_ok=True)

class savemodelcallback(BaseCallback):

    def __init__(self, check_freq: int, name=None, verbose=1):
        super(savemodelcallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = os.path.join("./models/", str(time.strftime("%Y%m%d-%H%M%S"))+ "_ddpg_airsim_drone_callback")


    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # print(self.n_calls)
        if self.n_calls % self.check_freq == 0:
            print("************************************************************** \
                  **************************************************************")
            print("Saving new best model to {}".format(self.save_path))
            self.model.save(self.save_path)

        return True

model = DDPG("MlpPolicy",
             AirSimDroneEnv(),
             policy_kwargs=  policy_kwargs,
             batch_size=1280,
             train_freq=500,
             gradient_steps=500,
             buffer_size=50000,
             learning_rate=0.001,
             learning_starts=2000,
             gamma=0.99,
             seed=0,
             action_noise=noise,
             tensorboard_log=log_dir,
             verbose=2)

name= r".\models\ "+str(time.strftime("%Y%m%d-%H%M%S"))+ "_ddpg_airsim_drone"
model.learn(callback=savemodelcallback(check_freq=5000),total_timesteps=1000,progress_bar=True,log_interval=1)
model.save(name)
