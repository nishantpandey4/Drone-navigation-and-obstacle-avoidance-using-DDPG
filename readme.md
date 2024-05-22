# Drone Obstacle Avoidance with AirSim and DDPG

## Authors
- Nishant Pandey
- Jayasuriya Suresh
## Overview

This project implements a drone obstacle avoidance system using AirSim and the Deep Deterministic Policy Gradient (DDPG) algorithm. The goal is to train a drone to navigate through an environment while avoiding obstacles in real-time.

## Pacakges Used
- stable-baselines3 v1.7.0 (pip install stable-baselines3[extra]==1.7.0)
- airsim v1.8.1
- gym 0.21.0
- Packages as required by the airsim package.
A requirements.txt has been attached in case of versions mismatches or the conda env fails to be imported.
## Features

- Utilizes the AirSim simulator for realistic drone flight dynamics and sensor data.
- Implements the DDPG algorithm for training the drone to avoid obstacles.
- Provides a user-friendly interface for  visualizing the obstacle avoidance behavior.


## Contents
There are three folders, lidar, depth , lidar+depth which have train.py, eval.py and drone_env.py. The instructions to run all have been documented below.
Other folders and files are :
- readme.md
- env.yml
- requirements.txt
- setting.json

## Installation

1. Unzip the folder if you have not done so

2. Install the required packages from the env.yml file using conda
    ```shell
    conda env create -f env.yml
    ```

3. Download and setup the AirSim simulator by following the instructions in the [AirSim documentation](https://microsoft.github.io/AirSim/).
For this project we used the biniaries provided by airsim under [releases](https://github.com/microsoft/AirSim/releases) , V1.8.1 , [AirsimNH.zip](https://github.com/microsoft/AirSim/releases/download/v1.1.10/AirSimNH.zip).Extract the zip and run the airsimnh.exe for the simulator to work. Also paste the provided config file under Documents/Airsim/settings.json

## Usage

1. Launch the AirSim simulator.

2. Navigate to Lidar/Lidar+depth folder, run the main script to start the drone obstacle avoidance system:

    ```shell
    python train.py
    ```

3. The drone will start training. Once training is over, it will be saved under /models of the root dir. Logs are saved under /tmp of the root dir which can be viewed using

```shell
tensorboard --logdir=/tmp/name_of_folder
```
## Evaluation
1. Launch the AirSim simulator.

2. Modify the model_path in the eval.py file to the path of your model. Run the main script to start the drone obstacle avoidance system:

    ```shell
    python  eval.py
    ```

3. Evaluation starts and you can see the output in the airsim window. After evaluation is done, we can see the metrics which show up in a matplotlib window and also is saved in the root directory.

## Bugs
- If the code does not connect to the simulator, please change the port in settings.json under ApiServerPort as well as on the code , in the files drone_env_ddpg.py and ddpg_lidar.py in line number 19 for both eval and the normal files.
## Acknowledgements

- [AirSim](https://microsoft.github.io/AirSim/) - Open-source simulator for drones and cars developed by Microsoft.
- [DDPG](https://arxiv.org/abs/1509.02971) - Deep Deterministic Policy Gradient algorithm for continuous control tasks.

-  [UAV_Navigation_DRL_AirSim](https://github.com/heleidsn/UAV_Navigation_DRL_AirSim/tree/main) for opensourcing his code as it helped us understand a lot and we have used parts of his code in our codebase.

- [DDPG-AirSim-Drone-Obstacle-Avoidance](https://github.com/John-Venti/DDPG-AirSim-Drone-Obstacle-Avoidance) for helping us understand DDPG and how to control a drone in airsim using RL.

- [Reinforcement-learning](https://github.com/RahulSajnani/Reinforcement-learning)
