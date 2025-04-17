# Traffic Optimisation using Reinforcement Learning
This project implements an intelligent traffic light control system using reinforcement learning (RL) to optimize traffic flow. The system uses SUMO (Simulation of Urban MObility) to simulate traffic environments and trains a Proximal Policy Optimisation (PPO) agent to control traffic signals more efficiently than traditional fixed-time controllers.

## Features
Train reinforcement learning agents to control traffic lights
Run simulations with either trained models or default SUMO controllers
Collect and visualize performance metrics
Compare performance between RL models and traditional controllers

## Requirements
- Python 3.8+
- SUMO traffic simulator 1.8+
- Dependencies listed in src/requirements.txt:
    - stable-baselines3
    - gymnasium
    - numpy
    - matplotlib
    - pandas

## Installation
1. Install SUMO following the [official installation guide](https://sumo.dlr.de/docs/Downloads.php)
2. Clone this repository
3. Install Python dependencies:
```pip install -r requirements.txt```

## Training
To train a new agent:
```python -m src.agent.training```
The trained model will be saved to [src/environment/sumo_runner.py](src/environment/sumo_runner.py)

## Running Simulations
Run a simulation with default phase timings:
```python -m src.environment.sumo_runner --net src/networks/2lane_junc/single.net.xml --route src/networks/2lane_junc/single_horizontal.rou.xml --default --steps 8000```
Run a simulation with a trained model:
```python -m src.environment.sumo_runner --net src/networks/2lane_junc/single.net.xml --route src/networks/2lane_junc/single_horizontal.rou.xml --model src/agent/ppo_traffic_light_model.zip --steps 8000```

Command line options:
- `--net`: Path to SUMO .net.xml file
- `--route`: Path to SUMO .rou.xml file
- `--model`: Path to trained model .zip file
- `--steps`: Simulation duration in steps
- `--no-gui`: Run without SUMO GUI
- `--default`: Use default SUMO traffic light controller

## Compare Results
After running simulations with both the trained model and default controller:
```python -m src.compare_results --model ppo_traffic_light_model```
This will generate comparative visualizations and statistics in the [graphs](src/graphs) directory.

## Results
After running simulations, you will find results in the [graphs](src/graphs) directory:
- CSV files with metrics data
- Performance plots showing waiting time, queue length, average speed, and rewards
- Comparative visualizations between models

## Agent Hyperparameters
To train with custom parameters, modify [training.py](src/agent/training.py).
You can adjust:
- Learning rate
- Batch size
- Neural network architecture
- Reward function components

To create custom traffic scenarios, add new network files to the [networks](src/networks) directory and refer to the SUMO documentation for network creation.