#implementation of the traffic environment using SUMO and traci

import traceback
import gymnasium as gym
import traci
from gymnasium.utils import EzPickle

from src.environment.traffic_light_update import TrafficLights
from src.environment.observation_processor import Observation

class SumoEnvironment(gym.Env, EzPickle):
    def __init__(
        self,
        net_file=None,
        route_file=None,
        config_file=None,
        use_gui=False,
        num_seconds=3600,
        max_green=40,
        min_green=10,
        yellow_time=3,
        reward_func="waiting_time_diff",
        render_mode=None,
        has_pedestrians=False
    ):
        
        EzPickle.__init__(self, net_file, route_file, use_gui, num_seconds, 
                         max_green, min_green, yellow_time, reward_func, render_mode)
        
        self.net_file = net_file
        self.route_file = route_file
        self.config_file = config_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.max_green = max_green
        self.min_green = min_green
        self.yellow_time = yellow_time
        self.reward_func = reward_func
        self.render_mode = render_mode
        self.has_pedestrians = has_pedestrians
    
        # Set up sumo command
        self.sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        
        if self.config_file:
            self.sumo_cmd = [
                self.sumo_binary,
                "-c", self.config_file,
                "--no-step-log", "true",
                "--random", "false",
                "--start", "true",
                "--quit-on-end", "true"
            ]
        else:
            self.sumo_cmd = [
                self.sumo_binary, 
                "-n", self.net_file,
                "-r", self.route_file,
                "--no-step-log", "true",
                "--random", "false",
                "--start", "true",
                "--quit-on-end", "true"
            ]
        
        traci.start(self.sumo_cmd)
        self.traffic_signal_ids = list(traci.trafficlight.getIDList())
        
        #create traffic light objects with pedestrian support if needed
        self.traffic_lights = {}
        for ts_id in self.traffic_signal_ids:
            self.traffic_lights[ts_id] = TrafficLights(
                ts_id, 
                yellow_time=self.yellow_time,
                max_green_time=self.max_green,
                min_green_time=self.min_green,
                reward_func=self.reward_func,
                has_pedestrians=self.has_pedestrians
            )
            
        #define observation and action spaces (for multiple traffic lights)
        self.obs_processor = Observation()
        self.define_spaces()
        
        #initial state
        self.sim_step = 0
        self.episode_reward = 0
        
        #close initial connection
        traci.close()
        
    def define_spaces(self):
        #define observation and action spaces
        if len(self.traffic_signal_ids) == 1:
            #single traffic light case
            ts = self.traffic_lights[self.traffic_signal_ids[0]]
            self.observation_space = ts.observation_space
            self.action_space = ts.action_space
        else:
            #multiple traffic lights
            self.observation_space = gym.spaces.Dict({
                ts_id: ts.observation_space 
                for ts_id, ts in self.traffic_lights.items()
            })
            self.action_space = gym.spaces.Dict({
                ts_id: ts.action_space
                for ts_id, ts in self.traffic_lights.items()
            })
            
    def reset(self, seed=None, options=None):
        #reset environment
        if seed is not None:
            super().reset(seed=seed)
        
        #start a new sumo simulation
        if traci.isLoaded():
            traci.close()
            
        try:
            traci.start(self.sumo_cmd)
            
            #check if the traffic light IDs exist
            available_tls = traci.trafficlight.getIDList()
            print(f"Available traffic lights: {available_tls}")
            
            for ts_id in self.traffic_signal_ids:
                if ts_id not in available_tls:
                    print(f"Warning: Traffic light ID {ts_id} not found in SUMO network!")
            
            #reset all traffic lights
            for ts in self.traffic_lights.values():
                ts.reset()
                
            #reset episode variables
            self.sim_step = 0
            self.episode_reward = 0
            
            #get initial observation
            obs = self.get_observation()
            info = {}
            
            return obs, info
        except Exception as e:
            print(f"Error during reset: {e}")
            traceback.print_exc()
            raise
        
    def step(self, action):
        # apply actions to traffic lights
        if action is None:
            # Default controller mode - update without specific actions
            for ts_id in self.traffic_signal_ids:
                self.traffic_lights[ts_id].update(None)
        elif len(self.traffic_signal_ids) == 1:
            # single traffic light case
            ts_id = self.traffic_signal_ids[0]
            self.traffic_lights[ts_id].update(action)
        else:
            # multiple traffic lights case
            for ts_id, act in action.items():
                self.traffic_lights[ts_id].update(act)
                
        traci.simulationStep()
        self.sim_step += 1
        
        #calculate rewards
        rewards = {}
        total_reward = 0
        for ts_id, ts in self.traffic_lights.items():
            reward = ts.calculate_reward()
            rewards[ts_id] = reward
            total_reward += reward
            
        #for single agent case to simplify reward
        if len(self.traffic_signal_ids) == 1:
            rewards = rewards[self.traffic_signal_ids[0]]
            
        #update episode reward
        self.episode_reward += total_reward
            
        #check if simulation is done
        done = self.sim_step >= self.num_seconds or traci.simulation.getMinExpectedNumber() == 0
        
        #get new observation
        obs = self.get_observation()
        
        #create info dict
        info = {
            'step': self.sim_step,
            'total_reward': self.episode_reward
        }
        
        #for gymnasium compatibility
        terminated = done
        truncated = False
            
        return obs, rewards, terminated, truncated, info
    
    def get_observation(self):
        #get observations from all traffic lights
        if len(self.traffic_signal_ids) == 1:
            # Single traffic light case
            ts_id = self.traffic_signal_ids[0]
            obs = self.traffic_lights[ts_id].get_observation()
            return self.obs_processor.normalise(obs)
        else:
            #multiple traffic lights case
            obs = {}
            for ts_id, ts in self.traffic_lights.items():
                obs[ts_id] = self.obs_processor.normalise(ts.get_observation())
            return obs
        
    def close(self):
        #close env and connection
        if traci.isLoaded():
            traci.close()