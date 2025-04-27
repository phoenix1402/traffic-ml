import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MultiTrafficLightWrapper(gym.Wrapper):
    """
    Wrapper to handle multiple traffic lights in SumoEnvironment.
    Makes the environment compatible with Stable Baselines 3.
    """
    def __init__(self, env):
        super().__init__(env)
        
        #set up observation and action spaces based on traffic light count
        self._setup_spaces()
        
        self._needs_reset = True
        
        self.traffic_signal_ids = getattr(env, 'traffic_signal_ids', [])
        self.traffic_lights = getattr(env, 'traffic_lights', {})
        self.traci = getattr(env, 'traci', None)
    
    def _setup_spaces(self):
        #handle observation space conversion
        if isinstance(self.observation_space, spaces.Dict):
            #check if this is a multi-traffic light env or single traffic light
            if any(isinstance(space, spaces.Dict) for space in self.observation_space.spaces.values()):
                # Multi-traffic light with nested Dicts - convert to single Dict
                self.observation_space = self._flatten_observation_space()

        if isinstance(self.action_space, spaces.Dict):
            #get number of actions for each traffic light
            n_actions = []
            for space in self.action_space.spaces.values():
                n_actions.append(space.n)
            
            #convert to MultiDiscrete
            self.action_space = spaces.MultiDiscrete(n_actions)
    
    def _flatten_observation_space(self):
        #get lane data shape from first traffic light
        first_ts_id = list(self.observation_space.spaces.keys())[0]
        first_ts_space = self.observation_space.spaces[first_ts_id]
        
        #get dimensions from the first traffic light
        lane_shape = first_ts_space.spaces['lane_data'].shape
        
        #create a flattened space with increased lane capacity
        num_traffic_lights = len(self.observation_space.spaces)
        
        return spaces.Dict({
            'lane_data': spaces.Box(
                low=0, high=1.0, 
                shape=(lane_shape[0] * num_traffic_lights, lane_shape[1]),
                dtype=np.float32
            ),
            'current_phases': spaces.Box(
                low=0, high=1.0, 
                shape=(num_traffic_lights,),
                dtype=np.float32
            ),
            'time_since_change': spaces.Box(
                low=0, high=1.0, 
                shape=(num_traffic_lights,),
                dtype=np.float32
            )
        })
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._needs_reset = False
        return self._process_observation(obs), info
    
    def step(self, action):
        if self._needs_reset:
            return self.reset()
        
        if isinstance(self.env.action_space, spaces.Dict) and isinstance(action, (list, np.ndarray)):
            dict_action = {}
            for i, ts_id in enumerate(self.env.traffic_signal_ids):
                dict_action[ts_id] = int(action[i])
            action = dict_action
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated or truncated:
            self._needs_reset = True
            
        #process reward
        if isinstance(reward, dict):
            info['traffic_light_rewards'] = reward.copy()
            reward = sum(reward.values())
            
        return self._process_observation(obs), reward, terminated, truncated, info
    
    def _process_observation(self, obs):
        #handle single traffic light case
        if not isinstance(obs, dict) or not any(isinstance(v, dict) for v in obs.values()):
            return obs
            
        #for multiple traffic lights, flatten observations
        all_lane_data = []
        all_phases = []
        all_times = []
        
        for ts_id, ts_obs in obs.items():
            #add lane data
            all_lane_data.append(ts_obs['lane_data'])
            
            #process phase
            if isinstance(ts_obs['current_phase'], np.ndarray):
                phase_val = ts_obs['current_phase'][0]
            else:
                phase_val = float(ts_obs['current_phase'])
                
            all_phases.append(phase_val)
            
            #process time since change
            if isinstance(ts_obs['time_since_change'], np.ndarray):
                time_val = float(ts_obs['time_since_change'][0])
            else:
                time_val = float(ts_obs['time_since_change'])
                
            all_times.append(time_val)
            
        combined_lane_data = np.vstack(all_lane_data)
        
        return {
            'lane_data': combined_lane_data,
            'current_phases': np.array(all_phases, dtype=np.float32),
            'time_since_change': np.array(all_times, dtype=np.float32)
        }