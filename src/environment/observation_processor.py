import numpy as np

class Observation:
    def __init__(self, max_vehicles=30, max_speed=15.0, max_wait_time=300.0):
        self.max_vehicles = max_vehicles
        self.max_speed = max_speed
        self.max_wait_time = max_wait_time
    
    def normalise(self, observation):
        #normalise all observation values to [0,1] range
        norm_obs = observation.copy()
        
        #normalise lane data
        lane_data = observation['lane_data']
        normalised_data = np.zeros_like(lane_data, dtype=np.float32)
        
        #for each lane, normalise metrics
        for i, lane in enumerate(lane_data):
            # [vehicle_count, mean_speed, occupancy, waiting_time]
            normalised_data[i, 0] = min(1.0, lane[0] / self.max_vehicles)
            normalised_data[i, 1] = min(1.0, lane[1] / self.max_speed)
            normalised_data[i, 2] = lane[2]
            normalised_data[i, 3] = min(1.0, lane[3] / self.max_wait_time)
            
        norm_obs['lane_data'] = normalised_data
        
        #normalise time since change (as fraction of max_green_time)
        norm_obs['time_since_change'] = np.array([
            min(1.0, observation['time_since_change'][0] / 60.0)
        ], dtype=np.float32)
        
        return norm_obs