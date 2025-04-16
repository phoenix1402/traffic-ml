# updates the traffic light phase based on the action taken by the agent

import traci
import gymnasium as gym
import numpy as np

class TrafficLights:
    def __init__(self, signal_id, yellow_time=2, max_green_time=40, min_green_time=5, reward_func="waiting_time_diff"):
        #TrafficLights manages a single traffic light in the sumo environment.
    
        #traffic light parameters
        self.id = signal_id #sumo traffic light id
        self.yellow_time = yellow_time #duration of yellow phase in seconds
        self.max_green_time = max_green_time #maximum green phase duration
        self.min_green_time = min_green_time #minimum green phase duration
        self.reward_func = reward_func #type of reward function to use
        self.last_wait_time = 0
        
        #define action and observation spaces for this traffic light
        self.action_space = gym.spaces.Discrete(2)
        
        self.observation_space = None
        self.initialise_phases()
        
    def initialise_phases(self):
        #initialise traffic light phases from sumo network data
        program_logics = traci.trafficlight.getAllProgramLogics(self.id) #get all phases from sumo
        if not program_logics:
            raise ValueError(f"No program logics found for traffic light {self.id}. Check your network file.")
                
        self.all_phases = program_logics[0].phases
        
        #get the green phases (exclude yellow/red phases)
        self.green_phases = []
        for i, phase in enumerate(self.all_phases):
            #check if this is a green phase
            if 'G' in phase.state:
                self.green_phases.append(i)
        
        #if no green phases found, signal an error
        if not self.green_phases:
            raise ValueError(f"No green phases found for traffic light {self.id}. Check your network file.")
                
        #initialise phase tracking variables
        self.current_phase_index = 0
        self.time_since_last_change = 0
        self.current_phase_duration = 0
        
        #define observation space based on controlled lanes
        self.setup_observation_space()

    def setup_observation_space(self):
        #setup observation space based on number of lanes
        controlled_lanes = traci.trafficlight.getControlledLanes(self.id)
        num_lanes = len(controlled_lanes)
        
        #for each lane: vehicle_count, mean_speed, occupancy, waiting_time
        #plus current phase and time since change
        self.observation_space = gym.spaces.Dict({
            'lane_data': gym.spaces.Box(
                low=0, 
                high=np.inf, 
                shape=(num_lanes, 4), 
                dtype=np.float32
            ),
            'current_phase': gym.spaces.Discrete(len(self.green_phases)),
            'time_since_change': gym.spaces.Box(
                low=0, 
                high=np.inf, 
                shape=(1,), 
                dtype=np.float32
            )
        })

    def update(self, action=None):
        #update traffic light based on action
        self.time_since_last_change += 1
        
        #if action is provided (agent decided to change phase)
        if action is not None and action > 0:
            #only change if minimum green time has passed
            if self.time_since_last_change >= self.min_green_time:
                try:
                    current_sumo_phase = traci.trafficlight.getPhase(self.id)
                    next_phase_index = (current_sumo_phase + 1) % len(self.all_phases)
                    traci.trafficlight.setPhase(self.id, next_phase_index)
                    
                    #update internal tracking variables
                    if next_phase_index in self.green_phases:
                        self.current_phase_index = self.green_phases.index(next_phase_index)
                    self.time_since_last_change = 0
                    
                except Exception as e:
                    print(f"Error updating traffic light {self.id}: {e}")
        
        #force phase change if max green time exceeded
        elif self.time_since_last_change >= self.max_green_time:
            self.update(action=1)

    def get_observation(self):
        #generate observations for the traffic light state
        controlled_lanes = traci.trafficlight.getControlledLanes(self.id)
        
        #collect metrics for each lane
        lane_data = []
        for lane in controlled_lanes:
            #get vehicle counts
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane)
            
            #get average speed
            mean_speed = traci.lane.getLastStepMeanSpeed(lane)
            if mean_speed < 0:
                mean_speed = traci.lane.getMaxSpeed(lane)
                
            #get lane occupancy percentage
            occupancy = traci.lane.getLastStepOccupancy(lane)
            
            #get waiting time on this lane
            waiting_time = sum(traci.vehicle.getAccumulatedWaitingTime(veh) 
                             for veh in traci.lane.getLastStepVehicleIDs(lane))
            
            lane_data.append([vehicle_count, mean_speed, occupancy, waiting_time])
        
        #add current phase index and time since last change
        observation = {
            'lane_data': np.array(lane_data, dtype=np.float32),
            'current_phase': self.current_phase_index,
            'time_since_change': np.array([self.time_since_last_change], dtype=np.float32)
        }
        
        return observation
    
    def calculate_reward(self):
        #calculate current metrics
        current_wait_time = self.get_accumulated_waiting_time()
        current_queue = self.get_queue_length()
        current_speed = self.get_average_speed()
        current_throughput = self.get_throughput()
        
        #track previous metrics if not already stored
        if not hasattr(self, 'last_queue'):
            self.last_queue = current_queue
            self.last_speed = current_speed
            self.last_throughput = current_throughput
        
        #calculate individual rewards
        wait_reward = (self.last_wait_time - current_wait_time)
        queue_reward = (self.last_queue - current_queue) * 0.5
        speed_reward = (current_speed - self.last_speed) * 2.0
        throughput_reward = (current_throughput - self.last_throughput) * 1.0
        
        #combine rewards with weights
        total_reward = (
            wait_reward * 1.0 +      # Reduce waiting time
            queue_reward * 0.5 +     # Reduce queue length
            speed_reward * 0.3 +     # Increase average speed
            throughput_reward * 0.2  # Increase throughput
        )
        
        #update metrics for next comparison
        self.last_wait_time = current_wait_time
        self.last_queue = current_queue
        self.last_speed = current_speed
        self.last_throughput = current_throughput
        
        return total_reward

    def get_accumulated_waiting_time(self):
        #get all lanes controlled by this traffic light
        controlled_lanes = traci.trafficlight.getControlledLanes(self.id)
        
        #calculate total waiting time across all lanes
        total_waiting_time = 0
        for lane in controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for vehicle in vehicles:
                total_waiting_time += traci.vehicle.getAccumulatedWaitingTime(vehicle)
        
        return total_waiting_time / 100.0  #normalise reward
    
    def get_queue_length(self):
        controlled_lanes = traci.trafficlight.getControlledLanes(self.id)
        total_queue = 0
        
        for lane in controlled_lanes:
            total_queue += traci.lane.getLastStepHaltingNumber(lane)
            
        return total_queue
    
    def get_average_speed(self):
        controlled_lanes = traci.trafficlight.getControlledLanes(self.id)
        total_speed = 0
        total_vehicles = 0
        
        for lane in controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleNumber(lane)
            if vehicles > 0:
                total_speed += traci.lane.getLastStepMeanSpeed(lane) * vehicles
                total_vehicles += vehicles
                
        return total_speed / max(1, total_vehicles)
    
    def get_throughput(self):
        #get traffic throughput at this intersection
        controlled_lanes = traci.trafficlight.getControlledLanes(self.id)
        return sum(traci.lane.getLastStepVehicleNumber(lane) for lane in controlled_lanes)
    
    def reset(self):
        #reset traffic light state for a new episode
        self.time_since_last_change = 0
        self.current_phase_index = 0
        self.last_wait_time = self.get_accumulated_waiting_time()