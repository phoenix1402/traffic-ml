# updates the traffic light phase based on the action taken by the agent

import traci
import gymnasium as gym
import numpy as np

class TrafficLights:
    def __init__(self, signal_id, yellow_time=4, max_green_time=40, min_green_time=10, reward_func="waiting_time_diff", has_pedestrians=False):
        #TrafficLights manages a single traffic light in the sumo environment.
    
        #traffic light parameters
        self.id = signal_id #sumo traffic light id
        self.yellow_time = yellow_time #duration of yellow phase in seconds
        self.max_green_time = max_green_time #maximum green phase duration
        self.min_green_time = min_green_time #minimum green phase duration
        self.reward_func = reward_func
        self.last_wait_time = 0
        self.last_queue = 0
        self.last_speed = 0
        self.last_pressure = 0
        self.last_emergency_stops = 0
        self.is_yellow = False
        self.next_green_phase = 0
        self.has_pedestrians = has_pedestrians
        
        self.initialise_phases()
        
        #define action and observation spaces for this traffic light
        self.action_space = gym.spaces.Discrete(len(self.green_phases))
        
    def initialise_phases(self):
        #initialise traffic light phases from sumo network data
        program_logics = traci.trafficlight.getAllProgramLogics(self.id) # get all phases from sumo
        if not program_logics:
            raise ValueError(f"No program logics found for traffic light {self.id}. Check your network file.")
                
        self.original_phases = program_logics[0].phases
        
        #get the green phases (exclude yellow/red phases)
        self.green_phases = []
        green_phase_objects = []
        for i, phase in enumerate(self.original_phases):
            if 'G' in phase.state:
                self.green_phases.append(i)
                green_phase_objects.append(phase)
        
        if not self.green_phases:
            raise ValueError(f"No green phases found for traffic light {self.id}. Check your network file.")
        
        self.yellow_dict = {}
        self.all_phases = list(green_phase_objects)
        
        for i, p1 in enumerate(green_phase_objects):
            for j, p2 in enumerate(green_phase_objects):
                if i == j: continue
                
                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] in ["G", "g"]) and (p2.state[s] in ["r", "s"]):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]
                
                self.yellow_dict[(i, j)] = len(self.all_phases)
                self.all_phases.append(traci.trafficlight.Phase(self.yellow_time * 1000, yellow_state))
                
        self.current_green_phase = 0
        self.time_since_last_change = 0
        
        self.setup_observation_space()
        self.custom_phases()
        
    def custom_phases(self):
        #update the traffic light program in SUMO with custom phases
        try:
            program = traci.trafficlight.getAllProgramLogics(self.id)[0]
            
            self.original_phase_count = len(program.phases)
            print(f"Traffic light {self.id} has {self.original_phase_count} phases in SUMO")
            
            self.phase_map = {}
            for i, green_idx in enumerate(self.green_phases):
                self.phase_map[i] = green_idx
            
            traci.trafficlight.setPhase(self.id, self.green_phases[0])
            
            self.yellow_phase()
        except Exception as e:
            print(f"Error updating traffic light program: {e}")
    
    def yellow_phase(self):
        self.yellow_dict = {}
        
        #check if the original phases already include yellow phases
        yellow_phases = []
        for i, phase in enumerate(self.original_phases):
            if 'y' in phase.state and 'G' not in phase.state and 'g' not in phase.state:
                yellow_phases.append(i)
        
        print(f"Found {len(yellow_phases)} yellow phases in original program")
        
        for i, src_idx in enumerate(self.green_phases):
            for j, dst_idx in enumerate(self.green_phases):
                if i == j:
                    continue
                    
                yellow_idx = None
                for y_idx in yellow_phases:
                    if y_idx > src_idx and y_idx < dst_idx:
                        yellow_idx = y_idx
                        break
                
                if yellow_idx is not None:
                    self.yellow_dict[(i, j)] = yellow_idx
                else:
                    self.yellow_dict[(i, j)] = dst_idx
        
        print(f"Created simplified yellow_dict with {len(self.yellow_dict)} transitions")

    def setup_observation_space(self):
        #setup observation space based on number of lanes
        controlled_lanes = traci.trafficlight.getControlledLanes(self.id)
        num_lanes = len(controlled_lanes)
        
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
        
        #handle yellow phase transition completion
        if self.is_yellow and self.time_since_last_change >= self.yellow_time:
            actual_phase_index = self.phase_map.get(int(self.next_green_phase), int(self.next_green_phase))
            
            max_phase = len(self.original_phases) - 1
            if actual_phase_index > max_phase:
                print(f"Warning: Using target phase {self.next_green_phase} directly as {actual_phase_index} exceeds {max_phase}")
                actual_phase_index = min(max_phase, int(self.next_green_phase))
            
            #transition to the next green phase
            traci.trafficlight.setPhase(self.id, actual_phase_index)
            self.current_green_phase = self.next_green_phase
            self.time_since_last_change = 0
            self.is_yellow = False
            return
            
        if action is None or self.is_yellow:
            return

        if action >= 0 and action < len(self.green_phases):
            target_green_index = int(action)
            
            #only change if minimum green time has passed and it's a different phase
            if self.time_since_last_change >= self.min_green_time and target_green_index != int(self.current_green_phase):
                try:
                    #get yellow transition phase
                    key = (int(self.current_green_phase), int(target_green_index))
                    if key not in self.yellow_dict:
                        print(f"Warning: Missing yellow transition for {key}, using target directly")
                        yellow_phase_index = self.phase_map.get(int(target_green_index), int(target_green_index))
                    else:
                        yellow_phase_index = self.yellow_dict[key]
                    
                    #validate the phase index
                    max_phase = len(self.original_phases) - 1
                    if yellow_phase_index > max_phase:
                        print(f"Warning: Phase {yellow_phase_index} exceeds maximum {max_phase}, using direct transition")
                        yellow_phase_index = self.phase_map.get(int(target_green_index), int(target_green_index))
                        
                    #set the phase and update tracking
                    self.next_green_phase = target_green_index
                    traci.trafficlight.setPhase(self.id, yellow_phase_index)
                    self.time_since_last_change = 0
                    self.is_yellow = True
                    
                except Exception as e:
                    print(f"Error updating traffic light {self.id}: {e}")
        
        #force phase change if max green time exceeded
        elif self.time_since_last_change >= self.max_green_time:
            #choose the next phase (cycling through green phases)
            next_green = (int(self.current_green_phase) + 1) % len(self.green_phases)
            self.update(next_green)

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
            'current_phase': self.current_green_phase,
            'time_since_change': np.array([self.time_since_last_change], dtype=np.float32)
        }
        
        return observation
    
    def get_pedestrian_waiting_time(self):
        controlled_crossings = []
        controlled_lanes = traci.trafficlight.getControlledLinks(self.id)
        
        #find pedestrian crossings controlled by this light
        for link in controlled_lanes:
            for connection in link:
                if len(connection) > 0 and connection[0].startswith(':') and 'w' in connection[0]:
                    controlled_crossings.append(connection[0])
        
        total_waiting = 0
        for person_id in traci.person.getIDList():
            try:
                person_edge = traci.person.getRoadID(person_id)
                if person_edge in controlled_crossings or person_edge.startswith(':'):
                    # If person is at a controlled junction and not moving
                    if traci.person.getSpeed(person_id) < 0.2:
                        total_waiting += 1
            except:
                pass
                
        return total_waiting
    
    def calculate_reward(self):
        #calculate current metrics
        current_wait_time = self.get_accumulated_waiting_time()
        current_queue = self.get_queue_length()
        current_speed = self.get_average_speed()
        current_pressure = self.get_pressure()
        current_emergency_stops = self.get_emergency_stops()
        
        #add pedestrian metrics if relevant
        current_pedestrian_wait = 0
        if self.has_pedestrians:
            try:
                current_pedestrian_wait = self.get_pedestrian_waiting_time()
            except:
                pass
        
        #track previous metrics if not already stored
        if not hasattr(self, 'last_queue'):
            self.last_queue = current_queue
            self.last_speed = current_speed
            self.last_pressure = current_pressure
            self.last_emergency_stops = current_emergency_stops
            self.last_pedestrian_wait = current_pedestrian_wait if self.has_pedestrians else 0
        
        #calculate individual rewards
        wait_reward = (self.last_wait_time - current_wait_time)
        queue_reward = (self.last_queue - current_queue) * 0.5
        speed_reward = (current_speed - self.last_speed) * 2.0
        pressure_reward = (current_pressure - self.last_pressure) * 1.0
        
        #calculate emergency stop penalty
        emergency_stop_penalty = current_emergency_stops * -2.0
        
        #calculate pedestrian waiting time penalty
        pedestrian_wait_penalty = 0
        if self.has_pedestrians:
            pedestrian_wait_penalty = (current_pedestrian_wait - self.last_pedestrian_wait) * -1.0
        
        #combine rewards with weights - include pedestrian metrics when available
        if self.has_pedestrians:
            total_reward = (
                wait_reward * 1.0 + #reduce waiting time
                queue_reward * 0.5 + #reduce queue length
                speed_reward * 0.3 + #increase average speed
                pressure_reward * 0.2 + #improve traffic balance (pressure)
                pedestrian_wait_penalty * 0.8 #pedestrian waiting time
            )
        else:
            total_reward = (
                wait_reward * 1.0 +
                queue_reward * 0.5 +
                speed_reward * 0.3 +
                pressure_reward * 0.2
            )
        
        #update metrics for next comparison
        self.last_wait_time = current_wait_time
        self.last_queue = current_queue
        self.last_speed = current_speed
        self.last_pressure = current_pressure  # Changed from throughput to pressure
        self.last_emergency_stops = current_emergency_stops
        if self.has_pedestrians:
            self.last_pedestrian_wait = current_pedestrian_wait
        
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
    
    def get_pressure(self):
        #get controlled lanes
        incoming_lanes = traci.trafficlight.getControlledLanes(self.id)
        
        #get outgoing lanes
        outgoing_lanes = []
        for connection in traci.trafficlight.getControlledLinks(self.id):
            if connection and len(connection) > 0 and connection[0][1]:  # If there's a valid connection
                outgoing_lane = connection[0][1]
                if outgoing_lane not in outgoing_lanes and not outgoing_lane.startswith(':'):
                    outgoing_lanes.append(outgoing_lane)
        
        #calculate pressure as difference between vehicles on outgoing vs incoming lanes
        outgoing_count = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in outgoing_lanes)
        incoming_count = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in incoming_lanes)
        
        return outgoing_count - incoming_count
    
    def get_emergency_stops(self):
        controlled_lanes = traci.trafficlight.getControlledLanes(self.id)
        
        emergency_stops = 0
        for lane in controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for veh_id in vehicles:
                if traci.vehicle.getAcceleration(veh_id) < -4.5:
                    emergency_stops += 1
                    
        return emergency_stops
    
    def reset(self):
        #reset traffic light state for a new episode
        self.time_since_last_change = 0
        self.current_green_phase = 0
        self.is_yellow = False
        self.last_wait_time = self.get_accumulated_waiting_time()