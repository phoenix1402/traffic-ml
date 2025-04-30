import os
import argparse
import traceback
from stable_baselines3 import PPO
import pandas as pd
import matplotlib.pyplot as plt
import traci
import numpy as np

from src.environment.sumo_env import SumoEnvironment
from src.environment.env_wrappers import MultiTrafficLightWrapper

def run_simulation(net_file=None, route_file=None, config_file=None, model_path=None, steps=3600, gui=True, use_default=False):
    if config_file:
        print(f"Starting simulation with config file: {config_file}")
        # Check if the config likely contains pedestrian data
        has_pedestrians = "persontrips" in config_file or "peds" in config_file
        print(f"Pedestrian simulation detected: {has_pedestrians}")
    else:
        print(f"Starting simulation with network: {net_file}")
        print(f"Route file: {route_file}")
        has_pedestrians = "persontrips" in route_file or "peds" in route_file
    
    print(f"GUI enabled: {gui}")
    print(f"Using default traffic light controller: {use_default}")
    
    seed = 42
    #create environment
    try:
        env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            config_file=config_file,
            use_gui=gui,
            num_seconds=steps,
            max_green=30,
            min_green=5,
            yellow_time=4,
            has_pedestrians=has_pedestrians,
            seed=seed
        )

        env = MultiTrafficLightWrapper(env)
        
        #reset environment
        obs, _ = env.reset()
        
        #load model if provided
        model = None
        if model_path and os.path.exists(model_path):
            model = PPO.load(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print("Using random actions")
        
        #run simulation
        total_reward = 0
        step_count = 0
        done = False

        metrics = {
            'step': [],
            'pressure': [],
            'waiting_time': [],
            'queue_length': [], 
            'average_speed': [],
            'collisions': [],
            'emergency_stops': [],
            'cumulative_reward': [],
            'co2_emissions': []
        }
        
        #add pedestrian metrics if needed
        if has_pedestrians:
            metrics['pedestrian_waiting_time'] = []
            metrics['pedestrian_count'] = []

        while not done:
            if not use_default:
                if model:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
                
                obs, reward, terminated, truncated, info = env.step(action)
            else:
                obs, reward, terminated, truncated, info = env.step(None)
            
            #update metrics
            total_reward += reward if isinstance(reward, (int, float)) else sum(reward.values())
            metrics['cumulative_reward'].append(total_reward)
            
            step_count += 1
            done = terminated or truncated
        
            if len(env.traffic_signal_ids) > 0:
                ts_id = env.traffic_signal_ids[0]
                ts = env.traffic_lights[ts_id]
                
                # Update step metric
                metrics['step'].append(step_count)
                
                #collect existing metrics
                metrics['waiting_time'].append(ts.get_accumulated_waiting_time())
                metrics['queue_length'].append(ts.get_queue_length())
                metrics['average_speed'].append(ts.get_average_speed())
                metrics['pressure'].append(ts.get_pressure())
                metrics['co2_emissions'].append(ts.get_co2_emissions())
                
                #get collision data from SUMO simulation
                metrics['collisions'].append(traci.simulation.getCollidingVehiclesNumber())
                
                #calculate emergency stops (vehicles with deceleration > 4.5 m/s²)
                emergency_stops = 0
                for veh_id in traci.vehicle.getIDList():
                    if traci.vehicle.getAcceleration(veh_id) < -4.5:
                        emergency_stops += 1
                metrics['emergency_stops'].append(emergency_stops)
                
                # Collect pedestrian metrics if available
                if has_pedestrians:
                    try:
                        ped_wait_time = ts.get_pedestrian_waiting_time()
                        metrics['pedestrian_waiting_time'].append(ped_wait_time)
                        metrics['pedestrian_count'].append(len(traci.person.getIDList()))
                    except Exception as e:
                        # Fallback if pedestrian data access fails
                        metrics['pedestrian_waiting_time'].append(0)
                        metrics['pedestrian_count'].append(0)
                        print(f"Error collecting pedestrian data: {e}")
            
            #print occasional progress
            if step_count % 100 == 0:
                print(f"Step {step_count}, pressure: {metrics['pressure'][-1] if metrics['pressure'] else 0}")
        
        #save metrics to csv
        if use_default:
            model_name = "default"
        else:
            model_name = os.path.basename(model_path).split('.')[0] if model_path else "random"
        
        #add pedestrian info to model name if applicable
        if has_pedestrians:
            model_name += "_with_peds"

        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        graphs_dir = os.path.join(src_dir, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)

        df = pd.DataFrame(metrics)
        csv_path = os.path.join(graphs_dir, f"sim_results_{model_name}.csv")
        df.to_csv(csv_path, index=False)
        
        plt.figure(figsize=(15, 15))
        metrics_to_plot = ['waiting_time', 'queue_length', 'average_speed', 'pressure', 'cumulative_reward', 'co2_emissions']
        
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(3, 2, i+1)
            plt.plot(metrics['step'], metrics[metric], 'b-', linewidth=2)
            plt.title(metric.replace('_', ' ').title())
            plt.xlabel('Simulation Step')
            
            if metric in ['waiting_time', 'queue_length']:
                plt.ylabel('Count')
            elif metric == 'average_speed':
                plt.ylabel('m/s')
            elif metric == 'pressure':
                plt.ylabel('Vehicles (out-in)')
            elif metric == 'cumulative_reward':
                plt.ylabel('Total Reward')
            elif metric == 'co2_emissions':
                plt.ylabel('g CO2')
                
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f"performance_{model_name}.png"))
        
        #create a second figure for safety metrics
        plt.figure(figsize=(15, 5))
        safety_metrics = ['collisions', 'emergency_stops']
        
        for i, metric in enumerate(safety_metrics):
            plt.subplot(1, 2, i+1)
            plt.plot(metrics['step'], metrics[metric], 'r-' if metric == 'collisions' else 'y-', linewidth=2)
            plt.title(f'{metric.replace("_", " ").title()} Over Time')
            plt.xlabel('Simulation Step')
            plt.ylabel('Count')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f"safety_{model_name}.png"))
        
        #add pedestrian charts if data is available
        if has_pedestrians:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.plot(metrics['step'], metrics['pedestrian_waiting_time'], 'm-', linewidth=2)
            plt.title('Pedestrian Waiting Time')
            plt.xlabel('Simulation Step')
            plt.ylabel('Time (seconds)')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(metrics['step'], metrics['pedestrian_count'], 'c-', linewidth=2)
            plt.title('Pedestrian Count')
            plt.xlabel('Simulation Step')
            plt.ylabel('Number of Pedestrians')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, f"pedestrians_{model_name}.png"))
        
        print(f"Performance metrics saved to:")
        print(f"  - {csv_path}")
        print(f"  - {os.path.join(graphs_dir, f'performance_{model_name}.png')}")
        print(f"  - {os.path.join(graphs_dir, f'safety_{model_name}.png')}")
        if has_pedestrians:
            print(f"  - {os.path.join(graphs_dir, f'pedestrians_{model_name}.png')}")
        
        print(f"\nSimulation completed: {step_count} steps with total reward: {total_reward}")
        env.close()
        
    except Exception as e:
        print(f"Error in simulation: {e}")
        traceback.print_exc()
        if 'env' in locals():
            env.close()
        return -1

def main():
    parser = argparse.ArgumentParser(description="Run SUMO traffic simulation")
    parser.add_argument("--net", help="Path to .net.xml file")
    parser.add_argument("--route", help="Path to .rou.xml file")
    parser.add_argument("--config", help="Path to .sumocfg file (alternative to --net and --route)")
    parser.add_argument("--model", help="Path to trained model (.zip)")
    parser.add_argument("--steps", type=int, default=5000, help="Simulation steps")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    parser.add_argument("--default", action="store_true", help="Use default SUMO traffic light controller")
    
    args = parser.parse_args()
    
    #check that we have either a config file or both net and route files
    if not args.config and (not args.net or not args.route):
        parser.error("Either --config or both --net and --route must be specified")
    
    run_simulation(
        net_file=args.net,
        route_file=args.route,
        config_file=args.config,
        model_path=args.model,
        steps=args.steps,
        gui=not args.no_gui,
        use_default=args.default
    )

if __name__ == "__main__":
    main()

# 2 lane junc gen
# python -m src.environment.sumo_runner --net src/networks/2lane_junc/single.net.xml --route src/networks/2lane_junc/single_gen.rou.xml --model src/agent/trained_models/ppo_model_single_gen.zip --steps 5000
# python -m src.environment.sumo_runner --net src/networks/2lane_junc/single.net.xml --route src/networks/2lane_junc/single_gen.rou.xml --default --steps 5000

# cross3ltl
# python -m src.environment.sumo_runner --net src/networks/cross3ltl/net.net.xml --route src/networks/cross3ltl/cross3ltl.rou.xml --model src/agent/trained_models/ppo_model_cross3ltl.zip --steps 5000
# python -m src.environment.sumo_runner --net src/networks/cross3ltl/net.net.xml --route src/networks/cross3ltl/cross3ltl.rou.xml --default --steps 5000

# 2x2
# python -m src.environment.sumo_runner --net src/networks/2x2/2x2.net.xml --route src/networks/2x2/2x2.rou.xml --model src/agent/trained_models/ppo_model_2x2.zip --steps 5000
# python -m src.environment.sumo_runner --net src/networks/2x2/2x2.net.xml --route src/networks/2x2/2x2.rou.xml --default --steps 5000