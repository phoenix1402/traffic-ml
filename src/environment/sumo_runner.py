import os
import argparse
import traceback
from stable_baselines3 import PPO
import pandas as pd
import matplotlib.pyplot as plt

from src.environment.sumo_env import SumoEnvironment

def run_simulation(net_file, route_file, model_path=None, steps=3600, gui=True, use_default=False):
    print(f"Starting simulation with network: {net_file}")
    print(f"Route file: {route_file}")
    print(f"GUI enabled: {gui}")
    print(f"Using default traffic light controller: {use_default}")
    
    #create environment
    try:
        env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            use_gui=gui,
            num_seconds=steps,
            max_green=30,
            min_green=5,
            yellow_time=3
        )
        
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
            'reward': [], 
            'waiting_time': [],
            'queue_length': [], 
            'average_speed': []
        }

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
            step_count += 1
            done = terminated or truncated
        
            #collect metrics
            metrics['step'].append(step_count)
            metrics['reward'].append(reward)
        
            #get traffic performance metrics
            if len(env.traffic_signal_ids) > 0:
                ts_id = env.traffic_signal_ids[0]
                metrics['waiting_time'].append(env.traffic_lights[ts_id].get_accumulated_waiting_time())
                metrics['queue_length'].append(env.traffic_lights[ts_id].get_queue_length())
                metrics['average_speed'].append(env.traffic_lights[ts_id].get_average_speed())
            
            #print occasional progress
            if step_count % 100 == 0:
                print(f"Step {step_count}, reward: {reward}")
        
        #save metrics to csv
        if use_default:
            model_name = "default"
        else:
            model_name = os.path.basename(model_path).split('.')[0] if model_path else "random"

        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        graphs_dir = os.path.join(src_dir, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)

        df = pd.DataFrame(metrics)
        csv_path = os.path.join(graphs_dir, f"sim_results_{model_name}.csv")
        df.to_csv(csv_path, index=False)
        
        #plot results
        plt.figure(figsize=(15, 10))
        metrics_to_plot = ['reward', 'waiting_time', 'queue_length', 'average_speed']
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(2, 2, i+1)
            plt.plot(metrics['step'], metrics[metric])
            plt.title(metric.replace('_', ' ').title())
            plt.xlabel('Simulation Step')
            plt.ylabel('Value')
        plt.tight_layout()
        plt.savefig(os.path.join(graphs_dir, f"performance_{model_name}.png"))
        print(f"Performance metrics saved to sim_results_{model_name}.csv and performance_{model_name}.png")
        
        print(f"Simulation completed: {step_count} steps with total reward: {total_reward}")
        env.close()
        
    except Exception as e:
        print(f"Error in simulation: {e}")
        traceback.print_exc()
        if 'env' in locals():
            env.close()
        return -1

def main():
    parser = argparse.ArgumentParser(description="Run SUMO traffic simulation")
    parser.add_argument("--net", required=True, help="Path to .net.xml file")
    parser.add_argument("--route", required=True, help="Path to .rou.xml file")
    parser.add_argument("--model", help="Path to trained model (.zip)")
    parser.add_argument("--steps", type=int, default=3600, help="Simulation steps")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI")
    parser.add_argument("--default", action="store_true", help="Use default SUMO traffic light controller")
    
    args = parser.parse_args()
    
    run_simulation(
        net_file=args.net,
        route_file=args.route,
        model_path=args.model,
        steps=args.steps,
        gui=not args.no_gui,
        use_default=args.default
    )
if __name__ == "__main__":
    main()

# Run with PPO model
#python -m src.environment.sumo_runner --net src/networks/2lane_junc/single.net.xml --route src/networks/2lane_junc/single_horizontal.rou.xml --model src/agent/ppo_traffic_light_model.zip --steps 8000

# Run with default controller
#python -m src.environment.sumo_runner --net src/networks/2lane_junc/single.net.xml --route src/networks/2lane_junc/single_horizontal.rou.xml --default --steps 8000