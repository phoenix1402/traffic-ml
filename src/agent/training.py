import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.environment.sumo_env import SumoEnvironment
from src.environment.env_wrappers import MultiTrafficLightWrapper

class CustomCallback(BaseCallback):
    def __init__(self, eval_env, n_eval_episodes=5, eval_freq=1000, verbose=1, has_pedestrians=False):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.has_pedestrians = has_pedestrians
        
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        graphs_dir = os.path.join(src_dir, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        
        #track all metrics
        self.rewards = []
        self.waiting_times = []
        self.queue_lengths = []
        self.avg_speeds = []
        self.throughputs = []
        
        #add pedestrian metrics if needed
        if has_pedestrians:
            self.ped_waiting_times = []
            self.ped_counts = []

        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.approx_kl_divs = []
        self.explained_variances = []
        self.learning_rates = []

    def _on_step(self):
        if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
            metrics_dict = self.model.logger.name_to_value
            
            #store the metrics if they're available in the current step
            if "train/policy_loss" in metrics_dict:
                self.policy_losses.append(metrics_dict["train/policy_loss"])
            if "train/value_loss" in metrics_dict:
                self.value_losses.append(metrics_dict["train/value_loss"])
            if "train/entropy_loss" in metrics_dict:
                self.entropy_losses.append(metrics_dict["train/entropy_loss"])
            elif "train/entropy" in metrics_dict:
                self.entropy_losses.append(metrics_dict["train/entropy"])
            if "train/approx_kl" in metrics_dict:
                self.approx_kl_divs.append(metrics_dict["train/approx_kl"])
            if "train/explained_variance" in metrics_dict:
                self.explained_variances.append(metrics_dict["train/explained_variance"])
            if "train/learning_rate" in metrics_dict:
                self.learning_rates.append(metrics_dict["train/learning_rate"])

        if self.n_calls % self.eval_freq == 0:
            #evaluate agent with all metrics
            metrics = self.evaluate()
            self.rewards.append(metrics['reward'])
            self.waiting_times.append(metrics['waiting_time'])
            self.queue_lengths.append(metrics['queue_length'])
            self.avg_speeds.append(metrics['avg_speed'])
            self.throughputs.append(metrics['throughput'])
            
            #log results
            self.logger.record('eval/mean_reward', metrics['reward'])
            self.logger.record('eval/mean_waiting_time', metrics['waiting_time'])
            self.logger.record('eval/mean_queue_length', metrics['queue_length'])
            self.logger.record('eval/mean_avg_speed', metrics['avg_speed'])
            self.logger.record('eval/mean_throughput', metrics['throughput'])
            
            #log pedestrian metrics if available
            if self.has_pedestrians:
                self.ped_waiting_times.append(metrics['pedestrian_waiting_time'])
                self.ped_counts.append(metrics['pedestrian_count'])
                self.logger.record('eval/mean_ped_waiting_time', metrics['pedestrian_waiting_time'])
                self.logger.record('eval/mean_ped_count', metrics['pedestrian_count'])
            
            #save plots
            self.save_metrics()
            
        return True
    
    def evaluate(self):
        rewards = []
        waiting_times = []
        queue_lengths = []
        avg_speeds = []
        throughputs = []
        
        #initialise pedestrian metrics if needed
        if self.has_pedestrians:
            pedestrian_waiting_times = []
            pedestrian_counts = []
        
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_waiting_time = 0
            episode_queue_length = 0
            episode_avg_speed = 0
            episode_throughput = 0
            
            #initialise episode pedestrian metrics
            if self.has_pedestrians:
                episode_ped_waiting_time = 0
                episode_ped_count = 0
            
            while not done:
                #get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                
                #take step in environment
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                #update metrics
                episode_reward += reward
                
                #get metrics from traffic light
                if len(self.eval_env.traffic_signal_ids) > 0:
                    ts_id = self.eval_env.traffic_signal_ids[0]
                    traffic_light = self.eval_env.traffic_lights[ts_id]
                    
                    #collect all metrics
                    waiting_time = traffic_light.get_accumulated_waiting_time()
                    queue_length = traffic_light.get_queue_length()
                    avg_speed = traffic_light.get_average_speed()
                    throughput = traffic_light.get_throughput()
                    
                    episode_waiting_time += waiting_time
                    episode_queue_length += queue_length
                    episode_avg_speed += avg_speed
                    episode_throughput += throughput
                    
                    if self.has_pedestrians:
                        try:
                            ped_wait_time = traffic_light.get_pedestrian_waiting_time()
                            ped_count = len(self.eval_env.traci.person.getIDList())
                            
                            episode_ped_waiting_time += ped_wait_time
                            episode_ped_count = max(episode_ped_count, ped_count)  # Use max count during episode
                        except Exception as e:
                            if self.verbose > 0:
                                print(f"Warning: Error collecting pedestrian data: {e}")
            
            rewards.append(episode_reward)
            waiting_times.append(episode_waiting_time)
            queue_lengths.append(episode_queue_length)
            avg_speeds.append(episode_avg_speed)
            throughputs.append(episode_throughput)
            
            #add pedestrian metrics if available
            if self.has_pedestrians:
                pedestrian_waiting_times.append(episode_ped_waiting_time)
                pedestrian_counts.append(episode_ped_count)
        
        #return mean reward and a dictionary with all metrics
        result = {
            'reward': np.mean(rewards),
            'waiting_time': np.mean(waiting_times),
            'queue_length': np.mean(queue_lengths),
            'avg_speed': np.mean(avg_speeds),
            'throughput': np.mean(throughputs)
        }
        
        if self.has_pedestrians:
            result['pedestrian_waiting_time'] = np.mean(pedestrian_waiting_times)
            result['pedestrian_count'] = np.mean(pedestrian_counts)
        
        return result
    
    def save_metrics(self):
        #determine number of subplots based on whether we have pedestrian data
        num_rows = 3 if self.has_pedestrians else 2
        num_cols = 3
        
        plt.figure(figsize=(15, 5 * num_rows))
        
        plt.subplot(num_rows, num_cols, 1)
        plt.plot(self.rewards)
        plt.title('Mean Reward per Evaluation')
        plt.xlabel('Evaluation')
        plt.ylabel('Mean Reward')
        
        plt.subplot(num_rows, num_cols, 2)
        plt.plot(self.waiting_times)
        plt.title('Mean Waiting Time')
        plt.xlabel('Evaluation')
        plt.ylabel('Waiting Time')
        
        plt.subplot(num_rows, num_cols, 3)
        plt.plot(self.queue_lengths)
        plt.title('Mean Queue Length')
        plt.xlabel('Evaluation')
        plt.ylabel('Queue Length')
        
        plt.subplot(num_rows, num_cols, 4)
        plt.plot(self.avg_speeds)
        plt.title('Mean Average Speed')
        plt.xlabel('Evaluation')
        plt.ylabel('Speed (m/s)')
        
        plt.subplot(num_rows, num_cols, 5)
        plt.plot(self.throughputs)
        plt.title('Mean Throughput')
        plt.xlabel('Evaluation')
        plt.ylabel('Throughput (vehicles)')
        
        if self.has_pedestrians:
            plt.subplot(num_rows, num_cols, 7)
            plt.plot(self.ped_waiting_times)
            plt.title('Mean Pedestrian Waiting Time')
            plt.xlabel('Evaluation')
            plt.ylabel('Waiting Time')
            
            plt.subplot(num_rows, num_cols, 8)
            plt.plot(self.ped_counts)
            plt.title('Mean Pedestrian Count')
            plt.xlabel('Evaluation')
            plt.ylabel('Number of Pedestrians')

        if len(self.policy_losses) > 0:
            plt.figure(figsize=(15, 10))
            
            #policy Loss
            plt.subplot(2, 3, 1)
            plt.plot(self.policy_losses)
            plt.title('Policy Loss')
            plt.xlabel('Updates')
            plt.ylabel('Loss')
            plt.grid(True)
            
            #value Loss
            plt.subplot(2, 3, 2)
            plt.plot(self.value_losses)
            plt.title('Value Function Loss')
            plt.xlabel('Updates') 
            plt.ylabel('Loss')
            plt.grid(True)
            
            #entropy
            plt.subplot(2, 3, 3)
            plt.plot(self.entropy_losses)
            plt.title('Entropy')
            plt.xlabel('Updates')
            plt.ylabel('Entropy')
            plt.grid(True)
            
            #approximate KL divergence
            if len(self.approx_kl_divs) > 0:
                plt.subplot(2, 3, 4)
                plt.plot(self.approx_kl_divs)
                plt.title('Approximate KL Divergence')
                plt.xlabel('Updates')
                plt.ylabel('KL Divergence')
                plt.grid(True)
            
            #explained Variance
            if len(self.explained_variances) > 0:
                plt.subplot(2, 3, 6)
                plt.plot(self.explained_variances)
                plt.title('Explained Variance')
                plt.xlabel('Updates')
                plt.ylabel('Variance')
                plt.grid(True)
            
        plt.tight_layout()
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        graphs_dir = os.path.join(src_dir, "graphs")
        model_suffix = "_with_peds" if self.has_pedestrians else ""
        plt.savefig(os.path.join(graphs_dir, f"training_progress_{self.n_calls}{model_suffix}.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent for traffic light control")
    parser.add_argument("--net", 
                        help="Path to .net.xml file")
    parser.add_argument("--route", 
                        help="Path to .rou.xml file")
    parser.add_argument("--config", 
                        help="Path to .sumocfg file (alternative to --net and --route)")
    parser.add_argument("--steps", type=int, default=2500, 
                        help="Simulation steps per episode")
    parser.add_argument("--total", type=int, default=100000, 
                        help="Total training timesteps")
    parser.add_argument("--eval-freq", type=int, default=5000, 
                        help="Evaluation frequency")
    parser.add_argument("--output", default="src/agent/trained_models/ppo_traffic_light_model", 
                        help="Path to save trained model")
    
    args = parser.parse_args()
    
    #set default net and route if no config file is provided
    if not args.config and not args.net:
        args.net = "src/networks/2lane_junc/single.net.xml"
        args.route = "src/networks/2lane_junc/single_horizontal.rou.xml" 
    
    #check for pedestrian data in route file or config file
    has_pedestrians = False
    if args.config:
        print(f"Training with config file: {args.config}")
        has_pedestrians = "persontrips" in args.config or "peds" in args.config
    else:
        print(f"Training with network: {args.net}")
        print(f"Route file: {args.route}")
        has_pedestrians = "persontrips" in args.route or "peds" in args.route
    
    if has_pedestrians:
        print("Pedestrian simulation detected - will track pedestrian metrics")
    
    print(f"Steps per episode: {args.steps}")
    print(f"Total training steps: {args.total}")
    
    #create environment with either config file or net/route files
    env_kwargs = {
        "use_gui": False,
        "num_seconds": args.steps,
        "max_green": 40,
        "min_green": 10,
        "yellow_time": 4,
        "has_pedestrians": has_pedestrians
    }
    
    if args.config:
        env_kwargs["config_file"] = args.config
    else:
        env_kwargs["net_file"] = args.net
        env_kwargs["route_file"] = args.route
    
    #create environment
    env = SumoEnvironment(**env_kwargs)
    env = MultiTrafficLightWrapper(env)
    
    #create evaluation environment
    eval_env = SumoEnvironment(**env_kwargs)
    eval_env = MultiTrafficLightWrapper(eval_env)
    
    #create callback for evaluation
    callback = CustomCallback(eval_env, eval_freq=args.eval_freq, has_pedestrians=has_pedestrians)
    
    #create agent
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.9,
        clip_range=0.2,
        ent_coef=0.1
    )
    
    try:
        #train agent
        print("Starting training with composite reward (press Ctrl+C to stop early)...")
        model.learn(total_timesteps=args.total, callback=callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    finally:
        #create trained_models directory if it doesn't exist
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        agent_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(agent_dir, "trained_models")
        os.makedirs(models_dir, exist_ok=True)
        
        #extract the route name from the file path
        route_name = ""
        if args.config:
            #if using a config file, try to extract route name from it
            config_path = args.config
            try:
                #extract primary route file name from config if possible
                with open(config_path, 'r') as f:
                    for line in f:
                        if '<route-files' in line:
                            #extract first route file from the list
                            route_files = line.split('value="')[1].split('"')[0]
                            first_route = route_files.split(',')[0]
                            route_name = os.path.basename(first_route).split('.rou.xml')[0]
                            break
            except Exception as e:
                print(f"Could not extract route name from config: {e}")
                route_name = os.path.basename(config_path).split('.sumocfg')[0]
        else:
            #using direct route file
            route_name = os.path.basename(args.route).split('.rou.xml')[0]
            
        #build the model path with the new pattern
        model_name = f"ppo_model_{route_name}"
        if has_pedestrians:
            model_name += "_with_peds"
            
        model_path = os.path.join(models_dir, model_name)
        print(f"Saving model to {model_path}")
        model.save(model_path)
        
        #close environment
        env.close()
        eval_env.close()

if __name__ == "__main__":
    main()

# Run with default parameters:
# python -m src.agent.training
#
# Run with custom parameters:
# python -m src.agent.training --net src/networks/2lane_junc/single.net.xml --route src/networks/2lane_junc/single_vertical.rou.xml --steps 3000 --total 150000
#
# Run with config file:
# python -m src.agent.training --config src/networks/acosta_persontrips/run.sumocfg --steps 5000 --total 100000