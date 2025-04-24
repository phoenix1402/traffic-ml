import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.environment.sumo_env import SumoEnvironment

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

    def _on_step(self):
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
        
        #initialize pedestrian metrics if needed
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
            
            #initialize episode pedestrian metrics
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
    parser.add_argument("--output", default="src/agent/ppo_traffic_light_model", 
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
        "min_green": 5,
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
    
    #create evaluation environment
    eval_env = SumoEnvironment(**env_kwargs)
    
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
        #save trained model
        model_path = args.output
        if has_pedestrians:
            model_path += "_with_peds"
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
# python -m src.agent.training --config src/networks/acosta_persontrips/run.sumocfg --steps 3000 --total 150000