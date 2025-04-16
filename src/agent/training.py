import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.environment.sumo_env import SumoEnvironment

class CustomCallback(BaseCallback):
    def __init__(self, eval_env, n_eval_episodes=5, eval_freq=1000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        
        #track all metrics
        self.rewards = []
        self.waiting_times = []
        self.queue_lengths = []
        self.avg_speeds = []
        self.throughputs = []

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
            
            #save plots
            self.save_metrics()
            
        return True
    
    def evaluate(self):
        rewards = []
        waiting_times = []
        queue_lengths = []
        avg_speeds = []
        throughputs = []
        
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_waiting_time = 0
            episode_queue_length = 0
            episode_avg_speed = 0
            episode_throughput = 0
            
            while not done:
                #get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                
                #take step in environment
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                # Update metrics
                episode_reward += reward
                
                #get metrics from traffic light
                if len(self.eval_env.traffic_signal_ids) == 1:
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
            
            rewards.append(episode_reward)
            waiting_times.append(episode_waiting_time)
            queue_lengths.append(episode_queue_length)
            avg_speeds.append(episode_avg_speed)
            throughputs.append(episode_throughput)
        
        #return mean reward and a dictionary with all metrics
        return {
            'reward': np.mean(rewards),
            'waiting_time': np.mean(waiting_times),
            'queue_length': np.mean(queue_lengths),
            'avg_speed': np.mean(avg_speeds),
            'throughput': np.mean(throughputs)
        }
    
    def save_metrics(self):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.plot(self.rewards)
        plt.title('Mean Reward per Evaluation')
        plt.xlabel('Evaluation')
        plt.ylabel('Mean Reward')
        
        plt.subplot(2, 3, 2)
        plt.plot(self.waiting_times)
        plt.title('Mean Waiting Time')
        plt.xlabel('Evaluation')
        plt.ylabel('Waiting Time')
        
        plt.subplot(2, 3, 3)
        plt.plot(self.queue_lengths)
        plt.title('Mean Queue Length')
        plt.xlabel('Evaluation')
        plt.ylabel('Queue Length')
        
        plt.subplot(2, 3, 4)
        plt.plot(self.avg_speeds)
        plt.title('Mean Average Speed')
        plt.xlabel('Evaluation')
        plt.ylabel('Speed (m/s)')
        
        plt.subplot(2, 3, 5)
        plt.plot(self.throughputs)
        plt.title('Mean Throughput')
        plt.xlabel('Evaluation')
        plt.ylabel('Throughput (vehicles)')
        
        plt.tight_layout()
        plt.savefig(f'src/agent/plots/training_progress_{self.n_calls}.png')
        plt.close()

def main():
    env = SumoEnvironment(
        net_file='src/networks/2way_single/single.net.xml',
        route_file='src/networks/2way_single/single_horizontal.rou.xml',
        use_gui=False,
        num_seconds=3600,
        max_green=40,
        min_green=5,
        yellow_time=2
    )
    
    #create evaluation environment
    eval_env = SumoEnvironment(
        net_file='src/networks/2way_single/single.net.xml',
        route_file='src/networks/2way_single/single_horizontal.rou.xml',
        use_gui=False,
        num_seconds=3600,
        max_green=40,
        min_green=5,
        yellow_time=2
    )
    
    #create callback for evaluation
    callback = CustomCallback(eval_env, eval_freq=10000)
    
    #create agent
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="src/agent/ppo_traffic_light_tensorboard/"
    )
    
    try:
        #train agent
        print("Starting training with composite reward (press Ctrl+C to stop early)...")
        model.learn(total_timesteps=500000, callback=callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    finally:
        #save trained model
        model_path = "src/agent/models/ppo_traffic_light_model"
        print(f"Saving model to {model_path}")
        model.save(model_path)
        
        #close environment
        env.close()
        eval_env.close()

if __name__ == "__main__":
    main()

# run with: python -m src.agent.training