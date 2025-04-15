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
        self.rewards = []
        self.waiting_times = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            #evaluate agent
            mean_reward, mean_waiting_time = self.evaluate()
            self.rewards.append(mean_reward)
            self.waiting_times.append(mean_waiting_time)
            
            #log results
            self.logger.record('eval/mean_reward', mean_reward)
            self.logger.record('eval/mean_waiting_time', mean_waiting_time)
            
            #save plots
            self.save_metrics()
            
        return True
    
    def evaluate(self):
        rewards = []
        waiting_times = []
        
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_waiting_time = 0
            
            while not done:
                #get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                
                #take step in environment
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                
                #update metrics
                episode_reward += reward
                
                #get waiting time from traffic light
                if len(self.eval_env.traffic_signal_ids) == 1:
                    ts_id = self.eval_env.traffic_signal_ids[0]
                    waiting_time = self.eval_env.traffic_lights[ts_id].get_accumulated_waiting_time()
                    episode_waiting_time += waiting_time
            
            rewards.append(episode_reward)
            waiting_times.append(episode_waiting_time)
        
        return np.mean(rewards), np.mean(waiting_times)
    
    def save_metrics(self):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards)
        plt.title('Mean Reward per Evaluation(higher is better)')
        plt.xlabel('Evaluation')
        plt.ylabel('Mean Reward')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.waiting_times)
        plt.title('Mean Waiting Time per Evaluation')
        plt.xlabel('Evaluation')
        plt.ylabel('Mean Waiting Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(f'src/agent/plots/training_progress_{self.n_calls}.png')
        plt.close()

def main():
    #set up environment
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
        print("Starting training (press Ctrl+C to stop early)...")
        model.learn(total_timesteps=500000, callback=callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    finally:
        #save trained model
        print("Saving model to src/agent/ppo_traffic_light_model.zip")
        model.save("src/agent/ppo_traffic_light_model")
        
        #close environment
        env.close()
        eval_env.close()

if __name__ == "__main__":
    main()

#run with: python -m src.agent.training