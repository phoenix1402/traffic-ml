import pandas as pd
import matplotlib.pyplot as plt
import os

def compare_results(model_name, default_name="default"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    graphs_dir = os.path.join(script_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    model_csv = os.path.join(graphs_dir, f"sim_results_{model_name}.csv")
    default_csv = os.path.join(graphs_dir, f"sim_results_{default_name}.csv")
    
    if not os.path.exists(model_csv) or not os.path.exists(default_csv):
        print(f"Error: CSV files not found. Run both simulations first.")
        print(f"Looking for files at:")
        print(f"  {model_csv}")
        print(f"  {default_csv}")
        return
    
    model_df = pd.read_csv(model_csv)
    default_df = pd.read_csv(default_csv)
    
    plt.figure(figsize=(15, 10))
    metrics_to_plot = ['waiting_time', 'queue_length', 'average_speed', 'reward']
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        plt.plot(model_df['step'], model_df[metric], label=f'PPO Model')
        plt.plot(default_df['step'], default_df[metric], label='Default SUMO Controller')
        plt.title(metric.replace('_', ' ').title())
        plt.xlabel('Simulation Step')
        plt.ylabel('Value')
        plt.legend()
    
    plt.tight_layout()
    comparison_path = os.path.join(graphs_dir, f"comparison_{model_name}_vs_default.png")
    plt.savefig(comparison_path)
    print(f"Comparison saved to {comparison_path}")
    
    print("\nSummary Statistics:")
    for metric in metrics_to_plot:
        model_avg = model_df[metric].mean()
        default_avg = default_df[metric].mean()
        
        if metric in ['waiting_time', 'queue_length']:
            improvement = ((default_avg - model_avg) / default_avg * 100)
            better = "lower" if model_avg < default_avg else "higher"
        else:
            improvement = ((model_avg - default_avg) / default_avg * 100)
            better = "higher" if model_avg > default_avg else "lower"
        
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  PPO Model: {model_avg:.2f}")
        print(f"  Default: {default_avg:.2f}")
        print(f"  Difference: {improvement:.2f}% {better}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare simulation results")
    parser.add_argument("--model", default="ppo_traffic_light_model", help="Model name (without extension)")
    args = parser.parse_args()
    
    compare_results(args.model)