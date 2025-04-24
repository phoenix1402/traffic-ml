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
    
    #print available columns to help debugging
    print(f"Available columns in model CSV: {model_df.columns.tolist()}")
    print(f"Available columns in default CSV: {default_df.columns.tolist()}")
    
    plt.figure(figsize=(15, 10))

    metrics_to_plot = ['waiting_time', 'queue_length', 'average_speed', 'throughput']
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        plt.plot(model_df['step'], model_df[metric], label=f'PPO Model')
        plt.plot(default_df['step'], default_df[metric], label='Default SUMO Controller')
        plt.title(metric.replace('_', ' ').title())
        plt.xlabel('Simulation Step')
        
        if metric in ['waiting_time', 'queue_length']:
            plt.ylabel('Count')
        elif metric == 'average_speed':
            plt.ylabel('m/s')
        elif metric == 'throughput':
            plt.ylabel('Vehicles per step')
            
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    comparison_path = os.path.join(graphs_dir, f"comparison_{model_name}_vs_default.png")
    plt.savefig(comparison_path)
    print(f"Comparison saved to {comparison_path}")
    
    #create a second figure for safety metrics
    plt.figure(figsize=(15, 5))
    safety_metrics = ['collisions', 'emergency_stops']
    
    for i, metric in enumerate(safety_metrics):
        plt.subplot(1, 2, i+1)
        
        #calculate totals for each model
        model_total = model_df[metric].sum()
        default_total = default_df[metric].sum()
        
        #create bar chart
        x = ['PPO Model', 'Default Controller']
        values = [model_total, default_total]
        
        #choose color based on metric
        color = 'r' if metric == 'collisions' else 'y'
        
        #create the bar chart
        bars = plt.bar(x, values, color=[color, 'gray'])
        
        #add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.title(f'Total {metric.replace("_", " ").title()}')
        plt.ylabel('Count')
        plt.grid(True, axis='y')
    
    plt.tight_layout()
    safety_comparison_path = os.path.join(graphs_dir, f"safety_comparison_{model_name}_vs_default.png")
    plt.savefig(safety_comparison_path)
    print(f"Safety comparison saved to {safety_comparison_path}")
    
    print("\nSummary Statistics:")
    all_metrics = metrics_to_plot + safety_metrics
    
    for metric in all_metrics:
        model_avg = model_df[metric].mean()
        default_avg = default_df[metric].mean()
        
        if metric in ['waiting_time', 'queue_length', 'collisions', 'emergency_stops']:
            improvement = ((default_avg - model_avg) / default_avg * 100) if default_avg != 0 else 0
            better = "lower" if model_avg < default_avg else "higher"
        else:  #for average_speed and throughput, higher is better
            improvement = ((model_avg - default_avg) / default_avg * 100) if default_avg != 0 else 0
            better = "higher" if model_avg > default_avg else "lower"
        
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  PPO Model: {model_avg:.2f}")
        print(f"  Default: {default_avg:.2f}")
        print(f"  Difference: {improvement:.2f}% {better}")
        
        #add cumulative comparisons for throughput
        if metric == 'throughput':
            model_total = model_df[metric].sum()
            default_total = default_df[metric].sum()
            diff_pct = ((model_total - default_total) / default_total * 100) if default_total != 0 else 0
            print(f"  Total throughput - PPO: {model_total} vehicles")
            print(f"  Total throughput - Default: {default_total} vehicles")
            print(f"  Throughput improvement: {diff_pct:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare simulation results")
    parser.add_argument("--model", default="ppo_traffic_light_model", help="Model name (without extension)")
    args = parser.parse_args()
    
    compare_results(args.model)