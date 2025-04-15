# Run the training script
python -m src.agent.training

# After training, you can run a simulation with the trained model using the run_sim.py script:
python run_sim.py --net src/networks/2way_single/single.net.xml --route src/networks/2way_single/single_horizontal.rou.xml --model ppo_traffic_light_model.zip