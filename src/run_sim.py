import sys
import os

#make sure SUMO_HOME environment variable is set
if "SUMO_HOME" not in os.environ:
    print("Error: SUMO_HOME environment variable not set.")
    print("Please set it to your SUMO installation directory")
    sys.exit(1)

#add SUMO tools to python path
if os.environ.get("SUMO_HOME"):
    tools_dir = os.path.join(os.environ.get("SUMO_HOME"), "tools")
    sys.path.append(tools_dir)

from src.environment.sumo_runner import main

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error running simulation: {e}")
        sys.exit(1)

#python -c "import os,sys; sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools')); import traci; traci.start(['sumo-gui', '-n', 'src/networks/2way_single/single.net.xml']); print(traci.trafficlight.getIDList()); print(traci.trafficlight.getRedYellowGreenState('t')); print(traci.trafficlight.getPhaseDefinition('t')); traci.close()"
#python run_sim.py --net src/networks/2way_single/single.net.xml --route src/networks/2way_single/single_horizontal.rou.xml