"""
run_parallel_MPI.py
Author: Sacha Guerrini

A script to run the forward model on different cosmology in parallel on the cluster using MPI.
"""
import subprocess
import yaml
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(size)

#Path to the base config file
BASE_CONFIG = "config.yaml"

#Path to the script that needs to be run
RUN_SCRIPT = "run.py"

#Total number of simulations
NUM_SIMULATIONS = 791

#Number of parallel processes
NUM_CORES = 20

def run_simulation(sim_number):
    """
    Function to run a single simulation with a given sim_number.
    It modifies the config file and runs run.py with it.
    """

    print(f"Rank {rank}:Simulation {sim_number} starting...")
    
    #Load base configuration
    with open(BASE_CONFIG, "r") as f:
        config = yaml.safe_load(f)

    #Modify simulation number
    config["simulation"]["sim_number"] = sim_number

    #Create a temporary config file
    temp_config_file = f"temp_config_{sim_number}.yaml"
    with open(temp_config_file, 'w') as f:
        yaml.safe_dump(config, f)

    command = ["python", RUN_SCRIPT, "--config", temp_config_file]
    result = subprocess.run(command, capture_output=True, text=True)

    print(f"Rank {rank}:Simulation {sim_number} completed with output:\n")
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")

    #Cleanup
    os.remove(temp_config_file)

#Distribution simulations among available ranks
for i in range(rank, NUM_SIMULATIONS, size):
    if os.path.exists(f'/lustre/fswork/projects/rech/prk/commun/GowerStreetSims/UNIONS_processing/forward_model_sim{i+1:05d}_nside0512_rot44_noisereal1.npy'):
        continue
    run_simulation(i + 1)

#Finalize MPI
MPI.Finalize()
