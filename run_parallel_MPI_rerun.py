import subprocess
import yaml
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

BASE_CONFIG = "config.yaml"
RUN_SCRIPT = "run.py"
NUM_SIMULATIONS = 791
RESULTS_PATH = "/lustre/fswork/projects/rech/prk/commun/GowerStreetSims/UNIONS_processing"
TAG_WORK = 1
TAG_STOP = 2

def run_simulation(sim_number):
    print(f"Rank {rank}: Simulation {sim_number} starting...")

    with open(BASE_CONFIG, "r") as f:
        config = yaml.safe_load(f)

    config["simulation"]["sim_number"] = sim_number

    temp_config_file = f"temp_config_{sim_number}.yaml"
    with open(temp_config_file, 'w') as f:
        yaml.safe_dump(config, f)

    command = ["python", RUN_SCRIPT, "--config", temp_config_file]
    result = subprocess.run(command, capture_output=True, text=True)

    print(f"Rank {rank}: Simulation {sim_number} completed")
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")

    os.remove(temp_config_file)

if rank == 0:
    # MASTER
    missing_sims = []
    for i in range(1, NUM_SIMULATIONS + 1):
        result_file = os.path.join(RESULTS_PATH, f"forward_model_sim{i:05d}_nside0512_rot44_noisereal1.npy")
        if not os.path.exists(result_file):
            missing_sims.append(i)

    print(f"Rank 0: {len(missing_sims)} simulations to run.")

    sim_iter = iter(missing_sims)
    num_workers = size - 1
    active_workers = 0

    # Initial distribution
    for worker in range(1, size):
        try:
            sim_number = next(sim_iter)
            comm.send(sim_number, dest=worker, tag=TAG_WORK)
            active_workers += 1
        except StopIteration:
            break

    while active_workers > 0:
        status = MPI.Status()
        _ = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        worker = status.Get_source()

        try:
            sim_number = next(sim_iter)
            comm.send(sim_number, dest=worker, tag=TAG_WORK)
        except StopIteration:
            comm.send(None, dest=worker, tag=TAG_STOP)
            active_workers -= 1

else:
    # WORKER
    while True:
        status = MPI.Status()
        sim_number = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == TAG_STOP:
            break
        
        
        print(f"Running Simulation number {sim_number}")
        run_simulation(sim_number)
        comm.send(None, dest=0)