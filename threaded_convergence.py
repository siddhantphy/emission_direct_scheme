# Convergence plots with multiprocessing for supercomputer use
import numpy as np
from circuit_simulation.circuit_simulator import QuantumCircuit
import json
from datetime import datetime
import multiprocessing as mp
import os

# Get current date and time in the format YYYYMMDDHHMMSS
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

coh_time = 10
pg = 0.001
bell_pair_parameters = {"ent_prot":"single_click","F_prep":0.999,"p_DE":0.01,"mu":0.95,"lambda":1,"eta":0.4474,"alpha":0.05}

# Define the number of iterations
shots = np.array([1,2,3,4])  # Example: 1 to 20 shots

# Number of runs per shot value (for error bars)
runs_per_shot = 100  # Increase for more statistics

def simulate_one_run(args):
    shot, coh_time, pg, bell_pair_parameters = args
    qc = QuantumCircuit(
        1, p_g=pg, network_noise_type=103, only_GHZ=True, shots_emission_direct=shot,
        bell_pair_parameters=bell_pair_parameters, T2n_idle=coh_time, T1n_idle=coh_time,
        T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )
    return qc.p_link, 1 - qc.F_link

if __name__ == "__main__":
    # Use all available CPUs
    num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", mp.cpu_count()))
    pool = mp.Pool(processes=num_cpus)

    success_probabilities = []
    infidelities = []
    success_prob_std = []
    infidelity_std = []

    for shot in shots:
        # Prepare arguments for each run
        args = [(shot, coh_time, pg, bell_pair_parameters) for _ in range(runs_per_shot)]
        results = pool.map(simulate_one_run, args)
        success_probs, infidelities_iter = zip(*results)
        success_probabilities.append(np.mean(success_probs))
        success_prob_std.append(np.std(success_probs))
        infidelities.append(np.mean(infidelities_iter))
        infidelity_std.append(np.std(infidelities_iter))

    pool.close()
    pool.join()

    # Save results to a JSON file for later plotting
    results_dict = {
        "iterations": shots.tolist(),
        "success_probabilities": success_probabilities,
        "success_prob_std": success_prob_std,
        "infidelities": infidelities,
        "infidelity_std": infidelity_std
    }

    os.makedirs("./output_data/simulation_data", exist_ok=True)
    convergence_data = rf"./output_data/simulation_data/{timestamp}_data_convergence.json"
    with open(convergence_data, "w") as f:
        json.dump(results_dict, f, indent=2)