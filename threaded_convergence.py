# Convergence plots (threaded)
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
shots = np.array([5, 10, 50, 100])

def simulate_one_shot(shot):
    qc = QuantumCircuit(
        0, p_g=pg, network_noise_type=103, only_GHZ=True, shots_emission_direct=shot,
        bell_pair_parameters=bell_pair_parameters, T2n_idle=coh_time, T1n_idle=coh_time,
        T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )
    # Return all needed statistics for later std calculation
    return (
        shot,
        qc.p_link,
        1 - qc.F_link,
        qc.emission_direct_statistics
    )

if __name__ == "__main__":
    num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", mp.cpu_count()))
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.map(simulate_one_shot, shots)

    # Unpack results
    shots_out, success_probabilities, infidelities, statistics = map(list, zip(*results))

    # Calculate stds
    success_prob_std = [np.std(np.array(i["p_link"])) for i in statistics]
    infidelity_std = [np.std(1 - np.array(i["F_link"])) for i in statistics]

    # Save results to a JSON file for later plotting
    results_dict = {
        "iterations": [int(x) for x in shots_out],
        "success_probabilities": [float(x) for x in success_probabilities],
        "success_prob_std": [float(x) for x in success_prob_std],
        "infidelities": [float(x) for x in infidelities],
        "infidelity_std": [float(x) for x in infidelity_std]
    }

    convergence_data = f".\\output_data\\simulation_data\\{timestamp}_data_convergence.json"
    with open(convergence_data, "w") as f:
        json.dump(results_dict, f, indent=2)