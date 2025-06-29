import numpy as np
from circuit_simulation.circuit_simulator import QuantumCircuit
import json
from datetime import datetime
import multiprocessing as mp
import os

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

shots = 4
coh_time = 10
pg = 0.001

alpha_range = np.arange(0.05, 0.525, 0.125)
alpha_distill_range = np.arange(0.05, 0.525, 0.125)

protocols = [
    ("bell_sc", 102),
    ("basic", 103),
    ("w", 104),
    ("w_to_GHZ", 107) 
]

def simulate_one(params):
    i, j, alpha, alpha_distill = params
    bell_pair_parameters = {
        "ent_prot": "single_click",
        "F_prep": 0.999,
        "p_DE": 0.01,
        "mu": 0.95,
        "lambda": 1,
        "eta": 0.4474,
        "alpha": alpha
    }
    infidelities = []
    successes = []
    statistics = []
    for _, network_noise_type in protocols:
        qc = QuantumCircuit(
            1, p_g=pg, only_GHZ=True, shots_emission_direct=shots,
            network_noise_type=network_noise_type, alpha_distill=alpha_distill,
            bell_pair_parameters=bell_pair_parameters,
            T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time,
            T2e_idle=coh_time, T1e_idle=coh_time
        )
        infidelities.append(1 - qc.F_link)
        successes.append(qc.p_link)
        statistics.append(qc.emission_direct_statistics)
    return (i, j, infidelities, successes, statistics)

if __name__ == "__main__":
    params_list = []
    for i, alpha in enumerate(alpha_range):
        for j, alpha_distill in enumerate(alpha_distill_range):
            params_list.append((i, j, alpha, alpha_distill))

    num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", mp.cpu_count()))
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.map(simulate_one, params_list)

    infidelity_data = {name: np.zeros((len(alpha_range), len(alpha_distill_range))) for name, _ in protocols}
    success_data = {name: np.zeros((len(alpha_range), len(alpha_distill_range))) for name, _ in protocols}
    statistics_data = {name: [[[] for _ in range(len(alpha_distill_range))] for _ in range(len(alpha_range))] for name, _ in protocols}

    for i, j, infidelities, successes, statistics in results:
        for k, (name, _) in enumerate(protocols):
            infidelity_data[name][i, j] = infidelities[k]
            success_data[name][i, j] = successes[k]
            statistics_data[name][i][j] = statistics[k]

    bell_pair_parameters = {
        "F_prep": 0.999,
        "p_DE": 0.01,
        "mu": 0.95
    }
    results_dict = {
        "alpha_range": alpha_range.tolist(),
        "alpha_distill_range": alpha_distill_range.tolist(),
        "infidelity_bell_sc": infidelity_data["bell_sc"].tolist(),
        "infidelity_basic": infidelity_data["basic"].tolist(),
        "infidelity_w": infidelity_data["w"].tolist(),
        "infidelity_w_to_GHZ": infidelity_data["w_to_GHZ"].tolist(),  
        "success_bell_sc": success_data["bell_sc"].tolist(),
        "success_basic": success_data["basic"].tolist(),
        "success_w": success_data["w"].tolist(),
        "success_w_to_GHZ": success_data["w_to_GHZ"].tolist(),  
        "statistics_bell_sc": statistics_data["bell_sc"],
        "statistics_basic": statistics_data["basic"],
        "statistics_w": statistics_data["w"],
        "statistics_w_to_GHZ": statistics_data["w_to_GHZ"],  
    }
    
    bright_state_raw_distillation_heatmap = rf'.\output_data\simulation_data\{timestamp}_data_bright_state_raw_distillation_heatmap_shots_{shots}_Fprep_{bell_pair_parameters["F_prep"]}_pDE_{bell_pair_parameters["p_DE"]}_mu_{bell_pair_parameters["mu"]}_cohtime_{coh_time}_pg_{pg}.json'
    with open(bright_state_raw_distillation_heatmap, 'w') as f:
        json.dump(results_dict, f, indent=4)