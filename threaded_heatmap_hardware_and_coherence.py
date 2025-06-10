import numpy as np
from circuit_simulation.circuit_simulator import QuantumCircuit
import json
from datetime import datetime
import multiprocessing as mp
import os

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

shots = 1
alpha = 0.05
pg = 0.001

coh_times = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25, 50, 75, 100]

bell_pair_parameters_list = [
    {"ent_prot": "single_click", "F_prep": 0.999, "p_DE": 0.01, "mu": 0.95, "lambda": 1, "eta": 0.4474, "alpha": alpha},
    {"ent_prot": "single_click", "F_prep": 0.999, "p_DE": 0.008, "mu": 0.96, "lambda": 1, "eta": 0.62, "alpha": alpha},
    {"ent_prot": "single_click", "F_prep": 0.999, "p_DE": 0.006, "mu": 0.97, "lambda": 1, "eta": 0.8, "alpha": alpha},
    {"ent_prot": "single_click", "F_prep": 1, "p_DE": 0.004, "mu": 0.98, "lambda": 1, "eta": 0.9, "alpha": alpha},
    {"ent_prot": "single_click", "F_prep": 1, "p_DE": 0.002, "mu": 0.99, "lambda": 1, "eta": 0.95, "alpha": alpha},
    {"ent_prot": "single_click", "F_prep": 1, "p_DE": 0.001, "mu": 0.99, "lambda": 1, "eta": 0.98, "alpha": alpha},
    {"ent_prot": "single_click", "F_prep": 1, "p_DE": 0.0, "mu": 1, "lambda": 1, "eta": 0.99, "alpha": alpha},
    {"ent_prot": "single_click", "F_prep": 1, "p_DE": 0.0, "mu": 1, "lambda": 1, "eta": 1, "alpha": alpha}
]

protocols = ["Direct Raw", "Direct DC", "Distill Basic", "Distill W", "Distill SC Bell", "Distill DC Bell"]

def simulate_one(params):
    i, j, coh_time, bell_pair_parameters = params
    infidelities = []
    success_rates = []
    statistics = []
    for protocol in protocols:
        bell_params = bell_pair_parameters.copy()
        # Set alpha=0.5 only for the "Direct DC" protocol
        if protocol == "Direct DC":
            bell_params["alpha"] = 0.5
        else:
            bell_params["alpha"] = alpha
        if protocol == "Distill DC Bell":
            bell_params["ent_prot"] = "double_click"
        else:
            bell_params["ent_prot"] = "single_click"
        network_noise_type = {
            "Direct Raw": 100,
            "Direct DC": 101,
            "Distill Basic": 103,
            "Distill W": 104,
            "Distill SC Bell": 102,
            "Distill DC Bell": 102
        }[protocol]
        qc = QuantumCircuit(
            1, p_g=pg, network_noise_type=network_noise_type, only_GHZ=True, shots_emission_direct=shots,
            bell_pair_parameters=bell_params,
            T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time,
            T2e_idle=coh_time, T1e_idle=coh_time
        )
        infidelities.append(1 - qc.F_link)
        success_rates.append(qc.p_link)
        statistics.append(qc.emission_direct_statistics)
    return (i, j, infidelities, success_rates, statistics)

if __name__ == "__main__":
    params_list = []
    for i, coh_time in enumerate(coh_times):
        for j, bell_pair_parameters in enumerate(bell_pair_parameters_list):
            params_list.append((i, j, coh_time, bell_pair_parameters))

    num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", mp.cpu_count()))
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.map(simulate_one, params_list)

    infidelity_data = {protocol: np.zeros((len(coh_times), len(bell_pair_parameters_list))) for protocol in protocols}
    success_rate_data = {protocol: np.zeros((len(coh_times), len(bell_pair_parameters_list))) for protocol in protocols}
    statistics_data = {protocol: [[[] for _ in range(len(bell_pair_parameters_list))] for _ in range(len(coh_times))] for protocol in protocols}

    for i, j, infidelities, success_rates, statistics in results:
        for k, protocol in enumerate(protocols):
            infidelity_data[protocol][i, j] = infidelities[k]
            success_rate_data[protocol][i, j] = success_rates[k]
            statistics_data[protocol][i][j] = statistics[k]

    results_dict = {
        "coh_times": coh_times,
        "bell_pair_parameters_list": bell_pair_parameters_list,
        "protocols": protocols,
        "infidelity_data": {k: v.tolist() for k, v in infidelity_data.items()},
        "success_rate_data": {k: v.tolist() for k, v in success_rate_data.items()},
        "statistics_data": {k: v for k, v in statistics_data.items()}
    }
    heatmap_data_file = rf'.\output_data\simulation_data\{timestamp}_heatmap_hardware_coherence_parameters_shots_{shots}_alpha_{alpha}_pg_{pg}.json'
    with open(heatmap_data_file, 'w') as f:
        json.dump(results_dict, f, indent=2)