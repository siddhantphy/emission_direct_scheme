import numpy as np
from circuit_simulation.circuit_simulator import QuantumCircuit
import json
from datetime import datetime
import multiprocessing as mp
import os

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

shots = 1
coh_time = 10
pg = 0.001

alpha_range = np.arange(0.0010101, 0.5, 0.1)
alpha_distill_range = np.arange(0.0010101, 0.5, 0.1)

protocols = [
    ("bell_sc", 102),
    ("basic", 103),
    ("w", 104)
]

def simulate_one(params):
    i, j, alpha, alpha_distill = params
    bell_pair_parameters = {"ent_prot": "single_click", "F_prep": 0.999, "p_DE": 0.01, "mu": 0.95, "lambda": 1, "eta": 0.4474, "alpha": alpha}
    results = []
    for name, network_noise_type in protocols:
        qc = QuantumCircuit(
            1, p_g=pg, network_noise_type=network_noise_type, alpha_distill=alpha_distill,
            bell_pair_parameters=bell_pair_parameters,
            T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time,
            T2e_idle=coh_time, T1e_idle=coh_time
        )
        infidelity = 1 - qc.F_link
        success = qc.p_link
        statistics = qc.emission_direct_statistics
        results.append((infidelity, success, statistics))
    return (i, j, results)

if __name__ == "__main__":
    params_list = []
    for i, alpha in enumerate(alpha_range):
        for j, alpha_distill in enumerate(alpha_distill_range):
            params_list.append((i, j, alpha, alpha_distill))

    num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", mp.cpu_count()))
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.map(simulate_one, params_list)

    infidelity_bell_sc = np.zeros((len(alpha_range), len(alpha_distill_range)))
    infidelity_basic = np.zeros((len(alpha_range), len(alpha_distill_range)))
    infidelity_w = np.zeros((len(alpha_range), len(alpha_distill_range)))

    success_bell_sc = np.zeros((len(alpha_range), len(alpha_distill_range)))
    success_basic = np.zeros((len(alpha_range), len(alpha_distill_range)))
    success_w = np.zeros((len(alpha_range), len(alpha_distill_range)))

    statistics_bell_sc = [[[] for _ in range(len(alpha_distill_range))] for _ in range(len(alpha_range))]
    statistics_basic = [[[] for _ in range(len(alpha_distill_range))] for _ in range(len(alpha_range))]
    statistics_w = [[[] for _ in range(len(alpha_distill_range))] for _ in range(len(alpha_range))]

    for i, j, results_tuple in results:
        (inf_bell_sc, succ_bell_sc, stat_bell_sc), (inf_basic, succ_basic, stat_basic), (inf_w, succ_w, stat_w) = results_tuple
        infidelity_bell_sc[i, j] = inf_bell_sc
        infidelity_basic[i, j] = inf_basic
        infidelity_w[i, j] = inf_w

        success_bell_sc[i, j] = succ_bell_sc
        success_basic[i, j] = succ_basic
        success_w[i, j] = succ_w

        statistics_bell_sc[i][j] = stat_bell_sc
        statistics_basic[i][j] = stat_basic
        statistics_w[i][j] = stat_w

    results_dict = {
        "alpha_range": alpha_range.tolist(),
        "alpha_distill_range": alpha_distill_range.tolist(),
        "infidelity_bell_sc": infidelity_bell_sc.tolist(),
        "infidelity_basic": infidelity_basic.tolist(),
        "infidelity_w": infidelity_w.tolist(),
        "success_bell_sc": success_bell_sc.tolist(),
        "success_basic": success_basic.tolist(),
        "success_w": success_w.tolist(),
        "statistics_bell_sc": statistics_bell_sc,
        "statistics_basic": statistics_basic,
        "statistics_w": statistics_w,
    }

    bell_pair_parameters = {"ent_prot": "single_click", "F_prep": 0.999, "p_DE": 0.01, "mu": 0.95, "lambda": 1, "eta": 0.4474, "alpha": alpha_range[0]}
    bright_state_raw_distillation_heatmap = rf'.\output_data\simulation_data\{timestamp}_data_bright_state_raw_distillation_heatmap_shots_{shots}_Fprep_{bell_pair_parameters["F_prep"]}_pDE_{bell_pair_parameters["p_DE"]}_mu_{bell_pair_parameters["mu"]}_cohtime_{coh_time}_pg_{pg}.json'
    with open(bright_state_raw_distillation_heatmap, 'w') as f:
        json.dump(results_dict, f, indent=4)