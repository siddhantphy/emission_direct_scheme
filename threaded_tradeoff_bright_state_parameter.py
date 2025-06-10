import numpy as np
from circuit_simulation.circuit_simulator import QuantumCircuit
import json
from datetime import datetime
import multiprocessing as mp
import os

# Get current date and time in the format YYYYMMDDHHMMSS
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Shots
shots = 1

# Coherence times
coh_time = 10  # Coherence times

# Gate error in GHZ state preparation
pg = 0.001

# Define the range for alpha and alpha_distill for infidelity
alpha_range = np.arange(0.0010101, 0.5, 0.1)
alpha_distill_range = np.arange(0.0010101, 0.5, 0.1)

def simulate_one_alpha(params):
    i, j, alpha, alpha_distill = params
    bell_pair_parameters = {"ent_prot": "single_click", "F_prep": 0.999, "p_DE": 0.01, "mu": 0.95, "lambda": 1, "eta": 0.4474, "alpha": alpha}
    # Bell state protocol with single click
    bell_sc_distilled_state_qc = QuantumCircuit(1, p_g=pg, network_noise_type=102, alpha_distill=alpha_distill, bell_pair_parameters=bell_pair_parameters, T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    infidelity_bell_sc = 1 - bell_sc_distilled_state_qc.F_link
    success_bell_sc = bell_sc_distilled_state_qc.p_link

    # Basic state protocol
    basic_distilled_state_qc = QuantumCircuit(1, p_g=pg, network_noise_type=103, alpha_distill=alpha_distill, bell_pair_parameters=bell_pair_parameters, T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    infidelity_basic = 1 - basic_distilled_state_qc.F_link
    success_basic = basic_distilled_state_qc.p_link

    # W state protocol
    w_distilled_state_qc = QuantumCircuit(1, p_g=pg, network_noise_type=104, alpha_distill=alpha_distill, bell_pair_parameters=bell_pair_parameters, T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    infidelity_w = 1 - w_distilled_state_qc.F_link
    success_w = w_distilled_state_qc.p_link

    return (i, j, infidelity_bell_sc, infidelity_basic, infidelity_w, success_bell_sc, success_basic, success_w)

if __name__ == "__main__":
    params_list = []
    for i, alpha in enumerate(alpha_range):
        for j, alpha_distill in enumerate(alpha_distill_range):
            params_list.append((i, j, alpha, alpha_distill))

    num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", mp.cpu_count()))
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.map(simulate_one_alpha, params_list)

    # Initialize matrices to store infidelity and success probability values
    infidelity_bell_sc = np.zeros((len(alpha_range), len(alpha_distill_range)))
    infidelity_basic = np.zeros((len(alpha_range), len(alpha_distill_range)))
    infidelity_w = np.zeros((len(alpha_range), len(alpha_distill_range)))
    success_bell_sc = np.zeros((len(alpha_range), len(alpha_distill_range)))
    success_basic = np.zeros((len(alpha_range), len(alpha_distill_range)))
    success_w = np.zeros((len(alpha_range), len(alpha_distill_range)))

    for i, j, inf_bell_sc, inf_basic, inf_w, succ_bell_sc, succ_basic, succ_w in results:
        infidelity_bell_sc[i, j] = inf_bell_sc
        infidelity_basic[i, j] = inf_basic
        infidelity_w[i, j] = inf_w
        success_bell_sc[i, j] = succ_bell_sc
        success_basic[i, j] = succ_basic
        success_w[i, j] = succ_w

    # Store the results in a dictionary for easy access
    results_dict = {
        "infidelity_bell_sc": infidelity_bell_sc.tolist(),
        "infidelity_basic": infidelity_basic.tolist(),
        "infidelity_w": infidelity_w.tolist(),
        "success_bell_sc": success_bell_sc.tolist(),
        "success_basic": success_basic.tolist(),
        "success_w": success_w.tolist()
    }
    # Save the data to a json file
    bell_pair_parameters = {"ent_prot": "single_click", "F_prep": 0.999, "p_DE": 0.01, "mu": 0.95, "lambda": 1, "eta": 0.4474, "alpha": alpha_range[0]}
    bright_state_raw_distillation_heatmap = rf'.\output_data\simulation_data\{timestamp}_data_bright_state_raw_distillation_heatmap_shots_{shots}_Fprep_{bell_pair_parameters["F_prep"]}_pDE_{bell_pair_parameters["p_DE"]}_mu_{bell_pair_parameters["mu"]}_cohtime_{coh_time}_pg_{pg}.json'
    with open(bright_state_raw_distillation_heatmap, 'w') as f:
        json.dump(results_dict, f, indent=4)