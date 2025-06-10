import numpy as np
from circuit_simulation.circuit_simulator import QuantumCircuit
from datetime import datetime
import json
import multiprocessing as mp
import os

# Get current date and time in the format YYYYMMDDHHMMSS
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Coherence times improvement
coh_time = 10

# Operational point for alpha
alpha = 0.05

# Gate error in GHZ generation
pg = 0.001

# Shots
shots = 1

# Bell-state parameters
bell_pair_parameters_list = [
    {"ent_prot":"single_click","F_prep":0.999,"p_DE":0.01,"mu":0.95,"lambda":1,"eta":0.4474,"alpha":alpha},
    {"ent_prot":"single_click","F_prep":0.999,"p_DE":0.008,"mu":0.96,"lambda":1,"eta":0.62,"alpha":alpha},
    {"ent_prot":"single_click","F_prep":0.999,"p_DE":0.006,"mu":0.97,"lambda":1,"eta":0.8,"alpha":alpha},
    {"ent_prot":"single_click","F_prep":1,"p_DE":0.004,"mu":0.98,"lambda":1,"eta":0.9,"alpha":alpha},
    {"ent_prot":"single_click","F_prep":1,"p_DE":0.0025,"mu":0.99,"lambda":1,"eta":0.95,"alpha":alpha},
    {"ent_prot":"single_click","F_prep":1,"p_DE":0.001,"mu":0.99,"lambda":1,"eta":0.98,"alpha":alpha},
    {"ent_prot":"single_click","F_prep":1,"p_DE":0.0,"mu":1,"lambda":1,"eta":0.99,"alpha":alpha},
    {"ent_prot":"single_click","F_prep":1,"p_DE":0.0,"mu":1,"lambda":1,"eta":1,"alpha":alpha}
]
x_positions = list(range(len(bell_pair_parameters_list)))

def simulate_one_hardware_param(idx_and_param):
    idx, bell_pair_parameter_set = idx_and_param
    # Standard
    raw_state_qc = QuantumCircuit(1, p_g=pg, network_noise_type=100, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_pair_parameter_set, T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    dc_bell_pair_parameters = bell_pair_parameter_set.copy()
    dc_bell_pair_parameters["alpha"] = 0.5
    dc_state_qc = QuantumCircuit(1, p_g=pg, network_noise_type=101, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=dc_bell_pair_parameters, T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    basic_distilled_state_qc = QuantumCircuit(1, p_g=pg, network_noise_type=103, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_pair_parameter_set, T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    w_distilled_state_qc = QuantumCircuit(1, p_g=pg, network_noise_type=104, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_pair_parameter_set, T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    bell_sc_distilled_state_qc = QuantumCircuit(1, p_g=pg, network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_pair_parameter_set, T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    bell_dc_pair_parameters = bell_pair_parameter_set.copy()
    bell_dc_pair_parameters["ent_prot"] = "double_click"
    bell_dc_distilled_state_qc = QuantumCircuit(1, p_g=pg, network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_dc_pair_parameters, T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    # PNR
    pnr_raw_state_qc = QuantumCircuit(1, p_g=pg, network_noise_type=100, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True, bell_pair_parameters=bell_pair_parameter_set, T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    pnr_dc_bell_pair_parameters = bell_pair_parameter_set.copy()
    pnr_dc_bell_pair_parameters["alpha"] = 0.5
    pnr_dc_state_qc = QuantumCircuit(1, p_g=pg, network_noise_type=101, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=pnr_dc_bell_pair_parameters, T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    pnr_basic_distilled_state_qc = QuantumCircuit(1, p_g=pg, network_noise_type=103, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True, bell_pair_parameters=bell_pair_parameter_set, T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    pnr_w_distilled_state_qc = QuantumCircuit(1, p_g=pg, network_noise_type=104, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True, bell_pair_parameters=bell_pair_parameter_set, T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    pnr_bell_sc_distilled_state_qc = QuantumCircuit(1, p_g=pg, network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True, bell_pair_parameters=bell_pair_parameter_set, T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    pnr_bell_dc_pair_parameters = bell_pair_parameter_set.copy()
    pnr_bell_dc_pair_parameters["ent_prot"] = "double_click"
    pnr_bell_dc_distilled_state_qc = QuantumCircuit(1, p_g=pg, network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True, bell_pair_parameters=pnr_bell_dc_pair_parameters, T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    return (
        idx,
        raw_state_qc.p_link, 1-raw_state_qc.F_link,
        dc_state_qc.p_link, 1-dc_state_qc.F_link,
        basic_distilled_state_qc.p_link, 1-basic_distilled_state_qc.F_link,
        w_distilled_state_qc.p_link, 1-w_distilled_state_qc.F_link,
        bell_sc_distilled_state_qc.p_link, 1-bell_sc_distilled_state_qc.F_link,
        bell_dc_distilled_state_qc.p_link, 1-bell_dc_distilled_state_qc.F_link,
        pnr_raw_state_qc.p_link, 1-pnr_raw_state_qc.F_link,
        pnr_dc_state_qc.p_link, 1-pnr_dc_state_qc.F_link,
        pnr_basic_distilled_state_qc.p_link, 1-pnr_basic_distilled_state_qc.F_link,
        pnr_w_distilled_state_qc.p_link, 1-pnr_w_distilled_state_qc.F_link,
        pnr_bell_sc_distilled_state_qc.p_link, 1-pnr_bell_sc_distilled_state_qc.F_link,
        pnr_bell_dc_distilled_state_qc.p_link, 1-pnr_bell_dc_distilled_state_qc.F_link
    )

if __name__ == "__main__":
    num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", mp.cpu_count()))
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.map(simulate_one_hardware_param, list(enumerate(bell_pair_parameters_list)))
    (
        x_positions_out,
        raw_p, raw_inf,
        dc_state_p, dc_state_inf,
        basic_state_p, basic_state_inf,
        w_state_p, w_state_inf,
        bell_sc_distilled_state_p, bell_sc_distilled_state_inf,
        bell_dc_distilled_state_p, bell_dc_distilled_state_inf,
        pnr_raw_p, pnr_raw_inf,
        pnr_dc_state_p, pnr_dc_state_inf,
        pnr_basic_state_p, pnr_basic_state_inf,
        pnr_w_state_p, pnr_w_state_inf,
        pnr_bell_sc_distilled_state_p, pnr_bell_sc_distilled_state_inf,
        pnr_bell_dc_distilled_state_p, pnr_bell_dc_distilled_state_inf
    ) = map(list, zip(*results))

    x_positions_array = np.array(x_positions_out)
    raw_inf_arr = np.array(raw_inf)
    dc_state_inf_arr = np.array(dc_state_inf)
    basic_state_inf_arr = np.array(basic_state_inf)
    w_state_inf_arr = np.array(w_state_inf)
    bell_sc_distilled_state_inf_arr = np.array(bell_sc_distilled_state_inf)
    bell_dc_distilled_state_inf_arr = np.array(bell_dc_distilled_state_inf)
    pnr_raw_inf_arr = np.array(pnr_raw_inf)
    pnr_dc_state_inf_arr = np.array(pnr_dc_state_inf)
    pnr_basic_state_inf_arr = np.array(pnr_basic_state_inf)
    pnr_w_state_inf_arr = np.array(pnr_w_state_inf)
    pnr_bell_sc_distilled_state_inf_arr = np.array(pnr_bell_sc_distilled_state_inf)
    pnr_bell_dc_distilled_state_inf_arr = np.array(pnr_bell_dc_distilled_state_inf)

    valid_indices = (~np.isnan(raw_inf_arr) & ~np.isnan(dc_state_inf_arr) & ~np.isnan(basic_state_inf_arr) &
                     ~np.isnan(w_state_inf_arr) & ~np.isnan(bell_sc_distilled_state_inf_arr) & ~np.isnan(bell_dc_distilled_state_inf_arr))
    x_positions_filtered = x_positions_array[valid_indices]
    raw_inf_filtered = raw_inf_arr[valid_indices]
    dc_state_inf_filtered = dc_state_inf_arr[valid_indices]
    basic_state_inf_filtered = basic_state_inf_arr[valid_indices]
    w_state_inf_filtered = w_state_inf_arr[valid_indices]
    bell_sc_distilled_state_inf_filtered = bell_sc_distilled_state_inf_arr[valid_indices]
    bell_dc_distilled_state_inf_filtered = bell_dc_distilled_state_inf_arr[valid_indices]

    valid_indices_pnr = (~np.isnan(pnr_raw_inf_arr) & ~np.isnan(pnr_dc_state_inf_arr) & ~np.isnan(pnr_basic_state_inf_arr) &
                         ~np.isnan(pnr_w_state_inf_arr) & ~np.isnan(pnr_bell_sc_distilled_state_inf_arr) & ~np.isnan(pnr_bell_dc_distilled_state_inf_arr))
    x_positions_filtered_pnr = x_positions_array[valid_indices_pnr]
    pnr_raw_inf_filtered = pnr_raw_inf_arr[valid_indices_pnr]
    pnr_dc_state_inf_filtered = pnr_dc_state_inf_arr[valid_indices_pnr]
    pnr_basic_state_inf_filtered = pnr_basic_state_inf_arr[valid_indices_pnr]
    pnr_w_state_inf_filtered = pnr_w_state_inf_arr[valid_indices_pnr]
    pnr_bell_sc_distilled_state_inf_filtered = pnr_bell_sc_distilled_state_inf_arr[valid_indices_pnr]
    pnr_bell_dc_distilled_state_inf_filtered = pnr_bell_dc_distilled_state_inf_arr[valid_indices_pnr]

    results_dict = {
        "x_positions": list(x_positions_out),
        "raw_p": raw_p,
        "raw_inf": raw_inf,
        "dc_state_p": dc_state_p,
        "dc_state_inf": dc_state_inf,
        "basic_state_p": basic_state_p,
        "basic_state_inf": basic_state_inf,
        "w_state_p": w_state_p,
        "w_state_inf": w_state_inf,
        "bell_sc_distilled_state_p": bell_sc_distilled_state_p,
        "bell_sc_distilled_state_inf": bell_sc_distilled_state_inf,
        "bell_dc_distilled_state_p": bell_dc_distilled_state_p,
        "bell_dc_distilled_state_inf": bell_dc_distilled_state_inf,
        "pnr_raw_p": pnr_raw_p,
        "pnr_raw_inf": pnr_raw_inf,
        "pnr_dc_state_p": pnr_dc_state_p,
        "pnr_dc_state_inf": pnr_dc_state_inf,
        "pnr_basic_state_p": pnr_basic_state_p,
        "pnr_basic_state_inf": pnr_basic_state_inf,
        "pnr_w_state_p": pnr_w_state_p,
        "pnr_w_state_inf": pnr_w_state_inf,
        "pnr_bell_sc_distilled_state_p": pnr_bell_sc_distilled_state_p,
        "pnr_bell_sc_distilled_state_inf": pnr_bell_sc_distilled_state_inf,
        "pnr_bell_dc_distilled_state_p": pnr_bell_dc_distilled_state_p,
        "pnr_bell_dc_distilled_state_inf": pnr_bell_dc_distilled_state_inf
    }

    hardware_parameters_data = rf'.\output_data\simulation_data\{timestamp}_data_hardware_parameters_improvement_success_rates_shots_{shots}_alpha_{alpha}_cohtime_{coh_time}_pg_{pg}.json'
    with open(hardware_parameters_data, 'w') as f:
        json.dump(results_dict, f, indent=4)
