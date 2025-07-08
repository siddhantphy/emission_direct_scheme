import numpy as np
from circuit_simulation.circuit_simulator import QuantumCircuit
from datetime import datetime
import multiprocessing as mp
import os
import json

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

coh_time = 10
alpha = 0.05
pg = 0.001
shots = 5

bell_pair_parameters_list = [{"ent_prot":"single_click","F_prep":0.999,"p_DE":0.01,"mu":0.95,"lambda":1,"eta":0.4474,"alpha":alpha}, # ES-1 FP Emission-based from Modular architectures paper https://arxiv.org/abs/2408.02837 
                             {"ent_prot":"single_click","F_prep":0.999,"p_DE":0.0,"mu":0.95,"lambda":1,"eta":0.4474,"alpha":alpha}, # ES-2 Assuming no photon double excitation error, inline with the assumptions
                             {"ent_prot":"single_click","F_prep":0.999,"p_DE":0.0,"mu":0.96,"lambda":1,"eta":0.5,"alpha":alpha}, # ES-3 Further improvements thereon
                             {"ent_prot":"single_click","F_prep":0.999,"p_DE":0.0,"mu":0.97,"lambda":1,"eta":0.6,"alpha":alpha}, # ES-4
                             {"ent_prot":"single_click","F_prep":1,"p_DE":0.0,"mu":0.975,"lambda":1,"eta":0.65,"alpha":alpha}, # ES-5
                             {"ent_prot":"single_click","F_prep":1,"p_DE":0.0,"mu":0.98,"lambda":1,"eta":0.7,"alpha":alpha}, # ES-6
                             {"ent_prot":"single_click","F_prep":1,"p_DE":0.0,"mu":0.9825,"lambda":1,"eta":0.75,"alpha":alpha}, # ES-7
                             {"ent_prot":"single_click","F_prep":1,"p_DE":0.0,"mu":0.985,"lambda":1,"eta":0.8,"alpha":alpha}, # ES-8
                             {"ent_prot":"single_click","F_prep":1,"p_DE":0.0,"mu":0.9875,"lambda":1,"eta":0.85,"alpha":alpha}, # ES-9
                             {"ent_prot":"single_click","F_prep":1,"p_DE":0.0,"mu":0.99,"lambda":1,"eta":0.9,"alpha":alpha}, # ES-10
                             {"ent_prot":"single_click","F_prep":1,"p_DE":0.0,"mu":0.9925,"lambda":1,"eta":0.95,"alpha":alpha}, # ES-11
                             {"ent_prot":"single_click","F_prep":1,"p_DE":0.0,"mu":0.995,"lambda":1,"eta":0.96,"alpha":alpha}, # ES-12
                             {"ent_prot":"single_click","F_prep":1,"p_DE":0.0,"mu":0.9975,"lambda":1,"eta":0.97,"alpha":alpha}, # ES-13
                             {"ent_prot":"single_click","F_prep":1,"p_DE":0.0,"mu":0.998,"lambda":1,"eta":0.98,"alpha":alpha}, # ES-14
                             {"ent_prot":"single_click","F_prep":1,"p_DE":0.0,"mu":0.9985,"lambda":1,"eta":0.985,"alpha":alpha}, # ES-15
                             {"ent_prot":"single_click","F_prep":1,"p_DE":0.0,"mu":0.999,"lambda":1,"eta":0.999,"alpha":alpha}, # ES-16
                             {"ent_prot":"single_click","F_prep":1,"p_DE":0.0,"mu":1,"lambda":1,"eta":0.999,"alpha":alpha}, # ES-17
                             {"ent_prot":"single_click","F_prep":1,"p_DE":0.0,"mu":1,"lambda":1,"eta":1,"alpha":alpha}] # ES-18 Ideal - noiseless case

x_positions = list(range(len(bell_pair_parameters_list)))

def simulate_one_hardware_param(idx_and_param):
    idx, bell_pair_parameter_set = idx_and_param

    # Non-photon-number-resolving
    raw_state_qc = QuantumCircuit(
        0, p_g=pg, network_noise_type=100, only_GHZ=True, shots_emission_direct=shots,
        bell_pair_parameters=bell_pair_parameter_set,
        T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )
    dc_bell_pair_parameters = bell_pair_parameter_set.copy()
    dc_bell_pair_parameters["alpha"] = 0.5
    dc_state_qc = QuantumCircuit(
        0, p_g=pg, network_noise_type=101, only_GHZ=True, shots_emission_direct=shots,
        bell_pair_parameters=dc_bell_pair_parameters,
        T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )
    basic_distilled_state_qc = QuantumCircuit(
        0, p_g=pg, network_noise_type=103, only_GHZ=True, shots_emission_direct=shots,
        bell_pair_parameters=bell_pair_parameter_set,
        T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )
    w_distilled_state_qc = QuantumCircuit(
        0, p_g=pg, network_noise_type=104, only_GHZ=True, shots_emission_direct=shots,
        bell_pair_parameters=bell_pair_parameter_set,
        T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )
    bell_sc_distilled_state_qc = QuantumCircuit(
        0, p_g=pg, network_noise_type=102, only_GHZ=True, shots_emission_direct=shots,
        bell_pair_parameters=bell_pair_parameter_set,
        T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )
    bell_dc_pair_parameters = bell_pair_parameter_set.copy()
    bell_dc_pair_parameters["ent_prot"] = "double_click"
    bell_dc_distilled_state_qc = QuantumCircuit(
        0, p_g=pg, network_noise_type=102, only_GHZ=True, shots_emission_direct=shots,
        bell_pair_parameters=bell_dc_pair_parameters,
        T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )
    w_to_GHZ_distilled_state_qc = QuantumCircuit(
        0, p_g=pg, network_noise_type=107, only_GHZ=True, shots_emission_direct=shots,
        bell_pair_parameters=bell_pair_parameter_set,
        T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )
    w_to_GHZ_dc_state_qc = QuantumCircuit(
        0, p_g=pg, network_noise_type=106, only_GHZ=True, shots_emission_direct=shots,
        bell_pair_parameters=bell_pair_parameter_set,
        T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )

    # Photon-number-resolving
    pnr_raw_state_qc = QuantumCircuit(
        0, p_g=pg, network_noise_type=100, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True,
        bell_pair_parameters=bell_pair_parameter_set,
        T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )
    dc_bell_pair_parameters_pnr = bell_pair_parameter_set.copy()
    dc_bell_pair_parameters_pnr["alpha"] = 0.5
    pnr_dc_state_qc = QuantumCircuit(
        0, p_g=pg, network_noise_type=101, only_GHZ=True, shots_emission_direct=shots,
        bell_pair_parameters=dc_bell_pair_parameters_pnr, photon_number_resolution=True,
        T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )
    pnr_basic_distilled_state_qc = QuantumCircuit(
        0, p_g=pg, network_noise_type=103, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True,
        bell_pair_parameters=bell_pair_parameter_set,
        T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )
    pnr_w_distilled_state_qc = QuantumCircuit(
        0, p_g=pg, network_noise_type=104, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True,
        bell_pair_parameters=bell_pair_parameter_set,
        T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )
    pnr_bell_sc_distilled_state_qc = QuantumCircuit(
        0, p_g=pg, network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True,
        bell_pair_parameters=bell_pair_parameter_set,
        T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )
    bell_dc_pair_parameters_pnr = bell_pair_parameter_set.copy()
    bell_dc_pair_parameters_pnr["ent_prot"] = "double_click"
    pnr_bell_dc_distilled_state_qc = QuantumCircuit(
        0, p_g=pg, network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True,
        bell_pair_parameters=bell_dc_pair_parameters_pnr,
        T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )
    pnr_w_to_GHZ_distilled_state_qc = QuantumCircuit(
        0, p_g=pg, network_noise_type=107, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True,
        bell_pair_parameters=bell_pair_parameter_set,
        T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )
    pnr_w_to_GHZ_dc_state_qc = QuantumCircuit(
        0, p_g=pg, network_noise_type=106, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True,
        bell_pair_parameters=bell_pair_parameter_set,
        T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time
    )

    return (
        idx,
        raw_state_qc.p_link, 1-raw_state_qc.F_link,
        dc_state_qc.p_link, 1-dc_state_qc.F_link,
        basic_distilled_state_qc.p_link, 1-basic_distilled_state_qc.F_link, basic_distilled_state_qc.emission_direct_statistics,
        w_distilled_state_qc.p_link, 1-w_distilled_state_qc.F_link, w_distilled_state_qc.emission_direct_statistics,
        bell_sc_distilled_state_qc.p_link, 1-bell_sc_distilled_state_qc.F_link, bell_sc_distilled_state_qc.emission_direct_statistics,
        bell_dc_distilled_state_qc.p_link, 1-bell_dc_distilled_state_qc.F_link, bell_dc_distilled_state_qc.emission_direct_statistics,
        w_to_GHZ_distilled_state_qc.p_link, 1-w_to_GHZ_distilled_state_qc.F_link, w_to_GHZ_distilled_state_qc.emission_direct_statistics,
        w_to_GHZ_dc_state_qc.p_link, 1-w_to_GHZ_dc_state_qc.F_link,
        pnr_raw_state_qc.p_link, 1-pnr_raw_state_qc.F_link,
        pnr_dc_state_qc.p_link, 1-pnr_dc_state_qc.F_link,
        pnr_basic_distilled_state_qc.p_link, 1-pnr_basic_distilled_state_qc.F_link, pnr_basic_distilled_state_qc.emission_direct_statistics,
        pnr_w_distilled_state_qc.p_link, 1-pnr_w_distilled_state_qc.F_link, pnr_w_distilled_state_qc.emission_direct_statistics,
        pnr_bell_sc_distilled_state_qc.p_link, 1-pnr_bell_sc_distilled_state_qc.F_link, pnr_bell_sc_distilled_state_qc.emission_direct_statistics,
        pnr_bell_dc_distilled_state_qc.p_link, 1-pnr_bell_dc_distilled_state_qc.F_link, pnr_bell_dc_distilled_state_qc.emission_direct_statistics,
        pnr_w_to_GHZ_distilled_state_qc.p_link, 1-pnr_w_to_GHZ_distilled_state_qc.F_link, pnr_w_to_GHZ_distilled_state_qc.emission_direct_statistics,
        pnr_w_to_GHZ_dc_state_qc.p_link, 1-pnr_w_to_GHZ_dc_state_qc.F_link
    )

if __name__ == "__main__":
    num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", mp.cpu_count()))
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.map(simulate_one_hardware_param, list(enumerate(bell_pair_parameters_list)))

    (
        x_positions_out,
        raw_p, raw_inf,
        dc_state_p, dc_state_inf,
        basic_state_p, basic_state_inf, basic_state_statistics,
        w_state_p, w_state_inf, w_state_statistics,
        bell_sc_distilled_state_p, bell_sc_distilled_state_inf, bell_sc_distilled_state_statistics,
        bell_dc_distilled_state_p, bell_dc_distilled_state_inf, bell_dc_distilled_state_statistics,
        w_to_GHZ_distilled_state_p, w_to_GHZ_distilled_state_inf, w_to_GHZ_distilled_state_statistics,
        w_to_GHZ_dc_state_p, w_to_GHZ_dc_state_inf,
        pnr_raw_p, pnr_raw_inf,
        pnr_dc_state_p, pnr_dc_state_inf,
        pnr_basic_state_p, pnr_basic_state_inf, pnr_basic_state_statistics,
        pnr_w_state_p, pnr_w_state_inf, pnr_w_state_statistics,
        pnr_bell_sc_distilled_state_p, pnr_bell_sc_distilled_state_inf, pnr_bell_sc_distilled_state_statistics,
        pnr_bell_dc_distilled_state_p, pnr_bell_dc_distilled_state_inf, pnr_bell_dc_distilled_state_statistics,
        pnr_w_to_GHZ_distilled_state_p, pnr_w_to_GHZ_distilled_state_inf, pnr_w_to_GHZ_distilled_state_statistics,
        pnr_w_to_GHZ_dc_state_p, pnr_w_to_GHZ_dc_state_inf
    ) = map(list, zip(*results))

    arr = lambda x: np.array(x)
    valid_indices = (~np.isnan(arr(raw_inf)) & ~np.isnan(arr(dc_state_inf)) & ~np.isnan(arr(basic_state_inf)) &
                     ~np.isnan(arr(w_state_inf)) & ~np.isnan(arr(bell_sc_distilled_state_inf)) & ~np.isnan(arr(bell_dc_distilled_state_inf)))
    x_positions_array = arr(x_positions_out)
    x_positions_filtered = x_positions_array[valid_indices]
    raw_inf_filtered = arr(raw_inf)[valid_indices]
    raw_p_filtered = arr(raw_p)[valid_indices]
    dc_state_inf_filtered = arr(dc_state_inf)[valid_indices]
    dc_state_p_filtered = arr(dc_state_p)[valid_indices]
    basic_state_inf_filtered = arr(basic_state_inf)[valid_indices]
    basic_state_p_filtered = arr(basic_state_p)[valid_indices]
    basic_state_statistics_filtered = arr(basic_state_statistics)[valid_indices]
    w_state_inf_filtered = arr(w_state_inf)[valid_indices]
    w_state_p_filtered = arr(w_state_p)[valid_indices]
    w_state_statistics_filtered = arr(w_state_statistics)[valid_indices]
    bell_sc_distilled_state_inf_filtered = arr(bell_sc_distilled_state_inf)[valid_indices]
    bell_sc_distilled_state_p_filtered = arr(bell_sc_distilled_state_p)[valid_indices]
    bell_sc_distilled_state_statistics_filtered = arr(bell_sc_distilled_state_statistics)[valid_indices]
    bell_dc_distilled_state_inf_filtered = arr(bell_dc_distilled_state_inf)[valid_indices]
    bell_dc_distilled_state_p_filtered = arr(bell_dc_distilled_state_p)[valid_indices]
    bell_dc_distilled_state_statistics_filtered = arr(bell_dc_distilled_state_statistics)[valid_indices]
    w_to_GHZ_distilled_state_inf_filtered = arr(w_to_GHZ_distilled_state_inf)[valid_indices]
    w_to_GHZ_distilled_state_p_filtered = arr(w_to_GHZ_distilled_state_p)[valid_indices]
    w_to_GHZ_distilled_state_statistics_filtered = arr(w_to_GHZ_distilled_state_statistics)[valid_indices]
    w_to_GHZ_dc_state_inf_filtered = arr(w_to_GHZ_dc_state_inf)[valid_indices]
    w_to_GHZ_dc_state_p_filtered = arr(w_to_GHZ_dc_state_p)[valid_indices]

    valid_indices_pnr = (~np.isnan(arr(pnr_raw_inf)) & ~np.isnan(arr(pnr_dc_state_inf)) & ~np.isnan(arr(pnr_basic_state_inf)) &
                         ~np.isnan(arr(pnr_w_state_inf)) & ~np.isnan(arr(pnr_bell_sc_distilled_state_inf)) & ~np.isnan(arr(pnr_bell_dc_distilled_state_inf)))
    x_positions_filtered_pnr = x_positions_array[valid_indices_pnr]
    pnr_raw_inf_filtered = arr(pnr_raw_inf)[valid_indices_pnr]
    pnr_raw_p_filtered = arr(pnr_raw_p)[valid_indices_pnr]
    pnr_dc_state_p_filtered = arr(pnr_dc_state_p)[valid_indices_pnr]
    pnr_dc_state_inf_filtered = arr(pnr_dc_state_inf)[valid_indices_pnr]
    pnr_basic_state_p_filtered = arr(pnr_basic_state_p)[valid_indices_pnr]
    pnr_basic_state_inf_filtered = arr(pnr_basic_state_inf)[valid_indices_pnr]
    pnr_basic_state_statistics_filtered = arr(pnr_basic_state_statistics)[valid_indices_pnr]
    pnr_w_state_inf_filtered = arr(pnr_w_state_inf)[valid_indices_pnr]
    pnr_w_state_p_filtered = arr(pnr_w_state_p)[valid_indices_pnr]
    pnr_w_state_statistics_filtered = arr(pnr_w_state_statistics)[valid_indices_pnr]
    pnr_bell_sc_distilled_state_inf_filtered = arr(pnr_bell_sc_distilled_state_inf)[valid_indices_pnr]
    pnr_bell_sc_distilled_state_p_filtered = arr(pnr_bell_sc_distilled_state_p)[valid_indices_pnr]
    pnr_bell_sc_distilled_state_statistics_filtered = arr(pnr_bell_sc_distilled_state_statistics)[valid_indices_pnr]
    pnr_bell_dc_distilled_state_inf_filtered = arr(pnr_bell_dc_distilled_state_inf)[valid_indices_pnr]
    pnr_bell_dc_distilled_state_p_filtered = arr(pnr_bell_dc_distilled_state_p)[valid_indices_pnr]
    pnr_bell_dc_distilled_state_statistics_filtered = arr(pnr_bell_dc_distilled_state_statistics)[valid_indices_pnr]
    pnr_w_to_GHZ_distilled_state_inf_filtered = arr(pnr_w_to_GHZ_distilled_state_inf)[valid_indices_pnr]
    pnr_w_to_GHZ_distilled_state_p_filtered = arr(pnr_w_to_GHZ_distilled_state_p)[valid_indices_pnr]
    pnr_w_to_GHZ_distilled_state_statistics_filtered = arr(pnr_w_to_GHZ_distilled_state_statistics)[valid_indices_pnr]
    pnr_w_to_GHZ_dc_state_inf_filtered = arr(pnr_w_to_GHZ_dc_state_inf)[valid_indices_pnr]
    pnr_w_to_GHZ_dc_state_p_filtered = arr(pnr_w_to_GHZ_dc_state_p)[valid_indices_pnr]

    results_dict = {
        "x_positions": x_positions_filtered.tolist(),
        "raw_inf": raw_inf_filtered.tolist(),
        "raw_p": raw_p_filtered.tolist(),
        "dc_state_inf": dc_state_inf_filtered.tolist(),
        "dc_state_p": dc_state_p_filtered.tolist(),
        "basic_state_inf": basic_state_inf_filtered.tolist(),
        "basic_state_p": basic_state_p_filtered.tolist(),
        "basic_state_statistics": basic_state_statistics_filtered.tolist(),
        "w_state_inf": w_state_inf_filtered.tolist(),
        "w_state_p": w_state_p_filtered.tolist(),
        "w_state_statistics": w_state_statistics_filtered.tolist(),
        "bell_sc_distilled_state_inf": bell_sc_distilled_state_inf_filtered.tolist(),
        "bell_sc_distilled_state_p": bell_sc_distilled_state_p_filtered.tolist(),
        "bell_sc_distilled_state_statistics": bell_sc_distilled_state_statistics_filtered.tolist(),
        "bell_dc_distilled_state_inf": bell_dc_distilled_state_inf_filtered.tolist(),
        "bell_dc_distilled_state_p": bell_dc_distilled_state_p_filtered.tolist(),
        "bell_dc_distilled_state_statistics": bell_dc_distilled_state_statistics_filtered.tolist(),
        "w_to_GHZ_distilled_state_inf": w_to_GHZ_distilled_state_inf_filtered.tolist(),
        "w_to_GHZ_distilled_state_p": w_to_GHZ_distilled_state_p_filtered.tolist(),
        "w_to_GHZ_distilled_state_statistics": w_to_GHZ_distilled_state_statistics_filtered.tolist(),
        "w_to_GHZ_dc_state_inf": w_to_GHZ_dc_state_inf_filtered.tolist(),
        "w_to_GHZ_dc_state_p": w_to_GHZ_dc_state_p_filtered.tolist(),
        "x_positions_pnr": x_positions_filtered_pnr.tolist(),
        "pnr_raw_inf": pnr_raw_inf_filtered.tolist(),
        "pnr_raw_p": pnr_raw_p_filtered.tolist(),
        "pnr_dc_state_inf": pnr_dc_state_inf_filtered.tolist(),
        "pnr_dc_state_p": pnr_dc_state_p_filtered.tolist(),
        "pnr_basic_state_inf": pnr_basic_state_inf_filtered.tolist(),
        "pnr_basic_state_p": pnr_basic_state_p_filtered.tolist(),
        "pnr_basic_state_statistics": pnr_basic_state_statistics_filtered.tolist(),
        "pnr_w_state_inf": pnr_w_state_inf_filtered.tolist(),
        "pnr_w_state_p": pnr_w_state_p_filtered.tolist(),
        "pnr_w_state_statistics": pnr_w_state_statistics_filtered.tolist(),
        "pnr_bell_sc_distilled_state_inf": pnr_bell_sc_distilled_state_inf_filtered.tolist(),
        "pnr_bell_sc_distilled_state_p": pnr_bell_sc_distilled_state_p_filtered.tolist(),
        "pnr_bell_sc_distilled_state_statistics": pnr_bell_sc_distilled_state_statistics_filtered.tolist(),
        "pnr_bell_dc_distilled_state_inf": pnr_bell_dc_distilled_state_inf_filtered.tolist(),
        "pnr_bell_dc_distilled_state_p": pnr_bell_dc_distilled_state_p_filtered.tolist(),
        "pnr_bell_dc_distilled_state_statistics": pnr_bell_dc_distilled_state_statistics_filtered.tolist(),
        "pnr_w_to_GHZ_distilled_state_inf": pnr_w_to_GHZ_distilled_state_inf_filtered.tolist(),
        "pnr_w_to_GHZ_distilled_state_p": pnr_w_to_GHZ_distilled_state_p_filtered.tolist(),
        "pnr_w_to_GHZ_distilled_state_statistics": pnr_w_to_GHZ_distilled_state_statistics_filtered.tolist(),
        "pnr_w_to_GHZ_dc_state_inf": pnr_w_to_GHZ_dc_state_inf_filtered.tolist(),
        "pnr_w_to_GHZ_dc_state_p": pnr_w_to_GHZ_dc_state_p_filtered.tolist(),
    }

    hardware_parameters_data = rf'.\output_data\simulation_data\{timestamp}_data_hardware_parameters_improvement_success_rates_shots_{shots}_alpha_{alpha}_cohtime_{coh_time}_pg_{pg}.json'
    with open(hardware_parameters_data, 'w') as f:
        json.dump(results_dict, f, indent=4)
