import numpy as np
from circuit_simulation.circuit_simulator import QuantumCircuit
from datetime import datetime
import multiprocessing as mp
import os
import json

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

shots = 2
alpha = 0.05
pg = 0.001
coh_times = [0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25, 50, 75, 100]

bell_pair_parameters = {"ent_prot":"single_click","F_prep":0.999,"p_DE":0.01,"mu":0.95,"lambda":1,"eta":0.4474,"alpha":alpha}
dc_bell_pair_parameters = {"ent_prot":"single_click","F_prep":0.999,"p_DE":0.01,"mu":0.95,"lambda":1,"eta":0.4474,"alpha":0.5}
bell_dc_pair_parameters = {"ent_prot":"double_click","F_prep":0.999,"p_DE":0.01,"mu":0.95,"lambda":1,"eta":0.4474,"alpha":alpha}

def simulate_one_coh_time(coh_time):
    # Non-photon-number-resolving
    raw_state_qc = QuantumCircuit(0, p_g=pg, network_noise_type=100, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_pair_parameters,
                                  T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    dc_state_qc = QuantumCircuit(0, p_g=pg, network_noise_type=101, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True, bell_pair_parameters=dc_bell_pair_parameters,
                                 T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    basic_distilled_state_qc = QuantumCircuit(0, p_g=pg, network_noise_type=103, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_pair_parameters,
                                              T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    w_distilled_state_qc = QuantumCircuit(0, p_g=pg, network_noise_type=104, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_pair_parameters,
                                          T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    bell_sc_distilled_state_qc = QuantumCircuit(0, p_g=pg, network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_pair_parameters,
                                                T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    bell_dc_distilled_state_qc = QuantumCircuit(0, p_g=pg, network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_dc_pair_parameters,
                                                T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    w_to_GHZ_distilled_state_qc = QuantumCircuit(0, p_g=pg, network_noise_type=107, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_pair_parameters,
                                                 T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    w_to_GHZ_dc_state_qc = QuantumCircuit(0, p_g=pg, network_noise_type=106, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_pair_parameters,
                                          T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)

    # Photon-number-resolving
    pnr_raw_state_qc = QuantumCircuit(0, p_g=pg, network_noise_type=100, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True, bell_pair_parameters=bell_pair_parameters,
                                      T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    pnr_dc_state_qc = QuantumCircuit(0, p_g=pg, network_noise_type=101, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True, bell_pair_parameters=dc_bell_pair_parameters,
                                     T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    pnr_basic_distilled_state_qc = QuantumCircuit(0, p_g=pg, network_noise_type=103, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True, bell_pair_parameters=bell_pair_parameters,
                                                  T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    pnr_w_distilled_state_qc = QuantumCircuit(0, p_g=pg, network_noise_type=104, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True, bell_pair_parameters=bell_pair_parameters,
                                              T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    pnr_bell_sc_distilled_state_qc = QuantumCircuit(0, p_g=pg, network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True, bell_pair_parameters=bell_pair_parameters,
                                                    T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    pnr_bell_dc_distilled_state_qc = QuantumCircuit(0, p_g=pg, network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True, bell_pair_parameters=bell_dc_pair_parameters,
                                                    T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    pnr_w_to_GHZ_distilled_state_qc = QuantumCircuit(0, p_g=pg, network_noise_type=107, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True, bell_pair_parameters=bell_pair_parameters,
                                                     T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)
    pnr_w_to_GHZ_dc_state_qc = QuantumCircuit(0, p_g=pg, network_noise_type=106, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True, bell_pair_parameters=bell_pair_parameters,
                                              T2n_idle=coh_time, T1n_idle=coh_time, T2n_link=coh_time, T1n_link=coh_time, T2e_idle=coh_time, T1e_idle=coh_time)

    return (
        coh_time,
        raw_state_qc.p_link, 1-raw_state_qc.F_link, raw_state_qc.emission_direct_statistics,
        dc_state_qc.p_link, 1-dc_state_qc.F_link, dc_state_qc.emission_direct_statistics,
        basic_distilled_state_qc.p_link, 1-basic_distilled_state_qc.F_link, basic_distilled_state_qc.emission_direct_statistics,
        w_distilled_state_qc.p_link, 1-w_distilled_state_qc.F_link, w_distilled_state_qc.emission_direct_statistics,
        bell_sc_distilled_state_qc.p_link, 1-bell_sc_distilled_state_qc.F_link, bell_sc_distilled_state_qc.emission_direct_statistics,
        bell_dc_distilled_state_qc.p_link, 1-bell_dc_distilled_state_qc.F_link, bell_dc_distilled_state_qc.emission_direct_statistics,
        w_to_GHZ_distilled_state_qc.p_link, 1-w_to_GHZ_distilled_state_qc.F_link, w_to_GHZ_distilled_state_qc.emission_direct_statistics,
        w_to_GHZ_dc_state_qc.p_link, 1-w_to_GHZ_dc_state_qc.F_link, w_to_GHZ_dc_state_qc.emission_direct_statistics,
        pnr_raw_state_qc.p_link, 1-pnr_raw_state_qc.F_link, pnr_raw_state_qc.emission_direct_statistics,
        pnr_dc_state_qc.p_link, 1-pnr_dc_state_qc.F_link, pnr_dc_state_qc.emission_direct_statistics,
        pnr_basic_distilled_state_qc.p_link, 1-pnr_basic_distilled_state_qc.F_link, pnr_basic_distilled_state_qc.emission_direct_statistics,
        pnr_w_distilled_state_qc.p_link, 1-pnr_w_distilled_state_qc.F_link, pnr_w_distilled_state_qc.emission_direct_statistics,
        pnr_bell_sc_distilled_state_qc.p_link, 1-pnr_bell_sc_distilled_state_qc.F_link, pnr_bell_sc_distilled_state_qc.emission_direct_statistics,
        pnr_bell_dc_distilled_state_qc.p_link, 1-pnr_bell_dc_distilled_state_qc.F_link, pnr_bell_dc_distilled_state_qc.emission_direct_statistics,
        pnr_w_to_GHZ_distilled_state_qc.p_link, 1-pnr_w_to_GHZ_distilled_state_qc.F_link, pnr_w_to_GHZ_distilled_state_qc.emission_direct_statistics,
        pnr_w_to_GHZ_dc_state_qc.p_link, 1-pnr_w_to_GHZ_dc_state_qc.F_link, pnr_w_to_GHZ_dc_state_qc.emission_direct_statistics
    )

if __name__ == "__main__":
    num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", mp.cpu_count()))
    with mp.Pool(processes=num_cpus) as pool:
        results = pool.map(simulate_one_coh_time, coh_times)

    (
        coh_times_out,
        raw_p, raw_inf, raw_stats,
        dc_state_p, dc_state_inf, dc_state_stats,
        basic_state_p, basic_state_inf, basic_state_stats,
        w_state_p, w_state_inf, w_state_stats,
        bell_sc_distilled_state_p, bell_sc_distilled_state_inf, bell_sc_distilled_state_stats,
        bell_dc_distilled_state_p, bell_dc_distilled_state_inf, bell_dc_distilled_state_stats,
        w_to_GHZ_distilled_state_p, w_to_GHZ_distilled_state_inf, w_to_GHZ_distilled_state_stats,
        w_to_GHZ_dc_state_p, w_to_GHZ_dc_state_inf, w_to_GHZ_dc_state_stats,
        pnr_raw_p, pnr_raw_inf, pnr_raw_stats,
        pnr_dc_state_p, pnr_dc_state_inf, pnr_dc_state_stats,
        pnr_basic_state_p, pnr_basic_state_inf, pnr_basic_state_stats,
        pnr_w_state_p, pnr_w_state_inf, pnr_w_state_stats,
        pnr_bell_sc_distilled_state_p, pnr_bell_sc_distilled_state_inf, pnr_bell_sc_distilled_state_stats,
        pnr_bell_dc_distilled_state_p, pnr_bell_dc_distilled_state_inf, pnr_bell_dc_distilled_state_stats,
        pnr_w_to_GHZ_distilled_state_p, pnr_w_to_GHZ_distilled_state_inf, pnr_w_to_GHZ_distilled_state_stats,
        pnr_w_to_GHZ_dc_state_p, pnr_w_to_GHZ_dc_state_inf, pnr_w_to_GHZ_dc_state_stats
    ) = map(list, zip(*results))

    arr = lambda x: np.array(x)
    valid_indices = (~np.isnan(arr(raw_inf)) & ~np.isnan(arr(dc_state_inf)) & ~np.isnan(arr(basic_state_inf)) &
                     ~np.isnan(arr(w_state_inf)) & ~np.isnan(arr(bell_sc_distilled_state_inf)) & ~np.isnan(arr(bell_dc_distilled_state_inf)) &
                     ~np.isnan(arr(w_to_GHZ_distilled_state_inf)) & ~np.isnan(arr(w_to_GHZ_dc_state_inf)))
    coh_times_filtered = arr(coh_times_out)[valid_indices]
    raw_inf_filtered = arr(raw_inf)[valid_indices]
    raw_p_filtered = arr(raw_p)[valid_indices]
    dc_state_inf_filtered = arr(dc_state_inf)[valid_indices]
    dc_state_p_filtered = arr(dc_state_p)[valid_indices]
    basic_state_inf_filtered = arr(basic_state_inf)[valid_indices]
    basic_state_p_filtered = arr(basic_state_p)[valid_indices]
    basic_state_stats_filtered = arr(basic_state_stats)[valid_indices]
    w_state_inf_filtered = arr(w_state_inf)[valid_indices]
    w_state_p_filtered = arr(w_state_p)[valid_indices]
    w_state_stats_filtered = arr(w_state_stats)[valid_indices]
    bell_sc_distilled_state_inf_filtered = arr(bell_sc_distilled_state_inf)[valid_indices]
    bell_sc_distilled_state_p_filtered = arr(bell_sc_distilled_state_p)[valid_indices]
    bell_sc_distilled_state_stats_filtered = arr(bell_sc_distilled_state_stats)[valid_indices]
    bell_dc_distilled_state_inf_filtered = arr(bell_dc_distilled_state_inf)[valid_indices]
    bell_dc_distilled_state_p_filtered = arr(bell_dc_distilled_state_p)[valid_indices]
    bell_dc_distilled_state_stats_filtered = arr(bell_dc_distilled_state_stats)[valid_indices]
    w_to_GHZ_distilled_state_inf_filtered = arr(w_to_GHZ_distilled_state_inf)[valid_indices]
    w_to_GHZ_distilled_state_p_filtered = arr(w_to_GHZ_distilled_state_p)[valid_indices]
    w_to_GHZ_distilled_state_stats_filtered = arr(w_to_GHZ_distilled_state_stats)[valid_indices]
    w_to_GHZ_dc_state_inf_filtered = arr(w_to_GHZ_dc_state_inf)[valid_indices]
    w_to_GHZ_dc_state_p_filtered = arr(w_to_GHZ_dc_state_p)[valid_indices]
    w_to_GHZ_dc_state_stats_filtered = arr(w_to_GHZ_dc_state_stats)[valid_indices]

    valid_indices_pnr = (~np.isnan(arr(pnr_raw_inf)) & ~np.isnan(arr(pnr_dc_state_inf)) & ~np.isnan(arr(pnr_basic_state_inf)) &
                         ~np.isnan(arr(pnr_w_state_inf)) & ~np.isnan(arr(pnr_bell_sc_distilled_state_inf)) & ~np.isnan(arr(pnr_bell_dc_distilled_state_inf)) &
                         ~np.isnan(arr(pnr_w_to_GHZ_distilled_state_inf)) & ~np.isnan(arr(pnr_w_to_GHZ_dc_state_inf)))
    coh_times_filtered_pnr = arr(coh_times_out)[valid_indices_pnr]
    pnr_raw_inf_filtered = arr(pnr_raw_inf)[valid_indices_pnr]
    pnr_raw_p_filtered = arr(pnr_raw_p)[valid_indices_pnr]
    pnr_raw_stats_filtered = arr(pnr_raw_stats)[valid_indices_pnr]
    pnr_dc_state_inf_filtered = arr(pnr_dc_state_inf)[valid_indices_pnr]
    pnr_dc_state_p_filtered = arr(pnr_dc_state_p)[valid_indices_pnr]
    pnr_dc_state_stats_filtered = arr(pnr_dc_state_stats)[valid_indices_pnr]
    pnr_basic_state_inf_filtered = arr(pnr_basic_state_inf)[valid_indices_pnr]
    pnr_basic_state_p_filtered = arr(pnr_basic_state_p)[valid_indices_pnr]
    pnr_basic_state_stats_filtered = arr(pnr_basic_state_stats)[valid_indices_pnr]
    pnr_w_state_inf_filtered = arr(pnr_w_state_inf)[valid_indices_pnr]
    pnr_w_state_p_filtered = arr(pnr_w_state_p)[valid_indices_pnr]
    pnr_w_state_stats_filtered = arr(pnr_w_state_stats)[valid_indices_pnr]
    pnr_bell_sc_distilled_state_inf_filtered = arr(pnr_bell_sc_distilled_state_inf)[valid_indices_pnr]
    pnr_bell_sc_distilled_state_p_filtered = arr(pnr_bell_sc_distilled_state_p)[valid_indices_pnr]
    pnr_bell_sc_distilled_state_stats_filtered = arr(pnr_bell_sc_distilled_state_stats)[valid_indices_pnr]
    pnr_bell_dc_distilled_state_inf_filtered = arr(pnr_bell_dc_distilled_state_inf)[valid_indices_pnr]
    pnr_bell_dc_distilled_state_p_filtered = arr(pnr_bell_dc_distilled_state_p)[valid_indices_pnr]
    pnr_bell_dc_distilled_state_stats_filtered = arr(pnr_bell_dc_distilled_state_stats)[valid_indices_pnr]
    pnr_w_to_GHZ_distilled_state_inf_filtered = arr(pnr_w_to_GHZ_distilled_state_inf)[valid_indices_pnr]
    pnr_w_to_GHZ_distilled_state_p_filtered = arr(pnr_w_to_GHZ_distilled_state_p)[valid_indices_pnr]
    pnr_w_to_GHZ_distilled_state_stats_filtered = arr(pnr_w_to_GHZ_distilled_state_stats)[valid_indices_pnr]
    pnr_w_to_GHZ_dc_state_inf_filtered = arr(pnr_w_to_GHZ_dc_state_inf)[valid_indices_pnr]
    pnr_w_to_GHZ_dc_state_p_filtered = arr(pnr_w_to_GHZ_dc_state_p)[valid_indices_pnr]
    pnr_w_to_GHZ_dc_state_stats_filtered = arr(pnr_w_to_GHZ_dc_state_stats)[valid_indices_pnr]

    results_dict = {
        "coherence_times": coh_times_filtered.tolist(),
        "raw_inf": raw_inf_filtered.tolist(),
        "raw_p": raw_p_filtered.tolist(),
        "dc_state_inf": dc_state_inf_filtered.tolist(),
        "dc_state_p": dc_state_p_filtered.tolist(),
        "basic_state_inf": basic_state_inf_filtered.tolist(),
        "basic_state_p": basic_state_p_filtered.tolist(),
        "basic_state_statistics": basic_state_stats_filtered.tolist(),
        "w_state_inf": w_state_inf_filtered.tolist(),
        "w_state_p": w_state_p_filtered.tolist(),
        "w_state_statistics": w_state_stats_filtered.tolist(),
        "bell_sc_distilled_state_inf": bell_sc_distilled_state_inf_filtered.tolist(),
        "bell_sc_distilled_state_p": bell_sc_distilled_state_p_filtered.tolist(),
        "bell_sc_distilled_state_statistics": bell_sc_distilled_state_stats_filtered.tolist(),
        "bell_dc_distilled_state_inf": bell_dc_distilled_state_inf_filtered.tolist(),
        "bell_dc_distilled_state_p": bell_dc_distilled_state_p_filtered.tolist(),
        "bell_dc_distilled_state_statistics": bell_dc_distilled_state_stats_filtered.tolist(),
        "w_to_GHZ_distilled_state_inf": w_to_GHZ_distilled_state_inf_filtered.tolist(),
        "w_to_GHZ_distilled_state_p": w_to_GHZ_distilled_state_p_filtered.tolist(),
        "w_to_GHZ_distilled_state_statistics": w_to_GHZ_distilled_state_stats_filtered.tolist(),
        "w_to_GHZ_dc_state_inf": w_to_GHZ_dc_state_inf_filtered.tolist(),
        "w_to_GHZ_dc_state_p": w_to_GHZ_dc_state_p_filtered.tolist(),
        "w_to_GHZ_dc_state_statistics": w_to_GHZ_dc_state_stats_filtered.tolist(),
        "pnr_coherence_times": coh_times_filtered_pnr.tolist(),
        "pnr_raw_inf": pnr_raw_inf_filtered.tolist(),
        "pnr_raw_p": pnr_raw_p_filtered.tolist(),
        "pnr_raw_statistics": pnr_raw_stats_filtered.tolist(),
        "pnr_dc_state_inf": pnr_dc_state_inf_filtered.tolist(),
        "pnr_dc_state_p": pnr_dc_state_p_filtered.tolist(),
        "pnr_dc_state_statistics": pnr_dc_state_stats_filtered.tolist(),
        "pnr_basic_state_inf": pnr_basic_state_inf_filtered.tolist(),
        "pnr_basic_state_p": pnr_basic_state_p_filtered.tolist(),
        "pnr_basic_state_statistics": pnr_basic_state_stats_filtered.tolist(),
        "pnr_w_state_inf": pnr_w_state_inf_filtered.tolist(),
        "pnr_w_state_p": pnr_w_state_p_filtered.tolist(),
        "pnr_w_state_statistics": pnr_w_state_stats_filtered.tolist(),
        "pnr_bell_sc_distilled_state_inf": pnr_bell_sc_distilled_state_inf_filtered.tolist(),
        "pnr_bell_sc_distilled_state_p": pnr_bell_sc_distilled_state_p_filtered.tolist(),
        "pnr_bell_sc_distilled_state_statistics": pnr_bell_sc_distilled_state_stats_filtered.tolist(),
        "pnr_bell_dc_distilled_state_inf": pnr_bell_dc_distilled_state_inf_filtered.tolist(),
        "pnr_bell_dc_distilled_state_p": pnr_bell_dc_distilled_state_p_filtered.tolist(),
        "pnr_bell_dc_distilled_state_statistics": pnr_bell_dc_distilled_state_stats_filtered.tolist(),
        "pnr_w_to_GHZ_distilled_state_inf": pnr_w_to_GHZ_distilled_state_inf_filtered.tolist(),
        "pnr_w_to_GHZ_distilled_state_p": pnr_w_to_GHZ_distilled_state_p_filtered.tolist(),
        "pnr_w_to_GHZ_distilled_state_statistics": pnr_w_to_GHZ_distilled_state_stats_filtered.tolist(),
        "pnr_w_to_GHZ_dc_state_inf": pnr_w_to_GHZ_dc_state_inf_filtered.tolist(),
        "pnr_w_to_GHZ_dc_state_p": pnr_w_to_GHZ_dc_state_p_filtered.tolist(),
        "pnr_w_to_GHZ_dc_state_statistics": pnr_w_to_GHZ_dc_state_stats_filtered.tolist()
    }

    coherence_data = rf".\output_data\simulation_data\{timestamp}_data_coherence_times_variation_shots_{shots}_Fprep_{bell_pair_parameters['F_prep']}_pDE_{bell_pair_parameters['p_DE']}_mu_{bell_pair_parameters['mu']}_eta_{bell_dc_pair_parameters['eta']}_alpha_{alpha}_pg_{pg}.json"
    with open(coherence_data, "w") as f:
        json.dump(results_dict, f, indent=2)