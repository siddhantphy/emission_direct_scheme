import numpy as np
from circuit_simulation.circuit_simulator import QuantumCircuit
from datetime import datetime
import multiprocessing as mp
import os
import json

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

alpha_range = np.arange(0.0101, 0.55, 0.05)
coh_time = 10
pg = 0.001
shots = 1

def simulate_one_alpha(alpha):
    bell_pair_parameters = {"ent_prot":"single_click","F_prep":1,"p_DE":0.01,"mu":0.97,"lambda":1,"eta":0.4474,"alpha":alpha}
    bell_dc_pair_parameters = bell_pair_parameters.copy()
    bell_dc_pair_parameters["ent_prot"] = "double_click"

    # Non-photon-number-resolving
    raw_state_qc = QuantumCircuit(1,network_noise_type=100, only_GHZ=True, shots_emission_direct=shots, p_g=pg, bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    dc_state_qc = QuantumCircuit(1,network_noise_type=101, only_GHZ=True, shots_emission_direct=shots, p_g=pg, bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    basic_distilled_state_qc = QuantumCircuit(1,network_noise_type=103, only_GHZ=True, shots_emission_direct=shots, p_g=pg, bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    w_distilled_state_qc = QuantumCircuit(1,network_noise_type=104, only_GHZ=True, shots_emission_direct=shots, p_g=pg, bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    bell_sc_distilled_state_qc = QuantumCircuit(1,network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, p_g=pg, bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    bell_dc_distilled_state_qc = QuantumCircuit(1,network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, p_g=pg, bell_pair_parameters=bell_dc_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)

    # Photon-number-resolving
    pnr_raw_state_qc = QuantumCircuit(1,network_noise_type=100, only_GHZ=True, shots_emission_direct=shots, p_g=pg, photon_number_resolution=True, bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    pnr_dc_state_qc = QuantumCircuit(1,network_noise_type=101, only_GHZ=True, shots_emission_direct=shots, p_g=pg, photon_number_resolution=True,bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    pnr_basic_distilled_state_qc = QuantumCircuit(1,network_noise_type=103, only_GHZ=True, shots_emission_direct=shots, p_g=pg, photon_number_resolution=True,bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    pnr_w_distilled_state_qc = QuantumCircuit(1,network_noise_type=104, only_GHZ=True, shots_emission_direct=shots, p_g=pg, photon_number_resolution=True,bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    pnr_bell_sc_distilled_state_qc = QuantumCircuit(1,network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, p_g=pg, photon_number_resolution=True,bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    pnr_bell_dc_distilled_state_qc = QuantumCircuit(1,network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, p_g=pg, photon_number_resolution=True,bell_pair_parameters=bell_dc_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)

    return (
        alpha,
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
        results = pool.map(simulate_one_alpha, alpha_range)

    (
        alpha_out,
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

    # Filter out nan values
    arr = lambda x: np.array(x)
    valid_indices = (~np.isnan(arr(raw_inf)) & ~np.isnan(arr(dc_state_inf)) & ~np.isnan(arr(basic_state_inf)) &
                     ~np.isnan(arr(w_state_inf)) & ~np.isnan(arr(bell_sc_distilled_state_inf)) & ~np.isnan(arr(bell_dc_distilled_state_inf)))
    alpha_filtered = arr(alpha_out)[valid_indices]
    raw_inf_filtered = arr(raw_inf)[valid_indices]
    dc_state_inf_filtered = arr(dc_state_inf)[valid_indices]
    basic_state_inf_filtered = arr(basic_state_inf)[valid_indices]
    w_state_inf_filtered = arr(w_state_inf)[valid_indices]
    bell_sc_distilled_state_inf_filtered = arr(bell_sc_distilled_state_inf)[valid_indices]
    bell_dc_distilled_state_inf_filtered = arr(bell_dc_distilled_state_inf)[valid_indices]

    valid_indices_pnr = (~np.isnan(arr(pnr_raw_inf)) & ~np.isnan(arr(pnr_dc_state_inf)) & ~np.isnan(arr(pnr_basic_state_inf)) &
                         ~np.isnan(arr(pnr_w_state_inf)) & ~np.isnan(arr(pnr_bell_sc_distilled_state_inf)) & ~np.isnan(arr(pnr_bell_dc_distilled_state_inf)))
    alpha_filtered_pnr = arr(alpha_out)[valid_indices_pnr]
    pnr_raw_inf_filtered = arr(pnr_raw_inf)[valid_indices_pnr]
    pnr_dc_state_inf_filtered = arr(pnr_dc_state_inf)[valid_indices_pnr]
    pnr_basic_state_inf_filtered = arr(pnr_basic_state_inf)[valid_indices_pnr]
    pnr_w_state_inf_filtered = arr(pnr_w_state_inf)[valid_indices_pnr]
    pnr_bell_sc_distilled_state_inf_filtered = arr(pnr_bell_sc_distilled_state_inf)[valid_indices_pnr]
    pnr_bell_dc_distilled_state_inf_filtered = arr(pnr_bell_dc_distilled_state_inf)[valid_indices_pnr]

    results_dict = {
        "alpha": list(alpha_out),
        "raw": {
            "p": raw_p,
            "inf": raw_inf,
            "pnr_p": pnr_raw_p,
            "pnr_inf": pnr_raw_inf
        },
        "dc_state": {
            "p": dc_state_p,
            "inf": dc_state_inf,
            "pnr_p": pnr_dc_state_p,
            "pnr_inf": pnr_dc_state_inf
        },
        "basic_state": {
            "p": basic_state_p,
            "inf": basic_state_inf,
            "pnr_p": pnr_basic_state_p,
            "pnr_inf": pnr_basic_state_inf
        },
        "w_state": {
            "p": w_state_p,
            "inf": w_state_inf,
            "pnr_p": pnr_w_state_p,
            "pnr_inf": pnr_w_state_inf
        },
        "bell_sc_distilled_state": {
            "p": bell_sc_distilled_state_p,
            "inf": bell_sc_distilled_state_inf,
            "pnr_p": pnr_bell_sc_distilled_state_p,
            "pnr_inf": pnr_bell_sc_distilled_state_inf
        },
        "bell_dc_distilled_state": {
            "p": bell_dc_distilled_state_p,
            "inf": bell_dc_distilled_state_inf,
            "pnr_p": pnr_bell_dc_distilled_state_p,
            "pnr_inf": pnr_bell_dc_distilled_state_inf
        },
        "filtered": {
            "alpha": alpha_filtered.tolist(),
            "raw_inf": raw_inf_filtered.tolist(),
            "dc_state_inf": dc_state_inf_filtered.tolist(),
            "basic_state_inf": basic_state_inf_filtered.tolist(),
            "w_state_inf": w_state_inf_filtered.tolist(),
            "bell_sc_distilled_state_inf": bell_sc_distilled_state_inf_filtered.tolist(),
            "bell_dc_distilled_state_inf": bell_dc_distilled_state_inf_filtered.tolist()
        },
        "filtered_pnr": {
            "alpha": alpha_filtered_pnr.tolist(),
            "pnr_raw_inf": pnr_raw_inf_filtered.tolist(),
            "pnr_dc_state_inf": pnr_dc_state_inf_filtered.tolist(),
            "pnr_basic_state_inf": pnr_basic_state_inf_filtered.tolist(),
            "pnr_w_state_inf": pnr_w_state_inf_filtered.tolist(),
            "pnr_bell_sc_distilled_state_inf": pnr_bell_sc_distilled_state_inf_filtered.tolist(),
            "pnr_bell_dc_distilled_state_inf": pnr_bell_dc_distilled_state_inf_filtered.tolist()
        }
    }

    bell_pair_parameters = {"ent_prot":"single_click","F_prep":1,"p_DE":0.01,"mu":0.97,"lambda":1,"eta":0.4474,"alpha":alpha_range[0]}
    bell_dc_pair_parameters = bell_pair_parameters.copy()
    bell_dc_pair_parameters["ent_prot"] = "double_click"
    bright_state_parameter_influence_data = rf'.\output_data\simulation_data\{timestamp}_data_bright_state_parameter_influence_shots_{shots}_Fprep_{bell_pair_parameters["F_prep"]}_pDE_{bell_pair_parameters["p_DE"]}_mu_{bell_pair_parameters["mu"]}_eta_{bell_dc_pair_parameters["eta"]}_cohtime_{coh_time}_pg_{pg}.json'
    with open(bright_state_parameter_influence_data, 'w') as f:
        json.dump(results_dict, f, indent=4)
