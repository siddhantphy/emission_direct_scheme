import numpy as np
from circuit_simulation.circuit_simulator import QuantumCircuit
from datetime import datetime
import json
import multiprocessing as mp
import os

# Get current date and time in the format YYYYMMDDHHMMSS
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Shots
shots = 10

# Operating parameter alpha
alpha = 0.05

# Gate error in GHZ generation
pg = 0.001

# Coherence times improvement
# coh_times = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
coh_times = [0.1, 1, 10, 100, 1000,10000] # Reduced for quick calculations

# Bell pair parameters
bell_pair_parameters = {"ent_prot":"single_click","F_prep":0.999,"p_DE":0.01,"mu":0.95,"lambda":1,"eta":0.4474,"alpha":alpha}
dc_bell_pair_parameters = {"ent_prot":"single_click","F_prep":0.999,"p_DE":0.01,"mu":0.95,"lambda":1,"eta":0.4474,"alpha":0.5}
bell_dc_pair_parameters = {"ent_prot":"double_click","F_prep":0.999,"p_DE":0.01,"mu":0.95,"lambda":1,"eta":0.4474,"alpha":alpha} # Another bell pair protocol with double-click elemetray links

# Plot the curves with specified markers and colors
raw_p = []
raw_inf = []
dc_state_p = []
dc_state_inf = []
basic_state_p = []                       
basic_state_inf = []
w_state_p = []
w_state_inf = []
bell_sc_distilled_state_p = []
bell_sc_distilled_state_inf = []
bell_dc_distilled_state_p = []
bell_dc_distilled_state_inf = []

pnr_raw_p = []
pnr_raw_inf = []
pnr_dc_state_p = []
pnr_dc_state_inf = []
pnr_basic_state_p = []
pnr_basic_state_inf = []
pnr_w_state_p = []
pnr_w_state_inf = []
pnr_bell_sc_distilled_state_p = []
pnr_bell_sc_distilled_state_inf = []
pnr_bell_dc_distilled_state_p = []
pnr_bell_dc_distilled_state_inf = []

def simulate_one_coh_time(coh_time):
    raw_state_qc = QuantumCircuit(1,p_g=pg,network_noise_type=100, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    dc_state_qc = QuantumCircuit(1,p_g=pg,network_noise_type=101, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True,bell_pair_parameters=dc_bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    basic_distilled_state_qc = QuantumCircuit(1,p_g=pg,network_noise_type=103, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    w_distilled_state_qc = QuantumCircuit(1,p_g=pg,network_noise_type=104, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    bell_sc_distilled_state_qc = QuantumCircuit(1,p_g=pg,network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    bell_dc_distilled_state_qc = QuantumCircuit(1,p_g=pg,network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_dc_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    pnr_raw_state_qc = QuantumCircuit(1,p_g=pg,network_noise_type=100, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True, bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    pnr_dc_state_qc = QuantumCircuit(1,p_g=pg,network_noise_type=101, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True,bell_pair_parameters=dc_bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    pnr_basic_distilled_state_qc = QuantumCircuit(1,p_g=pg,network_noise_type=103, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True,bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    pnr_w_distilled_state_qc = QuantumCircuit(1,p_g=pg,network_noise_type=104, only_GHZ=True, shots_emission_direct=shots, photon_number_resolution=True,bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    pnr_bell_sc_distilled_state_qc = QuantumCircuit(1,p_g=pg,network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    pnr_bell_dc_distilled_state_qc = QuantumCircuit(1,p_g=pg,network_noise_type=102, only_GHZ=True, shots_emission_direct=shots, bell_pair_parameters=bell_dc_pair_parameters,T2n_idle=coh_time,T1n_idle=coh_time,T2n_link=coh_time,T1n_link=coh_time,T2e_idle=coh_time,T1e_idle=coh_time)
    return (
        coh_time,
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
        results = pool.map(simulate_one_coh_time, coh_times)
    (
        coh_times_out,
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

    # Use the filtered indices to filter both infidelities and probabilities
    coh_times_array = np.array(coh_times_out)
    raw_p_arr = np.array(raw_p)
    dc_state_p_arr = np.array(dc_state_p)
    basic_state_p_arr = np.array(basic_state_p)
    w_state_p_arr = np.array(w_state_p)
    bell_sc_distilled_state_p_arr = np.array(bell_sc_distilled_state_p)
    bell_dc_distilled_state_p_arr = np.array(bell_dc_distilled_state_p)
    pnr_raw_p_arr = np.array(pnr_raw_p)
    pnr_dc_state_p_arr = np.array(pnr_dc_state_p)
    pnr_basic_state_p_arr = np.array(pnr_basic_state_p)
    pnr_w_state_p_arr = np.array(pnr_w_state_p)
    pnr_bell_sc_distilled_state_p_arr = np.array(pnr_bell_sc_distilled_state_p)
    pnr_bell_dc_distilled_state_p_arr = np.array(pnr_bell_dc_distilled_state_p)

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
    coh_times_filtered = coh_times_array[valid_indices]
    raw_p_filtered = raw_p_arr[valid_indices]
    dc_state_p_filtered = dc_state_p_arr[valid_indices]
    basic_state_p_filtered = basic_state_p_arr[valid_indices]
    w_state_p_filtered = w_state_p_arr[valid_indices]
    bell_sc_distilled_state_p_filtered = bell_sc_distilled_state_p_arr[valid_indices]
    bell_dc_distilled_state_p_filtered = bell_dc_distilled_state_p_arr[valid_indices]
    raw_inf_filtered = raw_inf_arr[valid_indices]
    dc_state_inf_filtered = dc_state_inf_arr[valid_indices]
    basic_state_inf_filtered = basic_state_inf_arr[valid_indices]
    w_state_inf_filtered = w_state_inf_arr[valid_indices]
    bell_sc_distilled_state_inf_filtered = bell_sc_distilled_state_inf_arr[valid_indices]
    bell_dc_distilled_state_inf_filtered = bell_dc_distilled_state_inf_arr[valid_indices]

    valid_indices_pnr = (~np.isnan(pnr_raw_inf_arr) & ~np.isnan(pnr_dc_state_inf_arr) & ~np.isnan(pnr_basic_state_inf_arr) &
                         ~np.isnan(pnr_w_state_inf_arr) & ~np.isnan(pnr_bell_sc_distilled_state_inf_arr) & ~np.isnan(pnr_bell_dc_distilled_state_inf_arr))
    coh_times_filtered_pnr = coh_times_array[valid_indices_pnr]
    pnr_raw_p_filtered = pnr_raw_p_arr[valid_indices_pnr]
    pnr_dc_state_p_filtered = pnr_dc_state_p_arr[valid_indices_pnr]
    pnr_basic_state_p_filtered = pnr_basic_state_p_arr[valid_indices_pnr]
    pnr_w_state_p_filtered = pnr_w_state_p_arr[valid_indices_pnr]
    pnr_bell_sc_distilled_state_p_filtered = pnr_bell_sc_distilled_state_p_arr[valid_indices_pnr]
    pnr_bell_dc_distilled_state_p_filtered = pnr_bell_dc_distilled_state_p_arr[valid_indices_pnr]
    pnr_raw_inf_filtered = pnr_raw_inf_arr[valid_indices_pnr]
    pnr_dc_state_inf_filtered = pnr_dc_state_inf_arr[valid_indices_pnr]
    pnr_basic_state_inf_filtered = pnr_basic_state_inf_arr[valid_indices_pnr]
    pnr_w_state_inf_filtered = pnr_w_state_inf_arr[valid_indices_pnr]
    pnr_bell_sc_distilled_state_inf_filtered = pnr_bell_sc_distilled_state_inf_arr[valid_indices_pnr]
    pnr_bell_dc_distilled_state_inf_filtered = pnr_bell_dc_distilled_state_inf_arr[valid_indices_pnr]

    results_dict = {
        "coherence_times": coh_times_filtered.tolist(),
        "raw_state_p": raw_p_filtered.tolist(),
        "raw_state_inf": raw_inf_filtered.tolist(),
        "dc_state_p": dc_state_p_filtered.tolist(),
        "dc_state_inf": dc_state_inf_filtered.tolist(),
        "basic_state_p": basic_state_p_filtered.tolist(),
        "basic_state_inf": basic_state_inf_filtered.tolist(),
        "w_state_p": w_state_p_filtered.tolist(),
        "w_state_inf": w_state_inf_filtered.tolist(),
        "bell_sc_distilled_state_p": bell_sc_distilled_state_p_filtered.tolist(),
        "bell_sc_distilled_state_inf": bell_sc_distilled_state_inf_filtered.tolist(),
        "bell_dc_distilled_state_p": bell_dc_distilled_state_p_filtered.tolist(),
        "bell_dc_distilled_state_inf": bell_dc_distilled_state_inf_filtered.tolist(),
        "pnr_raw_state_p": pnr_raw_p_filtered.tolist(),
        "pnr_raw_state_inf": pnr_raw_inf_filtered.tolist(),
        "pnr_dc_state_p": pnr_dc_state_p_filtered.tolist(),
        "pnr_dc_state_inf": pnr_dc_state_inf_filtered.tolist(),
        "pnr_basic_state_p": pnr_basic_state_p_filtered.tolist(),
        "pnr_basic_state_inf": pnr_basic_state_inf_filtered.tolist(),
        "pnr_w_state_p": pnr_w_state_p_filtered.tolist(),
        "pnr_w_state_inf": pnr_w_state_inf_filtered.tolist(),
        "pnr_bell_sc_distilled_state_p": pnr_bell_sc_distilled_state_p_filtered.tolist(),
        "pnr_bell_sc_distilled_state_inf": pnr_bell_sc_distilled_state_inf_filtered.tolist(),
        "pnr_bell_dc_distilled_state_p": pnr_bell_dc_distilled_state_p_filtered.tolist(),
        "pnr_bell_dc_distilled_state_inf": pnr_bell_dc_distilled_state_inf_filtered.tolist()
    }

    coherence_data = rf".\output_data\simulation_data\{timestamp}_data_coherence_times_variation_shots_{shots}_Fprep_{bell_pair_parameters['F_prep']}_pDE_{bell_pair_parameters['p_DE']}_mu_{bell_pair_parameters['mu']}_eta_{bell_dc_pair_parameters['eta']}_alpha_{alpha}_pg_{pg}.json"
    with open(coherence_data, "w") as f:
        json.dump(results_dict, f, indent=2)
