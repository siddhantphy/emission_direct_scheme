import pandas as pd
import os
import sys
import re
import pickle
import math
import numpy as np
from collections import defaultdict
from scipy.stats import sem
ACCURACY = 7


def confidence_interval(data, confidence=0.682, minus_mean=False, require_unique_min=False, return_new_bound=False):
    n = len(data)
    mean = np.mean(data) if minus_mean else 0
    data_sorted = sorted(data)

    lower_bound = math.floor(n * ((1 - confidence) / 2))
    upper_bound = math.ceil(n * (1 - (1 - confidence) / 2))
    upper_bound = upper_bound if upper_bound < len(data) else -1

    if require_unique_min:
        while lower_bound > 0 and round(abs(data_sorted[lower_bound] - data_sorted[lower_bound - 1]), ACCURACY) == 0:
            lower_bound -= 1
        while upper_bound < (n - 1) and round(abs(data_sorted[upper_bound + 1] - data_sorted[upper_bound]), ACCURACY) == 0:
            upper_bound += 1

    if return_new_bound:
        return abs(data_sorted[lower_bound] - mean), data_sorted[upper_bound] - mean, \
               lower_bound / n * 100, upper_bound / n * 100
    else:
        return abs(data_sorted[lower_bound] - mean), data_sorted[upper_bound] - mean


def GHZ_success_rate_for_cut_off_time(data):
    total_number_of_iterations = len(data)

    eval_dict = defaultdict(lambda: defaultdict(int))
    for duration in data:
        eval_dict[round(duration, ACCURACY)]["count_durations"] += 1
    eval_dict = dict(sorted(eval_dict.items(), key=lambda item: item[0]))

    iterations_counter = 0
    for duration in eval_dict.keys():
        iterations_counter += eval_dict[duration]["count_durations"]
        eval_dict[duration]["GHZ_success_rate"] = iterations_counter / total_number_of_iterations

    return eval_dict


def remove_identical_columns(data):
    data_no_index = data.reset_index()
    drop_cols = []
    for col in data_no_index:
        if data_no_index[col].nunique() == 1:
           drop_cols.append(col)

    return data.droplevel(drop_cols)


def get_all_files_from_folder(folder, folder_name, pkl=False):
    pattern = re.compile('^{}.*'.format(folder_name))
    files = []
    pkl_files = []
    for sub_dir in os.listdir(folder):
        if pattern.fullmatch(sub_dir):
            for file in os.listdir(os.path.join(folder, sub_dir)):
                if file.endswith(".csv") and "failed" not in file:
                    files.append(os.path.join(folder, sub_dir, file))
                elif file.endswith(".pkl") and pkl:
                    pkl_files.append(os.path.join(folder, sub_dir, file))

    if pkl:
        return files, pkl_files

    return files


def get_results_from_files(superoperator_files, pkl_files, name_csv):
    indices = ['protocol_name', 'node', 'pg', 'pm', 'pm_1', 'pn', 'decoherence', 'p_bell_success', 'pulse_duration',
               'network_noise_type', 'no_single_qubit_error', 'basis_transformation_noise', 'cut_off_time',
               'probabilistic', 'fixed_lde_attempts']
    result_df = pd.DataFrame(columns=indices)
    result_df = result_df.set_index(indices)

    for superoperator_file in superoperator_files:
        pkl_file = (superoperator_file.replace(".csv", ".pkl") if superoperator_file.replace(".csv", ".pkl") in
                    pkl_files else None)
        if pkl_file:
            full_data = pickle.load(open(pkl_file, "rb"))
        else:
            print("\n[!] Warning: Expected pickle file for file ({}) not found! No spread data can be shown."
                  .format(os.path.basename(superoperator_file)), file=sys.stderr)
            full_data = None
        result_df = result_df.sort_index()
        df = pd.read_csv(superoperator_file, sep=';', index_col=[0, 1], float_precision='round_trip')
        if df.iloc[0, df.columns.get_loc('pulse_duration')] == 0:
            df.iloc[0, df.columns.get_loc('fixed_lde_attempts')] = 0
        df.loc[:, 'node'] = "Natural Abundance" if re.match('.*[Nn]a', superoperator_file) else "Purified"
        index = tuple(df.iloc[0, df.columns.get_loc(index)] for index in indices)

        variables = ['written_to', 'avg_lde_attempts', 'avg_duration', 'ghz_fidelity']
        for variable in variables:
            if variable in df:
                result_df.loc[index, variable] = df.iloc[0, df.columns.get_loc(variable)]

        if full_data:
            interval_data = ['ghz_sem', "dur_sem", "stab_sem"]
            for interval in interval_data:
                kind = interval.split(sep="_")[0]
                key = kind if "dur" in kind else kind + "_fid"
                result_df.loc[index, interval] = sem(full_data[key])
                result_df.loc[index, kind + '_lspread'] = confidence_interval(full_data[key], minus_mean=True)[0]
                result_df.loc[index, kind + '_rspread'] = confidence_interval(full_data[key], minus_mean=True)[1]

            result_df.loc[index, '99_duration'] = confidence_interval(full_data["dur"], 0.98)[1]

        d = 2**4
        result_df.loc[index, 'IIII'] = (d * df['p'].iloc[0] + 1) / (d + 1)

    # result_df = remove_identical_columns(result_df)
    result_df = result_df.sort_index()
    result_df.to_csv(name_csv, sep=';')
    print(result_df)


if __name__ == '__main__':
    name_csv = "./notebooks/circuit_data_NV.csv"
    folder = "./results/sim_data_6"
    folder_name = "superoperator_cutoff_99_full"

    files, pkl_files = get_all_files_from_folder(folder, folder_name, pkl=True)

    files = [f for f in files if 'Old' not in f]
    pkl_files = [p for p in pkl_files if 'Old' not in p]
    get_results_from_files(files, pkl_files, name_csv)
