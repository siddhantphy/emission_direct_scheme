from genericpath import exists
import os
import sys
sys.path.insert(1, os.path.abspath(os.getcwd()))
from pprint import pprint
from multiprocessing import Pool, cpu_count
import threading
import pickle
import dill
import pandas as pd
import circuit_simulation.stabilizer_measurement_protocols.stabilizer_measurement_protocols as stab_protocols
import circuit_simulation.stabilizer_measurement_protocols.auto_generated_stabilizer_measurement_protocols as agsmp
from circuit_simulation.stabilizer_measurement_protocols.argument_parsing import compose_parser, group_arguments
from circuit_simulation.gates.gates import *
from circuit_simulation.circuit_simulator import QuantumCircuit
from utilities.files import get_full_path, detect_filenames
import itertools as it
import datetime
import time
import random
import re
from plot_superoperator.analyse_simulation_data import confidence_interval
from circuit_simulation.termcolor.termcolor import cprint
from collections import defaultdict
import numpy as np
import math
from tqdm import tqdm
from copy import copy, deepcopy
from itertools import product
import warnings
warnings.filterwarnings('ignore', message='.*Specify dtype option on import or set low_memory=False.*')
from scipy import sparse as sp
from circuit_simulation.basic_operations.basic_operations import fidelity, trace_distance
SUM_ACCURACY = 7
ACCURACY = 15


def print_signature():
    cprint("\nQuantum Circuit Simulator®", color='cyan')
    print("--------------------------")


def create_file_name(filename, **kwargs):
    protocol = kwargs.pop('protocol')
    protocol_recipe = kwargs.pop('protocol_recipe')
    use_swap = True if "_swap" in protocol else False
    protocol = protocol if "_swap" not in protocol else protocol.rstrip('_swap')
    protocol = "hc_" + protocol if protocol != "auto_generated" else protocol_recipe
    protocol = protocol if use_swap else protocol + "_no-swap"
    protocol = protocol if ('noiseless_swap' in kwargs and kwargs['noiseless_swap'] and use_swap) is False \
        else protocol + "_nl-swap"

    # protocol = protocol if not kwargs['noiseless_swap'] else protocol.strip('_swap')
    # protocol_swap_add = "_swap" if "_swap" in protocol else ""
    # protocol = protocol if protocol != "auto_generated_swap" else "ags_" + protocol_recipe + protocol_swap_add
    filename = "" if filename is None else filename
    filename = "{}{}{}".format(filename, "_" if len(filename) > 0 and filename[-1] not in "/_" else "", protocol)

    for key, value in kwargs.items():
        # Do not include if value is None, 0 or np.inf (default cut_off_time) or if key is t_pulse
        if value is None or value == np.inf or key in ['_node', 'noiseless_swap']:
            continue
        if 'bell_pair_parameters' in kwargs and kwargs['bell_pair_parameters'] is not None \
                and key in ['F_link', 'p_link']:
            continue
        if key == "bell_pair_parameters" or key == "gate_durations":
            for subkey, subvalue in kwargs[key].items():
                filename += "_" + str(subkey)
                if not isinstance(subvalue, tuple):
                    subvalue = (subvalue, )
                for subsubvalue in subvalue:
                    filename += "-" + str(subsubvalue)
        elif key == "combine_supop" and value is False:
            continue
        elif not (key in ['dec', 'prob'] and value is True):
            value = "-inf" if value in ['np.infty', 'infty', 'infinity', 'inf'] \
                else ("-" + str(value) if value is not True else "")
            filename += "_" + str(key) + value if key not in ['dec', 'prob'] else "_no-" + str(key)

    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + "_" + filename

    return filename.strip('_')


def _get_cut_off_dataframe(files: list):
    return_values = []
    for file in files:
        if file is None:
            return_values.append(None)
            break
        if file.lower() == 'auto':
            return_values.append(file)
            break

        try:
            percentage = float(file)
            if percentage < 0 or percentage > 100:
                raise ValueError('Specified automatic cut-off time does not fall between 0 and 100%.')
            return_values.append(percentage)
        except ValueError:
            if not os.path.exists(file):
                raise ValueError('File containing the cut-off times could not be found.')
            return_values.append(pd.read_csv(file, sep=";", float_precision='round_trip'))
    return return_values


def _get_cut_off_time(dataframe, run_dict, circuit_args, diff_params, **kwargs):
    cut_off_time = run_dict.pop('cut_off_time')

    auto_cut_off_percentages = {'plain': 99.9, 'modicum': 99.8, 'basic1': 99.4, 'basic2': 99.4, 'medium1': 98.6,
                                'medium2': 98.6, 'minimum4x_22': 98.0, 'expedient': 98.0, 'refined1': 96.2,
                                'refined2': 96.2, 'minimum4x_40_1': 96.2, 'minimum4x_40_2': 96.2, 'stringent': 96.0}
    if run_dict['protocol_recipe'] is not None:
        try:
            nr_Bell_states = run_dict['protocol_recipe'].split("_")[-2]
            auto_cut_off_perc = 100.2 - float(nr_Bell_states) / 10
        except IndexError:
            auto_cut_off_perc = 99.0
        except ValueError:
            auto_cut_off_perc = 99.0
    elif run_dict['protocol'] in auto_cut_off_percentages.keys():
        auto_cut_off_perc = auto_cut_off_percentages[run_dict['protocol']]
    elif 'swap' in run_dict['protocol'] and run_dict['protocol'][:-5] in auto_cut_off_percentages.keys():
        auto_cut_off_perc = auto_cut_off_percentages[run_dict['protocol'][:-5]]
    else:
        auto_cut_off_perc = 99.0

    if cut_off_time != np.inf or dataframe is None:
        return cut_off_time, False

    # file_n = create_file_name(kwargs['csv_filename'], dec=circuit_args['decoherence'], prob=circuit_args['probabilistic'],
    #                           node=run_dict['_node'], noiseless_swap=circuit_args['noiseless_swap'], **run_dict)
    diff_params_copy = deepcopy(diff_params)
    if 'noiseless_swap' in diff_params.keys():
        del diff_params_copy['noiseless_swap']
    fn_short = create_file_name(kwargs['csv_filename'], protocol=run_dict['protocol'],
                                protocol_recipe=run_dict['protocol_recipe'], node=run_dict['_node'],
                                **diff_params_copy, dec=circuit_args['decoherence'], prob=circuit_args['probabilistic'],
                                noiseless_swap=circuit_args['noiseless_swap'], combine_supop=circuit_args['combine'],
                                seed=run_dict['seed_number'], cut_off_time=cut_off_time)
    update_result_files(kwargs['cp_path'], fn_short)
    file_n = os.path.join(get_full_path(kwargs['cp_path']), fn_short[16:])
    if os.path.exists(file_n + '.csv'):
        print(f"CSV file used to determine cut-off time: {file_n}.")
        data = pd.read_csv(file_n + '.csv', sep=";", float_precision="round_trip", index_col=["error_config", "lie"])
        column_index_written_to = data.columns.get_loc('written_to')
        if data.iloc[0, column_index_written_to]*1.05 > circuit_args['iterations']:
            for dur_percentage in [float(col[4:]) for col in data if col.startswith('dur')]:
                if (dataframe == "auto" and dur_percentage == auto_cut_off_perc) or \
                        (isinstance(dataframe, float) and dataframe == dur_percentage):
                    print('dur_' + str(dur_percentage))
                    column_index_dur = data.columns.get_loc('dur_' + str(dur_percentage))
                    print('[INFO] Found cut-off time value for {}% from file: {}'.format(str(dur_percentage),
                                                                                         data.iloc[0,
                                                                                                   column_index_dur]))
                    return round(data.iloc[0, column_index_dur], ACCURACY), False
            if isinstance(dataframe, float) or dataframe == "auto":
                percentage = dataframe if isinstance(dataframe, float) else auto_cut_off_perc
                if os.path.exists(file_n + '.pkl'):
                    print(f"Pickle-file found for calculating cut-off: {file_n}.")
                    characteristics = pickle.load(open(file_n + '.pkl', 'rb'))
                    conf_int = confidence_interval(characteristics['dur'], 1 - 2 * (100 - percentage) / 100,
                                                   require_unique_min=True, return_new_bound=True)
                    add_column_values(data, ['dur_' + str(conf_int[3])], [conf_int[1]])
                    data.to_csv(file_n + '.csv', sep=';')
                    column_index_dur = data.columns.get_loc('dur_' + str(conf_int[3]))
                    print('[INFO] Calculated cut-off time value for {}% from file: {}'.format(str(str(conf_int[3])),
                                                                                              data.iloc[0,
                                                                                                        column_index_dur]))
                    return round(data.iloc[0, column_index_dur], ACCURACY), False
                else:
                    print(f"Pickle-file for calculating cut-off is missing: {file_n}.")
    else:
        print(f"CSV file for calculating cut-off is missing: {file_n}.")

    # # This makes sure the cut-off time in the "auto" mode is calculated with at least 10000 iterations of the circuit:
    # circuit_args['iterations'] += 10000 - circuit_args['iterations'] if circuit_args['iterations'] < 10000 else 0
    print('No records for requested cut-off time found or not enough iterations. First running for:\n{}'.format(fn_short))
    return np.inf, True


def _open_existing_superoperator_file(filename, addition=""):
    if filename is None:
        return
    if not os.path.exists(filename + addition):
        return

    existing_file = pd.read_csv(filename + addition, sep=';', float_precision='round_trip')
    index = ['error_config', 'lie'] if 'error_idle' not in existing_file else ['error_stab', 'error_idle', 'lie']

    existing_file = existing_file.set_index(index)

    return existing_file


def update_result_files(folder, fn, remove_main=False):
    main_file, main_file_present, sub_files = detect_filenames(folder, fn, export_full_names=True)
    print(f"Identified superoperator dataframe runs: {[sub_file[:15] for sub_file in sub_files]}.")
    if main_file_present:
        print(f"Main superoperator dataframe is present.")

    main_file_cot = main_file
    if "_cut_off_time-" in main_file:
        cut_off_time = float(main_file.split("_cut_off_time-")[1])
        main_file_cot = main_file.split("_cut_off_time-")[0] + "_cut_off_time-" + str(round(cut_off_time, 15))
    if main_file_present is False and sub_files:
        # Create main file from first file in sub_files
        main_data_frame_created = False
        for ext in ["", "_failed"]:
            filename = os.path.join(get_full_path(folder), sub_files[0] + ext)
            df = _open_existing_superoperator_file(filename + ".csv")
            if df is not None:
                # add_column_values(df, ['sub_files'], [sub_files[0]])
                filename_main_file = os.path.join(get_full_path(folder), main_file_cot + ext)
                df.to_csv(filename_main_file + ".csv", sep=';')
                if os.path.exists(filename + '.pkl'):
                    characteristics_old = pickle.load(open(filename + '.pkl', 'rb'))
                    pickle.dump(characteristics_old, file=open(filename_main_file + '.pkl', 'wb+'))
                main_data_frame_created = True
            elif ext == "":
                raise FileExistsError(f"File {filename} does not exist.")
        if main_data_frame_created:
            print(f"Main superoperator dataframe is created from sub_file {sub_files[0]}.")
            del sub_files[0]

    for ext in ["", "_failed"]:
        filename_main_file = os.path.join(get_full_path(folder), main_file_cot + ext)
        df = _open_existing_superoperator_file(filename_main_file + ".csv")
        if df is not None:
            sub_files_in_main = df.iloc[0, df.columns.get_loc("date_and_time")]
            sub_files_in_main_list = sub_files_in_main.split(" ")
            characteristics_main = None
            if os.path.exists(filename_main_file + '.pkl'):
                characteristics_main = pickle.load(open(filename_main_file + '.pkl', 'rb'))
            for sub_file in sub_files:
                if sub_file[:15] not in sub_files_in_main_list:
                    filename = os.path.join(get_full_path(folder), sub_file + ext)
                    print(filename)
                    df_sub = _open_existing_superoperator_file(filename + ".csv")
                    if df_sub is not None:
                        print(f"Sub files {sub_file} is added to the main frame.")
                        df = _combine_superoperator_dataframes(df, df_sub)
                        if os.path.exists(filename + '.pkl') and characteristics_main:
                            characteristics_sub = pickle.load(open(filename + '.pkl', 'rb'))
                            [characteristics_main[key].extend(value) for key, value in characteristics_sub.items() if key != 'index']
                        # sub_files_in_main += " " + str(sub_file)
                    elif ext == "":
                        raise FileExistsError(f"File {filename} does not exist.")

            # df.iloc[0, df.columns.get_loc("sub_files")] = sub_files_in_main
            df.to_csv(filename_main_file + ".csv", sep=';')
            if characteristics_main:
                pickle.dump(characteristics_main, file=open(filename_main_file + '.pkl', 'wb+'))
        # elif ext == "":
        #     raise FileExistsError(f"File {filename_main_file} does not exist.")

    if remove_main:
        print(f"Main superoperator dataframes are removed after the calculation.")
        os.remove(os.path.join(get_full_path(folder), main_file_cot) + ".pkl")
        os.remove(os.path.join(get_full_path(folder), main_file_cot) + ".csv")
        if os.path.exists(os.path.join(get_full_path(folder), main_file_cot) + '_failed.csv'):
            os.remove(os.path.join(get_full_path(folder), main_file_cot) + '_failed.csv')


def _formulate_distributed_stabilizer_and_idle_superoperator(dataframes, cutoff_search):
    """ This function combines the round noise and idling superoperator in the new form! """

    if len(dataframes) < 2 or cutoff_search:
        return dataframes[0]

    round_superoperator = dataframes[0]
    idle_superoperator = dataframes[1]

    error_configs = [''.join(comb) for comb in product(["I", "X", "Y", "Z"], repeat=4)]
    error_configs.pop(0) # Pop the first element as it will be done manually

    round_available_configs_s = list(set([config[0] for config in list(round_superoperator.index)]))
    round_available_configs_p = [c[1]+c[0]+c[3]+c[2] for c in round_available_configs_s]
    idle_available_configs = list(set([config[0] for config in list(idle_superoperator.index)]))

    columns = list(round_superoperator.columns)
    first_row_vals = round_superoperator.iloc[0, :].values.tolist()
    second_row_vals = round_superoperator.iloc[1, :].values.tolist()

    # Make all possible columns needed for the new superoperator
    columns.insert(0, "error_config")
    columns.insert(1, "lie")
    columns.insert(4, "idle")

    # Manually construct the first and second row from existing data of the two superoperators
    first_row_vals.insert(0, "IIII")
    first_row_vals.insert(1, False)
    first_row_vals.insert(4, idle_superoperator.iloc[0, 1])

    second_row_vals.insert(0, "IIII")
    second_row_vals.insert(1, True)
    second_row_vals.insert(4, idle_superoperator.iloc[1, 1])

    template_row_vals = deepcopy(second_row_vals)
    for i in range(5):
        template_row_vals.pop(0)

    combined_superoperator = pd.DataFrame(columns=columns) # Make the new dataframe with the new columns
    combined_superoperator.loc[0] = first_row_vals # Fill the first row manually!
    combined_superoperator.loc[1] = second_row_vals # Fill the second row manually!

    # Now loop over all the possible configs and add the data row by row based on availability
    for config in error_configs:
        p_false = p_true = s_false = s_true = idle = 0
        if config in idle_available_configs:
            if False in idle_superoperator['p'][config].keys():
                idle = idle_superoperator['p'][config][False]
            elif True in idle_superoperator['p'][config].keys():
                idle = idle_superoperator['p'][config][True]
        if config in round_available_configs_s:
            if False in round_superoperator['s'][config].keys():
                s_false = round_superoperator['s'][config][False]
            if True in round_superoperator['s'][config].keys():
                s_true = round_superoperator['s'][config][True]
        if config in round_available_configs_p:
            re_ordered_config = config[1] + config[0] + config[3] + config[2]
            if False in round_superoperator['p'][re_ordered_config].keys():
                p_false = round_superoperator['p'][re_ordered_config][False]
            if True in round_superoperator['p'][re_ordered_config].keys():
                p_true = round_superoperator['p'][re_ordered_config][True]

        if p_false == p_true == s_false == s_true == idle == 0:
            pass
        else:
            combined_superoperator.loc[len(combined_superoperator)] = [config, False, p_false, s_false, idle] + template_row_vals
            combined_superoperator.loc[len(combined_superoperator)] = [config, True, p_true, s_true, idle] + template_row_vals

    combined_superoperator.set_index(['error_config', 'lie'], inplace=True, drop=True) # Set the indexes as the error config and lie column

    return combined_superoperator
    

def _combine_idle_and_stabilizer_superoperator(dataframes, cutoff_search):
    def combine_dataframes(stab, stab_idle, type):
        for stab_item in stab.iteritems():
            for stab_idle_item in stab_idle.iteritems():
                (p_error, p_lie), p_value = stab_item
                (p_error_idle, p_lie_idle), p_value_idle = stab_idle_item

                # Combine items on same meas error value. Multiply idle prob by two, since no difference in meas error
                combined_prob = p_value * (p_value_idle*2)
                if p_lie == p_lie_idle and combined_prob > 1e-14:
                    combined_dataframe.loc[(p_error, p_error_idle, p_lie), type] = combined_prob

    # Do not combine if only one dataframe or only the cutoff time value is searched for
    if len(dataframes) < 2 or cutoff_search:
        return dataframes[0]

    superoperator_stab = dataframes[0]
    superoperator_idle = dataframes[1]

    index = pd.MultiIndex.from_product([[item[0] for item in superoperator_stab.index if item[1]],
                                        [item[0] for item in superoperator_idle.index if item[1]],
                                        [False, True]],
                                       names=['error_stab', 'error_idle', 'lie'])
    combined_dataframe = pd.DataFrame(columns=superoperator_stab.columns, index=index)

    combined_dataframe = combined_dataframe.sort_index()

    # WARNING!!!
    # The order of the errors on the data qubits in 'p' should be altered for the qubits to be attached to
    # consistent nodes in the network (compared to the 's' errors). I now only fixed that for the "_formulate..."
    # function above.

    combine_dataframes(superoperator_stab['s'], superoperator_idle['s'], 's')
    combine_dataframes(superoperator_stab['p'], superoperator_idle['p'], 'p')
    combined_dataframe.fillna({'p': 0., 's': 0.}, inplace=True)

    combined_dataframe.iloc[0, 2:] = superoperator_stab.iloc[0, 2:]
    combined_dataframe.iloc[0, combined_dataframe.columns.get_loc('qubit_order')] = \
        (superoperator_stab.iloc[0, superoperator_stab.columns.get_loc('qubit_order')] +
         superoperator_idle.iloc[0, superoperator_idle.columns.get_loc('qubit_order')])
    combined_dataframe = combined_dataframe[(combined_dataframe.T
                                             .applymap(lambda x: x != 0 and x is not None and not pd.isna(x))).any()]

    return combined_dataframe


def _init_random_seed(timestamp=None, worker=0, iteration=0):
    if timestamp is None:
        timestamp = time.time()
    seed = int("{:.0f}".format(timestamp * 10 ** 7) + str(worker) + str(iteration))
    random.seed(float(seed))
    return seed


def add_column_values(dataframe, columns, values):
    for column, value in zip(columns, values):
        dataframe[column] = None
        dataframe.iloc[0, dataframe.columns.get_loc(column)] = value


def _combine_superoperator_dataframes(dataframe_1, dataframe_2):
    """
        Combines two given superoperator dataframes into one dataframe

        Parameters
        ----------
        dataframe_1 : pd.DataFrame or None
            Superoperator dataframe to be combined
        dataframe_2 : pd.DataFrame or None
            Superoperator dataframe to be combined
    """
    if dataframe_1 is None and dataframe_2 is None:
        return None
    if dataframe_1 is None:
        return dataframe_2
    if dataframe_2 is None:
        return dataframe_1

    new_df = copy(dataframe_1) if dataframe_1.shape[0] > dataframe_2.shape[0] else copy(dataframe_2)
    other_df = copy(dataframe_2) if dataframe_1.shape[0] > dataframe_2.shape[0] else copy(dataframe_1)

    # First combine the total amount of iterations, such that it can be used later
    written_to_original = new_df.iloc[0, new_df.columns.get_loc("written_to")]
    written_to_new = other_df.iloc[0, other_df.columns.get_loc("written_to")]
    corrected_written_to = written_to_new + written_to_original
    new_df.iloc[0, new_df.columns.get_loc("written_to")] = corrected_written_to

    if round(sum(new_df['p']), SUM_ACCURACY) != 1.0 or round(sum(other_df['p']), SUM_ACCURACY) != 1.0:
        print("Warning: Probabilities of (one of) the dataframes does not sum to 1.", file=sys.stderr)

    # Calculate the average probability of the error configurations per stabilizer
    other_df[['p', 's']] = other_df[['p', 's']].mul(written_to_new)
    new_df[['p', 's']] = new_df[['p', 's']].mul(written_to_original)

    if 'idle' in new_df.columns.tolist():
        other_df[['idle']] = other_df[['idle']].mul(written_to_new)
        new_df[['idle']] = new_df[['idle']].mul(written_to_original)

    combined_elements = new_df[['p', 's']].add(other_df[['p', 's']], fill_value=0).div(corrected_written_to)
    new_df = new_df.assign(p=combined_elements['p'])
    new_df = new_df.assign(s=combined_elements['s'])

    if 'idle' in new_df.columns.tolist():
        combined_idle_elements = new_df[['idle']].add(other_df[['idle']], fill_value=0).div(corrected_written_to)
        new_df = new_df.assign(idle=combined_idle_elements['idle'])

    # Update the average of the other system characteristics
    new_df['total_duration'] = (new_df['total_duration'] + other_df['total_duration'])
    new_df['total_link_attempts'] = (new_df['total_link_attempts'] + other_df['total_link_attempts'])

    new_df['avg_link_attempts'] = new_df['total_link_attempts'] / corrected_written_to
    new_df['avg_duration'] = new_df['total_duration'] / corrected_written_to
    if 'dur_99' in new_df:
        new_df['dur_99'] = (new_df['dur_99'].mul(written_to_original) +
                            other_df['dur_99'].mul(written_to_new)) / corrected_written_to

    # Update fidelity
    other_df['ghz_fidelity'] = other_df['ghz_fidelity'].mul(written_to_new)
    new_df['ghz_fidelity'] = new_df['ghz_fidelity'].mul(written_to_original)

    new_df['ghz_fidelity'] = (new_df['ghz_fidelity'] + other_df['ghz_fidelity']) / corrected_written_to
    new_df = new_df[(new_df.T.applymap(lambda x: x != 0 and x is not None and not pd.isna(x))).any()]

    if 'date_and_time' in new_df:
        if 'date_and_time' in other_df:
            new_df.iloc[0, new_df.columns.get_loc("date_and_time")] += " " + other_df.iloc[0, other_df.columns.get_loc("date_and_time")]
    else:
        if 'date_and_time' in other_df:
            add_column_values(new_df, ['date_and_time'], [other_df.iloc[0, other_df.columns.get_loc("date_and_time")]])

    if round(sum(new_df['p']), SUM_ACCURACY) != 1.0:
        print("Warning: The combined dataframe sums to {}.".format(sum(new_df['p'])), file=sys.stderr)

    if 'idle' in new_df.columns.tolist():
        if round(sum(new_df['idle']), SUM_ACCURACY) != 1.0:
            print("Warning: The combined dataframe idle sums to {}.".format(sum(new_df['idle'])), file=sys.stderr)

    return new_df


def add_decoherence_if_cut_off(qc: QuantumCircuit):
    if qc.cut_off_time < np.inf and not qc.cut_off_time_reached:
        waiting_time = qc.cut_off_time - qc.total_duration
        if waiting_time > 0:
            qc._increase_duration(waiting_time, [], involved_nodes=list(qc.nodes.keys()), check=False)
            qc.end_current_sub_circuit(total=True, duration=waiting_time, sub_circuit="Waiting", apply_decoherence=True)


def _additional_qc_arguments(**kwargs):
    additional_arguments = {
        'noise': True,
        'basis_transformation_noise': False,
        'thread_safe_printing': True,
        'no_single_qubit_error': True
    }
    kwargs.update(additional_arguments)
    return kwargs


def print_circuit_parameters(operational_args, circuit_args, varational_circuit_args):
    print('\n' + 80*'#')
    for args_name, args_values in locals().items():
        print("\n{}:\n-----------------------".format(args_name.capitalize()))
        pprint(args_values)
    print('\n' + 80*'#' + '\n')


def additional_parsing_of_arguments(**args):
    # Pop the argument_file since it is no longer needed from this point
    args.pop("argument_file")

    if args['parameter_select'] is not None:
        par_select = args['parameter_select']
        args['p_g'] = [args['p_g'][par_select]]

    if 'auto_generated' in args['protocol'] and args['protocol_recipes_file'] is not None:
        # We are now going to overwrite the list with protocol recipes.
        # First we open the file with the protocol recipes:
        protocol_recipe_file = args['protocol_recipes_file']
        new_protocol_recipes = []
        if os.path.exists(protocol_recipe_file):
            with open(protocol_recipe_file, 'r') as protocol_recipes:
                lines = protocol_recipes.read().split('\n')
                for line in lines:
                    line = line.replace(" ", "")
                    if line:
                        if "(" in line:
                            line = line.strip("()")
                            full_settings = re.split(r",(?!(?:[^,\[\]]+,)*[^,\[\]]+])", line, 0)
                            print(full_settings)
                            if len(full_settings) > 1:
                                if "[" in full_settings[1]:
                                    cut_off_times = full_settings[1].strip("[]").split(",")
                                else:
                                    cut_off_times = [full_settings[1]] if full_settings[1] != 'None' else [None]
                            if len(full_settings) > 2:
                                if "[" in full_settings[2]:
                                    error_probs = full_settings[2].strip("[]").split(",")
                                else:
                                    error_probs = [full_settings[2]] if full_settings[2] != 'None' else [None]
                            if len(full_settings) == 1:
                                new_protocol_recipes.append(full_settings[0])
                            elif len(full_settings) == 2:
                                for cut_off_time in cut_off_times:
                                    cut_off_time = float(cut_off_time) if cut_off_time is not None else None
                                    new_protocol_recipes.append([full_settings[0], cut_off_time])
                            elif len(full_settings) == 3:
                                for cut_off_time in cut_off_times:
                                    cut_off_time = float(cut_off_time) if cut_off_time is not None else None
                                    for error_prob in error_probs:
                                        error_prob = float(error_prob) if error_prob is not None else None
                                        new_protocol_recipes.append([full_settings[0], cut_off_time, error_prob])
                        else:
                            new_protocol_recipes.append(line)
            args['protocol_recipe'] = new_protocol_recipes

    if args['protocol_recipe_select'] is not None:
        prot_select = args['protocol_recipe_select']
        selected_settings = args['protocol_recipe'][prot_select]
        if isinstance(selected_settings, list):
            args['protocol_recipe'] = [selected_settings[0]]
            if len(selected_settings) > 1 and selected_settings[1] is not None:
                args['cut_off_time'] = [selected_settings[1]]
            if len(selected_settings) > 2 and selected_settings[2] is not None:
                args['p_g'] = [selected_settings[2]]
        else:
            args['protocol_recipe'] = [selected_settings]

    if isinstance(args['protocol_recipe'], list) and any(isinstance(item, list) for item in args['protocol_recipe']):
        raise SyntaxError("It's not possible to import protocol recipes with cut-off time and/or error probability "
                          "settings from a file without selecting a protocol with 'protocol_recipe_select'.")

    # THIS IS NOT GENERIC, will error when directories are moved or renamed
    file_dir = os.path.dirname(__file__)
    look_up_table_dir = os.path.join(file_dir, '../gates', 'gate_lookup_tables')

    if args['single_qubit_gate_lookup'] is not None:
        with open(os.path.join(look_up_table_dir, args['single_qubit_gate_lookup']), 'rb') as obj:
            args['single_qubit_gate_lookup'] = pickle.load(obj)

    if args['two_qubit_gate_lookup'] is not None:
        with open(os.path.join(look_up_table_dir, args['two_qubit_gate_lookup']), "rb") as obj2:
            args['two_qubit_gate_lookup'] = pickle.load(obj2)

    gate_duration_file = args.get('gate_duration_file')
    gates_dict = {}
    if gate_duration_file is not None and os.path.exists(gate_duration_file):
        gates_dict = set_gate_durations_from_file(gate_duration_file, noiseless_swap=args['noiseless_swap'])
    elif gate_duration_file is not None:
        raise ValueError("Cannot find file to set gate durations with. File path: {}"
                         .format(os.path.abspath(gate_duration_file)))
    for gate_dur in range(len(args['gate_durations'])):
        if args['gate_durations'][gate_dur] is None:
            args['gate_durations'][gate_dur] = {}
        for key in gates_dict.keys():
            value = None
            if key not in args['gate_durations'][gate_dur]:
                if key in gates_dict:
                    value = args['gate_durations'][gate_dur][key] = gates_dict[key]
            else:
                value = args['gate_durations'][gate_dur][key]
            if value:
                if isinstance(value, list):
                    args['gate_durations'][gate_dur][key] = value
                else:
                    args['gate_durations'][gate_dur][key] = [value, None]
                for isv, subvalue in enumerate(args['gate_durations'][gate_dur][key]):
                    if subvalue is None and key in gates_dict and len(gates_dict[key]) > isv:
                        args['gate_durations'][gate_dur][key][isv] = gates_dict[key][isv]

    if args.get('protocol_recipe') is not None:
        for i_prf, prot_rec_file in enumerate(args.get('protocol_recipe')):
            file_name = get_full_path("circuit_simulation/protocol_recipes/" + prot_rec_file)
            # file_name = os.path.abspath(os.getcwd()) + "/ciruit_simulation/protocol_recipes/" + prot_rec_file
            if not os.path.exists(file_name):
                raise ValueError("Cannot find file to set the protocol recipe with. File path: {}"
                                 .format(os.path.abspath(file_name)))

    default_values = {'bell_pair_type': 3, 'network_noise_type': 1, 'F_link': 0.01, 'p_link': 0.0001,
                      'bell_pair_parameters': None, 't_link': 6e-06, 't_meas': 4e-06, 'T1n_idle': 300, 'T1n_link': 1.2,
                      'T1e_idle': 10000, 'T2n_idle': 10, 'T2n_link': 1.2, 'T2e_idle': 1, 't_pulse': 13e-03,
                      'n_DD': 2500, 'gate_durations': None, 'p_g': 0.01, 'p_m': 0.01, 'p_m_1': None,
                      'noiseless_swap': False, 'decoherence': True, 'probabilistic': True}

    varied_parameters = []
    for key, value in args.items():
        if isinstance(value, list) and len(value) > 1 and key in default_values.keys():
            varied_parameters.append(key)
        # if assign_default_values and value is None and key in default_values.keys():
        #     args[key] = default_values[key]
        # if assign_default_values and value == [None] and key in default_values.keys():
        #     args[key] = [default_values[key]]
    if 'p_g' in varied_parameters and 'p_m' not in varied_parameters and (args['p_m_equals_p_g']
                                                                          or args['p_m_equals_extra_noisy_measurement']):
        varied_parameters.append('p_m')
    args['default_values'] = default_values
    args['varied_parameters'] = varied_parameters

    if args['network_noise_type'] in [*range(10, 22)] + [99]:
        args['bell_pair_type'] = 0
    if args['network_noise_type'] == 3:
        args['bell_pair_type'] = 3

    return args


def update_parameters(run_dict, circuit_args, default_values, varied_parameters):
    bell_pair_parameter_list = ['ent_prot', 'F_prep', 'p_DE', 'mu', 'lambda', 'eta', 'alpha']
    gate_duration_parameter_list = {'X_gate': ('tn_X', 'te_X'), 'Y_gate': ('tn_Y', 'te_Y'), 'Z_gate': ('tn_Z', 'te_Z'),
                                    'H_gate': ('tn_H', 'te_H'), 'CNOT_gate': ('t_CX', None), 'CZ_gate': ('t_CZ', None),
                                    'CiY_gate': ('t_CiY', None), 'SWAP_gate': ('t_SWAP', None)}
    gate_duration_parameter_list_values = []
    for value in gate_duration_parameter_list.values():
        gate_duration_parameter_list_values.append(value[0])
        if value[1]:
            gate_duration_parameter_list_values.append(value[1])

    # Here we collect all parameters that describe a set in sets.csv, and put them in a list (parameters_in_df)
    node_df = pd.read_csv(get_full_path("circuit_simulation/node/sets.csv"), sep=";", float_precision="round_trip",
                          index_col="name")
    node_df.replace(np.nan, None, inplace=True)
    node_df.replace("None", None, inplace=True)
    node_df.replace("np.infty", np.infty, inplace=True)
    # node_df.replace("False", False, inplace=True)
    # node_df.replace("True", True, inplace=True)
    node_dict = node_df.drop(columns=["nickname"]+varied_parameters).to_dict("index")
    node_dict_full = node_df.drop(columns="nickname").to_dict("index")
    parameters_in_df = list(node_dict[list(node_dict.keys())[0]].keys())

    # run_dict['set_number'] = "TestSet"

    nicknames_to_names = {v.lower(): k for k, v in node_df["nickname"].to_dict().items()}

    create_new_set = True
    set_parameters = None
    if run_dict['set_number'] is not None:
        try:
            # set_parameters = node_dict[nicknames_to_names[run_dict['set_number'].lower()]]
            set_parameters = node_dict_full[nicknames_to_names[run_dict['set_number'].lower()]]
            run_dict['_node'] = nicknames_to_names[run_dict['set_number'].lower()]
            create_new_set = False
        except KeyError:
            pass

    diff_params = []
    new_set_parameters = dict((key, None) for key in parameters_in_df)

    non_emission_based_data = False
    if circuit_args['network_noise_type'] in range(10, 22):
        non_emission_based_i = circuit_args['network_noise_type'] - 10
        non_emission_based_data = True
        circuit_args['bell_pair_type'] = 0
    if (circuit_args['network_noise_type'] is None
            and set_parameters
            and set_parameters['network_noise_type'] in range(10, 22)):
        non_emission_based_i = set_parameters['network_noise_type'] - 10
        non_emission_based_data = True
        circuit_args['bell_pair_type'] = 0
    if non_emission_based_data:
        data = np.load('circuit_simulation/states/non_emission_based_99_fidelity_Bell_states.npy', allow_pickle=True)
        density_matrix_target = sp.lil_matrix((4, 4))
        density_matrix_target[0, 0] = 0.5
        density_matrix_target[0, 3] = 0.5
        density_matrix_target[3, 0] = 0.5
        density_matrix_target[3, 3] = 0.5
        non_emission_based = {}
        non_emission_based['F_link'] = [None] * len(data)
        non_emission_based['p_link'] = [None] * len(data)
        non_emission_based['t_link'] = [None] * len(data)
        for i in range(len(data)):
            non_emission_based['F_link'][i] = fidelity(sp.lil_matrix(data[i]['super_simulation']['density_matrix']), density_matrix_target)
            non_emission_based['p_link'][i] = data[i]['super_simulation']['p_success']
            non_emission_based['t_link'][i] = data[i]['super_simulation']['eff_time']

    for parameter, default_value in default_values.items():
        update_dict = circuit_args if parameter in ['bell_pair_type', 'network_noise_type', 'noiseless_swap', 'decoherence', 'probabilistic'] \
            else run_dict
        if parameter == 'bell_pair_parameters':
            if circuit_args['network_noise_type'] in [3, 99] or \
                    (circuit_args['network_noise_type'] is None and set_parameters and
                     set_parameters['network_noise_type'] in [3, 99]):
                if update_dict['bell_pair_parameters'] is None:
                    update_dict['bell_pair_parameters'] = {}
                for subparameter in bell_pair_parameter_list:
                    if subparameter in update_dict['bell_pair_parameters']:
                        if set_parameters and set_parameters[subparameter] != update_dict['bell_pair_parameters'][subparameter]:
                            diff_params.append(subparameter)
                    else:
                        update_dict['bell_pair_parameters'][subparameter] = None
                    subvalue = update_dict['bell_pair_parameters'][subparameter]
                    if subvalue is None and set_parameters and set_parameters[subparameter]:
                        update_dict['bell_pair_parameters'][subparameter] = set_parameters[subparameter]
                    elif subvalue is None and not (subparameter == 'alpha' and
                                                   update_dict['bell_pair_parameters']['ent_prot'] == "double_click"):
                        raise ValueError("Invalid combination of Bell pair parameters.")
                    # if isinstance(update_dict['bell_pair_parameters'][subparameter], int) and not isinstance(update_dict['bell_pair_parameters'][subparameter], bool):
                    #     new_set_parameters[subparameter] = float(update_dict['bell_pair_parameters'][subparameter])
                    # else:
                    new_set_parameters[subparameter] = update_dict['bell_pair_parameters'][subparameter]
                update_dict['F_link'] = None
                new_set_parameters['F_link'] = None
                update_dict['p_link'] = None
                new_set_parameters['p_link'] = None
            else:
                for subparameter in ['F_link', 'p_link']:
                    if non_emission_based_data:
                        update_dict[subparameter] = non_emission_based[subparameter][non_emission_based_i]
                        new_set_parameters[subparameter] = non_emission_based[subparameter][non_emission_based_i]
                    elif update_dict[subparameter] is None:
                        if set_parameters and set_parameters[subparameter]:
                            update_dict[subparameter] = set_parameters[subparameter]
                        else:
                            update_dict[subparameter] = default_values[subparameter]
                        new_set_parameters[subparameter] = update_dict[subparameter]
                    elif set_parameters and update_dict[subparameter] != set_parameters[subparameter]:
                        diff_params.append(subparameter)
                update_dict['bell_pair_parameters'] = None

        elif parameter == 'gate_durations':
            gate_times_in_set_parameters = False
            if set_parameters:
                for subparameter in gate_duration_parameter_list_values:
                    if set_parameters[subparameter]:
                        gate_times_in_set_parameters = True
                        break
            if gate_times_in_set_parameters:
                if run_dict['gate_durations'] is None or len(run_dict['gate_durations']) == 0:
                    run_dict['gate_durations'] = {}
                for subkey, subparameter in gate_duration_parameter_list.items():
                    for i_ssp, subsubparameter in enumerate(subparameter):
                        # if subsubparameter and ((subkey not in run_dict['gate_durations'] and set_parameters[subsubparameter]) or
                        #                         (subkey in run_dict['gate_durations'] and run_dict['gate_durations'][subkey] != set_parameters[subsubparameter])):
                        #     diff_params.append(subsubparameter)
                        if subsubparameter is not None and set_parameters[subsubparameter]:
                            if subkey not in run_dict['gate_durations']:
                                run_dict['gate_durations'][subkey] = [None, None]
                            if run_dict['gate_durations'][subkey][i_ssp] is None:
                                run_dict['gate_durations'][subkey][i_ssp] = set_parameters[subsubparameter]
            for subkey, subparameter in gate_duration_parameter_list.items():
                for i_ssp, subsubparameter in enumerate(subparameter):
                    if subsubparameter is not None \
                            and run_dict['gate_durations'] and subkey in run_dict['gate_durations'] \
                            and len(run_dict['gate_durations'][subkey]) > i_ssp:
                        new_set_parameters[subsubparameter] = float(run_dict['gate_durations'][subkey][i_ssp])
                    elif subsubparameter is not None:
                        new_set_parameters[subsubparameter] = None
                    if subsubparameter is not None \
                            and set_parameters \
                            and new_set_parameters[subsubparameter] != set_parameters[subsubparameter]:
                        diff_params.append(subsubparameter)

        elif update_dict[parameter] is None and parameter not in ['F_link', 'p_link']:
            if parameter == 't_link' and non_emission_based_data:
                update_dict[parameter] = non_emission_based[parameter][non_emission_based_i]
            elif set_parameters and set_parameters[parameter] is not None:
                update_dict[parameter] = set_parameters[parameter]
            else:
                update_dict[parameter] = default_value

        elif parameter in ['noiseless_swap', 'decoherence', 'probabilistic']:
            if update_dict[parameter] is False \
                    and set_parameters is not None \
                    and set_parameters[parameter] is not False:
                update_dict[parameter] = set_parameters[parameter]

        if parameter not in ['F_link', 'p_link', 'gate_durations', 'bell_pair_parameters']:
            # CHANGE THINGS HERE!
            # if isinstance(update_dict[parameter], int) and not isinstance(update_dict[parameter], bool):
            #     new_set_parameters[parameter] = float(update_dict[parameter])
            # else:
            if parameter in ['T1e_idle', 'T1n_idle', 'T1n_link', 'T2e_idle', 'T2n_idle', 'T2n_link'] \
                    and update_dict[parameter] in ['infty', 'infinity', 'inf', 'np.infty']:
                # ['T1n_idle', 'T1e_idle']
                new_set_parameters[parameter] = 'np.infty'
            else:
                new_set_parameters[parameter] = update_dict[parameter]
            if set_parameters and set_parameters[parameter] != update_dict[parameter]:
                if ('T1' in parameter or 'T2' in parameter or 't_' in parameter) \
                        and circuit_args['decoherence'] is False:
                    pass
                elif parameter in ['T1e_idle', 'T1n_idle', 'T1n_link', 'T2e_idle', 'T2n_idle', 'T2n_link']:     # ['T1n_idle', 'T1e_idle']:
                    try:
                        float1 = float(update_dict[parameter]) if update_dict[parameter] is not None else None
                        float2 = float(set_parameters[parameter]) if set_parameters[parameter] is not None else None
                        if float1 != float2:
                            diff_params.append(parameter)
                    except ValueError:
                        if update_dict[parameter] != set_parameters[parameter]:
                            diff_params.append(parameter)
                else:
                    diff_params.append(parameter)

    if run_dict['gate_durations'] is not None:
        set_duration_of_known_gates(run_dict['gate_durations'])

    new_set_parameters_full = copy(new_set_parameters)
    for parameter in varied_parameters:
        del new_set_parameters[parameter]

    # for parameter, value in new_set_parameters_full.items():
    #     run_dict[parameter] = value

    # print(f"new_set_parameters = {new_set_parameters}.")
    # Check if the parameters that we have now accidentally coincide with a set

    if create_new_set and run_dict['set_number']:
        # Save the parameters that we have as a new set
        node_nickname = run_dict['set_number']
        row_to_add = [[node_nickname] + list(new_set_parameters.values())]
        column_names = ["nickname"] + list(new_set_parameters.keys())
        index_to_add = ["s" + node_nickname[3:]] if node_nickname[:3].lower() == "set" else [node_nickname[:6]]
        node_df = pd.concat([node_df, pd.DataFrame(row_to_add, columns=column_names, index=index_to_add)])
        node_df.to_csv(get_full_path("circuit_simulation/node/sets.csv"), sep=";", index_label="name")
        run_dict['_node'] = index_to_add[0]

    if create_new_set and run_dict['set_number'] is None:
        # First we check if the parameter values that we have right now overlap with an existing set in our table
        # print(f"new_set_para = {new_set_parameters}.")
        # for parameter, value in new_set_parameters.items():
        #     print(parameter, value, type(value))
        # print(f"node_dict[5] = {list(node_dict.values())[3]}.")
        # for parameter, value in list(node_dict.values())[3].items():
        #     print(parameter, value, type(value))
        #     print(parameter, new_set_parameters[parameter], type(new_set_parameters[parameter]))
        try:
            node_name = list(node_dict.keys())[list(node_dict.values()).index(new_set_parameters)]
            identified_set_parameters = node_dict_full[node_name]
        except:
            node_name = "s" + str(len(node_dict))
            identified_set_parameters = copy(new_set_parameters_full)
            for parameter in varied_parameters:
                identified_set_parameters[parameter] = None
            row_to_add = [["Set" + str(len(node_dict))] + list(identified_set_parameters.values())]
            column_names = ["nickname"] + list(identified_set_parameters.keys())
            index_to_add = [node_name]
            node_df = pd.concat([node_df, pd.DataFrame(row_to_add, columns=column_names, index=index_to_add)])
            node_df.to_csv(get_full_path("circuit_simulation/node/sets.csv"), sep=";", index_label="name")
        for parameter in varied_parameters:
            if new_set_parameters_full[parameter] != identified_set_parameters[parameter]:
                diff_params.append(parameter)
        run_dict['_node'] = node_name

    # For the time-being, the |01> + |10> state is hard-coded under "network_noise_type = 3" as a noisy state of the
    # type "bell_pair_type = 3". Therefore, if "network_noise_type = 3" is selected, we have to also set
    # "Bell_pair_type" to 3 for the time being. The same is true for "network_noise_type = 99": in that case we always
    # have to use "bell_pair_type = 0".
    circuit_args['bell_pair_type'] = 3 if circuit_args['network_noise_type'] == 3 else circuit_args['bell_pair_type']
    circuit_args['bell_pair_type'] = 0 if circuit_args['network_noise_type'] == 99 else circuit_args['bell_pair_type']

    if circuit_args['network_noise_type'] in [3]:
        mu = run_dict['bell_pair_parameters']['mu']
        F_prep = run_dict['bell_pair_parameters']['F_prep']
        labda = run_dict['bell_pair_parameters']['lambda']
        p_DE = run_dict['bell_pair_parameters']['p_DE']
        eta = run_dict['bell_pair_parameters']['eta']

        phi = math.sqrt(mu) * ((2 * F_prep - 1) ** 2) * (2 * labda - 1) * ((1 - p_DE) ** 2)

        if run_dict['bell_pair_parameters']['ent_prot'] == 'single_click':
            alpha = run_dict['bell_pair_parameters']['alpha']
            p_link = (2 * eta * (1 - eta) + (1 + mu)/2 * eta ** 2) * alpha ** 2 + 2 * eta * alpha * (1 - alpha)
            coeff_psi_p = 1 / p_link * (1 + phi) * eta * alpha * (1 - alpha)
        else:
            alpha = 1/2
            p_link = 2 * alpha * (1 - alpha) * eta ** 2
            coeff_psi_p = 1 / p_link * (1 + phi ** 2) * alpha * (1 - alpha) * eta ** 2
            if 'alpha' in run_dict['bell_pair_parameters']:
                del run_dict['bell_pair_parameters']['alpha']

        run_dict['F_link'] = 1 - coeff_psi_p
        run_dict['p_link'] = p_link

        # print(f"phi = {phi}.")
        # print(f"Fidelity = {coeff_psi_p}.")
        # print(f"Success probabilty = {p_link}.")

        circuit_args['bell_pair_type'] = 3
    elif circuit_args['network_noise_type'] in [99]:
        circuit_args['bell_pair_type'] = 0
    else:
        run_dict['bell_pair_parameters'] = None

    for coh_time in ['T1e_idle', 'T1n_idle', 'T1n_link', 'T2e_idle', 'T2n_idle', 'T2n_link']:   # ['T1n_idle', 'T1e_idle']:
        try:
            run_dict[coh_time] = float(run_dict[coh_time])
        except ValueError:
            if run_dict[coh_time] in ['np.infty', 'infinity', 'infty', 'inf']:
                run_dict[coh_time] = np.infty
            else:
                raise ValueError(f"The coherence time set for {coh_time} is not understood.")

    diff_params = dict((param, new_set_parameters_full[param]) for param in diff_params)

    return diff_params


def _merge_success_failed(succeeded, cut_off, fn):
    """ Merges the success and failed superoperator into one in the new format! """
    
    ghz_success = succeeded['written_to'][0] if succeeded is not None else None
    ghz_failed = cut_off['written_to'][0] if cut_off is not None else None

    if ghz_success is None:
        success_rate = 0
        iterations = ghz_failed
        failed_rate = 1
        nr_data_qubits = len(cut_off['qubit_order'][0].split(","))
    elif ghz_failed is None:
        failed_rate = 0
        iterations = ghz_success
        success_rate = 1
        nr_data_qubits = len(succeeded['qubit_order'][0].split(","))
    else:
        success_rate = ghz_success / (ghz_success + ghz_failed)
        failed_rate = ghz_failed / (ghz_success + ghz_failed)
        iterations = ghz_success + ghz_failed
        nr_data_qubits = len(succeeded['qubit_order'][0].split(","))

    error_configs = [''.join(comb) for comb in product(["I", "X", "Y", "Z"], repeat=nr_data_qubits)]

    # Make the new dataframe with the new essential columns:
    merged_superoperator = pd.DataFrame(columns=["error_config", "ghz_success", "lie", "p", "s", "idle"])

    if (ghz_success is not None) and (ghz_failed is not None): # When both superoperators are present
        success_available_configs = list(set([config[0] for config in list(succeeded.index)]))
        failed_available_configs = list(set([config[0] for config in list(cut_off.index)]))
        for config in error_configs:
            p_TF = p_TT = p_FF = p_FT = s_TF = s_TT = s_FF = s_FT = idle_T = idle_F = 0
            if (config in success_available_configs) and (config in failed_available_configs):
                if False in succeeded['p'][config].keys():
                    p_TF = succeeded['p'][config][False] * success_rate
                if True in succeeded['p'][config].keys():
                    p_TT = succeeded['p'][config][True] * success_rate
                if False in cut_off['p'][config].keys():
                    p_FF = cut_off['p'][config][False] * failed_rate
                if True in cut_off['p'][config].keys():
                    p_FT = cut_off['p'][config][True] * failed_rate

                if False in succeeded['s'][config].keys():
                    s_TF = succeeded['s'][config][False] * success_rate
                if True in succeeded['s'][config].keys():
                    s_TT = succeeded['s'][config][True] * success_rate
                if False in cut_off['s'][config].keys():
                    s_FF = cut_off['s'][config][False] * failed_rate
                if True in cut_off['s'][config].keys():
                    s_FT = cut_off['s'][config][True] * failed_rate

                if "idle" in succeeded and False in succeeded['idle'][config].keys():
                    idle_T = succeeded['idle'][config][False] * success_rate
                else:
                    idle_T = 0
                if "idle" in cut_off and True in cut_off['idle'][config].keys():
                    idle_F = cut_off['idle'][config][False] * failed_rate
                else:
                    idle_F = 0

            if (config in success_available_configs) and (config not in failed_available_configs):
                if False in succeeded['p'][config].keys():
                    p_TF = succeeded['p'][config][False] * success_rate
                if True in succeeded['p'][config].keys():
                    p_TT = succeeded['p'][config][True] * success_rate

                if False in succeeded['s'][config].keys():
                    s_TF = succeeded['s'][config][False] * success_rate
                if True in succeeded['s'][config].keys():
                    s_TT = succeeded['s'][config][True] * success_rate

                if "idle" in succeeded and False in succeeded['idle'][config].keys():
                    idle_T = succeeded['idle'][config][False] * success_rate
                else:
                    idle_T = 0
            
            if (config not in success_available_configs) and (config in failed_available_configs):
                if False in cut_off['p'][config].keys():
                    p_FF = cut_off['p'][config][False] * failed_rate
                if True in cut_off['p'][config].keys():
                    p_FT = cut_off['p'][config][True] * failed_rate

                if False in cut_off['s'][config].keys():
                    s_FF = cut_off['s'][config][False] * failed_rate
                if True in cut_off['s'][config].keys():
                    s_FT = cut_off['s'][config][True] * failed_rate

                if "idle" in cut_off and True in cut_off['idle'][config].keys():
                    idle_F = cut_off['idle'][config][False] * failed_rate
                else:
                    idle_F = 0

            if p_TF == p_TT == p_FF == p_FT == s_TF == s_TT == s_FF == s_FT == idle_T == idle_F == 0:
                pass
            else:
                if not (p_TF == s_TF == idle_T == 0):
                    merged_superoperator.loc[len(merged_superoperator)] = [config, True, False, p_TF, s_TF, idle_T]
                if not (p_TT == s_TT == idle_T == 0):
                    merged_superoperator.loc[len(merged_superoperator)] = [config, True, True, p_TT, s_TT, idle_T]
                if not (p_FF == s_FF == idle_F == 0):
                    merged_superoperator.loc[len(merged_superoperator)] = [config, False, False, p_FF, s_FF, idle_F]
                if not (p_FT == s_FT == idle_F == 0):
                    merged_superoperator.loc[len(merged_superoperator)] = [config, False, True, p_FT, s_FT, idle_F]

    if (ghz_success is not None) and (ghz_failed is None): # When only success superoperator is present
        success_available_configs = list(set([config[0] for config in list(succeeded.index)]))
        for config in error_configs:
            p_TF = p_TT = p_FF = p_FT = s_TF = s_TT = s_FF = s_FT = idle_T = idle_F = 0
            if config in success_available_configs:
                if False in succeeded['p'][config].keys():
                    p_TF = succeeded['p'][config][False] * success_rate
                if True in succeeded['p'][config].keys():
                    p_TT = succeeded['p'][config][True] * success_rate

                if False in succeeded['s'][config].keys():
                    s_TF = succeeded['s'][config][False] * success_rate
                if True in succeeded['s'][config].keys():
                    s_TT = succeeded['s'][config][True] * success_rate

                if "idle" in succeeded and False in succeeded['idle'][config].keys():
                    idle_T = succeeded['idle'][config][False] * success_rate
                else:
                    idle_T = 0

                if p_TF == p_TT == p_FF == p_FT == s_TF == s_TT == s_FF == s_FT == idle_T == idle_F == 0:
                    pass
                else:
                    if not (p_TF == s_TF == idle_T == 0):
                        merged_superoperator.loc[len(merged_superoperator)] = [config, True, False, p_TF, s_TF, idle_T]
                    if not (p_TT == s_TT == idle_T == 0):
                        merged_superoperator.loc[len(merged_superoperator)] = [config, True, True, p_TT, s_TT, idle_T]
                    if not (p_FF == s_FF == idle_F == 0):
                        merged_superoperator.loc[len(merged_superoperator)] = [config, False, False, p_FF, s_FF, idle_F]
                    if not (p_FT == s_FT == idle_F == 0):
                        merged_superoperator.loc[len(merged_superoperator)] = [config, False, True, p_FT, s_FT, idle_F]

    if (ghz_success is None) and (ghz_failed is not None): # When only failed superoperator is present
        failed_available_configs = list(set([config[0] for config in list(cut_off.index)]))
        for config in error_configs:
            p_TF = p_TT = p_FF = p_FT = s_TF = s_TT = s_FF = s_FT = idle_T = idle_F = 0
            if config in failed_available_configs:
                if False in cut_off['p'][config].keys():
                    p_FF = cut_off['p'][config][False] * failed_rate
                if True in cut_off['p'][config].keys():
                    p_FT = cut_off['p'][config][True] * failed_rate

                if False in cut_off['s'][config].keys():
                    s_FF = cut_off['s'][config][False] * failed_rate
                if True in cut_off['s'][config].keys():
                    s_FT = cut_off['s'][config][True] * failed_rate

                if "idle" in cut_off and True in cut_off['idle'][config].keys():
                    idle_F = cut_off['idle'][config][False] * failed_rate
                else:
                    idle_F = 0

                if p_TF == p_TT == p_FF == p_FT == s_TF == s_TT == s_FF == s_FT == idle_T == idle_F == 0:
                    pass
                else:
                    if not (p_TF == s_TF == idle_T == 0):
                        merged_superoperator.loc[len(merged_superoperator)] = [config, True, False, p_TF, s_TF, idle_T]
                    if not (p_TT == s_TT == idle_T == 0):
                        merged_superoperator.loc[len(merged_superoperator)] = [config, True, True, p_TT, s_TT, idle_T]
                    if not (p_FF == s_FF == idle_F == 0):
                        merged_superoperator.loc[len(merged_superoperator)] = [config, False, False, p_FF, s_FF, idle_F]
                    if not (p_FT == s_FT == idle_F == 0):
                        merged_superoperator.loc[len(merged_superoperator)] = [config, False, True, p_FT, s_FT, idle_F]

    # Now we put back the metadata in the first row!
    if ghz_success is not None:
        additional_columns = list(succeeded.columns)
        first_row_vals = succeeded.iloc[0, :].values.tolist()
        remove_columns = 3 if "idle" in succeeded else 2
    else:
        additional_columns = list(cut_off.columns)
        first_row_vals = cut_off.iloc[0, :].values.tolist()
        remove_columns = 3 if "idle" in cut_off else 2

    for i in range(remove_columns):
        # Remove the first 2 columns as they have already been added! First two columns became index in previous form
        # of the superoperator:
        additional_columns.pop(0)
        # Remove the first 2 column data as it has been already added:
        first_row_vals.pop(0)

    new_first_row = merged_superoperator.iloc[0, :].values.tolist()
    new_first_row = new_first_row + first_row_vals
    new_first_row[6] = iterations

    for new_col in additional_columns:
        merged_superoperator[f"{new_col}"] = np.nan

    merged_superoperator.loc[0] = new_first_row

    # Set the indexes back to the three columns:
    merged_superoperator.set_index(['error_config', 'ghz_success', 'lie'], inplace=True, drop=True)

    # Remove the 'idle' column if there are no idling qubits:
    if (succeeded is None or "idle" not in succeeded) and (cut_off is None or "idle" not in cut_off):
        merged_superoperator.drop('idle', axis=1, inplace=True)

    # # Print for verification of probability sum
    # print("##### Statistics of the final superoperator #####")
    # print("Plaquette stabilizers probability sum: {}".format(sum(merged_superoperator['p'])))
    # print("Star stabilizers probability sum: {}".format(sum(merged_superoperator['s'])))
    # print("Idling qubits probability sum: {}".format(sum(merged_superoperator['idle'])))

    merged_superoperator.to_csv(fn + "_merged.csv", sep=';')


def _save_superoperator_dataframe(fn_short, characteristics, succeeded, cut_off, cut_off_info,
                                  do_not_merge_superoperators):
    if fn_short != (None, None) and len(fn_short) > 1:
        for dataframe in [succeeded, cut_off]:
            if dataframe is not None:
                add_column_values(dataframe, ['date_and_time'], [fn_short[1][:15]])
        # Adding confidence intervals to the superoperator
        print("Stabilizers probability sum: {}".format(sum(succeeded['p'])))

        succeeded = _add_interval_to_dataframe(succeeded, characteristics, cut_off_info)

        fn = os.path.join(get_full_path(fn_short[0]), fn_short[1])
        # Save pickle the characteristics file
        if os.path.exists(fn + '.pkl') and characteristics:
            characteristics_old = pickle.load(open(fn + '.pkl', 'rb'))
            [characteristics[key].extend(value) for key, value in characteristics_old.items() if key != 'index']
        pickle.dump(characteristics, file=open(fn + '.pkl', 'wb+')) if characteristics else None

        # Save the superoperators to a csv file
        for result, fn_add in zip([succeeded, cut_off], ['.csv', '_failed.csv']):
            fn_new = fn + fn_add
            existing_file = _open_existing_superoperator_file(fn_new)
            result = _combine_superoperator_dataframes(result, existing_file)
            if result is not None:
                result.to_csv(fn_new, sep=';')

        update_result_files(fn_short[0], fn_short[1][16:])

        if do_not_merge_superoperators is False:
            _merge_success_failed(succeeded, cut_off, fn)


def _add_interval_to_dataframe(dataframe, characteristics, cut_off_percentage=None):
    try:
        cut_off_percentage = float(cut_off_percentage)
    except TypeError:
        cut_off_percentage = 99
    except ValueError:
        cut_off_percentage = 99
    if dataframe is not None:
        conf_int = confidence_interval(characteristics['dur'], 1 - 2 * (100 - cut_off_percentage) / 100,
                                       require_unique_min=True, return_new_bound=True)
        add_column_values(dataframe, ['dur_' + str(conf_int[3])],
                          [conf_int[1]])
    return dataframe


def main_threaded(*, iterations, fn, **kwargs):
    # Run main method asynchronously with each worker getting an equal amount of iterations to run
    results = []
    workers = iterations if 0 < iterations < cpu_count() else cpu_count()
    if 'threads' in kwargs and kwargs['threads'] is not None:
        workers = kwargs['threads']
    thread_pool = Pool(workers)
    iterations, remaining_iterations = divmod(iterations, workers)
    kwargs['iter_pw'] = iterations
    print(f"Number of workers used = {workers}.")

    # for worker in range(1, workers + 1):
    #     thr_kwargs = deepcopy(kwargs)
    #     thr_kwargs['calc_id'] = (worker - 1)
    #     thr_kwargs['iterations'] = iterations + remaining_iterations * int(worker == workers)
    #     results.append(thread_pool.apply_async(main, kwds=thr_kwargs))
    for worker in range(workers):
        thr_kwargs = deepcopy(kwargs)
        thr_kwargs['calc_id'] = worker
        thr_kwargs['iterations'] = iterations + 1 if worker < remaining_iterations else iterations
        thr_kwargs['remaining_iterations'] = remaining_iterations
        results.append(thread_pool.apply_async(main, kwds=thr_kwargs))
    thread_pool.close()

    # Collect all the results from the workers
    succeeded = None
    cut_off = None
    print_lines_results = []
    tot_characteristics = defaultdict(list)
    for res in results:
        (succeeded_res, cut_off_res), print_lines, characteristics = res.get()
        succeeded = _combine_superoperator_dataframes(succeeded, succeeded_res)
        cut_off = _combine_superoperator_dataframes(cut_off, cut_off_res)
        print_lines_results.extend(print_lines)
        [tot_characteristics[key].extend(value) for key, value in characteristics.items()]

    # print(*print_lines_results)

    # Save superoperator dataframe to csv if exists and requested by user
    _save_superoperator_dataframe(fn, tot_characteristics, succeeded, cut_off, kwargs['cut_off'],
                                  kwargs['do_not_merge_superoperators'])
    return (succeeded, cut_off), print_lines_results, tot_characteristics


def main_series(fn, **kwargs):
    pbar_2 = tqdm(total=kwargs['iterations']) if kwargs.get('progress_bar') else None
    (succeeded, cut_off), print_lines, characteristics = main(pbar_2=pbar_2, **kwargs)
    # print(*print_lines)
    # if not kwargs['draw_circuit']:
    #     print(f"Durations: {characteristics['dur']}.")
    #     print(f"Stabilizer fidelities: {characteristics['stab_fid']}.")
    #     print(f"GHZ fidelities: {characteristics['ghz_fid']}.\n")

    # Save the superoperator to the according csv files (options: normal, cut-off)
    _save_superoperator_dataframe(fn, characteristics, succeeded, cut_off, kwargs['cut_off'],
                                  kwargs['do_not_merge_superoperators'])
    return (succeeded, cut_off), print_lines, characteristics



def main(*, iterations, protocol, stabilizer_type, threaded=False, gate_duration_file=None, cutoff_search=False,
         color=False, draw_circuit=True, save_latex_pdf=False, to_console=False, pbar_2=None, seed_number=None,
         protocol_recipe=None, calc_id=0, iter_pw=0, iter_offset=0, remaining_iterations=0, use_swap_gates=False,
         **kwargs):
    seeds_used_offset = int(iter_offset) + calc_id * iter_pw + min(calc_id, remaining_iterations)
    seeds_used = [*range(seeds_used_offset, seeds_used_offset + int(iterations))]
    supop_dataframe_failed = None
    supop_dataframe_succeed = None
    total_print_lines = []
    calc_avg_supop_state = kwargs["calc_avg_supop_state"] if "calc_avg_supop_state" in kwargs.keys() else False
    # calc_avg_supop_state = kwargs["calc_avg_supop_state"]
    if calc_avg_supop_state:
        characteristics = {'dur': [], 'stab_fid': [], 'ghz_fid': [], 'trace_dist_X': [], 'trace_dist_Z': [],
                           'cut_off_time': [], 'weighted_sum': [], 'avg_supop': [], 'avg_supop_iters': [],
                           'avg_supop_idle': [], 'avg_supop_idle_iters': []}
    else:
        characteristics = {'dur': [], 'stab_fid': [], 'ghz_fid': [], 'cut_off_time': [], 'weighted_sum': []}


    # CHECK HERE IF ANY OF THE PARAMETERS ARE DIFFERENT THEN IN THE SPECIFIED "SET" AND ADD THESE PARAMETERS TO THE
    # FILE NAME.

    if isinstance(protocol_recipe, str) and "auto_generated" in protocol:
        protocol_name_csv = "recipe_" + protocol_recipe + "_swap" if use_swap_gates else "recipe_" + protocol_recipe
    else:
        protocol_name_csv = protocol

    prot_recipe_folder = get_full_path("circuit_simulation/protocol_recipes/")
    if (protocol_recipe is not None) and isinstance(protocol_recipe, str) \
            and os.path.isfile(prot_recipe_folder + protocol_recipe):
        protocol_recipe = dill.load(open(prot_recipe_folder + protocol_recipe, 'rb'))

    # Progress bar initialisation
    pbar = None
    if pbar_2:
        # Second bar not working properly within PyCharm. Uncomment when using in normal terminal
        pass
        #pbar = tqdm(total=100, position=1, desc='Current circuit simulation')

    # Set the gate durations (when threaded, each thread needs its own modified copy of the gate duration file)
    if threaded:
        if kwargs['gate_durations']:
            set_duration_of_known_gates(kwargs['gate_durations'])
        else:
            set_gate_durations_from_file(gate_duration_file)

    # Get the QuantumCircuit object corresponding to the protocol and the protocol method by its name
    kwargs = _additional_qc_arguments(**kwargs)
    supop_qubits = None
    if "auto_generated" in protocol:
        qc, supop_qubits = agsmp.create_protocol_recipe_quantum_circuit(protocol_recipe, pbar, **kwargs)
    else:
        qc, supop_qubits = stab_protocols.create_quantum_circuit(protocol, pbar, **kwargs)

    if calc_avg_supop_state:
        nr_data_qubits = len(qc.data_qubits) if supop_qubits is None else len(supop_qubits[0])
        nr_q = 2 ** (nr_data_qubits * 2)
        avg_supop = {"X": {"succ": sp.csr_matrix((nr_q, nr_q)), "fail": sp.csr_matrix((nr_q, nr_q))},
                     "Z": {"succ": sp.csr_matrix((nr_q, nr_q)), "fail": sp.csr_matrix((nr_q, nr_q))}}
        avg_supop_iters = {"X": {"succ": 0, "fail": 0}, "Z": {"succ": 0, "fail": 0}}
        avg_supop_idle = {"X": {"succ": sp.csr_matrix((nr_q, nr_q)), "fail": sp.csr_matrix((nr_q, nr_q))},
                     "Z": {"succ": sp.csr_matrix((nr_q, nr_q)), "fail": sp.csr_matrix((nr_q, nr_q))}}
        avg_supop_idle_iters = {"X": {"succ": 0, "fail": 0}, "Z": {"succ": 0, "fail": 0}}
        trace_distance_dict = {"X": {"succ": None, "fail": None}, "Z": {"succ": None, "fail": None}}

    # Run iterations of the protocol
    for iter in range(iterations):
        pbar.reset() if pbar else None
        if pbar_2 is not None:
            pbar_2.update(1) if pbar_2 else None
        elif not kwargs['progress_bar']:
            pass
            # print(">>> At iteration {}/{}.".format(iter + 1, iterations), end='\r', flush=True)

        if seed_number is not None:
            seed_used = int(seed_number)
        else:
            seed_used = int(seeds_used[iter])
            # _init_random_seed(worker=threading.get_ident(), iteration=iter)
        # print(seed_used)
        random.seed(seed_used)

        # Run the user requested protocol
        operation = CZ_gate if stabilizer_type == "Z" else CNOT_gate
        if "auto_generated" in protocol:
            qc, stab_meas_nodes = agsmp.auto_generated_swap(qc, operation=operation, prot_rec=protocol_recipe)
        else:
            protocol_method = getattr(stab_protocols, protocol)
            stab_meas_nodes = protocol_method(qc, operation=operation)

        superoperator_qubits_list = [qc.data_qubits] if supop_qubits is None else supop_qubits
        inv_data_qubits = None if supop_qubits is None else superoperator_qubits_list[0]

        qcircuits_diff_stab_types = dict.fromkeys(stabilizer_type, None)
        if len(stabilizer_type) > 1:
            qc_copy = deepcopy(qc)
        for i, stab_type in enumerate(stabilizer_type):
            qc_used = qc if i == 0 else deepcopy(qc_copy)
            operation = CZ_gate if stab_type == "Z" else CNOT_gate
            qc_used.stabilizer_measurement(operation, nodes=stab_meas_nodes, swap=use_swap_gates, tqubit=inv_data_qubits)
            pbar.update(10 / len(stabilizer_type)) if pbar is not None else None
            qc_used.end_current_sub_circuit(total=True, forced_level=True, apply_decoherence=True)
            add_decoherence_if_cut_off(qc_used)
            if i == 0:
                qc_used.draw_circuit(no_color=not color, color_nodes=True) if draw_circuit else None
                qc_used.draw_circuit_latex() if save_latex_pdf else None

            pbar.update(10 / len(stabilizer_type)) if pbar is not None else None
            qcircuits_diff_stab_types[stab_type] = qc_used

        # If no superoperator qubits are returned, take the data qubits as such:
        # THIS IS NOW MOVED TO EARLIER IN THE CODE (THE CIRCUIT DEFINITION)
        # superoperator_qubits_list = [qc.data_qubits] if superoperator_qubits_list is None else superoperator_qubits_list

        # Obtain the superoperator in a dataframe format
        supop_dataframe = []
        for i, superoperator_qubits in enumerate(superoperator_qubits_list):
            idle_data_qubit = 4 if i != 0 else False
            superoperator_dict = {}
            for stab_type in qcircuits_diff_stab_types:
                # print(stab_type)
                qc_used = qcircuits_diff_stab_types[stab_type]
                if calc_avg_supop_state and idle_data_qubit is False:
                    succ_fail = "succ" if qc.cut_off_time_reached is False else "fail"
                    avg_supop[stab_type][succ_fail] = (qc_used.get_combined_density_matrix(superoperator_qubits)[0]
                                                       + avg_supop[stab_type][succ_fail]
                                                       * avg_supop_iters[stab_type][succ_fail]) \
                                                      / (avg_supop_iters[stab_type][succ_fail] + 1)
                    avg_supop_iters[stab_type][succ_fail] += 1
                    # if bef_aft == "aft":
                    #     trace_distance_dict[stab_type][succ_fail] = trace_distance(avg_supop["bef"][stab_type][succ_fail],
                    #                                                                avg_supop["aft"][stab_type][succ_fail])
                if calc_avg_supop_state and idle_data_qubit == 4:
                    succ_fail = "succ" if qc.cut_off_time_reached is False else "fail"
                    avg_supop_idle[stab_type][succ_fail] = (qc_used.get_combined_density_matrix(superoperator_qubits)[0]
                                                       + avg_supop_idle[stab_type][succ_fail]
                                                       * avg_supop_idle_iters[stab_type][succ_fail]) \
                                                      / (avg_supop_idle_iters[stab_type][succ_fail] + 1)
                    avg_supop_idle_iters[stab_type][succ_fail] += 1
                superoperator = qc_used.get_superoperator(superoperator_qubits, stab_type, no_color=(not color),
                                                          stabilizer_protocol=True, print_to_console=to_console,
                                                          idle_data_qubit=idle_data_qubit,
                                                          protocol_name=protocol_name_csv, return_dataframe=False)
                superoperator_dict[stab_type] = superoperator

                # for supel in superoperator:
                #     if supel.error_array == ['I'] * 4:
                #         print(superoperator_qubits, supel.error_array, supel.lie, supel.p)
            dataframe = qc._superoperator_to_dataframe(superoperator_dict, protocol_name=protocol_name_csv,
                                                       qubit_order=superoperator_qubits, **kwargs)
            # print(i)
            # print(dataframe.to_string())
            # print("\n\n\n\n\n\n")
            supop_dataframe.append(dataframe)

        # if ((not qc.cut_off_time_reached and qc.ghz_fidelity is None) or (qc.cut_off_time_reached and qc.ghz_fidelity)
        #    or round(sum(dataframe['p']), SUM_ACCURACY) != 1.0 or (qc.ghz_fidelity and qc.ghz_fidelity < 0.45)):
        #     print("Warning: Unexpected combination of parameters experienced:", file=sys.stderr, flush=True)
        #     print({'cutoff_reached': qc.cut_off_time_reached, 'duration': qc.total_duration, 'seed_used': seed_used,
        #            'ghz_fid': qc.ghz_fidelity, 'sum': sum(dataframe['p'])}, flush=True)
        #     # total_print_lines.append("Warning: Unexpected combination of parameters experienced:")
        #     # total_print_lines.append({'cutoff_reached': qc.cut_off_time_reached,
        #     #                           'duration': qc.total_duration, 'seed_used': seed_used,
        #     #                           'ghz_fid': qc.ghz_fidelity, 'sum': sum(dataframe['p'])})

        if kwargs['do_not_merge_superoperators'] is True:
            supop_dataframe = _combine_idle_and_stabilizer_superoperator(supop_dataframe, cutoff_search)
        else:
            supop_dataframe = _formulate_distributed_stabilizer_and_idle_superoperator(supop_dataframe, cutoff_search)

        pbar.update(10) if pbar is not None else None

        # if not qc.cut_off_time_reached:
        characteristics['dur'] += [qc.total_duration]
        characteristics['ghz_fid'] += [qc.ghz_fidelity]
        characteristics['stab_fid'] += [supop_dataframe.iloc[0, 0]]
        if not qc.cut_off_time_reached:
            # characteristics['trace_dist_X'] += [trace_distance_dict["X"]["succ"]]
            # characteristics['trace_dist_Z'] += [trace_distance_dict["Z"]["succ"]]
            characteristics['cut_off_time'] += [False]
        else:
            # characteristics['trace_dist_X'] += [trace_distance_dict["X"]["fail"]]
            # characteristics['trace_dist_Z'] += [trace_distance_dict["Z"]["fail"]]
            characteristics['cut_off_time'] += [True]

            # full_weight_prob_Z = 0
            # full_weight_prob_X = 0
            # full_weight_prob_Y = 0
            # full_weight_prob_meas = 0
            # for (error_array, meas), prob in supop_dataframe['p'].items():
            #     full_weight_prob_Z += prob * error_array.count("Z")
            #     full_weight_prob_X += prob * error_array.count("X")
            #     full_weight_prob_Y += prob * error_array.count("Y")
            #     if meas is True:
            #         full_weight_prob_meas += prob
            #
            # for (error_array, meas), prob in supop_dataframe['s'].items():
            #     full_weight_prob_Z += prob * error_array.count("Z")
            #     full_weight_prob_X += prob * error_array.count("X")
            #     full_weight_prob_Y += prob * error_array.count("Y")
            #     if meas is True:
            #         full_weight_prob_meas += prob
            #
            # characteristics['weighted_sum'] += [1 - (full_weight_prob_X + full_weight_prob_Z
            #                                          + 2 * full_weight_prob_Y + full_weight_prob_meas / 2)]

        # Fuse the superoperator dataframes obtained in each iteration
        if qc.cut_off_time_reached:
            # print(f"Cut-off reached for {seed_used}.")
            supop_dataframe_failed = _combine_superoperator_dataframes(supop_dataframe_failed, supop_dataframe)
        else:
            # print(f"Cut-off not reached for {seed_used}.")
            supop_dataframe_succeed = _combine_superoperator_dataframes(supop_dataframe_succeed, supop_dataframe)

        total_print_lines.extend(qc.print_lines)
        total_print_lines.append("\nStab fidelity: {}".format(supop_dataframe.iloc[0, 0])) if draw_circuit else None
        total_print_lines.append("\nGHZ fidelity: {} ".format(qc.ghz_fidelity)) if draw_circuit else None
        total_print_lines.append("\nTotal circuit duration: {} s".format(qc.total_duration)) if draw_circuit else None
        qc.reset()

    if calc_avg_supop_state:
        characteristics['avg_supop'] += [avg_supop]
        characteristics['avg_supop_iters'] += [avg_supop_iters]
        characteristics['avg_supop_idle'] += [avg_supop_idle]
        characteristics['avg_supop_idle_iters'] += [avg_supop_idle_iters]

    pbar_2.close() if pbar_2 else None
    pbar.close() if pbar is not None else None
    return (supop_dataframe_succeed, supop_dataframe_failed), total_print_lines, characteristics


def run_for_arguments(operational_args, circuit_args, var_circuit_args, **kwargs):
    default_values = kwargs['default_values']
    varied_parameters = kwargs['varied_parameters']
    filenames = []
    fn = None
    # cut_off_dataframe = _get_cut_off_dataframe(operational_args['cut_off'])
    var_circuit_args['cut_off'] = _get_cut_off_dataframe(var_circuit_args['cut_off'])

    # node = {2: 'Pur', 0.021: 'NatAb', 0: 'Ideal'}

    iterations = circuit_args['iterations']
    var_circuit_args['seed_number'] = [None] if iterations > 1 else var_circuit_args['seed_number']

    new_protocol_list, list_of_recipes = [], []
    if "auto_generated" in var_circuit_args['protocol']:
        # If there is a "auto_generated" protocol listed, and there are also "protocol_recipe"'s listed, we have to
        # make sure we don't loop extra over the protocols that are not "auto_generated".
        for protocol_name in var_circuit_args['protocol']:
            if "auto_generated" not in protocol_name:
                new_protocol_list.append(protocol_name)
        if var_circuit_args['protocol_recipe'] != None:
            for i_prot, protocol_recipe in enumerate(var_circuit_args['protocol_recipe']):
                new_protocol_list.append(("auto_generated", i_prot))
                list_of_recipes.append(protocol_recipe)
        var_circuit_args['protocol'] = new_protocol_list
    var_circuit_args["protocol_recipe"] = [None]

    back_up_F_link, back_up_p_link = [], []
    if var_circuit_args['bell_pair_parameters'][0] is not None:
        back_up_F_link = var_circuit_args['F_link']
        back_up_p_link = var_circuit_args['p_link']
        var_circuit_args['F_link'] = [None]
        var_circuit_args['p_link'] = [None]
        for i_bpp in range(len(var_circuit_args['bell_pair_parameters'])):
            var_circuit_args['bell_pair_parameters'][i_bpp] = (var_circuit_args['bell_pair_parameters'][i_bpp], i_bpp)

    # Loop over command line arguments
    for run in it.product(*(it.product([key], var_circuit_args[key]) for key in var_circuit_args.keys())):
        count = 0
        run_dict = dict(run)

        if run_dict['protocol'][0] == "auto_generated" and list_of_recipes:
            run_dict['protocol_recipe'] = list_of_recipes[run_dict['protocol'][1]]
            run_dict['protocol'] = run_dict['protocol'][0]

        if run_dict['bell_pair_parameters'] is not None:
            run_dict['F_link'] = back_up_F_link[run_dict['bell_pair_parameters'][1]]
            run_dict['p_link'] = back_up_p_link[run_dict['bell_pair_parameters'][1]]
            run_dict['bell_pair_parameters'] = run_dict['bell_pair_parameters'][0]

        if run_dict['protocol_recipe'] is not None \
                and isinstance(run_dict['protocol_recipe'], str) \
                and not os.path.isfile(get_full_path("circuit_simulation/protocol_recipes/") + run_dict['protocol_recipe']):
            run_dict['protocol'] = run_dict['protocol_recipe']
            run_dict['protocol_recipe'] = None

        # Set run_dict values based on circuit arguments
        # run_dict['p_link'] = run_dict['p_link'] if circuit_args['probabilistic'] else 1
        # run_dict['n_DD'] = run_dict['n_DD'] if (run_dict['t_pulse'] is None or run_dict['t_pulse'] > 0) else 0
        if circuit_args['p_m_equals_extra_noisy_measurement']:
            run_dict['p_m'] = 5 * run_dict['p_g'] / 3 - 4 * (run_dict['p_g']) ** 2 / 9 - 2 * (run_dict['p_g']) ** 3 / 9
        elif circuit_args['p_m_equals_p_g']:
            run_dict['p_m'] = run_dict['p_g']
        run_dict['protocol'] = (run_dict['protocol'] + "_swap" if circuit_args['use_swap_gates']
                                else run_dict['protocol'])

        cut_off_dataframe = run_dict['cut_off']
        ##### UPDATE THE PARAMETERS
        diff_params = update_parameters(run_dict, circuit_args, default_values, varied_parameters)

        # If cutoff time is not found in auto mode, it first does simulations to find this and then reruns with cutoff time
        while (run_dict['cut_off_time'] == np.inf and (cut_off_dataframe == 'auto'
                                                       or isinstance(cut_off_dataframe, float))) or count == 0:

            count += 1
            circuit_args['iterations'] = iterations
            run_dict['cut_off_time'], circuit_args['cutoff_search'] = _get_cut_off_time(cut_off_dataframe, run_dict,
                                                                                        circuit_args, diff_params,
                                                                                        **operational_args)

            fn_short = None
            if operational_args['csv_filename'] or operational_args['cp_path']:
                # Create parameter specific filename
                # fn = create_file_name(operational_args['csv_filename'], dec=circuit_args['decoherence'],
                #                       prob=circuit_args['probabilistic'], node=run_dict['_node'],
                #                       noiseless_swap=circuit_args['noiseless_swap'], **run_dict)
                fn_short = create_file_name(operational_args['csv_filename'], protocol=run_dict['protocol'],
                                            protocol_recipe=run_dict['protocol_recipe'], node=run_dict['_node'],
                                            **diff_params, dec=circuit_args['decoherence'],
                                            prob=circuit_args['probabilistic'], # noiseless_swap=circuit_args['noiseless_swap'],
                                            combine_supop=circuit_args['combine'], seed=run_dict['seed_number'],
                                            cut_off_time=run_dict['cut_off_time'])

                print(fn_short)

                update_result_files(operational_args['cp_path'], fn_short)

                fn = os.path.join(get_full_path(operational_args['cp_path']), fn_short)
                fn_main = os.path.join(get_full_path(operational_args['cp_path']), fn_short[16:])

                print(f"Going to calculate superoperator {fn}.")

                filenames.append(fn_main) if not (run_dict['cut_off_time'] == np.inf
                                                  and (cut_off_dataframe == 'auto'
                                                       or isinstance(cut_off_dataframe, float))) else None

                # Check if parameter settings has not yet been evaluated, else skip
                if fn is not None and os.path.exists(fn_main + ".csv"):
                    data = pd.read_csv(fn_main + '.csv', sep=";", float_precision='round_trip')
                    iterations_carried_out = (data.loc[0, 'written_to'])
                    if os.path.exists(fn_main + "_failed.csv"):
                        data_failed = pd.read_csv(fn_main + "_failed.csv", sep=";", float_precision='round_trip')
                        iterations_carried_out += int(data_failed.loc[0, 'written_to'])
                    res_iterations = int(circuit_args['iterations'] - iterations_carried_out)
                    # iterations within 5% margin
                    # if not circuit_args['probabilistic'] or circuit_args['iterations'] * 0.05 >= res_iterations:
                    # if not circuit_args['probabilistic'] or res_iterations <= 0:
                    if res_iterations <= 0 and not operational_args['force_run']:
                        print("\n[INFO] Skipping circuit for file '{}', since {} already has enough iterations.".format(fn, fn_main))
                        continue
                    else:
                        if not operational_args['force_run']:
                            print("\nFile found with too less iterations. Running for {} iterations\n".format(
                                res_iterations))
                            circuit_args['iterations'] = res_iterations
                        circuit_args['iter_offset'] = int(iterations_carried_out)

            print("\nRunning {} iteration(s) with values for the variational arguments:"
                  .format(circuit_args['iterations']))
            pprint({**run_dict})
            print(f"decoherence: {circuit_args['decoherence']}")
            print(f"probabilistic: {circuit_args['probabilistic']}")
            print(f"noiseless_swap: {circuit_args['noiseless_swap']}")
            if operational_args['threaded']:
                _, print_lines, characteristics = main_threaded(fn=(operational_args['cp_path'], fn_short),
                                                                **operational_args, **run_dict, **circuit_args)
            else:
                _, print_lines, characteristics = main_series(fn=(operational_args['cp_path'], fn_short),
                                                              **operational_args, **run_dict, **circuit_args)
            print(*print_lines)
            if not kwargs['draw_circuit'] and not operational_args['threaded']:
                print(f"Durations: {characteristics['dur']}.")
                print(f"Stabilizer fidelities: {characteristics['stab_fid']}.")
                print(f"GHZ fidelities: {characteristics['ghz_fid']}.")
                # print(f"Trace distance X: {characteristics['trace_dist_X']}.")
                # print(f"Trace distance Z: {characteristics['trace_dist_Z']}.\n")

    for i_prot in range(len(var_circuit_args['protocol'])):
        var_circuit_args['protocol'][i_prot] = var_circuit_args['protocol'][i_prot][0]

    if var_circuit_args['bell_pair_parameters'] != [None]:
        for i_bpp in range(len(var_circuit_args['bell_pair_parameters'])):
            var_circuit_args['bell_pair_parameters'][i_bpp] = var_circuit_args['bell_pair_parameters'][i_bpp][0]

    return filenames


if __name__ == "__main__":
    parser = compose_parser()
    args = vars(parser.parse_args())
    args = additional_parsing_of_arguments(**args)
    grouped_arguments = group_arguments(parser, **args)
    print_signature()
    print_circuit_parameters(*grouped_arguments)

    # Loop over all possible combinations of the user determined parameters
    run_for_arguments(*grouped_arguments, **args)
