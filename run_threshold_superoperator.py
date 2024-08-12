from circuit_simulation.stabilizer_measurement_protocols.argument_parsing import compose_parser, group_arguments
from run_threshold import add_arguments
from circuit_simulation.stabilizer_measurement_protocols.run_protocols import run_for_arguments, \
    additional_parsing_of_arguments, print_circuit_parameters
from circuit_simulation._superoperator.superoperator_methods import create_iid_superoperator
from utilities.files import detect_filenames_folder, filename_contains_datestamp
from oopsc.threshold.sim import sim_thresholds, update_result_files, check_if_index_in_main_file
from circuit_simulation.stabilizer_measurement_protocols.run_protocols import update_result_files as update_supop_files
from itertools import product
from collections import defaultdict
import os
import numpy as np
import pandas as pd
import time
from pprint import pprint
import warnings
warnings.filterwarnings('ignore', message='.*Specify dtype option on import or set low_memory=False.*')


def get_iid_superoperators(operational_args, circuit_args, var_circuit_args):
    filenames = []
    qubit_errors = var_circuit_args['pg']
    meas_error = bool(var_circuit_args['pm'][0]) if not circuit_args['pm_equals_pg'] else True
    for error in qubit_errors:
        m_error = error if meas_error else 0
        superoperator = create_iid_superoperator(error, m_error, num_qubits=4)
        filename = f'{operational_args["csv_filename"]}iid_{error}{"_meas_error" if meas_error else ""}.csv'
        superoperator.to_csv(filename, sep=";")
        filenames.append(filename)

    print(f"[INFO] Created iid superoperators for pg={qubit_errors}")
    return filenames


def create_index_slice(df, column, begin=None, end=None):
    idx = tuple()
    column = [df.index.names.index(value) for value in column]
    for i in range(len(df.index.names)):
        if i in column:
            index = column.index(i)
            cur_begin = begin[index] if begin is not None else None
            cur_end = end[index] if end is not None else None
            idx += (slice(cur_begin, cur_end, None),)
        else:
            idx += (slice(None, None, None),)
    return idx


def determine_superoperators(superoperator_filenames, args):
    primary_superoperators = []
    primary_superoperators_failed = []
    secondary_superoperators = []
    secondary_superoperators_failed = []

    for filename in superoperator_filenames:
        if 'secondary' in filename:
            secondary_superoperators.append(filename)
            secondary_superoperators_failed.append(filename + "_failed") if 'time' in filename else None
        else:
            primary_superoperators.append(filename)
            primary_superoperators_failed.append(filename + "_failed") if 'time' in filename else None

    args['superoperator_filenames'] = primary_superoperators
    args['superoperator_filenames_failed'] = primary_superoperators_failed if primary_superoperators_failed else None
    args['superoperator_filenames_additional'] = secondary_superoperators if secondary_superoperators else None
    args['superoperator_filenames_additional_failed'] = (secondary_superoperators_failed
                                                         if secondary_superoperators_failed else None)

    args['GHZ_successes'] = []
    args['supop_date_and_time_values'] = []
    for so in range(len(primary_superoperators)):
        data_success = pd.read_csv(primary_superoperators[so] + '.csv', sep=";", float_precision="round_trip",
                                   index_col=["error_config", "lie"])
        if primary_superoperators_failed and os.path.exists(primary_superoperators_failed[so] + '.csv'):
            successful_iterations = data_success.loc[('IIII', False), 'written_to']
            data_failed = pd.read_csv(primary_superoperators_failed[so] + '.csv', sep=";", float_precision="round_trip",
                                      index_col=["error_config", "lie"])
            failed_iterations = data_failed.loc[('IIII', False), 'written_to']
            args['GHZ_successes'].append(float(successful_iterations) /
                                         (float(successful_iterations) + float(failed_iterations)))
        else:
            args['GHZ_successes'].append(1.0)
        try:
            args['supop_date_and_time_values'].append(data_success.loc[('IIII', False), 'date_and_time'])
        except KeyError:
            args['supop_date_and_time_values'].append("00000000_000000")

    args['folder'] = os.path.join(os.path.dirname(primary_superoperators[0]), "threshold_sim")
    args['save_result'] = True

    return args


def determine_lattice_evaluation_by_result(surface_args, opp_args, circuit_args, var_circuit_args):
    folder = surface_args['folder']

    # print('\n\n\n\n\n')
    # print(surface_args)
    # print(opp_args)
    # print(circuit_args)
    # print(var_circuit_args)
    # print('\n\n\n\n\n')

    var_circuit_args['GHZ_success'] = [0.99 if cut != np.inf else 1.1 for cut in var_circuit_args['cut_off_time']]
    var_circuit_args['node'] = "FixThis" # ['Purified'] if circuit_args['T1_lde'] == 2 else ["Natural Abundance"]
    var_circuit_args['p_link'] = "FixThis" #var_circuit_args['lde_success'] if circuit_args['probabilistic'] else [1]
    var_circuit_args['protocol_name'] = set([p.strip("_secondary") + "_swap" if circuit_args['use_swap_gates'] else
                                            p.strip("_secondary") for p in var_circuit_args['protocol']])
    res_iters = defaultdict(int)
    parameters = {}

    if os.path.exists(folder):

        files = detect_filenames_folder(folder)
        for main, date_stamps in files.items():
            update_result_files(folder, main)

        for file in os.listdir(folder):
            if not filename_contains_datestamp(file[:-4]):
                data = pd.read_csv(os.path.join(folder, file), float_precision='round_trip')
                data['cut_off_time'] = data['cut_off_time'].round(decimals=12)
                data['p_link'] = data['p_link'].round(decimals=8)
                data.replace(np.nan, None, inplace=True)
                data.replace("None", None, inplace=True)
                # parameters = {}
                # for col in data:
                #     if col in var_circuit_args:
                #         parameters[col] = var_circuit_args[col]
        #             if col not in ['L', 'N', 'success', 'ent_prot', 'F_prep', 'p_DE', 'mu', 'eta', 'labda', 'alpha']:
        #                 parameters[col] = var_circuit_args[col]
                    # elif var_circuit_args['bell_pair_parameters'] == [None]:
                    #     parameters[col] = [None]
                    # elif col in ['ent_prot', 'F_prep', 'p_DE', 'mu', 'eta', 'labda', 'alpha']:
                    #     col_read = 'lambda' if col == 'labda' else col
                    #     collect_columns_bpp = []
                    #     for i_bpp in range(len(var_circuit_args['bell_pair_parameters'])):
                    #         if col_read in var_circuit_args['bell_pair_parameters'][i_bpp].keys():
                    #             collect_columns_bpp.append(var_circuit_args['bell_pair_parameters'][i_bpp][col_read])
                    #         else:
                    #             collect_columns_bpp.append(None)
                    #     parameters[col] = collect_columns_bpp

                # parameters = {col: var_circuit_args[col] for col in data if col not in ['L', 'N', 'success']}
                # data.set_index(['L'] + list(parameters.keys()), inplace=True)
                # data.sort_index()
                # for index in product(*[surface_args['lattices'], *parameters.values()]):
                #     res_iters[index[0]] += (1 if index in data.index and
                #                             (data.loc[index, "N"] * 1.05) >= surface_args['iters'] else 0)

                superoperator_parameters = []
                for i_fn, file_name in enumerate(surface_args['superoperator_filenames']):
                    superoperator_parameters.append({})
                    data2 = pd.read_csv(file_name + ".csv", float_precision='round_trip', sep=";")
                    data2['cut_off_time'] = data2['cut_off_time'].round(decimals=8)
                    data2['p_link'] = data2['p_link'].round(decimals=8)
                    data2.replace(np.nan, None, inplace=True)
                    data2.replace("None", None, inplace=True)

                    for col in data:
                        if col == 'labda':
                            superoperator_parameters[i_fn]['labda'] = data2.loc[0, 'lambda']
                        elif col not in ['L', 'cycles', 'N', 'success', 'date_and_time', 'supop_date_and_time',
                                         'GHZ_success_rate']:
                            superoperator_parameters[i_fn][col] = data2.loc[0, col]

                superoperator_values = [list(items.values()) for items in superoperator_parameters]

                dataframe_columns = ['L', 'cycles'] + list(superoperator_parameters[0].keys())
                data.set_index(dataframe_columns, inplace=True)
                data.sort_index(inplace=True)
                for index1, index2, index3 in product(surface_args['lattices'], [surface_args['cycles']], superoperator_values):
                    data2 = pd.DataFrame([[index1] + [index2] + index3], columns=dataframe_columns)
                    data2.set_index(dataframe_columns, inplace=True)
                    data2.sort_index(inplace=True)
                    # for i, value in enumerate(data2.index[0]):
                    #     print(value, type(value))
                    #     print(data.index[0][i], type(data.index[0][i]))
                    index = data2.index[0]
                    index, check = check_if_index_in_main_file(index, data)
                    res_iters[index[0]] += (1 if check and (data.loc[index, "N"] * 1.05) >= surface_args['iters'] else 0)

    # for L, count in res_iters.items():
    #     if count == len(list(product(*parameters.values()))):
    #         surface_args['lattices'].remove(L)
    #         print("\n[INFO] Skipping simulations for L={} since it has already run for all parameters".format(L))

    # If there are no lattices left to evaluate, the program can exit
    if not surface_args['lattices']:
        pprint(data)
        print("\nAll surface code simulations have already been performed. Exiting Surface Code simulations")
        exit(1)

    return surface_args


if __name__ == "__main__":
    parser = compose_parser()
    add_arguments(parser)
    args = vars(parser.parse_args())

    if args["protocol_recipe_select"] is not None:
        args["protocol_recipe_select"] *= args["protocol_recipe_select_per_node"]
    for prot_settings_number in range(args["protocol_recipe_select_per_node"]):
        # Get input arguments:
        circuit_sim_args = {action.dest: args[action.dest] for action in compose_parser()._actions if action.dest != 'help'}
        surface_code_args = {action.dest: args[action.dest] for action in add_arguments()._actions if action.dest != 'help'}

        # Run circuit simulation if superoperator file does not yet exist
        print('\n #############################################')
        print(' ############ CIRCUIT SIMULATIONS ############')
        print(' #############################################\n')
        circuit_sim_args = additional_parsing_of_arguments(**circuit_sim_args,
                                                           network_architecture_type=surface_code_args['network_architecture_type'])
        grouped_arguments = group_arguments(compose_parser(), **circuit_sim_args)
        print_circuit_parameters(*grouped_arguments)
        if args['iid']:
            superoperator_filenames = get_iid_superoperators(*grouped_arguments)
        else:
            superoperator_filenames = run_for_arguments(*grouped_arguments, **circuit_sim_args)
        print('\n -----------------------------------------------------------')

        # # PUT BACK LAST LAYER (ALSO [:-1] PART IN GRAPH_3D
        # # PUT BACK DECODING PART IN OOPSC.PY
        # # PUT BACK CIRCUIT CALC PART ABOVE
        # # PUT BACK ARGUMENT FILE NORMAL PARAMETERS
        # from os import listdir
        # from os.path import isfile, join
        # file_location = 'C:\\Users\\sebastiandebon\\surfdrive\\Code\\Sebastian\\oopsc\\superoperator\\Phenom_thresholds_Siddhant_meas_errs\\'
        # files = [f for f in listdir(file_location) if isfile(join(file_location, f))]
        # sup_files = [file_location + f[:-4] for f in files]
        # # error_rates = [float(file[file.find("px-") + len("px-"):file.rfind("_pxm")]) for file in sup_files]
        # # errors = sorted(error_rates, key=float)
        # superoperator_filenames = sup_files

        # # for prob in [0.0068, 0.0069, 0.0071, 0.0072, 0.0073, 0.0074]:
        # # superoperator_filenames.append(f"phenomenological_0.02_0.02_0.02_0.02_toric_only_two_qubits_new_3.csv")

        # Run surface code simulations
        if surface_code_args['skip_surface_code'] is False:
            print('\n ##################################################')
            print(' ############ SURFACE CODE SIMULATIONS ############')
            print(' ##################################################\n')
            surface_code_args = determine_superoperators(superoperator_filenames, surface_code_args)
            surface_code_args = determine_lattice_evaluation_by_result(surface_code_args, *grouped_arguments)

            decoder = surface_code_args.pop("decoder")
            # surface_code_args['cycles'] = 3

            decoders = __import__("oopsc.decoder", fromlist=[decoder])
            decode = getattr(decoders, decoder)

            decoder_names = {
                "mwpm": "minimum weight perfect matching (blossom5)",
                "uf": "union-find",
                "uf_uwg": "union-find non weighted growth",
                "ufbb": "union-find balanced bloom"
            }
            decoder_name = decoder_names[decoder] if decoder in decoder_names else decoder
            print(f"{'_' * 75}\n\ndecoder type: " + decoder_name)
            print(surface_code_args)
            surface_code_args['remove_main_dataframes'] = circuit_sim_args['remove_main_dataframes']

            sim_thresholds(decode, **surface_code_args)

        if (circuit_sim_args['csv_filename'] or circuit_sim_args['cp_path']) and circuit_sim_args['remove_main_dataframes']:
            for supop_filename in superoperator_filenames:
                print("")
                fol_fn = supop_filename.rsplit("\\", 1)
                if len(fol_fn) == 1:
                    fol_fn = fol_fn[0].rsplit("/", 1)
                update_supop_files(fol_fn[0], fol_fn[1], remove_main=True)
                if circuit_sim_args['cut_off'] != [None]:
                    update_supop_files(fol_fn[0], fol_fn[1].split("cut_off_time")[0][:-1], remove_main=True)

        if args["protocol_recipe_select"] is not None:
            args["protocol_recipe_select"] += 1
