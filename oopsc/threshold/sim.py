'''
2020 Mark Shui Hu, QuTech

www.github.com/watermarkhu/oop_surface_code
_____________________________________________

'''
from .. import oopsc
from oopsc.superoperator import superoperator as so
from utilities.files import get_full_path, detect_filenames
from pprint import pprint
import multiprocessing as mp
import pandas as pd
import sys, os
import math
import datetime
SUP_INDICES_1 = ['node', 'p_g', 'p_n', 'p_m', 'pm_1', 'p_link', 't_pulse', 'GHZ_success', 'protocol_name']
SUP_INDICES_2 = ['node', 'p_g', 'p_n', 'ent_prot', 'F_prep', 'p_DE', 'mu', 'labda', 'eta', 'alpha', 'p_m', 'p_m_1',
                 't_pulse', 'GHZ_success', 'protocol_name']
SUP_INDICES_3 = ['protocol_name', 'set_number', 'bell_pair_type', 'network_noise_type', 'F_link', 'p_link', 'ent_prot',
                 'F_prep', 'p_DE', 'mu', 'labda', 'eta', 'alpha', 't_link', 't_meas', 'T1n_idle', 'T1n_link',
                 'T1e_idle', 'T2n_idle', 'T2n_link', 'T2e_idle', 't_pulse', 'n_DD', 'te_X', 'te_Y', 'tn_X', 'tn_Y',
                 'te_Z', 'te_H', 'tn_Z', 'tn_H', 't_CZ', 't_CX', 't_CiY', 't_SWAP', 'p_g', 'p_m', 'p_m_1',
                 'noiseless_swap', 'basis_transformation_noise', 'combine', 'probabilistic', 'decoherence',
                 'cut_off_time']

def get_superoperator_indices(lattices, superoperators, cycles):
    index_dict = {"L": lattices, "cycles": cycles}
    indices = SUP_INDICES_3
    for att in indices:
        index_dict.update({att: list(set([getattr(s, att) for s in superoperators]))})

    return index_dict


def get_current_index(lattice, superoperator, cycles, current_data_index):
    index = (lattice, cycles)
    indices = SUP_INDICES_3
    for i_att, att in enumerate(indices):
        if isinstance(current_data_index[i_att+2], str):
        # if att in ['pn', 'F_prep', 'p_DE', 'mu', 'labda', 'eta', 'alpha']:
            value = str(getattr(superoperator, att)) if getattr(superoperator, att) is not None else "None"
        else:
            value = getattr(superoperator, att) if getattr(superoperator, att) is not None else "None"
            # print(current_data_index[i_att+2], value)
        index += (value,)

    return index

def read_data(file_path):
    try:
        data = pd.read_csv(file_path, header=0, float_precision='round_trip')
        indices = ["L", "p"] if "p" in data else ["L", "p_g"] #, "GHZ_success"]
        return data.set_index(indices)
    except FileNotFoundError:
        print("File not found")
        exit()


def get_data(data, latts, probs, P_store=1):

    if not latts: latts = []
    if not probs: probs = []
    fitL = data.index.get_level_values("L")
    fitp = data.index.get_level_values("p") if "p" in data.index else data.index.get_level_values("p_g")
    fitN = data.loc[:, "N"].values
    fitt = data.loc[:, "success"].values

    fitdata = [[] for i in range(4)]
    for L, P, N, t in zip(fitL, fitp, fitN, fitt):
        p = round(float(P)/P_store, 6)
        if all([N != 0, not latts or L in latts, not probs or p in probs]):
            fitdata[0].append(L)
            fitdata[1].append(p)
            fitdata[2].append(N)
            fitdata[3].append(t)

    return fitdata[0], fitdata[1], fitdata[2], fitdata[3]


def check_if_index_in_main_file(index, data_main):
    index = list(index)
    for i, object in enumerate(data_main.index[0]):
        if isinstance(object, str) and not isinstance(index[i], str):
            index[i] = str(index[i])
    index = tuple(index)
    if index in data_main.index:
        return index, True
    else:
        return index, False


def update_result_files(folder, fn, remove_main_dataframes=False):
    main_file, main_file_present, sub_files = detect_filenames(folder, fn)
    print(f"Identified surface code dataframe runs: {sub_files}.")
    if main_file_present:
        print(f"Main surface code dataframe is present.")
    filename_main = os.path.join(get_full_path(folder), main_file)

    columns = ['GHZ_success_rate', 'N', 'success', 'date_and_time', 'supop_date_and_time']

    if main_file_present is False and sub_files:
        # Create main file from first file in sub_files
        filename = os.path.join(get_full_path(folder), sub_files[-1] + "_" + main_file)
        if os.path.exists(filename + ".csv"):
            data = pd.read_csv(filename + ".csv", header=0, float_precision='round_trip')
            if data is not None:
                index_columns = [index_col for index_col in data if index_col not in columns]
                data = data.set_index(index_columns)
                data.sort_index(inplace=True)
                data = data[(data.T.applymap(lambda x: x != 0 and x is not None and not pd.isna(x))).any()]
                data.to_csv(filename_main + ".csv")
                print(f"Main surface code dataframe is created from sub_file {sub_files[-1]}.")
                del sub_files[-1]
        else:
            raise FileExistsError(f"File {filename} does not exist.")

    if os.path.exists(filename_main + ".csv"):
        data_main = pd.read_csv(filename_main + ".csv", header=0, float_precision='round_trip')
        if data_main is not None:
            index_columns = [index_col for index_col in data_main if index_col not in columns]
            data_main = data_main.set_index(index_columns)
            data_main.sort_index(inplace=True)

            if len(sub_files) > 0:
                sub_files.reverse()

            for sub_file in sub_files:
                filename = os.path.join(get_full_path(folder), sub_file + "_" + main_file)
                if os.path.exists(filename + ".csv"):
                    data_sub = pd.read_csv(filename + ".csv", header=0, float_precision='round_trip')
                    if data_sub is not None:
                        data_sub = data_sub.set_index(index_columns)
                        data_sub.sort_index(inplace=True)
                        for index in data_sub.index:
                            skip_index = False
                            replace_index = False
                            reason = 0

                            all_columns = list(set(data_main.columns) | set(data_sub.columns))
                            for column in all_columns:
                                if column not in data_main:
                                    data_main[column] = "" if "date_and_time" in column else 0
                                if column not in data_sub:
                                    data_sub[column] = "" if "date_and_time" in column else 0

                            index_in_style_m, check = check_if_index_in_main_file(index, data_main)
                            if not check:
                                data_main.loc[index_in_style_m, :] = 0
                                data_main.sort_index(inplace=True)
                            else:
                                sub_files_in_main = data_main.loc[index_in_style_m, "date_and_time"]
                                sub_files_in_main_list = sub_files_in_main.split(" ")
                                if sub_file in sub_files_in_main_list:
                                    skip_index = True
                                else:
                                    # The following only applies to surface code calculations that have a
                                    # "GHZ_success_rate" in their data frame (i.e., surface code files). In these files,
                                    # a later added column "supop_date_and_time" describes at what time stamps the
                                    # superoperator was created that was used for the surface code calculations. When
                                    # combining dataframes, we don't want to add surface code calculations based on an
                                    # "old" superoperator file. That is why we should exclude surface code calculations that
                                    # contain a "date_and_time" time stap that is older than the newest
                                    # "supop_date_and_time" time stamp in the main data frame.
                                    if "GHZ_success_rate" in all_columns:
                                        skip_index = True
                                        col = "supop_date_and_time"
                                        check_for_newest_index = False
                                        # Here, we have established this is a surface code file and we already have this
                                        # index in our main file, we are only going to add this index to the new file in
                                        # case:
                                        # 1) Both the new and the main index have (a) nonzero superoperator calculation
                                        #    date-time(s) and they completely overlap:
                                        if col in data_main \
                                                and data_main.loc[index_in_style_m, col] not in ["", 0] \
                                                and data_sub.loc[index, col] not in ["", 0]:
                                            reason = 1
                                            if data_main.loc[index_in_style_m, col] == data_sub.loc[index, col]:
                                                skip_index = False
                                            else:
                                                # If both new and main have supop date-time information, but they do not
                                                # match, we choose the index with the most recent surface code
                                                # calculation date-time:
                                                check_for_newest_index = True

                                        # 2) The main index has (a) nonzero superoperator calculation date-time(s) and the
                                        #    new index hasn't, but the earliest surface code calculation date-time(s) of the
                                        #    new index is still later in time than than the latest superoperator
                                        #    calculation date-time(s) of the main index.
                                        elif col in data_main \
                                                and data_main.loc[index_in_style_m, col] not in ["", 0] \
                                                and not (isinstance(data_main.loc[index_in_style_m, col], float)
                                                         and math.isnan(data_main.loc[index_in_style_m, col])) \
                                                and data_sub.loc[index, col] in ["", 0]:
                                            reason = 2
                                            if "date_and_time" in data_sub \
                                                    and data_sub.loc[index, "date_and_time"] not in ["", 0]:
                                                if data_main.loc[index_in_style_m, col] == "d":
                                                    print(data_main.loc[index_in_style_m, col])
                                                    raise ValueError

                                                try:
                                                    dates_in_main_supop = data_main.loc[index_in_style_m, col].split(" ")
                                                except:
                                                    print(index_in_style_m)
                                                    print(col)
                                                    print(data_main.loc[index_in_style_m, "supop_date_and_time"])
                                                    raise AttributeError
                                                dates_in_main_supop.sort(key=int)
                                                dates_in_sub = data_sub.loc[index, "date_and_time"].split(" ")
                                                dates_in_sub.sort(key=int)
                                                # Check if "dates_in_sub[0] > dates_in_main_supop[-1]"
                                                if len(dates_in_sub[0]) == 15 and len(dates_in_main_supop[-1]) == 15 \
                                                        and (int(dates_in_sub[0][:8]) > int(dates_in_main_supop[-1][:8])
                                                             or (int(dates_in_sub[0][:8]) == int(dates_in_main_supop[-1][:8])
                                                                 and int(dates_in_sub[0][9:16]) > int(dates_in_main_supop[-1][9:16]))):
                                                    skip_index = False
                                                else:
                                                    check_for_newest_index = True

                                        # 3) The other way around from scenario 2:
                                        elif col in data_main \
                                                and data_sub.loc[index, col] not in ["", 0] \
                                                and data_main.loc[index_in_style_m, col] in ["", 0]:
                                            reason = 3
                                            if "date_and_time" in data_main \
                                                    and data_main.loc[index_in_style_m, "date_and_time"] not in ["", 0]:
                                                dates_in_sub_supop = data_sub.loc[index, col].split(" ")
                                                dates_in_sub_supop.sort(key=int)
                                                dates_in_main = data_main.loc[index_in_style_m, "date_and_time"].split(" ")
                                                dates_in_main.sort(key=int)
                                                if len(dates_in_main[0]) == 15 and len(dates_in_sub_supop[-1]) == 15 \
                                                        and (int(dates_in_main[0][:8]) > int(dates_in_sub_supop[-1][:8])
                                                             or (int(dates_in_main[0][:8]) == int(dates_in_sub_supop[-1][:8])
                                                                 and int(dates_in_main[0][9:16]) > int(dates_in_sub_supop[-1][9:16]))):
                                                    skip_index = False
                                                else:
                                                    check_for_newest_index = True

                                        # In case neither the main index or the new index have information about when
                                        # their superoperator was created, we are going to keep the newest one:
                                        elif col in data_main \
                                                and data_sub.loc[index, col] in ["", 0] \
                                                and data_main.loc[index_in_style_m, col] in ["", 0]:
                                            reason = 4
                                            check_for_newest_index = True

                                        # Then there is of course the scenario where have to replace the main index
                                        # with the new index:
                                        if check_for_newest_index:
                                            if "date_and_time" in data_sub \
                                                    and "date_and_time" in data_main \
                                                    and data_sub.loc[index, "date_and_time"] not in [0, ""] \
                                                    and data_main.loc[index_in_style_m, "date_and_time"] not in [0, ""]:
                                                dates_in_main = data_main.loc[index_in_style_m, "date_and_time"].split(" ")
                                                dates_in_main.sort(key=int)
                                                dates_in_sub = data_sub.loc[index, "date_and_time"].split(" ")
                                                dates_in_sub.sort(key=int)
                                                if len(dates_in_sub[-1]) == 15 and len(dates_in_main[-1]) == 15 \
                                                        and (int(dates_in_sub[-1][:8]) > int(dates_in_main[-1][:8])
                                                             or (int(dates_in_sub[-1][:8]) == int(dates_in_main[-1][:8])
                                                                 and int(dates_in_sub[-1][9:16]) > int(dates_in_main[-1][9:16]))):
                                                    replace_index = True

                            # if skip_index is False:
                            #     print(f"statistics for {sub_file}:")
                            #     for i, value in enumerate(data_main.index[0]):
                            #         print(type(value), value)
                            #         print(type(index[i]), index[i])

                            if replace_index is True:
                                print(f"Replacement: Sub file {sub_file} is replacing the current value of index {index} in the main frame, because of reason {reason}.")
                                for column in data_main.columns:
                                    data_main.loc[index_in_style_m, column] = (data_sub.loc[index, column])

                            elif skip_index is False:
                                print(f"Add: Sub file {sub_file} is added to the main frame for index {index}.")
                                for column in data_main.columns:
                                    if index_in_style_m in data_main.index and \
                                            not pd.isna(data_main.loc[index_in_style_m, column]):
                                        if column == 'date_and_time' or column == "supop_date_and_time":
                                            if data_main.loc[index_in_style_m, column] in [0, ""]:
                                                data_main.loc[index_in_style_m, column] = (data_sub.loc[index, column])
                                            else:
                                                times = data_main.loc[index_in_style_m, column].split(" ")
                                                times += data_sub.loc[index, column].split(" ")
                                                times = list(dict.fromkeys([ind_time for ind_time in times if ind_time != ""]))
                                                times.sort(key=int)
                                                new_value = ""
                                                for time_value in times:
                                                    new_value += str(time_value) + " "
                                                data_main.loc[index_in_style_m, column] = (new_value[:-1])
                                        elif column == "GHZ_success_rate":
                                            if check is False:
                                                data_main.loc[index_in_style_m, column] = data_sub.loc[index, column]
                                            elif data_main.loc[index_in_style_m, column] != data_sub.loc[index, column]:
                                                raise IndexError(f"GHZ success rates should align, but don't: "
                                                                 f"{data_main.loc[index_in_style_m, column]} versus "
                                                                 f"{data_sub.loc[index, column]}.")
                                        else:
                                            data_main.loc[index_in_style_m, column] = (
                                                    data_main.loc[index_in_style_m, column]
                                                    + data_sub.loc[index, column])
                                    else:
                                        data_main.loc[index_in_style_m, column] = (data_sub.loc[index, column])

                            elif skip_index is True and reason != 0:
                                print(f"Neglect: Index {index} of sub file {sub_file} is neglected for the main frame because of reason {reason}.")

            data_main = data_main[
                (data_main.T.applymap(lambda x: x != 0 and x is not None and not pd.isna(x))).any()]
            data_main.to_csv(filename_main + ".csv")

        if remove_main_dataframes:
            print(f"Main surface code dataframes are removed after the calculation.")
            os.remove(filename_main + ".csv")


def sim_thresholds(
        decoder,
        lattice_type="toric",
        lattices = [],
        perror = [],
        superoperator_filenames=[],
        superoperator_filenames_additional=None,
        superoperator_filenames_additional_failed=None,
        superoperator_filenames_failed=None,
        GHZ_successes=[1.1],
        networked_architecture=False,
        space_weight=2,
        iters = 0,
        measurement_error=False,
        multithreading=False,
        threads=None,
        save_result=True,
        show_result=True,
        file_name="thres",
        folder=".",
        P_store=1000,
        debug=False,
        cycles=None,
        supop_date_and_time_values=None,
        network_architecture_type=None,
        **kwargs
        ):
    '''
    ############################################
    '''
    # print(f"\n\n\n\n\n{kwargs}\n{GHZ_successes}\n{superoperator_filenames}\n{superoperator_filenames_failed}\n\n\n\n")
    run_oopsc = oopsc.multiprocess if multithreading else oopsc.multiple

    network_architecture_type = "weight-4" if network_architecture_type is None else network_architecture_type

    if measurement_error:
        from ..graph import graph_3D as go
    else:
        from ..graph import graph_2D as go

    sys.setrecursionlimit(100000)

    get_name = lambda s: s[s.rfind(".")+1:]
    g_type = get_name(go.__name__)
    d_type = get_name(decoder.__name__)
    full_name = f"{lattice_type}_{g_type}_{d_type}_{file_name}"
    date_and_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + "_"

    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = folder + "/" + full_name + ".csv"

    progressbar = kwargs.pop("progressbar")

    data = None
    config = oopsc.default_config(**kwargs)

    superoperators = []
    if superoperator_filenames:
        perror = []
        for i, superoperator_filename in enumerate(superoperator_filenames):
            GHZ_success = GHZ_successes[i]
            supop_date_time = supop_date_and_time_values[i] if (supop_date_and_time_values is not None
                                                                or (isinstance(supop_date_and_time_values, list)
                                                                    and len(supop_date_and_time_values) == 0)) else ""
            # for GHZ_success in GHZ_successes:
            additional = [superoperator_filenames_additional[i]] if superoperator_filenames_additional is not None \
                else None
            additional.append(superoperator_filenames_additional_failed[i]) if \
                superoperator_filenames_additional_failed is not None else None
            failed = superoperator_filenames_failed[i] if superoperator_filenames_failed is not None else None
            superoperator = so.Superoperator(superoperator_filename, GHZ_success,
                                             additional_superoperators=additional, failed_ghz_superoperator=failed,
                                             supop_date_time=supop_date_time)
            superoperators.append(superoperator)
            perror.append(superoperator.p_g)

    data_s = {s.protocol_name: None for s in superoperators} if superoperators else None

    # Simulate and save results to file
    for lati in lattices:

        if multithreading:
            if threads is None:
                threads = mp.cpu_count()
            graph = [oopsc.lattice_type(lattice_type, config, decoder, go, lati, cycles=cycles) for _ in range(threads)]
            [g.decoder.set_space_weight(space_weight) for g in graph] if d_type == "mwpm" else None
        else:
            graph = oopsc.lattice_type(lattice_type, config, decoder, go, lati, cycles=cycles)
            graph.decoder.set_space_weight(space_weight) if d_type == "mwpm" else None

        for i, pi in enumerate(perror):

            superoperator = None
            if superoperators:
                superoperator = superoperators[i]
                superoperator.reset_stabilizer_rounds()
                networked_architecture = bool(superoperator.F_link) if not networked_architecture else True
                data = data_s[superoperator.protocol_name]

            if show_result:
                if superoperator:
                    superoperator_fn = "pd.DataFrame" if isinstance(superoperator.file_name, pd.DataFrame) else superoperator.file_name
                    print(f"Calculating {iters} iterations for L = {lati}, p = {pi}, cycles = {cycles}, "
                          f"GHZ_success = {superoperator.GHZ_success}, superoperator = {superoperator_fn}.")
                else:
                    print(f"Calculating {iters} iterations for L = {lati}, p = {pi}, cycles = {cycles}.")

            oopsc_args = dict(
                paulix=pi,
                superoperator=superoperator,
                networked_architecture=networked_architecture,
                lattice_type=lattice_type,
                debug=debug,
                processes=threads,
                progressbar=progressbar,
                network_architecture_type=network_architecture_type
            )
            if measurement_error and not superoperator:
                oopsc_args.update(measurex=pi)

            cycles_ind = ['None'] if cycles is None else [cycles]
            ind_dict = {"L": lattices, "p": perror, "cycles": cycles_ind} if \
                not superoperator else get_superoperator_indices(lattices, superoperators, cycles_ind)

            protocol_name = superoperator.protocol_name + "_" if superoperator and superoperator.protocol_name is not None else ""
            node_name = superoperator.node if superoperator else ""
            file_name_short = f"{protocol_name if protocol_name else ''}{node_name}_{full_name}"
            file_path = os.path.join(folder, date_and_time + file_name_short + ".csv")
            # file_path_short = os.path.join(folder, file_name_short + ".csv")

            output = run_oopsc(lati, config, iters, graph=graph, **oopsc_args)
            output['GHZ_success_rate'] = superoperator.GHZ_success
            output['date_and_time'] = date_and_time[:-1]
            output['supop_date_and_time'] = superoperator.supop_date_time

            if data is None:
                if os.path.exists(file_path):
                    data = pd.read_csv(file_path, header=0, float_precision='round_trip')
                    data = data.set_index(list(ind_dict.keys()))
                    data.sort_index(inplace=True)
                else:
                    columns = list(output.keys())
                    index = pd.MultiIndex.from_product([*ind_dict.values()], names=ind_dict.keys())
                    data = pd.DataFrame(0, index=index, columns=columns)

            cur_index = (lati, pi, cycles_ind[0]) if not superoperator else \
                get_current_index(lati, superoperator, cycles_ind[0], data.index[0])

            # print(cur_index[-1], type(cur_index[-1]))
            #
            # print(cur_index in data.index)
            #
            # print("\n\n\n\n")
            #
            # for i, value in enumerate(data.index[0]):
            #     print(value, cur_index[i], type(value), type(cur_index[i]))
            #
            # print("\n\n\n\n")

            for key, value in output.items():
                data.sort_index(inplace=True)
                if cur_index not in data.index:
                    data.loc[cur_index, :] = 0
                    data.sort_index(inplace=True)
                if cur_index in data.index and not pd.isna(data.loc[cur_index, key]):
                    if key == "date_and_time":
                        if data.loc[cur_index, key] == 0:
                            data.loc[cur_index, key] = (date_and_time[:-1])
                        else:
                            data.loc[cur_index, key] = (data.loc[cur_index, key] + " " + date_and_time[:-1])
                    elif key == "supop_date_and_time":
                        if data.loc[cur_index, key] == 0 or data.loc[cur_index, key] == value:
                            data.loc[cur_index, key] = (value)
                        else:
                            # This should not be possible (because we now create a new dataframe for each surface code
                            # calculation that contains a new date-time in front of the file name - i.e., this file
                            # should not exist yet).
                            raise IndexError(f"Cannot add surface code calculation based on superoperator with "
                                             f"date-time statistics {value} to dataframe with superoperator date-time "
                                             f"statistics {data.loc[cur_index, key]}.")
                    elif key == "GHZ_success_rate":
                        data.loc[cur_index, key] = (value)
                    else:
                        data.loc[cur_index, key] = (data.loc[cur_index, key] + value)
                else:
                    data.loc[cur_index, key] = (value)

                # data.sort_index(inplace=True)

            if save_result:
                data = data[(data.T.applymap(lambda x: x != 0 and x is not None and not pd.isna(x))).any()]
                data.to_csv(file_path)
                update_result_files(folder, date_and_time + file_name_short,
                                    remove_main_dataframes=kwargs['remove_main_dataframes'])
            data_s[superoperator.protocol_name] = data if superoperators else None

    if show_result:
        print(data.to_string())

    if save_result:
        print("file saved to {}".format(file_path))
        data.to_csv(file_path)

    return data
