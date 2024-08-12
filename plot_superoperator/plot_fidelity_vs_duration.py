import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import markers as mkrs
from itertools import product

TRANS = {'pg': '$p_g$', 'pn': '$p_n$', 'pm': '$p_m$', 'pm_1': '$p_{m1}$', 'p_bell_success': "$p_{LDE}$",
         "T1_lde": "$T_1^{LDE},T_2^{LDE}$"}


def create_file_name(kind: str, fn: str, fixed_vls: dict, spread: bool):
    fn += "_" + kind
    translation = {'decoherence': "dec", "p_bell_success": "lde_success"}
    for k, v in fixed_vls.items():
        if k in filename_skip_parameters or v == False:
            continue
        if k in translation:
            k = translation[k]

        fn += "_{}{}".format(k, v if not type(v) == bool else "")

    if cutoff_results:
        fn += "_cutoff"
    if spread:
        fn += "_spread"

    return fn


def get_marker_index(marker_cols, run_dict):
    marker_ind = tuple()
    for value in marker_cols:
        marker_ind += (run_dict[value] if value != "cut_off_time" or run_dict[value] == np.inf else 1,)

    return marker_ind


def get_label_name(run_dict):
    def translate_value(k, v):
        value_translation = {"decoherence": "dec", "fixed_lde_attempts": "decoupling", "probabilistic": "",
                             "cut_off_time": "cutoff",
                             "protocol_name": v.replace("_", "-")
                                               .replace("weight_3", "weight_3:Dyn-prot-3-8-1").capitalize()
                                               .replace("-secondary", ":Bipartite-4")
                                               .replace("Expedient", "EXPEDIENT").replace("Stringent", "STRINGENT")
                             if type(v) == str else None, "node": v.replace("Abundance", "abundance")
                             if type(v) == str else None}

        return value_translation.get(k, value)

    name = ""
    for key, value in run_dict.items():
        if value not in [False, 0, np.inf]:
            value = translate_value(key, value)
            name += "{}{}, ".format(TRANS[key] + "=" if key in TRANS else "", str(value).replace("-swap", ""))

    name = name.strip(", ")

    return name


def keep_rows_to_evaluate(df, evaluate_values, cutoff_results=None):
    if not cutoff_results:
        df = df[df['cut_off_time'] == np.inf]

    for key, values in evaluate_values.items():
        values = values + [str(i) for i in values]  # Usage: if dataframe parses the values as strings

        if values and 'REMOVE' not in values:
            df = df[df[key].isin(values)]
        elif 'REMOVE' not in values:
            evaluate_values[key] = set(df[key])

    if df.empty:
        print("\n[ERROR] No data to show for this set of parameters!", file=sys.stderr)
        exit(1)

    return df


def identify_indices(df: pd.DataFrame):
    no_index_idicators = ['99', 'ghz', 'avg', 'sem', 'spread', 'IIII', 'written', 'pulse']
    index_columns = {}
    fixed_values = {}
    print("\nThe following values are fixed:")
    for column in df:
        if all([indicator not in column for indicator in no_index_idicators]):
            unique_values = sorted(set(df[column]))
            if len(unique_values) > 1 or column in ['protocol_name', 'node']:
                index_columns[column] = unique_values
            elif len(unique_values) == 1:
                fixed_values[column] = unique_values[0]
                print("\t[+] {}={}".format(column, unique_values[0]))

    return index_columns, fixed_values


def plot_style(title=None, xlabel=None, ylabel=None, ymax=1., ymin=0., **kwargs):
    fig, ax = plt.subplots(figsize=(14, 16))
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.subplots_adjust(left=0.08, bottom=0.08, right=.95, top=.95)
    ax.grid(color='w', linestyle='-', linewidth=2)
    ax.set_title(title, fontsize=34)
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    for key, arg in kwargs.items():
        func = getattr(ax, f"set_{key}")
        func(arg)
    ax.patch.set_facecolor('0.95')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_ticks(np.arange(ymin, ymax+0.1, accuracy))
    ax.set_ylim(ymin=ymin, ymax=ymax)

    return fig, ax


def scatter_plot(y_value, title, xlabel, ylabel, df: pd.DataFrame, marker_cols, index_dict, spread=False,
                 no_dec_small=True, ymin=0., ymax=1., legend=True):
    colors = {}
    [colors.update({name: color}) for name, color in zip(index_dict['protocol_name'], mcolors.TABLEAU_COLORS)]
    points = list(mkrs.MarkerStyle.filled_markers)
    fig, ax = plot_style(title, xlabel, ylabel, ymin=ymin, ymax=ymax)
    i = 0
    protocol_markers = {}
    for index_tuple in product(*index_dict.values()):
        iteration_dict = dict(zip(index_dict.keys(), index_tuple))
        index = tuple(iteration_dict.values())

        if index in df.index:
            protocol = iteration_dict['protocol_name']
            node = iteration_dict['node']
            dec = iteration_dict['decoherence'] if 'decoherence' in iteration_dict else True
            cutoff = iteration_dict.get('cut_off_time', np.inf) < np.inf
            marker_index = get_marker_index(marker_cols, iteration_dict)
            if marker_index not in protocol_markers:
                protocol_markers[marker_index] = i
                i += 1
            color = colors[protocol]
            dataframe_new = df.loc[index, :]
            style = 'none' if node == 'Purified' else 'full'
            error = {'ghz_fidelity': 'ghz', "IIII": "stab"}
            y_err = [[dataframe_new[error[y_value] + '_lspread']], [dataframe_new[error[y_value] + '_rspread']]]
            x_err = [[dataframe_new['dur_lspread']], [dataframe_new['dur_rspread']]] if not cutoff else None
            ax.errorbar(dataframe_new['avg_duration'],
                        dataframe_new[y_value],
                        yerr=None if not spread or not dec else y_err,
                        xerr=None if not spread or not dec else x_err,
                        marker=points[protocol_markers[marker_index]],
                        color=color,
                        ms=20 if dec or not no_dec_small else 12,
                        capsize=12,
                        label=get_label_name(iteration_dict),
                        fillstyle=style,
                        linestyle='')

    if legend:
        ax.legend(prop={'size': 20})

    return fig, ax


def main(filename, evaluate_values, spread, no_dec_small, cutoff_results, save=False, file_path=None, ghz_title=None,
         stabilizer_title=None, ymin=0., ymax=1.):
    # Set standard title if none given
    ghz_title = ghz_title if ghz_title is not None else "GHZ fidelity"
    stabilizer_title = stabilizer_title if stabilizer_title is not None else "Stabilizer fidelity"

    # Read data and remove rows that do not intersect with the chosen parameters
    dataframe = pd.read_csv(filename, sep=';', float_precision='round_trip')
    dataframe = keep_rows_to_evaluate(dataframe, evaluate_values, cutoff_results)

    # Find fixed parameters and base markers and index on this data
    index_dict, fixed_values = identify_indices(dataframe)
    marker_index_cols = set(index_dict).difference(['node', 'protocol_name'])
    dataframe = dataframe.set_index(list(index_dict.keys()))

    # Plot the GHZ and stabilizer fidelity
    fig, ax = scatter_plot("ghz_fidelity", ghz_title, "Duration (s)",
                           "$\overline{F}$", dataframe, marker_index_cols, index_dict, spread=spread,
                           no_dec_small=no_dec_small, ymin=ymin, ymax=ymax)
    ax.hlines(0.79, 0, 1, linestyles='dashed', color='orange', label="Plain, no decoherence upper bound")
    ax.hlines(0.81, 0, 1,linestyles='dashed', color='blue', label='Dyn-prot-4-4-1, no decoherence upper bound')
    ax.legend(prop={'size': 20}, loc='lower right')
    fig2, ax2 = scatter_plot("IIII", stabilizer_title, "Duration (s)", "$\overline{F}$",
                             dataframe, marker_index_cols, index_dict, spread=spread, no_dec_small=no_dec_small,
                             ymin=ymin, ymax=ymax, legend=False)
    ax2.hlines(0.72, 0, 1, linestyles='dashed', color='orange', label="Plain, no decoherence upper bound")
    ax2.hlines(0.71, 0, 1, linestyles='dashed', color='blue', label='Dyn-prot-4-4-1, no decoherence upper bound')

    plt.show()

    if save:
        file_path_stab = create_file_name('stab', file_path, fixed_values, spread)
        file_path_ghz = create_file_name('ghz', file_path, fixed_values, spread)
        fig.savefig(file_path_ghz + ".pdf", transparent=False, format="pdf", bbox_inches="tight")
        fig2.savefig(file_path_stab + ".pdf", transparent=False, format="pdf", bbox_inches="tight")


if __name__ == '__main__':
    # General booleans
    spread = False          # Shows the 68.2% spread error bars
    save = True             # Saves the figures to the given filepath
    no_dec_small = True     # Plots the data points without decoherence smaller
    cutoff_results = False  # Show the results for the 99% cutoff time
    ymax = 1
    ymin = 0
    accuracy = 0.1

    # Input and output file parameters
    file_name = '../notebooks/circuit_data_NV.csv'
    filename_skip_parameters = ['basis_transformation_noise', 'network_noise_type', 'probabilistic',
                                'no_single_qubit_error']
    file_path = '../results/thesis_files/draft_figures/fidelity_vs_duration_current_pres'

    # Filter on the data of the input file
    evaluate_values = {'decoherence':           [],
                       'fixed_lde_attempts':    [2000],
                       'node':                  [],
                       'p_bell_success':        [0.0001],
                       'pg':                    [0.01],
                       'pm':                    [0.01],
                       'pm_1':                  [None],
                       'pn':                    [0.1],
                       'protocol_name':         ['dyn_prot_4_4_1_swap', "plain_swap"],
                       'pulse_duration':        []
                       }

    main(file_name, evaluate_values, spread, no_dec_small, cutoff_results, save, file_path, ymin=ymin, ymax=ymax)

