import argparse
import json
import numpy as np


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as v:
            parser.parse_args([argument for argument in v.read().split() if "#" not in argument], namespace)


def group_arguments(parser, **kwargs):
    opp_args = {ac.dest: kwargs[ac.dest] for ac in parser._actions
                if ac.container.description == 'Operation arguments' and ac.dest != 'argument_file'}
    circuit_args = {ac.dest: kwargs[ac.dest] for ac in parser._actions
                    if ac.container.description == 'Circuit arguments'}
    var_circuit_args = {ac.dest: kwargs[ac.dest] for ac in parser._actions
                        if ac.container.description == 'Variational circuit arguments'}
    if 'network_architecture_type' in kwargs.keys():
        circuit_args['network_architecture_type'] = kwargs['network_architecture_type']

    return opp_args, circuit_args, var_circuit_args


def compose_parser():
    parser = argparse.ArgumentParser(prog='Stabilizer measurement protocol simulations')
    opp_arg = parser.add_argument_group(description="Operation arguments")
    circuit_arg = parser.add_argument_group(description="Circuit arguments")
    var_circuit_arg = parser.add_argument_group(description="Variational circuit arguments")

    # Operational Arguments
    opp_arg.add_argument('-c',
                         '--color',
                         help='Specifies if the console output should display color. Optional',
                         required=False,
                         action='store_true')
    opp_arg.add_argument('-ltsv',
                         '--save_latex_pdf',
                         help='If given, a pdf containing a drawing of the noisy circuit in latex will be saved to the '
                              '`circuit_pdfs` folder. Optional',
                         required=False,
                         action='store_true')
    opp_arg.add_argument('-fn',
                         '--csv_filename',
                         required=False,
                         type=str,
                         default=None,
                         help='Give the file name of the csv file that will be saved.')
    opp_arg.add_argument('-cp',
                         '--cp_path',
                         required=False,
                         type=str,
                         default=None,
                         help='Give the path the csv file should be copied to (Cluster runs).')
    opp_arg.add_argument("-tr",
                         "--threaded",
                         help="Use when the program should run in multi-threaded mode. Optional",
                         required=False,
                         action="store_true")
    opp_arg.add_argument("--to_console",
                         help="Print the superoperator results to the console.",
                         required=False,
                         action="store_true")
    opp_arg.add_argument("-draw",
                         "--draw_circuit",
                         help="Print a drawing of the circuit to the console",
                         required=False,
                         action="store_true")
    opp_arg.add_argument("--calc_avg_supop_state",
                         help="Calculate the average superoperator Choi state",
                         required=False,
                         action="store_true")
    opp_arg.add_argument("-lkt_1q",
                         "--single_qubit_gate_lookup",
                         help="Name of a .pkl single-qubit gate lookup file.",
                         required=False,
                         type=str,
                         default=None)
    opp_arg.add_argument("-lkt_2q",
                         "--two_qubit_gate_lookup",
                         help="Name of a .pkl two-qubit gate lookup file.",
                         required=False,
                         type=str,
                         default=None)
    opp_arg.add_argument("--argument_file",
                         help="loads values from a file instead of the command line",
                         type=open,
                         action=LoadFromFile)
    opp_arg.add_argument("--gate_duration_file",
                         help="Specify the path to the file that contains the gate duration times.",
                         type=str,
                         required=False)
    opp_arg.add_argument("--progress_bar",
                         help="Displays no progress bar for simulation.",
                         action='store_true')
    # opp_arg.add_argument('-cut_off',
    #                      '--cut_off',
    #                      help='Specifies the file to load the cut-off time from, or sets cut-off percentage itself, '
    #                           'for performing a stabilizer measurement.',
    #                      type=str,
    #                      default=None)
    opp_arg.add_argument("-fr",
                         "--force_run",
                         help="Force simulation to run if file already exists",
                         required=False,
                         action="store_true")
    opp_arg.add_argument("-remove_main",
                         "--remove_main_dataframes",
                         help="Remove the main dataframes after the calculations (and only keep the ones that start "
                              "with a datetime-stamp. This is useful for supercomputer calculations).",
                         required=False,
                         action="store_true")
    opp_arg.add_argument("-no_merge",
                         "--do_not_merge_superoperators",
                         help="Adding this tag uses the old functionality that produces two superoperators: one for "
                              "successful runs within the cut-off time, and one for failed runs outside the cut-off "
                              "time. Without this tag these two superoperators are merged into a single superoperator "
                              "that contains a 'ghz_success' column. The tag also changes how idling qubits in the "
                              "nodes are treated: with this tag Pauli errors on both measured data columns and "
                              "idling data qubits are combined into a single probability; without this tag the noise "
                              "on the idling data qubits is processed in a separate column of the final superoperator.",
                         required=False,
                         action="store_true")
    opp_arg.add_argument("-par_select",
                         "--parameter_select",
                         help="Select one variational parameter out of a list",
                         type=int,
                         required=False,
                         default=None)
    opp_arg.add_argument("-prot_recipe_select",
                         "--protocol_recipe_select",
                         help="Select variational protocol recipes out of the list with protocol recipes called",
                         type=int,
                         required=False,
                         default=None)
    opp_arg.add_argument("-prot_recipe_select_per_node",
                         "--protocol_recipe_select_per_node",
                         help="Select how many variational protocol recipes should be selected per calculation",
                         type=int,
                         required=False,
                         default=1)
    opp_arg.add_argument("--protocol_recipes_file",
                         help="Specify the path to the file that contains the protocol recipes that we want to run.",
                         type=str,
                         required=False,
                         default=None)

    # Variational Circuit Arguments
    var_circuit_arg.add_argument('-p',
                                 '--protocol',
                                 help='Specifies which protocol should be used. - options: {'
                                      'monolithic/expedient/stringent}',
                                 nargs="*",
                                 choices=['monolithic', 'expedient', 'stringent', 'weight_2_4', 'weight_3',
                                          'dyn_prot_4_14_1', 'dyn_prot_4_22_1', 'bipartite_4',  'bipartite_6', 'plain',
                                          'dyn_prot_4_6_sym_1', 'dejmps_2_4_1', 'dejmps_2_6_1', 'dejmps_2_8_1',
                                          'modicum', 'dyn_prot_4_4_1_auto', 'dyn_prot_3_4_1', 'dyn_prot_3_8_1',
                                          'weight_2_4_secondary', 'auto_generated', 'bipartite_4_to_1',
                                          'bipartite_7_to_1', 'basic1', 'basic2', 'medium1', 'medium2', 'refined1',
                                          'refined2', 'minimum4x_40_1', 'minimum4x_40_2', 'minimum4x_22', 'direct_ghz',
                                          'weight_3_direct'],
                                 type=str.lower,
                                 default=['monolithic'])
    var_circuit_arg.add_argument('-prot_rec',
                                 '--protocol_recipe',
                                 help='Should contain the location of ProtocolRecipe object that describes a protocol '
                                      'that can be run as auto_generated.',
                                 nargs="*",
                                 required=False,
                                 default=None)
    var_circuit_arg.add_argument('-p_g',
                                 '--p_g',
                                 help='Specifies the amount of gate error present in the system',
                                 type=float,
                                 nargs="*",
                                 default=[None])
    var_circuit_arg.add_argument('-p_m',
                                 '--p_m',
                                 help='Specifies the amount of measurement error present in the system',
                                 type=float,
                                 nargs="*",
                                 default=[None])
    var_circuit_arg.add_argument('-p_m_1',
                                 '--p_m_1',
                                 help='The measurement error rate in case an 1-state is supposed to be measured',
                                 required=False,
                                 type=float,
                                 nargs="*",
                                 default=[None])
    var_circuit_arg.add_argument('-F_link',
                                 '--F_link',
                                 help='Specifies the amount of network error present in the system',
                                 type=float,
                                 nargs="*",
                                 default=[None])
    var_circuit_arg.add_argument('-bell_params',
                                 '--bell_pair_parameters',
                                 help='Specifies exactly parameter values for Bell states of type 3 and with network '
                                      'noise type 3.',
                                 type=json.loads,
                                 nargs="*",
                                 default=[None])
    var_circuit_arg.add_argument('-p_link',
                                 '--p_link',
                                 help='Specifies the success probability of the creation of a Bell pair (if '
                                      'probabilistic).',
                                 type=float,
                                 nargs='*',
                                 default=[None])
    var_circuit_arg.add_argument('-t_meas',
                                 '--t_meas',
                                 help='Specifies the duration of a measurement operation.',
                                 type=float,
                                 nargs='*',
                                 default=[None])
    var_circuit_arg.add_argument('-t_link',
                                 '--t_link',
                                 help='Specifies the duration of a single Bell pair entanglement attempt.',
                                 type=float,
                                 nargs='*',
                                 default=[None])
    var_circuit_arg.add_argument('-T1ni',
                                 '--T1n_idle',
                                 help='T1 relaxation time for a nuclear qubit.',
                                 type=str,
                                 nargs='*',
                                 default=[None])
    var_circuit_arg.add_argument('-T2ni',
                                 '--T2n_idle',
                                 help='T2 relaxation time for a nuclear qubit.',
                                 type=float,
                                 nargs='*',
                                 default=[None])
    var_circuit_arg.add_argument('-T1nl',
                                 '--T1n_link',
                                 help='T1 relaxation time for a nuclear qubit while link is performed.',
                                 type=float,
                                 nargs='*',
                                 default=[None])
    var_circuit_arg.add_argument('-T2nl',
                                '--T2n_link',
                                 help='T2 relaxation time for a nuclear qubit while link is performed.',
                                 type=float,
                                 nargs='*',
                                 default=[None])
    var_circuit_arg.add_argument('-T1ei',
                                 '--T1e_idle',
                                 help='T1 relaxation time for an electron qubit.',
                                 type=str,
                                 nargs='*',
                                 default=[None])
    var_circuit_arg.add_argument('-T2ei',
                                 '--T2e_idle',
                                 help='T2 relaxation time for an electron qubit.',
                                 type=float,
                                 nargs='*',
                                 default=[None])
    var_circuit_arg.add_argument('-gate_durations',
                                 '--gate_durations',
                                 help='Specifies (in dictionary form) gate duration times for common gates.',
                                 type=json.loads,
                                 nargs="*",
                                 default=[None])
    var_circuit_arg.add_argument("-set_number",
                                 "--set_number",
                                 help="Set all circuit parameters based on a default parameter list found in "
                                      "circuit_simulation/node/sets.csv (parameters will be overwritten by specific circuit "
                                      "parameters that are defined during parsing).",
                                 required=False,
                                 nargs='*',
                                 default=[None])
    var_circuit_arg.add_argument("-n_DD",
                                 "--n_DD",
                                 help="Specify the amount of fixed link attempts before a pulse is sent to the nuclear "
                                      "qubits.",
                                 type=float,
                                 nargs="*",
                                 default=[None])
    var_circuit_arg.add_argument('-t_pulse',
                                 '--t_pulse',
                                 help='Specifies the duration of a pulse used in the pulse sequence. If no pulse '
                                      'sequence is present, this should NOT be specified.',
                                 type=float,
                                 nargs='*',
                                 default=[None])
    var_circuit_arg.add_argument('-cut',
                                 '--cut_off_time',
                                 help='Specifies the cut-off time for performing a stabilizer measurement.',
                                 type=float,
                                 nargs="*",
                                 default=[np.inf])
    var_circuit_arg.add_argument('-cut_off',
                                 '--cut_off',
                                 help='Specifies the file to load the cut-off time from, or sets cut-off percentage '
                                      'itself, for performing a stabilizer measurement.',
                                 type=str,
                                 nargs="*",
                                 default=[None])
    var_circuit_arg.add_argument('-seed',
                                 '--seed_number',
                                 help='Contains a list of seeds used for the calculations. Is deactivated if the '
                                      'number of iterations is set to bigger than 1.',
                                 nargs="*",
                                 required=False,
                                 default=[None])

    # Constant Circuit arguments
    circuit_arg.add_argument('-it',
                             '--iterations',
                             help='Specifies the number of iterations that should be done (use only in combination '
                                  'with '
                             '--prb)',
                             type=int,
                             default=1)
    circuit_arg.add_argument('-s',
                             '--stabilizer_type',
                             help='Specifies what the kind of stabilizer should be.',
                             choices=['Z', 'X'],
                             nargs="*",
                             type=str.upper,
                             default='Z')
    circuit_arg.add_argument('-dec',
                             '--decoherence',
                             help='Specifies if decoherence is present in the system.',
                             required=False,
                             action='store_true')
    circuit_arg.add_argument('--p_m_equals_p_g',
                             help='Specify if measurement error equals the gate error. "-p_m" will then be disregarded',
                             required=False,
                             action='store_true')
    circuit_arg.add_argument('--p_m_equals_extra_noisy_measurement',
                             help='Specify if measurement error equals gate error, plus extra single-qubit depolarizing'
                                  ' channel with p_g before measurement. "-p_m" will then be disregarded',
                             required=False,
                             action='store_true')
    circuit_arg.add_argument('-prb',
                             '--probabilistic',
                             help='Specifies if the processes in the protocol are probabilistic.',
                             required=False,
                             action='store_true')
    circuit_arg.add_argument("-swap",
                             "--use_swap_gates",
                             help="A version of the protocol will be run that uses SWAP gates to ensure NV-center "
                                  "realism.",
                             required=False,
                             action="store_true")
    circuit_arg.add_argument("-no_swap",
                             "--noiseless_swap",
                             help="A version of the protocol will be run that uses SWAP gates to ensure NV-center "
                                  "realism.",
                             required=False,
                             action="store_true")
    circuit_arg.add_argument("-n_type",
                             "--network_noise_type",
                             help="Specify the network noise type. ",
                             type=int,
                             choices=[0, 1, 2, 3] + [*range(10, 22)] + [*range(30, 33)] + [*range(40, 43)]
                                     + [*range(50, 55)] + [*range(60, 65)] + [*range(70, 75)] + [*range(80, 85)] + [99]+[*range(100,105)])
    circuit_arg.add_argument("-dynamic_states",
                             "--dynamic_direct_states",
                             help="Specify if you want to use direct states with varying gate error.",
                             required=False,
                             action="store_true"
                             )
    circuit_arg.add_argument("-bp_type",
                             "--bell_pair_type",
                             help="Specify the type of Bell pair that is generated. ",
                             type=int,
                             choices=[0, 1, 2, 3, 30, 40])
    circuit_arg.add_argument("-combine",
                             "--combine",
                             help="Combine superoperator permutations (Used when twirling).",
                             required=False,
                             action="store_true")

    return parser