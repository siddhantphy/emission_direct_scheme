from circuit_simulation.circuit_simulator import *
from tqdm import tqdm
from copy import deepcopy
PBAR: tqdm = None
import time
# from simulate_protocol.protocol_recipe import remove_link_ids_from_qubit_memory


def convert_nmb2let(number_list):
    if type(number_list) == int:
        letter_list = chr(65 + number_list)
    elif type(number_list) == list:
        letter_list = ""
        for number in number_list:
            letter_list += chr(65 + number)
    else:
        letter_list = None
    return letter_list


def create_protocol_recipe_quantum_circuit(protocol_recipe, pbar, **kwargs):
    """
        Initialises a QuantumCircuit object corresponding to a protocol described by the object "ProtocolRecipe".

        Parameters
        ----------
        protocol_recipe : ProtocolRecipe object
            Object describing the protocol for which the QuantumCircuit object should be initialised.

        For other parameters, please see QuantumCircuit class for more information

    """

    global PBAR
    PBAR = pbar

    print_operations = False

    network_architecture_type = kwargs['network_architecture_type'] if 'network_architecture_type' in kwargs.keys() \
        else None
    if network_architecture_type is not None and network_architecture_type in ['weight-3', 'weight-4']:
        nodes_used = 4
        nr_data_qubits = 4 if network_architecture_type == 'weight-4' else 8
    else:
        nodes_used = protocol_recipe.n
        nr_data_qubits = nodes_used

    # Define the qubits that are used
    number_of_qubits_used = nr_data_qubits * 2
    for nodes in protocol_recipe.qubit_memory:
        number_of_qubits_used += len(nodes)
    init_type = nr_data_qubits * 2
    qc = QuantumCircuit(number_of_qubits_used, init_type, **kwargs)
    if print_operations:
        print(colored(f"Quantum circuit with {number_of_qubits_used} qubits in init type {2} is defined.", "green"))

    supop_qubits = None
    if network_architecture_type == 'weight-3':
        supop_qubits = [[number_of_qubits_used - 2, number_of_qubits_used - 6, number_of_qubits_used - 10,
                         number_of_qubits_used - 4],
                        [number_of_qubits_used - 8, number_of_qubits_used - 12, number_of_qubits_used - 14,
                         number_of_qubits_used - 16]]

    smallest_qubit = number_of_qubits_used - 2 * nr_data_qubits
    for i_n in range(nodes_used):
        nr_anc_qubits_node = len(protocol_recipe.qubit_memory[i_n]) if len(protocol_recipe.qubit_memory) > i_n else 0
        largest_qubit = smallest_qubit - 1
        smallest_qubit = largest_qubit - nr_anc_qubits_node + 1
        if network_architecture_type == 'weight-3':
            data_qubits = [number_of_qubits_used - 2 * (1 + 2 * i_n), number_of_qubits_used - 2 * (2 + 2 * i_n)]
            amount_data_qubits = 2
        else:
            data_qubits = [number_of_qubits_used - 2 * (1 + i_n)]
            amount_data_qubits = 1
        qc.define_node(convert_nmb2let(i_n),
                       qubits=data_qubits + [*range(largest_qubit, smallest_qubit - 1, -1)],
                       amount_data_qubits=amount_data_qubits)
        if print_operations:
            print(f"Node {convert_nmb2let(i_n)} is defined with qubits={data_qubits + [*range(largest_qubit, smallest_qubit - 1, -1)]}.")

    # Define the sub_circuits that are used, with (as values) their concurrent subcircuits
    sc_with_their_cscs = {}

    for ts in protocol_recipe.time_blocks:
        sc_in_ts = [convert_nmb2let(sc.subsystem.nodes) for sc in ts if sc.subsystem]
        if len(sc_in_ts) == 1 and sc_in_ts[0] not in sc_with_their_cscs:
            sc_with_their_cscs[sc_in_ts[0]] = []
        else:
            for sc in sc_in_ts:
                for csc in sc_in_ts:
                    if sc != csc:
                        if sc in sc_with_their_cscs:
                            if csc not in sc_with_their_cscs[sc]:
                                sc_with_their_cscs[sc].append(csc)
                        else:
                            sc_with_their_cscs[sc] = [csc]

    if print_operations:
        print(f"Subcircuits with their concurrent subcircuits are: {sc_with_their_cscs}.")

    defined_sub_circuits = []
    for sc, cscs in sc_with_their_cscs.items():
        all_concurrent_sub_systems_defined = True
        for csc in cscs:
            if csc not in defined_sub_circuits:
                all_concurrent_sub_systems_defined = False
                break
        if all_concurrent_sub_systems_defined and cscs:
            qc.define_sub_circuit(sc, concurrent_sub_circuits=cscs)
            if print_operations:
                print(f"Subcircuit {sc} is defined, with concurrent subcircuit(s) {cscs}.")
        else:
            qc.define_sub_circuit(sc)
            if print_operations:
                print(f"Subcircuit {sc} is defined.")
        defined_sub_circuits.append(sc)
    if "ABCDE"[:protocol_recipe.n] not in defined_sub_circuits:
        qc.define_sub_circuit("ABCDE"[:protocol_recipe.n])
        if print_operations:
            name = "ABCDE"[:protocol_recipe.n]
            print(f"Subcircuit {name} is defined.")

    # for subsystem in protocol_recipe.subsystems:
    #     ssys_name = convert_nmb2let(protocol_recipe.subsystems[subsystem].nodes)
    #     concurrent_subsystems = protocol_recipe.subsystems[subsystem].concurrent_subsystems
    #     defined_sub_circuits.append(ssys_name)
    #
    #     if concurrent_subsystems:
    #         all_concurrent_sub_systems_defined = True
    #         conc_ssys = []
    #         for concurrent_ss in concurrent_subsystems:
    #             conc_ssys.append(convert_nmb2let(concurrent_ss))
    #             if convert_nmb2let(concurrent_ss) not in defined_sub_circuits:
    #                 all_concurrent_sub_systems_defined = False
    #         if all_concurrent_sub_systems_defined:
    #             qc.define_sub_circuit(ssys_name, concurrent_sub_circuits=conc_ssys)
    #             if print_operations:
    #                 print(f"Subcircuit {ssys_name} is defined, with concurrent subcircuit(s) {conc_ssys}.")
    #         else:
    #             qc.define_sub_circuit(ssys_name)
    #             if print_operations:
    #                 print(f"Subcircuit {ssys_name} is defined.")
    #     else:
    #         qc.define_sub_circuit(ssys_name)
    #         if print_operations:
    #             print(f"Subcircuit {ssys_name} is defined.")

    qc.define_sub_circuit("A")
    if print_operations:
        print(f"Subcircuit A is defined.")
    if protocol_recipe.n == 2:
        qc.define_sub_circuit("B", concurrent_sub_circuits=["A"])
        if print_operations:
            print(f"Subcircuit B is defined, with concurrent subcircuit ['A'].")
    elif protocol_recipe.n == 3:
        qc.define_sub_circuit("B")
        qc.define_sub_circuit("C", concurrent_sub_circuits=["A", "B"])
        if print_operations:
            print(f"Subcircuit B is defined.")
            print(f"Subcircuit C is defined, with concurrent subcircuits ['A', 'B'].")
    elif protocol_recipe.n == 4:
        qc.define_sub_circuit("B")
        qc.define_sub_circuit("C")
        qc.define_sub_circuit("D", concurrent_sub_circuits=["A", "B", "C"])
        if print_operations:
            print(f"Subcircuit B is defined.")
            print(f"Subcircuit C is defined.")
            print(f"Subcircuit D is defined, with concurrent subcircuit(s) ['A', 'B', 'C'].")
    elif protocol_recipe.n == 5:
        qc.define_sub_circuit("B")
        qc.define_sub_circuit("C")
        qc.define_sub_circuit("D")
        qc.define_sub_circuit("E", concurrent_sub_circuits=["A", "B", "C", "D"])
        if print_operations:
            print(f"Subcircuit B is defined.")
            print(f"Subcircuit C is defined.")
            print(f"Subcircuit D is defined.")
            print(f"Subcircuit E is defined, with concurrent subcircuit(s) ['A', 'B', 'C', 'D'].")
        qc.define_sub_circuit("ABCDE")

    else:
        print("This functionality should be added.")
    if print_operations:
        print("")

    return qc, supop_qubits


def find_operation_that_can_be_switched(initialized_qubit_combinations, qubit_nr, operation_numbers, link_creation_analysis):
    qubit_index = initialized_qubit_combinations[qubit_nr][0].index(qubit_nr)
    involved_ops = initialized_qubit_combinations[qubit_nr][1][qubit_index]
    # print(f"involved_ops = {involved_ops}.")
    link_brother = None
    for i_op, op in enumerate([x[0] for x in involved_ops]):
        if operation_numbers[op][0] == "CREATE_LINK":
            for i_list, list_ops in enumerate(initialized_qubit_combinations[qubit_nr][1]):
                if i_list != qubit_index:
                    if op in [x[0] for x in list_ops]:
                        if link_brother is not None:
                            print(f"link_brother is not None, but {link_brother}. Operation {link_brother[0]} corresponds to {operation_numbers[link_brother[0]]}.")
                            # print(qubit_nr, qubit_index)
                            # print(initialized_qubit_combinations)
                        link_brother = (op, i_list)
    if qubit_index == 0 and link_brother is not None:
        # print(f"Link creation id {link_brother[0]} is in the right order.")
        if link_brother[0] not in link_creation_analysis:
            link_creation_analysis[link_brother[0]] = [1]
        else:
            link_creation_analysis[link_brother[0]].append(1)
    elif qubit_index >= 1 and link_brother is not None:
        # print(f"Link creation id {link_brother[0]} is in the wrong order: brother on index {link_brother[1]}.")
        # if link_brother[1] != 0:
        #     print(involved_ops)
        #     print(initialized_qubit_combinations[qubit_nr][1])
        if link_brother[1] < qubit_index:
            value = 1 if link_brother[1] == 0 else 2
            if link_brother[0] not in link_creation_analysis:
                link_creation_analysis[link_brother[0]] = [-1 * value]
            else:
                link_creation_analysis[link_brother[0]].append(-1 * value)
        else:
            if link_brother[0] not in link_creation_analysis:
                link_creation_analysis[link_brother[0]] = [2]
            else:
                link_creation_analysis[link_brother[0]].append(2)

    else:
        link_creation = None
        for i_op, op in enumerate([x[0] for x in involved_ops]):
            if operation_numbers[op][0] == "CREATE_LINK":
                if link_creation is not None:
                    print(f"link_brother is not None, but {link_brother}.")
                link_creation = op
        # print(f"Link creation id {link_creation} doesn't matter.")
        if link_creation not in link_creation_analysis:
            link_creation_analysis[link_creation] = [0]
        else:
            link_creation_analysis[link_creation].append(0)

    # if print_initialized_qubit_combinations and qubit_index != 0:
    # if qubit_index != 0:
    #     print(f"Qubit index {qubit_index} is measured, with qubit combinations {initialized_qubit_combinations[qubit_nr][0][qubit_index]}.")

    return link_creation_analysis


def test_protocol_on_initialization_order(prot_rec):
    dist_gates = {1: CZ_gate, 2: CNOT_gate, 3: CiY_gate}
    operation_numbers = {}
    link_creation_analysis = {}

    print_statements = [False] * 5
    print_operations = print_statements[0]
    print_qubit_numbers = print_statements[1]
    print_qubit_memory = print_statements[2]
    print_qubit_memory_local = print_statements[3]
    print_initialized_qubit_combinations = print_statements[4]

    if print_operations:
        print(colored("Executed protocol in the simulator:", "green"))
    first_line = True

    start = 0
    qubit_numbers = []
    qubit_memory_local = []
    for node in reversed(prot_rec.qubit_memory):
        qubit_numbers.append([*range(start, start + len(node))])
        qubit_memory_local.append([None] * len(node))
        start += len(node)
    qubit_numbers = list(reversed(qubit_numbers))
    qubit_memory_local = list(reversed(qubit_memory_local))
    initialized_qubit_combinations = {i: [[i], []] for i in range(qubit_numbers[0][-1] + 1)}

    i_ts = 0

    start_time = time.time()
    MAX_TIME = 1 * 60

    while i_ts < len(prot_rec.time_blocks) and time.time() - start_time < MAX_TIME:
        i_ssys = 0

        if print_operations:
            if not first_line:
                print("")
            else:
                first_line = False
            print(colored("Time step " + str(i_ts) + ":", "white"))

        while i_ssys < len(prot_rec.time_blocks[i_ts]) and time.time() - start_time < MAX_TIME:
            sub_circuit_opened = False
            if prot_rec.time_blocks[i_ts][i_ssys].list_of_operations:
                i_op = 0
                while i_op < len(prot_rec.time_blocks[i_ts][i_ssys].list_of_operations) and time.time() - start_time < MAX_TIME:
                    recipe_op = prot_rec.time_blocks[i_ts][i_ssys].list_of_operations[i_op]

                    if print_operations or print_qubit_memory_local or print_qubit_memory or print_qubit_numbers:
                        print("")

                    skip_operation = False
                    if not skip_operation:
                        if sub_circuit_opened is False:
                            if print_operations:
                                print(colored(
                                    "SUBSYSTEM " + str(prot_rec.time_blocks[i_ts][i_ssys].subsystem.nodes) + " ("
                                    + str(prot_rec.time_blocks[i_ts][i_ssys].elem_links) + " LDE attempts):", "red"))

                        sub_circuit_opened = True


                        if print_operations:
                            recipe_op.print_operation()




                        if recipe_op.type == "CREATE_LINK":
                            qubit_memory_local[recipe_op.nodes[0]][0] = recipe_op.link_id
                            qubit_memory_local[recipe_op.nodes[1]][0] = recipe_op.link_id
                            operation_numbers[recipe_op.i_op] = ("CREATE_LINK", )

                            # Extra structure to try to have the measurement qubit as first qubit as often as possible
                            qb1, qb2 = qubit_numbers[recipe_op.nodes[0]][0], qubit_numbers[recipe_op.nodes[1]][0]
                            for qb in [qb1, qb2]:
                                if initialized_qubit_combinations[qb][0] != [qb]:
                                    for qb3 in initialized_qubit_combinations[qb][0]:
                                        state3 = initialized_qubit_combinations[qb3]
                                        if qb in state3[0]:
                                            del state3[1][state3[0].index(qb)]
                                            del state3[0][state3[0].index(qb)]
                            state_object = [[qb2, qb1], [[(recipe_op.i_op, 0)], [(recipe_op.i_op, 1)]]]
                            initialized_qubit_combinations[qb1] = state_object
                            initialized_qubit_combinations[qb2] = state_object





                        elif recipe_op.type == "SWAP":
                            efficient_swap = True
                            if all(recipe_op.link_id):  # If this occurs, the SWAP might take place between two
                                                        # qubits that are already initialized. In that case we cannot
                                                        # the "efficient_swap" functionality.
                                qubit_number_1 = qubit_numbers[recipe_op.e_qubits[0][0]][recipe_op.e_qubits[0][1]]
                                qubit_number_2 = qubit_numbers[recipe_op.m_qubits[0][0]][recipe_op.m_qubits[0][1]]
                                efficient_swap = False if (initialized_qubit_combinations[qubit_number_1][1]
                                                           and initialized_qubit_combinations[qubit_number_2][1]) else True

                            node = recipe_op.nodes[0]
                            save_information = qubit_memory_local[node][0]
                            qubit_memory_local[node][0] = qubit_memory_local[node][recipe_op.m_qubits[0][1]]
                            qubit_memory_local[node][recipe_op.m_qubits[0][1]] = save_information
                            operation_numbers[recipe_op.i_op] = ("SWAP", efficient_swap)

                            # Extra structure to try to have the measurement qubit as first qubit as often as possible
                            qb1 = qubit_numbers[recipe_op.e_qubits[0][0]][0]
                            qb2 = qubit_numbers[recipe_op.m_qubits[0][0]][recipe_op.m_qubits[0][1]]
                            state1 = initialized_qubit_combinations[qb1]
                            state2 = initialized_qubit_combinations[qb2]
                            if efficient_swap is False:
                                if qb2 not in state1[0]:
                                    # Combine the states
                                    for i_qb, qb in enumerate(state1[0]):
                                        state1[1][i_qb].append((recipe_op.i_op, 0))
                                    for i_qb, qb in enumerate(state2[0]):
                                        state2[1][i_qb].append((recipe_op.i_op, 1))
                                        state1[0].append(qb)
                                        state1[1].append(state2[1][i_qb])
                                        initialized_qubit_combinations[qb] = state1
                                    initialized_qubit_combinations[qb2] = state1
                            else:
                                state1_index = state1[0].index(qb1)
                                state2_index = state2[0].index(qb2)
                                state1[0][state1_index] = qb2
                                state2[0][state2_index] = qb1
                                initialized_qubit_combinations[qb1] = state2
                                initialized_qubit_combinations[qb2] = state1






                        elif recipe_op.type == "DISTILL":
                            measurement_list = [None] * len(recipe_op.operator)
                            measurement_order = [None] * len(recipe_op.operator)
                            for i_dist_op, dist_op in enumerate(recipe_op.operator):
                                dist_gate = dist_gates[dist_op]
                                cqubit_nr = qubit_numbers[recipe_op.e_qubits[i_dist_op][0]][0]
                                measurement_list[i_dist_op] = (recipe_op.e_qubits[i_dist_op],
                                                               recipe_op.m_qubits[i_dist_op],
                                                               dist_gate,
                                                               i_dist_op)
                                measurement_order[i_dist_op] = initialized_qubit_combinations[cqubit_nr][0].index(cqubit_nr)

                            measurement_list = [x for _, x in sorted(zip(measurement_order, measurement_list))]
                            if print_initialized_qubit_combinations:
                                print(f"Control qubits are {[qubit_numbers[x[0][0]][x[0][1]] for x in measurement_list]}. Target qubits are {[qubit_numbers[x[1][0]][x[1][1]] for x in measurement_list]}.")

                            for i_qb, qubit in enumerate(measurement_list):
                                node_nmb = qubit[0][0]
                                qubit_nr = qubit_numbers[qubit[0][0]][qubit[0][1]]
                                link_creation_analysis = find_operation_that_can_be_switched(initialized_qubit_combinations, qubit_nr, operation_numbers, link_creation_analysis)

                                # Extra structure to try to have the measurement qubit as first qubit as often as possible
                                c_qubit = qubit_numbers[qubit[0][0]][qubit[0][1]]
                                t_qubit = qubit_numbers[qubit[1][0]][qubit[1][1]]
                                state1 = initialized_qubit_combinations[c_qubit]
                                state2 = initialized_qubit_combinations[t_qubit]
                                if t_qubit not in state1[0]:
                                    # Combine the states
                                    for i_q, qb in enumerate(state2[0]):
                                        state1[0].append(qb)
                                        state1[1].append(state2[1][i_q])
                                        initialized_qubit_combinations[qb] = state1
                                    initialized_qubit_combinations[t_qubit] = state1
                                # state1[1][state1[0].index(t_qubit)].append(recipe_op.link_id)
                                del state1[1][state1[0].index(c_qubit)]
                                del state1[0][state1[0].index(c_qubit)]
                                initialized_qubit_combinations[c_qubit] = [[c_qubit], []]
                                # print({k: v[0] for k, v in initialized_qubit_combinations.items()})

                                qubit_memory_local[qubit[0][0]][qubit[0][1]] = None
                                for qb_loc, id_number in enumerate(qubit_memory_local[node_nmb]):
                                    if id_number in recipe_op.family_tree:
                                        qubit_memory_local[node_nmb][qb_loc] = recipe_op.link_id

                            operation_numbers[recipe_op.i_op] = ("DISTILL", )







                        elif recipe_op.type == "FUSE":
                            qubit_nr = qubit_numbers[recipe_op.e_qubits[0][0]][0]
                            if print_initialized_qubit_combinations:
                                print(f"Control qubit is {qubit_nr}. Target qubit is {qubit_numbers[recipe_op.m_qubits[0][0]][recipe_op.m_qubits[0][1]]}.")

                            link_creation_analysis = find_operation_that_can_be_switched(initialized_qubit_combinations, qubit_nr, operation_numbers, link_creation_analysis)

                            for e_qubit in recipe_op.e_qubits:
                                qubit_memory_local[e_qubit[0]][e_qubit[1]] = None
                            for node in range(len(qubit_memory_local)):
                                for qb_loc, id_number in enumerate(qubit_memory_local[node]):
                                    if id_number in recipe_op.family_tree:
                                        qubit_memory_local[node][qb_loc] = recipe_op.link_id
                            # Extra structure to try to have the measurement qubit as first qubit as often as possible
                            c_qb = qubit_numbers[recipe_op.e_qubits[0][0]][0]
                            t_qb = qubit_numbers[recipe_op.m_qubits[0][0]][recipe_op.m_qubits[0][1]]
                            state1 = initialized_qubit_combinations[c_qb]
                            if t_qb not in state1[0]:
                                # Combine the states
                                state2 = initialized_qubit_combinations[t_qb]
                                for i_qb, qb in enumerate(state2[0]):
                                    state1[0].append(qb)
                                    state1[1].append(state2[1][i_qb])
                                    initialized_qubit_combinations[qb] = state1
                                initialized_qubit_combinations[t_qb] = state1
                            # state1[1][state1[0].index(t_qb)].append(recipe_op.i_op)
                            del state1[1][state1[0].index(c_qb)]
                            del state1[0][state1[0].index(c_qb)]
                            initialized_qubit_combinations[c_qb] = [[c_qb], []]

                        if print_qubit_memory_local:
                            print(f"Local qubit memory/register: {qubit_memory_local}.")
                        if print_qubit_numbers:
                            print(f"Initialized qubits         : {prot_rec.qubit_memory_per_time_step[i_ts][i_ssys][i_op]}.")
                            print(f"Qubit numbers              : {qubit_numbers}.")
                        if print_initialized_qubit_combinations:
                            print(f"My own list of init qubits : " + str({k: v[0] for k, v in initialized_qubit_combinations.items()}))
                            print(f"Full list of init qubits   : {initialized_qubit_combinations}.")
                            # print(f"My own list of init qubits : {initialized_qubit_combinations}.")


                    i_op += 1
            i_ssys += 1
        i_ts += 1

    # print(link_creation_analysis)
    final_link_creation_analysis = deepcopy(link_creation_analysis)
    for key, value in link_creation_analysis.items():
        if 1 in value and -1 not in value:
            del final_link_creation_analysis[key]
        elif -1 in value and 1 not in value:
            final_link_creation_analysis[key] = "Reverse"
        elif 2 in value and -2 not in value:
            del final_link_creation_analysis[key]
        elif -2 in value and 2 not in value:
            final_link_creation_analysis[key] = "Reverse"
        else:
            del final_link_creation_analysis[key]
    # print(final_link_creation_analysis)
    return final_link_creation_analysis


def auto_generated_swap(qc_input: QuantumCircuit, *, operation, prot_rec):
    dist_gates = {1: CZ_gate, 2: CNOT_gate, 3: CiY_gate}
    measurement_results = {}
    link_creation_attempts = {}
    # link_creation_analysis = {}
    link_creation_analysis = test_protocol_on_initialization_order(prot_rec)

    try_out_new_functionality = True

    # print("Put back print statements.")
    print_statements = [False] * 6
    print_operations = print_statements[0]
    print_qubit_numbers = print_statements[1]
    print_qubit_memory = print_statements[2]
    print_qubit_memory_local = print_statements[3]
    print_time_progression = print_statements[4]
    print_initialized_qubit_combinations = print_statements[5]

    if print_operations:
        print(colored("Executed protocol in the simulator:", "green"))
    first_line = True
    carried_out_operations = []

    start = 0
    qubit_numbers = []
    qubit_memory_local = []
    for node in reversed(prot_rec.qubit_memory):
        qubit_numbers.append([*range(start, start + len(node))])
        qubit_memory_local.append([None] * len(node))
        start += len(node)
    qubit_numbers = list(reversed(qubit_numbers))
    qc_input.qubit_numbers = qubit_numbers
    qubit_memory_local = list(reversed(qubit_memory_local))
    initialized_qubit_combinations = {i: [[i], []] for i in range(qubit_numbers[0][-1] + 1)}

    ids_carry_out = []
    ids_carry_out_reset = []

    i_ts = 0
    i_ssys = 0
    i_op = 0
    reset_from_failed_distillation = False
    time_step_cut_off_time_reached = False
    qc_back_up_failed_distillation_mode = False
    failed_distillation_time_mark = None
    failed_operations = None

    qc = deepcopy(qc_input)

    start_time = time.time()
    # print("Put MAX_TIME back to 2 minutes.")
    MAX_TIME = np.infty #10*60

    while i_ts < len(prot_rec.time_blocks) \
            and qc.cut_off_time_reached is False \
            and time.time() - start_time < MAX_TIME:
        reset_from_failed_distillation_previous_ts = reset_from_failed_distillation
        reset_from_failed_distillation = False
        if not reset_from_failed_distillation_previous_ts:
            i_ssys = 0

        if not qc_back_up_failed_distillation_mode:
            measurement_results_in_ts = {}

        if print_operations:
            if not first_line:
                print("")
            else:
                first_line = False
            print(colored("Time step " + str(i_ts) + ":", "white"))

        qc_back_up = deepcopy(qc)
        ids_carry_out_back_up = deepcopy(ids_carry_out)
        ids_carry_out_reset_back_up = deepcopy(ids_carry_out_reset)
        qubit_memory_local_back_up = deepcopy(qubit_memory_local)
        initialized_qubit_combinations_back_up = deepcopy(initialized_qubit_combinations)
        carried_out_operations_back_up = deepcopy(carried_out_operations)
        skipped_operations_outside_time_stamp = []

        while i_ssys < len(prot_rec.time_blocks[i_ts]) \
                and not reset_from_failed_distillation \
                and not qc.cut_off_time_reached \
                and time.time() - start_time < MAX_TIME:
            sub_circuit_opened = False
            if prot_rec.time_blocks[i_ts][i_ssys].list_of_operations:
                if qc_back_up_failed_distillation_mode or (not reset_from_failed_distillation_previous_ts):
                    i_op = 0
                sub_block_cut_off_time_reached = False
                while i_op < len(prot_rec.time_blocks[i_ts][i_ssys].list_of_operations) \
                        and not reset_from_failed_distillation \
                        and not qc.cut_off_time_reached \
                        and not sub_block_cut_off_time_reached \
                        and time.time() - start_time < MAX_TIME:
                    skip_i_op_counter = False
                    recipe_op = prot_rec.time_blocks[i_ts][i_ssys].list_of_operations[i_op]
                    reset_from_failed_distillation_previous_ts = False

                    if print_operations or print_time_progression or print_qubit_memory_local or print_qubit_memory or print_qubit_numbers:
                        print("")

                    node_times = {node: qc.nodes[node].sub_circuit_time + qc.total_duration for node in 'ABCDE'[:prot_rec.n]}
                    for node in "ABCDE"[:prot_rec.n]:
                        for sc in qc._sub_circuits.keys():
                            if node in sc:
                                node_times[node] += qc._sub_circuits[sc].total_duration
                    if recipe_op.type == "CREATE_LINK":
                        operation_time = max([node_times[convert_nmb2let(node)] for node in recipe_op.nodes])
                    else:
                        operation_time = min([node_times[convert_nmb2let(node)] for node in recipe_op.nodes])

                    skip_operation = False
                    if ids_carry_out:
                        if recipe_op.type == "SWAP":
                            if not (recipe_op.link_id[0] in ids_carry_out[-1]
                                    or recipe_op.link_id[1] in ids_carry_out[-1]
                                    or [recipe_op.i_op, recipe_op.link_id] in ids_carry_out[-1]):
                                skip_operation = True
                        elif recipe_op.link_id not in ids_carry_out[-1]:
                            skip_operation = True
                    while ids_carry_out \
                            and [i_ts, i_ssys, i_op] == ids_carry_out_reset[-1] \
                            and time.time() - start_time < MAX_TIME:
                        ids_carry_out.pop()
                        ids_carry_out_reset.pop()
                        if print_operations:
                            print(colored("New carry out = " + str(ids_carry_out) + ", Carry out reset = "
                                          + str(ids_carry_out_reset) + ".", "yellow"))

                    if not skip_operation and qc_back_up_failed_distillation_mode:
                        if round(failed_distillation_time_mark - operation_time, SUM_ACCURACY) <= 0:
                            if print_time_progression:
                                print(f"Operation {recipe_op.link_id} skipped, because after failed_distillation_time_mark.")
                            skip_operation = True
                            if recipe_op.type == "SWAP":
                                skipped_operations_outside_time_stamp.append([recipe_op.i_op, recipe_op.link_id])
                            else:
                                skipped_operations_outside_time_stamp.append(recipe_op.link_id)
                            if recipe_op.type == "CREATE_LINK":
                                inv_nodes_time = {node: node_times[convert_nmb2let(node)] for node in recipe_op.nodes}
                                other_node_time = min(inv_nodes_time.values())
                                if round(failed_distillation_time_mark - other_node_time, SUM_ACCURACY) > 0:
                                    link_waiting_time = inv_nodes_time[recipe_op.nodes[0]] - \
                                                        inv_nodes_time[recipe_op.nodes[1]]
                                    if link_waiting_time > 0:
                                        qc._increase_duration(link_waiting_time,
                                                              [qubit_numbers[recipe_op.nodes[1]][0]],
                                                              kind='idle',
                                                              print_time_progression=print_time_progression)
                                    else:
                                        qc._increase_duration(-1*link_waiting_time,
                                                              [qubit_numbers[recipe_op.nodes[0]][0]],
                                                              kind='idle',
                                                              print_time_progression=print_time_progression)
                                    node_times = {node: qc.nodes[node].sub_circuit_time + qc.total_duration for node in 'ABCDE'[:prot_rec.n]}
                                    for node in "ABCDE"[:prot_rec.n]:
                                        for sc in qc._sub_circuits.keys():
                                            if node in sc:
                                                node_times[node] += qc._sub_circuits[sc].total_duration

                    if print_time_progression:
                        print(f"Node times: {[qc.nodes[node].sub_circuit_time for node in 'ABCDE'[:prot_rec.n]]}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
                        print(f"Full node times at this point: {node_times}.")
                        print(f"Operation time: {operation_time}.")
                        print(f"Failed distillation time mark: {failed_distillation_time_mark}.")

                    if print_operations and skip_operation:
                        print(f"skip_operation for operation {recipe_op.link_id}: {skip_operation}.")
                        print(f"Operation should have taken place between nodes {recipe_op.nodes}.")

                    if not skip_operation:
                        if sub_circuit_opened is False:
                            # if print_time_progression:
                            #     print(f"Node times: {[qc.nodes[node].sub_circuit_time for node in 'ABCDE'[:prot_rec.n]]}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
                            forced_level = True if i_ssys == 0 else False
                            qc.start_sub_circuit(convert_nmb2let(prot_rec.time_blocks[i_ts][i_ssys].subsystem.nodes),
                                                 forced_level=forced_level)
                            if print_operations:
                                print(colored(
                                    "SUBSYSTEM " + str(prot_rec.time_blocks[i_ts][i_ssys].subsystem.nodes) + " ("
                                    + str(prot_rec.time_blocks[i_ts][i_ssys].elem_links) + " LDE attempts) at time "
                                                                                           f"{qc.total_duration}, "
                                                                                           f"with forced_level="
                                                                                           f"{forced_level}:", "red"))
                            if print_time_progression:
                                print(f"Node times: {[qc.nodes[node].sub_circuit_time for node in 'ABCDE'[:prot_rec.n]]}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")

                        sub_circuit_opened = True

                        if recipe_op.type != "SWAP":
                            carried_out_operations.append(recipe_op.link_id)

                        if print_operations:
                            recipe_op.print_operation()
                            print(f"Carried out operations: {carried_out_operations}.")
                            if not (recipe_op.type == "DISTILL" or recipe_op.type == "FUSE"):
                                print(f"Measurement results: {measurement_results}.")
                                print(f"Measurement results in time step: {measurement_results_in_ts}")
                            if not (recipe_op.type == "CREATE_LINK"):
                                print(f"Link creation attempts: {link_creation_attempts}.")





                        if recipe_op.type == "CREATE_LINK":
                            if recipe_op.i_op not in link_creation_analysis.keys():
                                qubit1 = convert_nmb2let(recipe_op.nodes[0]) + "-e"
                                qubit2 = convert_nmb2let(recipe_op.nodes[1]) + "-e"
                            else:
                                qubit1 = convert_nmb2let(recipe_op.nodes[1]) + "-e"
                                qubit2 = convert_nmb2let(recipe_op.nodes[0]) + "-e"
                            if try_out_new_functionality \
                                    and qc_back_up_failed_distillation_mode \
                                    and recipe_op.link_id in link_creation_attempts:
                                result = qc.create_bell_pair(qubit1,
                                                             qubit2,
                                                             print_time_progression=print_time_progression,
                                                             attempts=link_creation_attempts[recipe_op.link_id],
                                                             probabilistic=False)
                            else:
                                result = qc.create_bell_pair(qubit1,
                                                             qubit2,
                                                             print_time_progression=print_time_progression) #,
                                                             # probabilistic=False)
                            qubit_memory_local[recipe_op.nodes[0]][0] = recipe_op.link_id
                            qubit_memory_local[recipe_op.nodes[1]][0] = recipe_op.link_id
                            link_creation_attempts[recipe_op.link_id] = result
                            if print_operations:
                                print(f"Link creation attempts: {link_creation_attempts}.")

                            # Extra structure to try to have the measurement qubit as first qubit as often as possible
                            if recipe_op.i_op not in link_creation_analysis.keys():
                                qb1, qb2 = qubit_numbers[recipe_op.nodes[0]][0], qubit_numbers[recipe_op.nodes[1]][0]
                            else:
                                qb1, qb2 = qubit_numbers[recipe_op.nodes[1]][0], qubit_numbers[recipe_op.nodes[0]][0]
                            for qb in [qb1, qb2]:
                                if initialized_qubit_combinations[qb][0] != [qb]:
                                    for qb3 in initialized_qubit_combinations[qb][0]:
                                        state3 = initialized_qubit_combinations[qb3]
                                        if qb in state3[0]:
                                            del state3[1][state3[0].index(qb)]
                                            del state3[0][state3[0].index(qb)]
                            state_object = [[qb2, qb1], [[recipe_op.i_op], [recipe_op.i_op]]]
                            initialized_qubit_combinations[qb1] = state_object
                            initialized_qubit_combinations[qb2] = state_object





                        elif recipe_op.type == "SWAP":
                            efficient_swap = True
                            if all(recipe_op.link_id):  # If this occurs, the SWAP might take place between two
                                                        # qubits that are already initialized. In that case we cannot
                                                        # the "efficient_swap" functionality.
                                initialized_qubits = list(
                                    set(qc.qubits.keys()).difference(qc._uninitialised_qubits).difference(
                                        qc.data_qubits))
                                qubit_number_1 = qubit_numbers[recipe_op.e_qubits[0][0]][recipe_op.e_qubits[0][1]]
                                qubit_number_2 = qubit_numbers[recipe_op.m_qubits[0][0]][recipe_op.m_qubits[0][1]]
                                efficient_swap = False if (qubit_number_1 in initialized_qubits
                                                           and qubit_number_2 in initialized_qubits) else True

                            node = recipe_op.nodes[0]
                            result = qc.SWAP(convert_nmb2let(node) + "-e",
                                             convert_nmb2let(node) + "-e+" + str(recipe_op.m_qubits[0][1]),
                                             efficient=efficient_swap)
                            save_information = qubit_memory_local[node][0]
                            qubit_memory_local[node][0] = qubit_memory_local[node][recipe_op.m_qubits[0][1]]
                            qubit_memory_local[node][recipe_op.m_qubits[0][1]] = save_information

                            # Extra structure to try to have the measurement qubit as first qubit as often as possible
                            qb1 = qubit_numbers[recipe_op.e_qubits[0][0]][0]
                            qb2 = qubit_numbers[recipe_op.m_qubits[0][0]][recipe_op.m_qubits[0][1]]
                            state1 = initialized_qubit_combinations[qb1]
                            state2 = initialized_qubit_combinations[qb2]
                            # state1[1].append(recipe_op.link_id)
                            if efficient_swap is False:
                                if qb2 not in state1[0]:
                                    # Combine the states
                                    for i_qb, qb in enumerate(state2[0]):
                                        state1[0].append(qb)
                                        state1[1].append(state2[1][i_qb])
                                        initialized_qubit_combinations[qb] = state1
                                    initialized_qubit_combinations[qb2] = state1
                            else:
                                state1_index = state1[0].index(qb1)
                                state2_index = state2[0].index(qb2)
                                state1[0][state1_index] = qb2
                                state2[0][state2_index] = qb1
                                initialized_qubit_combinations[qb1] = state2
                                initialized_qubit_combinations[qb2] = state1






                        elif recipe_op.type == "DISTILL":
                            # measurement_list = [None] * len(recipe_op.operator)
                            # measurement_order = [None] * len(recipe_op.operator)
                            # for i_dist_op, dist_op in enumerate(recipe_op.operator):
                            #     dist_gate = dist_gates[dist_op]
                            #     cqubit_nr = qubit_numbers[recipe_op.e_qubits[i_dist_op][0]][0]
                            #     measurement_list[i_dist_op] = (recipe_op.e_qubits[i_dist_op][0], recipe_op.m_qubits[i_dist_op], dist_gate)
                            #     measurement_order[i_dist_op] = initialized_qubit_combinations[cqubit_nr][0].index(cqubit_nr)
                            #
                            # measurement_list = [x for _, x in sorted(zip(measurement_order, measurement_list))]
                            # if print_initialized_qubit_combinations:
                            #     print(f"Control qubits are {[qubit_numbers[x[0]][0] for x in measurement_list]}. Target qubits are {[qubit_numbers[x[1][0]][x[1][1]] for x in measurement_list]}.")
                            #
                            # read_results = True if (try_out_new_functionality
                            #                         and qc_back_up_failed_distillation_mode
                            #                         and recipe_op.link_id in measurement_results
                            #                         # and i_ssys <= failed_operations[0][1][1]
                            #                         ) else False
                            # result = measurement_results[recipe_op.link_id] if read_results \
                            #     else [None] * len(measurement_list)
                            #
                            # for i_qb, qubit in enumerate(measurement_list):
                            #     cqubit = convert_nmb2let(qubit[0]) + "-e"
                            #     tqubit = convert_nmb2let(qubit[1][0]) + "-e+" + str(qubit[1][1])
                            #     # Look into "reverse=True" statement here:
                            #     qc.apply_gate(qubit[2], cqubit=cqubit, tqubit=tqubit)
                            #     # qubit_nr = qubit_numbers[qubit[0]][0]
                            #     # qubit_index = initialized_qubit_combinations[qubit_nr][0].index(qubit_nr)
                            #     # involved_ops = initialized_qubit_combinations[qubit_nr][1][qubit_index]
                            #     # link_brother = None
                            #     # for op in involved_ops:
                            #     #     if op in link_creation_attempts.keys():
                            #     #         for i_list, list_ops in enumerate(initialized_qubit_combinations[qubit_nr][1]):
                            #     #             if i_list != qubit_index:
                            #     #                 if op in list_ops:
                            #     #                     if link_brother is not None:
                            #     #                         raise ValueError
                            #     #                     link_brother = (op, i_list)
                            #     # if qubit_index == 0 and link_brother is not None:
                            #     #     print(f"Link creation id {link_brother[0]} is in the right order.")
                            #     # elif qubit_index >= 1 and link_brother is not None:
                            #     #     print(f"Link creation id {link_brother[0]} is in the wrong order: brother on index {link_brother[1]}.")
                            #     # else:
                            #     #     link_creation = None
                            #     #     for op in involved_ops:
                            #     #         if op in link_creation_attempts.keys():
                            #     #             if link_creation is not None:
                            #     #                 raise ValueError
                            #     #             link_creation = op
                            #     #     print(f"Link creation id {link_creation} doesn't matter.")
                            #     # # if print_initialized_qubit_combinations and qubit_index != 0:
                            #     # if qubit_index != 0:
                            #     #     print(f"Qubit index {qubit_index} is measured, with qubit combinations {initialized_qubit_combinations[qubit_nr][0]}.")
                            #     if read_results:
                            #         qc.measure(cqubit, outcome=result[i_qb], probabilistic=False, basis="X")
                            #     else:
                            #         result_individual = qc.measure(cqubit, basis="X")[0]
                            #         if type(result_individual) == SKIP:
                            #             result = result_individual
                            #             break
                            #         else:
                            #             result[i_qb] = result_individual
                            #     measurement_results[recipe_op.link_id] = result if type(result) != SKIP else None

                            # New part instead of part above:
                            measurement_list = [] # [None] * len(recipe_op.operator)
                            measurement_order = [] # [None] * len(recipe_op.operator)
                            for i_dist_op, dist_op in enumerate(recipe_op.operator):
                                dist_gate = dist_gates[dist_op]
                                cqubit_nr = qubit_numbers[recipe_op.e_qubits[i_dist_op][0]][0]
                                node = convert_nmb2let(recipe_op.e_qubits[i_dist_op][0])
                                # Situations in which we need to carry out the part of the distillation operation in
                                # this node:
                                #   * The measurement is not yet carried out in the first place - i.e., there is no key
                                #     "recipe_op.link_id" in "measurement_results";
                                #   * There is a key "recipe_op.link_id" in "measurement_results", but this specific
                                #     part of the operation is not yet carried out (i.e., carries a None);
                                #   * We are in "qc_back_up_failed_distillation_mode" - meaning we have to read off
                                #     already executed measurements that are already done in the original iteration
                                #     of this time step (before the distillation failed). We now also allow for this
                                #     part of the measurement to be already carried out - this occurs when
                                #     "recipe_op.link_id" is already a key in "measurement_results" and this part of
                                #     distillation operation does not carry a None. In this case, however, we have
                                #     to make sure that we don't repeat measurements from previous time steps.
                                #     Therefore, we also have to make sure that this part of the distillation operation
                                #     was carried out in this time step particularly; this is checked by searching in a
                                #     dictionary with measurement results of this time step only.
                                # I think I don't need the two lines that are commented out (this should logically
                                # always be true if the if-statement reaches this part of the or-logic).

                                if (recipe_op.link_id not in measurement_results) \
                                        or (recipe_op.link_id in measurement_results
                                            and measurement_results[recipe_op.link_id][i_dist_op] is None) \
                                        or (try_out_new_functionality
                                            and qc_back_up_failed_distillation_mode
                                            # and recipe_op.link_id in measurement_results
                                            # and measurement_results[recipe_op.link_id][i_dist_op] is not None
                                            and recipe_op.link_id in measurement_results_in_ts
                                            and measurement_results_in_ts[recipe_op.link_id][i_dist_op] is not None):
                                    # If the operation is carried out, and we are in
                                    # "qc_back_up_failed_distillation_mode", we do still need to make sure the
                                    # "failed_distillation_time_mark" is not yet exceeded in this node.
                                    if not (try_out_new_functionality
                                            and qc_back_up_failed_distillation_mode
                                            and round(failed_distillation_time_mark - node_times[node], SUM_ACCURACY) <= 0):
                                        measurement_list.append((recipe_op.e_qubits[i_dist_op],
                                                                 recipe_op.m_qubits[i_dist_op],
                                                                 dist_gate,
                                                                 i_dist_op))
                                        measurement_order.append(initialized_qubit_combinations[cqubit_nr][0].index(cqubit_nr))
                                    else:
                                        # If the "failed_distillation_time_mark" is exceeded in this node, we skip this
                                        # part of the full distillation operation after all
                                        if print_operations:
                                            print(f"Sub-operation {i_dist_op} of operation {recipe_op.link_id} is "
                                                  f"skipped, because its node time {node_times[node]} fell outside "
                                                  f"failed_distillation_time_mark {failed_distillation_time_mark}.")

                            # Of all the sub-operations of this distillation operation that we DO carry out, we now
                            # determine in what place the qubits that need to be measured sit in their respectively
                            # density matrices. We sort them from smallest index to biggest:
                            if print_operations:
                                print(f"Indices of measured qubits in combined density matrices are: {measurement_order}.")
                            measurement_list = [x for _, x in sorted(zip(measurement_order, measurement_list))]
                            if print_initialized_qubit_combinations:
                                print(f"Control qubits are {[qubit_numbers[x[0][0]][x[0][1]] for x in measurement_list]}. Target qubits are {[qubit_numbers[x[1][0]][x[1][1]] for x in measurement_list]}.")

                            # We construct a placeholder in which we collect all measurement results:
                            results = [None] * len(recipe_op.operator)
                            if recipe_op.link_id not in measurement_results_in_ts.keys():
                                measurement_results_in_ts[recipe_op.link_id] = [None] * len(recipe_op.operator)

                            # We now apply the sub-operations of the full distillation operation that we DO carry out:
                            for i_qb, qubit in enumerate(measurement_list):
                                node_nmb = qubit[0][0]
                                cqubit = convert_nmb2let(qubit[0][0]) + "-e"
                                tqubit = convert_nmb2let(qubit[1][0]) + "-e+" + str(qubit[1][1])

                                # We apply the two-qubit gate that is part of the distillation operation:
                                # Look into "reverse=True" statement here:
                                qc.apply_gate(qubit[2], cqubit=cqubit, tqubit=tqubit)

                                # For the measurement of the distillation operation, we first determine if we should
                                # read out the measurement result, or we need to randomly apply a measurement. Reading
                                # off a result (and applying that deterministically) occurs if we are in
                                # "qc_back_up_failed_distillation_mode". In terms of the logic above (for when we
                                # proceed with a certain sub-operation), for this we only have to check if there is
                                # a measurement result available in the measurement results of this time step):
                                read_result = True if (recipe_op.link_id in measurement_results_in_ts
                                                       and measurement_results_in_ts[recipe_op.link_id] is not None
                                                       and measurement_results_in_ts[recipe_op.link_id][qubit[3]] is not None
                                                       ) else False
                                if print_operations and read_result:
                                    print(f"For sub-operation {qubit[3]}, the result is read-off from measurement_results_in_ts.")
                                if read_result:
                                    result = measurement_results_in_ts[recipe_op.link_id][qubit[3]]
                                    qc.measure(cqubit, outcome=result, probabilistic=False, basis="X")
                                else:
                                    result = qc.measure(cqubit, basis="X")
                                    if type(result) != SKIP:
                                        result = result[0]
                                    if type(result) != SKIP and measurement_results_in_ts[recipe_op.link_id] is not None:
                                        measurement_results_in_ts[recipe_op.link_id][qubit[3]] = result
                                    else:
                                        measurement_results_in_ts[recipe_op.link_id] = None
                                if type(result) == SKIP:
                                    results = result
                                elif type(results) != SKIP:
                                    results[qubit[3]] = result

                                # Extra structure to try to have the measurement qubit as first qubit as often as possible
                                c_qubit = qubit_numbers[qubit[0][0]][qubit[0][1]]
                                t_qubit = qubit_numbers[qubit[1][0]][qubit[1][1]]
                                state1 = initialized_qubit_combinations[c_qubit]
                                state2 = initialized_qubit_combinations[t_qubit]
                                if t_qubit not in state1[0]:
                                    # Combine the states
                                    for i_q, qb in enumerate(state2[0]):
                                        state1[0].append(qb)
                                        state1[1].append(state2[1][i_q])
                                        initialized_qubit_combinations[qb] = state1
                                    initialized_qubit_combinations[t_qubit] = state1
                                # state1[1][state1[0].index(t_qubit)].append(recipe_op.link_id)
                                del state1[1][state1[0].index(c_qubit)]
                                del state1[0][state1[0].index(c_qubit)]
                                initialized_qubit_combinations[c_qubit] = [[c_qubit], []]
                                # print({k: v[0] for k, v in initialized_qubit_combinations.items()})

                                qubit_memory_local[qubit[0][0]][qubit[0][1]] = None
                                for qb_loc, id_number in enumerate(qubit_memory_local[node_nmb]):
                                    if id_number in recipe_op.family_tree:
                                        qubit_memory_local[node_nmb][qb_loc] = recipe_op.link_id

                            # Collect earlier measurement results:
                            if type(results) != SKIP:
                                for i_result in range(len(results)):
                                    if results[i_result] is None and recipe_op.link_id in measurement_results:
                                        results[i_result] = measurement_results[recipe_op.link_id][i_result]

                            measurement_results[recipe_op.link_id] = results if type(results) != SKIP else None

                            # If there is still a None in the final list of measurement results, we need to perform this
                            # operation again while going through the operations again (to carry out the remaining sub-
                            # operations and parity of all measurement results):
                            if try_out_new_functionality \
                                    and qc_back_up_failed_distillation_mode \
                                    and measurement_results_in_ts[recipe_op.link_id] is not None \
                                    and None in measurement_results[recipe_op.link_id]:
                                skipped_operations_outside_time_stamp.append(recipe_op.link_id)

                            if print_operations:
                                print(colored("Measurements: " + str(results) + ". All results are: "
                                              + str(measurement_results) + ".", "yellow"))
                                print(f"Measurement results in this time step: {measurement_results_in_ts}.")

                            # # This was part of the old structure:
                            # qubit_memory_local[qubit[0][0]][qubit[0][1]] = None
                            # for node in range(len(qubit_memory_local)):
                            #     for qb_loc, id_number in enumerate(qubit_memory_local[node]):
                            #         if id_number in recipe_op.family_tree:
                            #             qubit_memory_local[node][qb_loc] = recipe_op.link_id


                            # If None is part of the measurement results, we do not yet evaluate the result:
                            if recipe_op.delay_after_sub_block is False \
                                    and measurement_results[recipe_op.link_id] is not None \
                                    and None not in measurement_results[recipe_op.link_id]:
                                success = 0
                                for succ_dep in recipe_op.success_dep:
                                    success = (success + measurement_results[succ_dep].count(1)) % 2 \
                                        if (qc.cut_off_time_reached is False
                                            and success is not None
                                            and succ_dep in measurement_results
                                            and measurement_results[succ_dep] is not None) else None
                                if success == 1:




                                    if try_out_new_functionality \
                                            and (recipe_op.frl[0] < i_ts or recipe_op.frl[1] < i_ssys):

                                        if print_operations:
                                            print(colored("Distillation failed at level " + str([i_ts, i_ssys, i_op]) + ".", "yellow"))
                                            # print(recipe_op.link_id)

                                        reset_qc = False
                                        if failed_operations is None:
                                            reset_qc = True
                                        elif [i_ts, i_ssys, i_op] != failed_operations[0][1]:
                                            node_times = {node: qc.nodes[node].sub_circuit_time + qc.total_duration
                                                          for node in 'ABCDE'[:prot_rec.n]}
                                            for node in "ABCDE"[:prot_rec.n]:
                                                for sc in qc._sub_circuits.keys():
                                                    if node in sc:
                                                        node_times[node] += qc._sub_circuits[sc].total_duration
                                            current_time = max([node_times[convert_nmb2let(node)] for node in recipe_op.nodes])
                                            # We check if the operation time of the distillation operation that failed
                                            # was after the current time for the newly encountered failed distillation.
                                            # If that was the case, we could actually have known that the newly
                                            # discovered distillation operation had failed at the time of the first
                                            # encountered operation. In that case, we have to reset the quantum circuit
                                            # again with the failed_distillation_time_mark of the newly encountered
                                            # distillation failure.
                                            if round(failed_operations[0][2] - current_time, SUM_ACCURACY) > 0:
                                                reset_qc = True
                                        else:
                                            reset_qc = False

                                        if reset_qc:
                                            if print_operations:
                                                print("\n\n\n\nSTART QC RECOVERY MODE")
                                            reset_from_failed_distillation = True
                                            qc_back_up_failed_distillation_mode = True

                                            node_times = {node: qc.nodes[node].sub_circuit_time + qc.total_duration
                                                          for node in 'ABCDE'[:prot_rec.n]}
                                            for node in "ABCDE"[:prot_rec.n]:
                                                for sc in qc._sub_circuits.keys():
                                                    if node in sc:
                                                        node_times[node] += qc._sub_circuits[sc].total_duration

                                            failed_distillation_time_mark = max([node_times[convert_nmb2let(node)]
                                                                                 for node in recipe_op.nodes])
                                            if print_time_progression:
                                                print(f"Node times: {[qc.nodes[node].sub_circuit_time for node in 'ABCDE'[:prot_rec.n]]}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
                                                print(f"Full node times at this point: {node_times}.")
                                                print(f"Failed distillation time mark: {failed_distillation_time_mark}.")

                                            qc = qc_back_up
                                            ids_carry_out = ids_carry_out_back_up
                                            ids_carry_out_reset = ids_carry_out_reset_back_up
                                            qubit_memory_local = qubit_memory_local_back_up
                                            initialized_qubit_combinations = initialized_qubit_combinations_back_up
                                            carried_out_operations = carried_out_operations_back_up

                                            failed_operations = [(recipe_op, [i_ts, i_ssys, i_op], operation_time, failed_distillation_time_mark)]

                                            i_ssys = 0
                                            i_op = 0
                                            skip_i_op_counter = True

                                            if print_operations:
                                                print(f"The quantum circuits is recovered from level [{i_ts}, {i_ssys}, {i_op}].")
                                                print(f"Carried out operations: {carried_out_operations}.")
                                                print(f"Measurement results: {measurement_results}.")
                                                print(f"Measurement results in time step: {measurement_results_in_ts}.")
                                                print(f"Link creation attempts: {link_creation_attempts}.")
                                            if print_qubit_numbers:
                                                print(f"Qubit numbers: {qubit_numbers}.")
                                                initialized_qubits = list(set(qc.qubits.keys()).difference(
                                                    qc._uninitialised_qubits).difference(
                                                    qc.data_qubits))
                                                print(f"Initialized qubits: {initialized_qubits}, number of initialized qubits: {len(initialized_qubits)}.")
                                            if print_qubit_memory_local:
                                                print(f"Local qubit memory/register: {qubit_memory_local}.")


                                        else:
                                            if [i_ts, i_ssys, i_op] != failed_operations[0][1]:
                                                failed_operations.append((recipe_op, [i_ts, i_ssys, i_op], operation_time, current_time))





                                    else:
                                        # if print_operations:
                                        #     print(colored(
                                        #         "Distillation failed: " + str([i_ts, i_ssys, i_op]) + " becomes " +
                                        #         str([recipe_op.frl[0], recipe_op.frl[1], recipe_op.frl[2]]) + ".",
                                        #         "yellow"))
                                        # skip_i_op_counter = True
                                        # ids_carry_out_reset.append([i_ts, i_ssys, i_op])
                                        # if ids_carry_out:
                                        #     ids_to_reset = [idnr for idnr in recipe_op.fr_list if idnr in ids_carry_out[-1]]
                                        # else:
                                        #     ids_to_reset = recipe_op.fr_list
                                        # ids_carry_out.append(ids_to_reset)
                                        # reset_qubits_after_failure(qc,
                                        #                            prot_rec.qubit_memory_per_time_step[i_ts][i_ssys][i_op],
                                        #                            ids_to_reset,
                                        #                            initialized_qubit_combinations,
                                        #                            print_initialized_qubit_combinations)
                                        #
                                        # carried_out_operations = [coo for coo in carried_out_operations if coo not in ids_to_reset]
                                        # measurement_results = {k: v for k, v in measurement_results.items() if k not in ids_to_reset}
                                        # measurement_results_in_ts = {k: v for k, v in measurement_results_in_ts.items() if k not in ids_to_reset}
                                        # link_creation_attempts = {k: v for k, v in link_creation_attempts.items() if k not in ids_to_reset}
                                        #
                                        # if not carried_out_operations:
                                        #     ids_carry_out.pop()
                                        #     ids_carry_out_reset.pop()
                                        # # [i_ts, i_ssys, i_op] = recipe_op.frl
                                        # if print_operations:
                                        #     print(colored("Carry out = " + str(ids_carry_out) + ", Carry out reset = "
                                        #                   + str(ids_carry_out_reset) + ".", "yellow"))
                                        #
                                        # for node in range(len(qubit_memory_local)):
                                        #     for qb_loc, id_number in enumerate(qubit_memory_local[node]):
                                        #         if id_number in ids_to_reset:
                                        #             qubit_memory_local[node][qb_loc] = None
                                        #
                                        # i_ts = recipe_op.frl[0]
                                        # i_ssys = recipe_op.frl[1]
                                        # i_op = recipe_op.frl[2]
                                        #
                                        # if recipe_op.frl[0] < i_ts or recipe_op.frl[1] < i_ssys:
                                        #     reset_from_failed_distillation = True

                                        skip_i_op_counter = True

                                        ids_carry_out_reset.append([i_ts, i_ssys, i_op])

                                        # Take the subtree of this operation and add it to the list of operations that need to be carried out:
                                        ids_to_reset = deepcopy(recipe_op.family_tree)
                                        ids_to_reset.append(recipe_op.link_id)

                                        # Check if operations THAT ARE NOT SKIPPED carried out after the operation have this operation in their
                                        # subtree. If that is the case, add this operation (and its subtree) to the list of operations that
                                        # need to be carried out:
                                        # I WILL CIRCUMVENT THIS STEP BY JUST CHECKING THE QUBIT_MEMORY/REGISTER FOR WHETHER PARENT NODE IDs OF
                                        # THIS OPERATION (THE NEXT STEP DIRECTLY BELOW): THIS WILL ALSO JUST GIVE US INFORMATION ABOUT WHAT
                                        # OPERATIONS HAVE THIS OPERATION IN THEIR SUBTREE THAT ARE CARRIED OUT IN THE MEANTIME.

                                        # Here we check whether any of the parents of the operation nodes that need to be recreated is already
                                        # executed at this point in time (which is most likely a fusion operation in that case). If that is the
                                        # case, we add this operation (and its subtree) to the list of operations that need to be carried out
                                        # ("ids_to_reset"). The way the qubit_register is created now, if this is the case, the id of the
                                        # parent should be present in the qubit register "qubit_memory_local":
                                        for id_to_reset in ids_to_reset:
                                            parent_id_reset = prot_rec.link_parent_id[id_to_reset]
                                            while parent_id_reset is not None:
                                                for i_node in range(len(qubit_memory_local)):
                                                    for i_qubit in range(len(qubit_memory_local[i_node])):
                                                        if qubit_memory_local[i_node][i_qubit] == parent_id_reset:
                                                            ids_to_reset.append(parent_id_reset)
                                                            pirl = prot_rec.id_link_structure[parent_id_reset]
                                                            ids_to_reset += deepcopy(prot_rec.time_blocks[pirl[0]][
                                                                                         pirl[1]].list_of_operations[
                                                                                         pirl[2]].family_tree)
                                                ids_to_reset = list(dict.fromkeys(ids_to_reset))
                                                parent_id_reset = prot_rec.link_parent_id[parent_id_reset]

                                        # Check if any of the operation nodes that stay in the qubit memory sit "in the way" of the operations
                                        # that need to be recreated; if they do, we also add them to the list of operations that need to be
                                        # recreated:
                                        qubit_memory_local_used_to_uninitialize_qubits = deepcopy(qubit_memory_local)
                                        frl_coor, ids_to_reset = check_conflicts_with_local_qubit_memory(prot_rec,
                                                                                                         qubit_memory_local,
                                                                                                         ids_to_reset,
                                                                                                         i_ts)

                                        carried_out_operations = [coo for coo in carried_out_operations if coo not in ids_to_reset]
                                        measurement_results = {k: v for k, v in measurement_results.items() if k not in ids_to_reset}
                                        measurement_results_in_ts = {k: v for k, v in measurement_results_in_ts.items() if k not in ids_to_reset}
                                        link_creation_attempts = {k: v for k, v in link_creation_attempts.items() if k not in ids_to_reset}

                                        reset_qubits_after_failure(qc, qubit_memory_local_used_to_uninitialize_qubits,
                                                                   ids_to_reset,
                                                                   initialized_qubit_combinations,
                                                                   print_initialized_qubit_combinations)

                                        # In principe, we don't need this for the new functionality (but only for the old functionality)
                                        if frl_coor[0] < i_ts or frl_coor[1] < i_ssys:
                                            reset_from_failed_distillation = True

                                        ids_carry_out.append(ids_to_reset)

                                        if print_operations:
                                            print(colored(
                                                "Because of failed distillation the recipe moves back from "
                                                + str([i_ts, i_ssys, i_op])
                                                + " to level " + str([frl_coor[0], frl_coor[1], frl_coor[2]]) + ".",
                                                "yellow"))
                                            print(colored("Carry out = " + str(ids_carry_out) + ", Carry out reset = "
                                                          + str(ids_carry_out_reset) + ".", "yellow"))
                                            print(f"Carried out operations: {carried_out_operations}.")
                                            print(f"Measurement results: {measurement_results}.")
                                            print(f"Link creation attempts: {link_creation_attempts}.")
                                            # if print_qubit_memory_local:
                                            print(f"Local qubit memory/register: {qubit_memory_local}.")
                                            print(f"Qubit numbers              : {qubit_numbers}.")

                                        i_ts = frl_coor[0]
                                        i_ssys = frl_coor[1]
                                        i_op = frl_coor[2]




                        elif recipe_op.type == "FUSE":
                            # # Option I:
                            cqubit = convert_nmb2let(recipe_op.e_qubits[0][0]) + "-e"
                            tqubit = convert_nmb2let(recipe_op.m_qubits[0][0]) + "-e+" + str(recipe_op.m_qubits[0][1])
                            result1 = qc.H(cqubit)
                            result2 = qc.apply_gate(CZ_gate, cqubit=cqubit, tqubit=tqubit) if type(result1) != SKIP \
                                else SKIP()
                            result3 = qc.H(cqubit) if type(result2) != SKIP else SKIP()
                            qubit_nr = qubit_numbers[recipe_op.e_qubits[0][0]][0]
                            qubit_combs = {qb: qc._qubit_density_matrix_lookup[qb][1] for qb in set(qc.qubits.keys()).difference(qc.data_qubits)}
                            qubit_index = qubit_combs[qubit_nr].index(qubit_nr)
                            if print_initialized_qubit_combinations:
                                print(f"Control qubit is {qubit_nr}. Target qubit is {qubit_numbers[recipe_op.m_qubits[0][0]][recipe_op.m_qubits[0][1]]}.")

                            # qubit_index = initialized_qubit_combinations[qubit_nr][0].index(qubit_nr)
                            # involved_ops = initialized_qubit_combinations[qubit_nr][1][qubit_index]
                            # link_brother = None
                            # for op in involved_ops:
                            #     if op in link_creation_attempts.keys():
                            #         for i_list, list_ops in enumerate(initialized_qubit_combinations[qubit_nr][1]):
                            #             if i_list != qubit_index:
                            #                 if op in list_ops:
                            #                     if link_brother is not None:
                            #                         raise ValueError
                            #                     link_brother = (op, i_list)
                            # if qubit_index == 0 and link_brother is not None:
                            #     print(f"Link creation id {link_brother[0]} is in the right order.")
                            # elif qubit_index >= 1 and link_brother is not None:
                            #     print(f"Link creation id {link_brother[0]} is in the wrong order: brother on index {link_brother[1]}.")
                            # else:
                            #     link_creation = None
                            #     for op in involved_ops:
                            #         if op in link_creation_attempts.keys():
                            #             if link_creation is not None:
                            #                 raise ValueError
                            #             link_creation = op
                            #     print(f"Link creation id {link_creation} doesn't matter.")
                            #
                            # # if print_initialized_qubit_combinations and qubit_index != 0:
                            # if qubit_index != 0:
                            #     print(f"Qubit index {qubit_index} is measured, with qubit combinations {qubit_combs[qubit_nr]}.")

                            if try_out_new_functionality \
                                    and qc_back_up_failed_distillation_mode \
                                    and recipe_op.link_id in measurement_results:
                                    # and i_ssys <= failed_operations[0][1][1]:
                                result = measurement_results[recipe_op.link_id]
                                qc.measure(cqubit, outcome=result[0], probabilistic=False, basis="Z") \
                                    if type(result3) != SKIP else SKIP()
                            else:
                                result = qc.measure(cqubit, basis="Z") if type(result3) != SKIP else SKIP()
                                measurement_results[recipe_op.link_id] = result if type(result) != SKIP else None
                            # # Option II:
                            # cqubit = convert_nmb2let(recipe_op.m_qubits[0][0]) + "-e+" + str(recipe_op.m_qubits[0][1])
                            # tqubit = convert_nmb2let(recipe_op.e_qubits[0][0]) + "-e"
                            # qc.apply_gate(CNOT_gate, cqubit=cqubit, tqubit=tqubit, electron_is_target=True)
                            # measurement_results[recipe_op.link_id] = qc.measure(tqubit, basis="Z")[0]
                            if print_operations:
                                print(colored("Measurement: " + str(measurement_results[recipe_op.link_id])
                                              + ". All results are: " + str(measurement_results) + ".", "yellow"))
                                print(f"Measurement results in this time step: {measurement_results_in_ts}.")
                            for e_qubit in recipe_op.e_qubits:
                                qubit_memory_local[e_qubit[0]][e_qubit[1]] = None
                            for node in range(len(qubit_memory_local)):
                                for qb_loc, id_number in enumerate(qubit_memory_local[node]):
                                    if id_number in recipe_op.family_tree:
                                        qubit_memory_local[node][qb_loc] = recipe_op.link_id
                            # Extra structure to try to have the measurement qubit as first qubit as often as possible
                            c_qb = qubit_numbers[recipe_op.e_qubits[0][0]][0]
                            t_qb = qubit_numbers[recipe_op.m_qubits[0][0]][recipe_op.m_qubits[0][1]]
                            state1 = initialized_qubit_combinations[c_qb]
                            # print(t_qb, state1)
                            # print(initialized_qubit_combinations)
                            # if recipe_op.link_id == 19:
                            #     raise ValueError
                            if t_qb not in state1[0]:
                                # Combine the states
                                state2 = initialized_qubit_combinations[t_qb]
                                for i_qb, qb in enumerate(state2[0]):
                                    state1[0].append(qb)
                                    state1[1].append(state2[1][i_qb])
                                    initialized_qubit_combinations[qb] = state1
                                initialized_qubit_combinations[t_qb] = state1
                            # state1[1][state1[0].index(t_qb)].append(recipe_op.link_id)
                            del state1[1][state1[0].index(c_qb)]
                            del state1[0][state1[0].index(c_qb)]
                            initialized_qubit_combinations[c_qb] = [[c_qb], []]



                        if print_time_progression and not reset_from_failed_distillation:
                            print(f"New node times after operation: {[qc.nodes[node].sub_circuit_time for node in 'ABCDE'[:prot_rec.n]]}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
                            node_times_new = {node: qc.nodes[node].sub_circuit_time + qc.total_duration for node in 'ABCDE'[:prot_rec.n]}
                            for node in "ABCDE"[:prot_rec.n]:
                                for sc in qc._sub_circuits.keys():
                                    if node in sc:
                                        node_times_new[node] += qc._sub_circuits[sc].total_duration
                            print(f"New full node times at this point: {node_times_new}.")
                            print(f"Failed distillation time mark: {failed_distillation_time_mark}.")
                            print(f"Current subcircuit: {qc._current_sub_circuit._name}.")

                        if print_qubit_memory_local and not reset_from_failed_distillation:
                            print(f"Local qubit memory/register: {qubit_memory_local}.")
                            print(f"Initialized qubits         : {prot_rec.qubit_memory_per_time_step[i_ts][i_ssys][i_op]}.")
                        if print_qubit_numbers and not reset_from_failed_distillation:
                            initialized_qubits = list(set(qc.qubits.keys()).difference(qc._uninitialised_qubits).difference(
                                qc.data_qubits))
                            local_list_initialized_qubits = []
                            for i_node, node in enumerate(qubit_memory_local):
                                for i_nmb, nmb in enumerate(node):
                                    if nmb is not None:
                                        local_list_initialized_qubits.append(qubit_numbers[i_node][i_nmb])
                            print(f"Local initialized qubit nrs: {sorted(local_list_initialized_qubits)}, number of initialized qubits: {len(local_list_initialized_qubits)}.")
                            print(f"Initialized qubit numbers  : {initialized_qubits}, number of initialized qubits: {len(initialized_qubits)}.")
                            print(f"Qubit numbers              : {qubit_numbers}.")
                        if print_initialized_qubit_combinations:
                            print(f"Automatic list init qubits : " + str({qb: qc._qubit_density_matrix_lookup[qb][1] for qb in set(qc.qubits.keys()).difference(qc.data_qubits)}))
                            print(f"My own list of init qubits : " + str({k: v[0] for k, v in initialized_qubit_combinations.items()}))
                            print(f"Full list of init qubits   : {initialized_qubit_combinations}.")
                            # print(f"My own list of init qubits : {initialized_qubit_combinations}.")


                        if type(result) == SKIP:
                            sub_block_cut_off_time_reached = True
                            time_step_cut_off_time_reached = True

                    if not skip_i_op_counter:
                        i_op += 1
            if not reset_from_failed_distillation:
                i_ssys += 1



        # if print_time_progression:
        #     print(f"Node times: {[qc.nodes[node].sub_circuit_time for node in 'ABCD']}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
        # This is the end of the time step.
        if time_step_cut_off_time_reached:
            qc.cut_off_time_reached = True






        while not reset_from_failed_distillation \
                and ids_carry_out \
                and [i_ts, "after"] == ids_carry_out_reset[-1] \
                and time.time() - start_time < MAX_TIME:
            ids_carry_out.pop()
            ids_carry_out_reset.pop()
            if print_operations:
                print(colored("New carry out = " + str(ids_carry_out) + ", Carry out reset = "
                              + str(ids_carry_out_reset) + ".", "yellow"))




        if qc_back_up_failed_distillation_mode \
                and not reset_from_failed_distillation \
                and time.time() - start_time < MAX_TIME:
            # The end of the qc recovery steps after a failed distillation step

            # We should have a list with skipped operations because of the failed distillation time stamp here
            # Let's call this list "skipped_operations_outside_time_stamp"
            # Watch out! I think this should only be the operations that are skipped because of the time stamp; not
            # the ones that are skipped because they weren't contained in carry_out[-1].

            if print_operations:
                print("Before qc recovery mode is ended:")
                print(f"Local qubit memory/register: {qubit_memory_local}.")

            ids_to_reset = []

            for failed_op in failed_operations:
                recipe_op = failed_op[0]
                [i_ts_f, i_ssys_f, i_op_f] = failed_op[1]

                # Take the subtree of this operation and add it to the list of operations that need to be carried out:
                ids_to_reset += deepcopy(recipe_op.family_tree)
                ids_to_reset.append(recipe_op.link_id)

                # Check if operations THAT ARE NOT SKIPPED carried out after the operation have this operation in their
                # subtree. If that is the case, add this operation (and its subtree) to the list of operations that
                # need to be carried out:
                # I WILL CIRCUMVENT THIS STEP BY JUST CHECKING THE QUBIT_MEMORY/REGISTER FOR WHETHER PARENT NODE IDs OF
                # THIS OPERATION (THE NEXT STEP DIRECTLY BELOW): THIS WILL ALSO JUST GIVE US INFORMATION ABOUT WHAT
                # OPERATIONS HAVE THIS OPERATION IN THEIR SUBTREE THAT ARE CARRIED OUT IN THE MEANTIME.

            # Here we check whether any of the parents of the operation nodes that need to be recreated is already
            # executed at this point in time (which is most likely a fusion operation in that case). If that is the
            # case, we add this operation (and its subtree) to the list of operations that need to be carried out
            # ("ids_to_reset"). The way the qubit_register is created now, if this is the case, the id of the
            # parent should be present in the qubit register "qubit_memory_local":
            for id_to_reset in ids_to_reset:
                parent_id_reset = prot_rec.link_parent_id[id_to_reset]
                while parent_id_reset is not None:
                    for i_node in range(len(qubit_memory_local)):
                        for i_qubit in range(len(qubit_memory_local[i_node])):
                            if qubit_memory_local[i_node][i_qubit] == parent_id_reset:
                                ids_to_reset.append(parent_id_reset)
                                pirl = prot_rec.id_link_structure[parent_id_reset]
                                ids_to_reset += deepcopy(prot_rec.time_blocks[pirl[0]][pirl[1]].list_of_operations[pirl[2]].family_tree)
                    ids_to_reset = list(dict.fromkeys(ids_to_reset))
                    parent_id_reset = prot_rec.link_parent_id[parent_id_reset]

            # We now add the operations that are skipped because they fall outside the failed distillation time stamp
            # to the list of "ids_to_reset". We don't have to add their family trees, because we just need these
            # operations to actually be carried out when we get back at this point in the recipe. (Unless, of course,
            # they sit on qubits that we need to recreate "ids_to_reset", in which case we also encounter them in the
            # functionality below and add their subtrees there.)
            if print_operations:
                print(f"ids_to_reset = {ids_to_reset}.")
                print(f"skipped_operations_outside_time_stamp = {skipped_operations_outside_time_stamp}.")
            # ids_to_reset = list(dict.fromkeys(ids_to_reset + skipped_operations_outside_time_stamp))

            keep_measurement_results = {}
            for skipped_op in skipped_operations_outside_time_stamp:
                if not isinstance(skipped_op, list):
                    if skipped_op not in ids_to_reset:
                        ids_to_reset.append(skipped_op)
                        if skipped_op in measurement_results.keys() and None in measurement_results[skipped_op]:
                            # Get the nodes of the operation
                            sol = prot_rec.id_link_structure[skipped_op]
                            skipped_operation = prot_rec.time_blocks[sol[0]][sol[1]].list_of_operations[sol[2]]
                            skipped_op_nodes = [qb[0] for qb in skipped_operation.e_qubits]
                            # Identify in which nodes we want to keep the measurement results
                            nodes_to_keep = []
                            for imr, meas_result in enumerate(measurement_results[skipped_op]):
                                if meas_result is not None:
                                    nodes_to_keep.append(skipped_op_nodes[imr])
                            keep_measurement_results[skipped_op] = nodes_to_keep

            for skipped_op in skipped_operations_outside_time_stamp:
                if isinstance(skipped_op, list):
                    add_swap_gate_entry = False
                    for skipped_id in skipped_op[1]:
                        if skipped_id is not None and skipped_id not in ids_to_reset:
                            add_swap_gate_entry = True
                            break
                    if add_swap_gate_entry:
                        ids_to_reset.append(skipped_op)
            if print_operations:
                print(f"New ids_to_reset after combining with skipped_operations_outside_time_stamp = {ids_to_reset}.")

            # Check if any of the operation nodes that stay in the qubit memory sit "in the way" of the operations
            # that need to be recreated; if they do, we also add them to the list of operations that need to be
            # recreated:
            qubit_memory_local_used_to_uninitialize_qubits = deepcopy(qubit_memory_local)
            frl_coor, ids_to_reset = check_conflicts_with_local_qubit_memory(prot_rec, qubit_memory_local,
                                                                             ids_to_reset, i_ts,
                                                                             keep_measurement_results)

            carried_out_operations = [coo for coo in carried_out_operations if coo not in ids_to_reset]
            measurement_results = {k: v for k, v in measurement_results.items()
                                   if (k not in ids_to_reset or k in keep_measurement_results.keys())}
            link_creation_attempts = {k: v for k, v in link_creation_attempts.items() if k not in ids_to_reset}

            if not carried_out_operations:
                ids_carry_out = []
                ids_carry_out_reset = []
            else:
                ids_carry_out_reset.append([i_ts, "after"])
                ids_carry_out.append(ids_to_reset)
            if print_operations:
                print(colored("Because of failed distillation the recipe moves back to level " + str([frl_coor[0], frl_coor[1], frl_coor[2]]) + ".", "yellow"))
                print(colored("Carry out = " + str(ids_carry_out) + ", Carry out reset = "
                              + str(ids_carry_out_reset) + ".", "yellow"))
                print(f"Carried out operations: {carried_out_operations}.")
                print(f"Measurement results: {measurement_results}.")
                print(f"Link creation attempts: {link_creation_attempts}.")
            # if print_qubit_memory_local:
                print(f"Local qubit memory/register: {qubit_memory_local}.")
                print(f"Qubit numbers              : {qubit_numbers}.")

            reset_qubits_after_failure(qc, qubit_memory_local_used_to_uninitialize_qubits, ids_to_reset,
                                       initialized_qubit_combinations, print_initialized_qubit_combinations)
            # for node in range(len(qubit_memory_local_used_to_uninitialize_qubits)):
            #     for qb_loc, id_number in enumerate(qubit_memory_local_used_to_uninitialize_qubits[node]):
            #         if id_number in ids_to_reset:
            #             qubit = convert_nmb2let(node) + "-e"
            #             if qb_loc != 0:
            #                 qubit += "+" + str(qb_loc)
            #             qc._reset_density_matrices([qubit])
                        # qubit_in_nr = qubit_numbers[node][qb_loc]
                        # state1 = initialized_qubit_combinations[qubit_in_nr]
                        # for qb in state1[0]:
                        #     initialized_qubit_combinations[qb] = [[qb], []]
                        #     # state2 = initialized_qubit_combinations[qb]
                        #     # if qubit_in_nr in state2[0]:
                        #     #     del state2[1][state2[0].index(qubit_in_nr)]
                        #     #     del state2[0][state2[0].index(qubit_in_nr)]
                        # initialized_qubit_combinations[qubit_in_nr] = [[qubit_in_nr], []]

            i_ts = frl_coor[0]
            i_ssys = frl_coor[1]
            i_op = frl_coor[2]

            qc_back_up_failed_distillation_mode = False
            failed_distillation_time_mark = None
            failed_operations = None
            reset_from_failed_distillation = True

            if print_operations:
                print("END QC RECOVERY MODE\n\n\n\n")

        if print_operations and time.time() - start_time < MAX_TIME:
            print(f"End of timestep: [i_ts, i_ssys, i_op] = [{i_ts, i_ssys, i_op}].")






        # DISTILLATION POST-SELECT OPERATIONS THAT HAVE TO BE CARRIED OUT AFTER A TIME STEP:
        if not reset_from_failed_distillation \
                and not qc_back_up_failed_distillation_mode \
                and time.time() - start_time < MAX_TIME:
            for dist_id in prot_rec.delayed_distillation_check[i_ts]:
                skip_evaluation = False
                if ids_carry_out:
                    if dist_id not in ids_carry_out[-1]:
                        skip_evaluation = True

                while ids_carry_out and [i_ts, dist_id, "after"] == ids_carry_out_reset[-1]:
                    ids_carry_out.pop()
                    ids_carry_out_reset.pop()
                    if print_operations:
                        print(colored("New carry out = " + str(ids_carry_out) + ", Carry out reset = "
                                      + str(ids_carry_out_reset) + ".", "yellow"))

                if not skip_evaluation:
                    oper_coor = prot_rec.id_link_structure[dist_id]
                    oper_frl = prot_rec.time_blocks[oper_coor[0]][oper_coor[1]].list_of_operations[oper_coor[2]].frl
                    oper_fr_list = prot_rec.time_blocks[oper_coor[0]][oper_coor[1]].list_of_operations[oper_coor[2]].fr_list
                    success = 0
                    for succ_dep in prot_rec.delayed_distillation_check[i_ts][dist_id]:
                        success = (success + measurement_results[succ_dep].count(1)) % 2 \
                                if (qc.cut_off_time_reached is False and success is not None and
                                    measurement_results[succ_dep] is not None) else None
                    if success != 0:
                        if print_operations:
                            print(colored("Distillation failed: " + str([i_ts, dist_id, "after"]) + " becomes " +
                                  str([oper_frl[0], oper_frl[1], oper_frl[2]]) + ".", "yellow"))
                        ids_carry_out_reset.append([i_ts, dist_id, "after"])
                        if ids_carry_out:
                            ids_to_reset = [idnr for idnr in oper_fr_list if idnr in ids_carry_out[-1]]
                        else:
                            ids_to_reset = oper_fr_list
                        ids_carry_out.append(ids_to_reset)

                        nr_sb = len(prot_rec.qubit_memory_per_time_step[i_ts])
                        nr_op = len(prot_rec.qubit_memory_per_time_step[i_ts][nr_sb-1])

                        reset_qubits_after_failure(qc,
                                                   prot_rec.qubit_memory_per_time_step[i_ts][nr_sb-1][nr_op-1],
                                                   ids_to_reset,
                                                   initialized_qubit_combinations,
                                                   print_initialized_qubit_combinations)
                        # for node in range(len(prot_rec.qubit_memory_per_time_step[i_ts][nr_sb-1][nr_op-1])):
                        #     for mem_qb in range(len(prot_rec.qubit_memory_per_time_step[i_ts][nr_sb-1][nr_op-1][node])):
                        #         if prot_rec.qubit_memory_per_time_step[i_ts][nr_sb-1][nr_op-1][node][mem_qb] in ids_to_reset:
                        #             qubit = convert_nmb2let(node) + "-e"
                        #             if mem_qb != 0:
                        #                 qubit += "+" + str(mem_qb)
                        #             qc._reset_density_matrices([qubit])
                                    # qubit_in_nr = qubit_numbers[node][mem_qb]
                                    # state1 = initialized_qubit_combinations[qubit_in_nr]
                                    # for qb in state1[0]:
                                    #     initialized_qubit_combinations[qb] = [[qb], []]
                                    #     # state2 = initialized_qubit_combinations[qb]
                                    #     # if qubit_in_nr in state2[0]:
                                    #     #     del state2[1][state2[0].index(qubit_in_nr)]
                                    #     #     del state2[0][state2[0].index(qubit_in_nr)]
                                    # initialized_qubit_combinations[qubit_in_nr] = [[qubit_in_nr], []]

                        for node in range(len(qubit_memory_local)):
                            for qb_loc, id_number in enumerate(qubit_memory_local[node]):
                                if id_number in ids_to_reset:
                                    qubit_memory_local[node][qb_loc] = None

                        # for node in range(len(prot_rec.qubit_memory_per_time_step[oper_coor[0]][oper_coor[1]][oper_coor[2]])):
                        #     for mem_qb in range(len(prot_rec.qubit_memory_per_time_step[oper_coor[0]][oper_coor[1]][oper_coor[2]][node])):
                        #         if prot_rec.qubit_memory_per_time_step[oper_coor[0]][oper_coor[1]][oper_coor[2]][node][mem_qb] in ids_to_reset:
                        #             qubit = convert_nmb2let(node) + "-e"
                        #             if mem_qb != 0:
                        #                 qubit += "+" + str(mem_qb)
                        #             qc._reset_density_matrices([qubit])
                        i_ts = oper_frl[0]
                        i_ssys = oper_frl[1]
                        i_op = oper_frl[2]
                        carried_out_operations = [coo for coo in carried_out_operations if coo not in ids_to_reset]
                        measurement_results = {k: v for k, v in measurement_results.items() if k not in ids_to_reset}
                        link_creation_attempts = {k: v for k, v in link_creation_attempts.items() if k not in ids_to_reset}
                        if not carried_out_operations:
                            ids_carry_out.pop()
                            ids_carry_out_reset.pop()
                        reset_from_failed_distillation = True
                        if print_operations:
                            print(colored("Carry out = " + str(ids_carry_out) + ", Carry out reset = "
                                          + str(ids_carry_out_reset) + ".", "yellow"))
                        if print_qubit_memory_local:
                            print(f"Local qubit memory/register: {qubit_memory_local}.")
                        break





        # FUSION CORRECTIONS:
        if not reset_from_failed_distillation \
                and not qc_back_up_failed_distillation_mode \
                and time.time() - start_time < MAX_TIME:
            sub_circuit_opened = False
            for fc_qubit in prot_rec.fusion_corrections[i_ts]:
                fusion_correction = prot_rec.fusion_corrections[i_ts][fc_qubit]

                carry_out_correction = [False, False]
                if ids_carry_out:
                    for corr_gate in range(2):
                        for succ_dep in fusion_correction.condition[corr_gate]:
                            if succ_dep in ids_carry_out[-1]:
                                carry_out_correction[corr_gate] = True
                                break
                else:
                    carry_out_correction = [True, True]
                # carry_out_correction = [True, True]

                if print_operations and True in carry_out_correction:
                    fusion_correction.print_fusion_correction()

                corr_qubit = convert_nmb2let(fusion_correction.qubit[0]) + "-e"
                if fusion_correction.qubit[1] != 0:
                    corr_qubit += "+" + str(fusion_correction.qubit[1])
                for corr_gate in range(2):
                    if carry_out_correction[corr_gate]:
                        correction = 0
                        for succ_dep in fusion_correction.condition[corr_gate]:
                            correction = (correction + measurement_results[succ_dep].count(1)) % 2 \
                                if (qc.cut_off_time_reached is False and correction is not None and
                                    measurement_results[succ_dep] is not None) else None
                        if correction != 0:
                            if sub_circuit_opened is False:
                                if print_time_progression:
                                    print(f"Node times: {[qc.nodes[node].sub_circuit_time for node in 'ABCDE'[:prot_rec.n]]}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
                                if prot_rec.n == 2:
                                    qc.start_sub_circuit("AB")
                                elif prot_rec.n == 3:
                                    qc.start_sub_circuit("ABC")
                                elif prot_rec.n == 4:
                                    qc.start_sub_circuit("ABCD")
                                elif prot_rec.n == 5:
                                    qc.start_sub_circuit("ABCDE")
                                if print_operations:
                                    print(colored("SUBSYSTEM " + str([*range(prot_rec.n)]) + ":", "red"))
                                if print_time_progression:
                                    print(f"Node times: {[qc.nodes[node].sub_circuit_time for node in 'ABCDE'[:prot_rec.n]]}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
                            sub_circuit_opened = True
                            if corr_gate == 0:
                                if print_operations:
                                    print(colored("Qubit " + str(fusion_correction.qubit)
                                                  + " is corrected with operator Z.", "yellow"))
                                if print_time_progression:
                                    node_time = qc.nodes[convert_nmb2let(fusion_correction.qubit[0])].sub_circuit_time
                                qc.Z(corr_qubit)
                            elif corr_gate == 1:
                                if print_operations:
                                    print(colored("Qubit " + str(fusion_correction.qubit)
                                                  + " is corrected with operator X.", "yellow"))
                                if print_time_progression:
                                    node_time = qc.nodes[convert_nmb2let(fusion_correction.qubit[0])].sub_circuit_time
                                qc.X(corr_qubit)




        # Move to the next time step:
        if not reset_from_failed_distillation and time.time() - start_time < MAX_TIME:
            i_ts += 1

    # We have to measure out some qubits that are not part of the final GHZ states, but are still initialized (because
    # they were entangled with a state that we didn't want to through away via an inefficient SWAP gate):
    # These are the GHZ state qubits:
    ghz_qubits = []
    for i_node, node in enumerate(prot_rec.qubit_memory_per_time_step[-1][-1][-1]):
        for i_qb, qb in enumerate(node):
            if qb is not None:
                ghz_qubits.append(qubit_numbers[i_node][i_qb])
    # for i_node, node in enumerate(qubit_memory_local):
    #     for i_qb, qb in enumerate(node):
    #         if qb is not None and prot_rec.qubit_memory_per_time_step[-1][-1][-1][i_node][i_qb] is None:
    #             qc.measure(qubit_numbers[i_node][i_qb], outcome=0, p_m=0.0, basis="Z", probabilistic=False, noise=False)
    initialized_qubits = list(set(qc.qubits.keys()).difference(qc._uninitialised_qubits).difference(qc.data_qubits))
    for qb in initialized_qubits:
        if qb not in ghz_qubits:
            qc.measure(qb, outcome=0, p_m=0.0, basis="Z", probabilistic=False, noise=False)

    if print_operations:
        print(f"Number of link creations: {len(link_creation_attempts.keys())}.")

    if print_time_progression and time.time() - start_time < MAX_TIME:
        print(f"\n\n\nNode times: {[qc.nodes[node].sub_circuit_time for node in 'ABCDE'[:prot_rec.n]]}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.\n\n\n")

    if time.time() - start_time >= MAX_TIME:
        raise TimeoutError("Circuit took too long to complete.")
    elif prot_rec.n == 2:
        # qc.stabilizer_measurement(operation, nodes=["A", "B"], swap=True)
        return qc, ["A", "B"]
    elif prot_rec.n == 3:
        # qc.stabilizer_measurement(operation, nodes=["A", "B", "C"], swap=True)
        return qc, ["A", "B", "C"]
    elif prot_rec.n == 4:
        # qc.stabilizer_measurement(operation, nodes=["A", "B", "C", "D"], swap=True)
        return qc, ["A", "B", "C", "D"]
    elif prot_rec.n == 5:
        return qc, ["A", "B", "C", "D", "E"]

    # print(qc._qubit_density_matrix_lookup)


def print_density_matrices(qc):
    # print("hello")
    objects = {}
    for object in qc._qubit_density_matrix_lookup.keys():
        if tuple(qc._qubit_density_matrix_lookup[object][1]) not in objects:
            objects[tuple(qc._qubit_density_matrix_lookup[object][1])] = qc.get_combined_density_matrix(qc._qubit_density_matrix_lookup[object][1])[0] # qc._qubit_density_matrix_lookup[object][0]
    for object in objects:
        print(f"{object}: \n{objects[object]}")


def reset_qubits_after_failure(qc, quantum_mem, ids_to_reset, initialized_qubit_combinations, print_initialized_qubit_combinations):
    qb_combs = {qb: qc._qubit_density_matrix_lookup[qb][1] for qb in set(qc.qubits.keys()).difference(qc.data_qubits)}

    to_reset = []
    to_reset_numbs = []
    for node in range(len(quantum_mem)):
        for mem_qb in range(len(quantum_mem[node])):
            if quantum_mem[node][mem_qb] in ids_to_reset:
                to_reset.append((node, mem_qb))
                to_reset_numbs.append(qc.qubit_numbers[node][mem_qb])
                # qubit = convert_nmb2let(node) + "-e"
                # if mem_qb != 0:
                #     qubit += "+" + str(mem_qb)
                # qc._reset_density_matrices([qubit])

    skip_reset = []
    for qb1 in to_reset:
        reset_qb = True
        for qb2 in qb_combs[qc.qubit_numbers[qb1[0]][qb1[1]]]:
            if qb2 not in to_reset_numbs:
                reset_qb = False
                break
        if reset_qb is False:
            skip_reset.append(qb1)

    for qb in to_reset:
        if qb not in skip_reset:
            qubit = convert_nmb2let(qb[0]) + "-e"
            if qb[1] != 0:
                qubit += "+" + str(qb[1])
            qc._reset_density_matrices([qubit])

            qubit_in_nr = qc.qubit_numbers[qb[0]][qb[1]]
            # state1 = initialized_qubit_combinations[qubit_in_nr]
            # for qb in state1[0]:
            #     initialized_qubit_combinations[qb] = [[qb], []]
            #     state2 = initialized_qubit_combinations[qb]
            # if qubit_in_nr in state2[0]:
            #     del state2[1][state2[0].index(qubit_in_nr)]
            # del state2[0][state2[0].index(qubit_in_nr)]
            initialized_qubit_combinations[qubit_in_nr] = [[qubit_in_nr], []]

    if print_initialized_qubit_combinations:
        # if skip_reset:
        #     print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nIt happens.\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        print(f"to_reset: {to_reset}.")
        print(f"to_reset (qubit numbers): {[qc.qubit_numbers[qb[0]][qb[1]] for qb in to_reset]}")
        print(f"skip_reset: {skip_reset}.")
        print(f"skip_reset (qubit numbers): {[qc.qubit_numbers[qb[0]][qb[1]] for qb in skip_reset]}")
        print(f"Automatic list init qubits : " + str({qb: qc._qubit_density_matrix_lookup[qb][1] for qb in set(qc.qubits.keys()).difference(qc.data_qubits)}))


#### REMOVE THIS FUNCTION IN "protocol_recipe.py"
def check_conflicts_with_local_qubit_memory(protocol_recipe, qubit_memory_local, links_to_be_reset, end_level_i_ts,
                                            operation_measurements_to_keep=None):
    # Now, we check if we also need to recreate any other links that happen to sit on the memory
    # locations that we require to make the links in links_to_be_reset:
    found_frl = False
    qubit_register = qubit_memory_local
    frl_coor = None

    # Here, we select the time step, sub circuit and operation number level of the last operation in this time step.
    # This part can be done simpler, but this is already made more general for when there can be more than 2 sub
    # circuits:
    end_level_i_ssys = 0
    while len(protocol_recipe.time_blocks[end_level_i_ts]) > end_level_i_ssys and \
            len(protocol_recipe.time_blocks[end_level_i_ts][end_level_i_ssys].list_of_operations) > 0:
            # len(protocol_recipe.time_blocks[end_level_i_ts][end_level_i_ssys + 1].list_of_operations) > 0:
        end_level_i_ssys += 1
    end_level_i_ssys -= 1
    end_level_i_op = len(protocol_recipe.time_blocks[end_level_i_ts][end_level_i_ssys].list_of_operations) - 1
    end_operation_level = [end_level_i_ts, end_level_i_ssys, end_level_i_op]

    while found_frl is False:
        # We remove the link_id's of all links that we are going to recreate from our local qubit register/memory:
        qubit_register = remove_link_ids_from_qubit_memory(qubit_register, links_to_be_reset,
                                                           operation_measurements_to_keep)
        failure_reset_level_link = None
        for l_id in protocol_recipe.id_link_structure:
            if l_id in links_to_be_reset:
                # This is the first link of "links_to_be_reset" that is created in the protocol.
                failure_reset_level_link = l_id
                break
        frl_coor = protocol_recipe.id_link_structure[failure_reset_level_link]

        # Here we collect all link_id's of links between "frl_coor" and the end of the time step "time_step"
        # that use the same memory locations as the links we are going to recreate. Therefore, we
        # also need to recreate these links.
        overlap_links = check_overlap_with_qubit_register(protocol_recipe, qubit_register, frl_coor,
                                                          end_operation_level, links_to_be_reset)

        if len(overlap_links) == 0:
            # If there were no overlapping links found, we know that this is the true failure reset
            # level for this version of the protocol.
            found_frl = True
        else:
            for link_id in overlap_links:
                link_c = protocol_recipe.id_link_structure[link_id]
                operation2 = protocol_recipe.time_blocks[link_c[0]][link_c[1]].list_of_operations[link_c[2]]
                if operation2.family_tree is not None:
                    # To recreate all overlapping links, we have to of course first create all the
                    # links in its subtree.
                    links_to_be_reset += deepcopy(operation2.family_tree)
                links_to_be_reset.append(link_id)

                if operation_measurements_to_keep:
                    # There exists a scenario where this state is already partially used up by a parent state (when this
                    # parent state couldn't complete the full operation because it was interrupted by a failed
                    # distillation operation in one the nodes). To check if this is the case, we first check if any of
                    # the parents of this state are currently in the qubit register. To do this, we must make sure that
                    # the parent found is actually in the same nodes as the current state (to prevent removing
                    # partially created parents that actually haven't used the current state yet):
                    nodes_operation = [qb[0] for qb in operation2.e_qubits]

                    # Now, we loop over these nodes in the qubit_register, and if we find any parents, we add the
                    # parent state id and their full family tree also the list of links to be reset:
                    parent_id_reset = protocol_recipe.link_parent_id[link_id]
                    while parent_id_reset is not None:
                        for i_node in nodes_operation:
                            for i_qubit in range(len(qubit_register[i_node])):
                                if qubit_register[i_node][i_qubit] == parent_id_reset:
                                    links_to_be_reset.append(parent_id_reset)
                                    pirl = protocol_recipe.id_link_structure[parent_id_reset]
                                    links_to_be_reset += deepcopy(protocol_recipe.time_blocks[pirl[0]][pirl[1]].
                                                                  list_of_operations[pirl[2]].family_tree)
                                    if parent_id_reset in operation_measurements_to_keep.keys() and i_node in operation_measurements_to_keep[parent_id_reset]:
                                        index_meas = operation_measurements_to_keep[parent_id_reset].index(i_node)
                                        del operation_measurements_to_keep[parent_id_reset][index_meas]
                                        if len(operation_measurements_to_keep[parent_id_reset]) == 0:
                                            del operation_measurements_to_keep[parent_id_reset]
                        parent_id_reset = protocol_recipe.link_parent_id[parent_id_reset]

                # We remove possible duplicates:
                links_to_be_reset_final = []
                for link in links_to_be_reset:
                    if link not in links_to_be_reset_final:
                        links_to_be_reset_final.append(link)
                # links_to_be_reset = list(dict.fromkeys(links_to_be_reset))
                links_to_be_reset = links_to_be_reset_final

    return frl_coor, links_to_be_reset


###### REMOVE THIS FUNCTION IN "protocol_recipe_sub_functions.py"
def remove_link_ids_from_qubit_memory(qubit_memory, link_id_list, operation_measurements_to_keep=None):
    if operation_measurements_to_keep is None:
        operation_measurements_to_keep = {}
    for i_node, node in enumerate(qubit_memory):
        for i_qubit, qubit in enumerate(node):
            id_in_mem = qubit_memory[i_node][i_qubit]
            if id_in_mem in link_id_list:
                if id_in_mem not in operation_measurements_to_keep.keys() \
                        or i_node not in operation_measurements_to_keep[id_in_mem]:
                    qubit_memory[i_node][i_qubit] = None
    return qubit_memory


###### REMOVE THIS FUNCTION IN "protocol_recipe.py"
def check_overlap_with_qubit_register(protocol_recipe, qubit_register, start_ts, end_ts, links_to_check):
    overlapping_links = []
    i_ts = start_ts[0]
    i_ssys = start_ts[1]
    i_op = start_ts[2]
    end_reached = False
    while not end_reached:
        if protocol_recipe.time_blocks[i_ts][i_ssys].list_of_operations:
            operation = protocol_recipe.time_blocks[i_ts][i_ssys].list_of_operations[i_op]
            if operation.type != "SWAP":
                if operation.link_id in links_to_check:
                    qubits = [operation.e_qubits, operation.m_qubits]
                    for type_qubits in range(2):
                        if qubits[type_qubits] is not None:
                            for [node, qubit] in qubits[type_qubits]:
                                if qubit_register[node][qubit] is not None: # and qubit_register[node][qubit] not in links_that_stay:
                                    overlapping_links.append(qubit_register[node][qubit])
            elif operation.type == "SWAP":
                if operation.link_id[0] in links_to_check or operation.link_id[1] in links_to_check:
                    # If one of the link_id's that is used in the SWAP operation is actually a link_id that we have
                    # to recreate, we have to carry out this SWAP gate, and we have to include the link_id's that
                    # are stored at the qubit memory locations that the SWAP acts on, if they aren't already a part
                    # of links_to_check:
                    [[node1, qubit1], [node2, qubit2]] = [operation.e_qubits[0], operation.m_qubits[0]]
                    if qubit_register[node1][qubit1] is not None and qubit_register[node1][qubit1] not in links_to_check:
                    # if qubit_register[node1][qubit1] is not None and qubit_register[node1][qubit1] not in links_that_stay:
                        overlapping_links.append(qubit_register[node1][qubit1])
                    if qubit_register[node2][qubit2] is not None and qubit_register[node2][qubit2] not in links_to_check:
                    # if qubit_register[node2][qubit2] is not None and qubit_register[node2][qubit2] not in links_that_stay:
                        overlapping_links.append(qubit_register[node2][qubit2])
            if [i_ts, i_ssys, i_op] == end_ts:
                end_reached = True
            i_op += 1
        if i_op == len(protocol_recipe.time_blocks[i_ts][i_ssys].list_of_operations):
            i_ssys += 1
            i_op = 0
            if i_ssys == len(protocol_recipe.time_blocks[i_ts]):
                i_ts += 1
                i_ssys = 0
    return list(dict.fromkeys(overlapping_links))
