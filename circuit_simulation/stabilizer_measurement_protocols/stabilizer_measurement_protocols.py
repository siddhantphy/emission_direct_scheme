from circuit_simulation.circuit_simulator import *
from tqdm import tqdm
PBAR: tqdm = None


def create_quantum_circuit(protocol, pbar, **kwargs):
    """
        Initialises a QuantumCircuit object corresponding to the protocol requested.

        Parameters
        ----------
        protocol : str
            Name of the protocol for which the QuantumCircuit object should be initialised

        For other parameters, please see QuantumCircuit class for more information

    """
    global PBAR
    PBAR = pbar

    supop_qubits = None

    if protocol == 'monolithic':
        kwargs.pop('basis_transformation_noise')
        kwargs.pop('no_single_qubit_error')
        qc = QuantumCircuit(9, 2, basis_transformation_noise=True, no_single_qubit_error=False, **kwargs)
        qc.define_node("A", qubits=[1, 3, 5, 7, 0], amount_data_qubits=4)
        qc.define_sub_circuit("A")

    if protocol == 'dejmps':
        qc = QuantumCircuit(8, 2, **kwargs)
        qc.define_node("A", qubits=[6, 3, 2])
        qc.define_node("B", qubits=[4, 1, 0])

        qc.define_sub_circuit("AB")
        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B", concurrent_sub_circuits=["A"])

    elif 'plain' in protocol:
        qc = QuantumCircuit(16, 2, **kwargs)

        qc.define_node("A", qubits=[14, 7, 6])
        qc.define_node("B", qubits=[12, 5, 4])
        qc.define_node("C", qubits=[10, 3, 2])
        qc.define_node("D", qubits=[8, 1, 0])

    elif protocol in ['weight_2_4_swap', 'weight_3_swap']:
        qc = QuantumCircuit(32, 16, **kwargs)

        qc.define_node("A", qubits=[30, 28, 15, 14, 13, 12], amount_data_qubits=2)
        qc.define_node("B", qubits=[26, 24, 11, 10, 9, 8], amount_data_qubits=2)
        qc.define_node("C", qubits=[22, 20, 7, 6, 5, 4], amount_data_qubits=2)
        qc.define_node("D", qubits=[18, 16, 3, 2, 1, 0], amount_data_qubits=2)

        supop_qubits = [[30, 26, 22, 28], [24, 20, 18, 16]]

    elif protocol in ['weight_3_direct', 'weight_3_direct_swap']:
        qc = QuantumCircuit(20, 16, **kwargs)

        qc.define_node("A", qubits=[18, 16, 2], amount_data_qubits=2)
        qc.define_node("B", qubits=[14, 12, 1], amount_data_qubits=2)
        qc.define_node("C", qubits=[10, 8, 0], amount_data_qubits=2)
        qc.define_node("D", qubits=[6, 4, 3], amount_data_qubits=2)

        supop_qubits = [[18, 14, 10, 16], [12, 8, 6, 4]]

    elif protocol in ['dyn_prot_3_4_1_swap']:
        qc = QuantumCircuit(16, 2, **kwargs)

        qc.define_node("A", qubits=[12, 14, 7, 6], amount_data_qubits=2)
        qc.define_node("B", qubits=[10, 4, 5, 3])
        qc.define_node("C", qubits=[8, 1, 2, 0])

    elif protocol in ['dyn_prot_3_8_1_swap']:
        qc = QuantumCircuit(19, 2, **kwargs)

        qc.define_node("A", qubits=[15, 17, 10, 9, 8], amount_data_qubits=2)
        qc.define_node("B", qubits=[13, 7, 6, 5, 4])
        qc.define_node("C", qubits=[11, 3, 2, 1, 0])

    elif protocol in ['modicum', 'modicum_swap', 'dyn_prot_4_4_1_auto_swap']:
        qc = QuantumCircuit(16, 2, **kwargs)

        qc.define_node("A", qubits=[14, 7, 6])
        qc.define_node("B", qubits=[12, 5, 4])
        qc.define_node("C", qubits=[10, 3, 2])
        qc.define_node("D", qubits=[8, 1, 0])

    elif 'dyn_prot_4_6_sym_1' in protocol:
        qc = QuantumCircuit(18, 2, **kwargs)

        qc.define_node("A", qubits=[16, 9, 8])
        qc.define_node("B", qubits=[14, 7, 6, 5])
        qc.define_node("C", qubits=[12, 4, 3])
        qc.define_node("D", qubits=[10, 2, 1, 0])

    elif 'dyn_prot_4_14_1' in protocol:
        qc = QuantumCircuit(22, 2, **kwargs)

        qc.define_node("A", qubits=[20, 13, 12, 11, 10])
        qc.define_node("B", qubits=[18, 9, 8, 7])
        qc.define_node("C", qubits=[16, 6, 5, 4, 3])
        qc.define_node("D", qubits=[14, 2, 1, 0])

    elif protocol == 'dyn_prot_4_22_1' or 'medium' in protocol or 'minimum4x_40' in protocol:
        qc = QuantumCircuit(24, 2, **kwargs)

        qc.define_node("A", qubits=[22, 15, 14, 13, 12])
        qc.define_node("B", qubits=[20, 11, 10, 9, 8])
        qc.define_node("C", qubits=[18, 7, 6, 5, 4])
        qc.define_node("D", qubits=[16, 3, 2, 1, 0])

    elif 'direct_ghz' in protocol:
        qc = QuantumCircuit(12, 2, **kwargs)

        qc.define_node("A", qubits=[10, 3])
        qc.define_node("B", qubits=[8, 2])
        qc.define_node("C", qubits=[6, 1])
        qc.define_node("D", qubits=[4, 0])

    elif protocol == 'dyn_prot_4_42_1' or 'refined' in protocol:
        qc = QuantumCircuit(28, 2, **kwargs)

        qc.define_node("A", qubits=[26, 19, 18, 17, 16, 15])
        qc.define_node("B", qubits=[24, 14, 13, 12, 11, 10])
        qc.define_node("C", qubits=[22, 9, 8, 7, 6, 5])
        qc.define_node("D", qubits=[20, 4, 3, 2, 1, 0])

    elif protocol in ['dejmps_2_4_1_swap', 'dejmps_2_6_1_swap', 'dejmps_2_8_1_swap', 'bipartite_4_swap',
                      'bipartite_6_swap', 'weight_2_4_secondary_swap', 'bipartite_4_to_1', 'bipartite_7_to_1']:
        qc = QuantumCircuit(32, 8, **kwargs)

        # If you don't specify which qubits are the data-qubits and electron-qubits, it is assumed that the first
        # qubit(s) in the list is (are) the data-qubit(s) and the last one is the electron_qubit.

        qc.define_node("A", qubits=[30, 28, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12], amount_data_qubits=2)
        qc.define_node("B", qubits=[26, 24, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], amount_data_qubits=2)

        qc.define_sub_circuit("AB")

        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B", concurrent_sub_circuits=["A"])

    # elif protocol in ['test_protocol_swap']:
    #     qc = QuantumCircuit(18, 2, **kwargs)
    #
    #     qc.define_node("A", qubits=[16, 9, 8, 7])
    #     qc.define_node("B", qubits=[14, 6, 5, 4])
    #     qc.define_node("C", qubits=[12, 3, 2])
    #     qc.define_node("D", qubits=[10, 1, 0])
    #
    #     qc.define_sub_circuit("CD")
    #     qc.define_sub_circuit("AB", concurrent_sub_circuits=["CD"])
    #     qc.define_sub_circuit("BC")
    #     qc.define_sub_circuit("AC")
    #     qc.define_sub_circuit("BD", concurrent_sub_circuits=["AC"])
    #
    #     qc.define_sub_circuit("ABCD")
    #
    #     qc.define_sub_circuit("A")
    #     qc.define_sub_circuit("B")
    #     qc.define_sub_circuit("C")
    #     qc.define_sub_circuit("D", concurrent_sub_circuits=["A", "B", "C"])

    else:
        qc = QuantumCircuit(20, 2, **kwargs)

        qc.define_node("A", qubits=[18, 11, 10, 9])
        qc.define_node("B", qubits=[16, 8, 7, 6])
        qc.define_node("C", qubits=[14, 5, 4, 3])
        qc.define_node("D", qubits=[12, 2, 1, 0])

    # Common sub circuit defining handled here
    if protocol in ['plain', 'plain_swap', 'weight_2_4_swap', 'expedient', 'expedient_swap', 'stringent',
                    'stringent_swap', 'dyn_prot_4_6_sym_1', 'dyn_prot_4_6_sym_1_swap', 'dyn_prot_4_14_1',
                    'modicum', 'modicum_swap', 'dyn_prot_4_4_1_auto_swap', 'dyn_prot_4_14_1_swap',
                    'dyn_prot_4_22_1', 'dyn_prot_4_42_1'] \
            or 'basic' in protocol \
            or 'medium' in protocol \
            or 'refined' in protocol \
            or 'minimum4x_40' in protocol:
        qc.define_sub_circuit("ABCD")
        qc.define_sub_circuit("AB")
        qc.define_sub_circuit("CD", concurrent_sub_circuits="AB")
        qc.define_sub_circuit("AC")
        if 'plain' not in protocol:
            qc.define_sub_circuit("BD", concurrent_sub_circuits="AC")
        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B")
        qc.define_sub_circuit("C")
        qc.define_sub_circuit("D", concurrent_sub_circuits=["A", "B", "C"])

    elif 'minimum4x_22' in protocol:
        qc.define_sub_circuit("ABCD")
        qc.define_sub_circuit("AB")
        qc.define_sub_circuit("CD", concurrent_sub_circuits="AB")
        qc.define_sub_circuit("AC")
        qc.define_sub_circuit("BD", concurrent_sub_circuits="AC")
        qc.define_sub_circuit("AD")
        qc.define_sub_circuit("BC", concurrent_sub_circuits="AD")
        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B")
        qc.define_sub_circuit("C")
        qc.define_sub_circuit("D", concurrent_sub_circuits=["A", "B", "C"])

    elif protocol in ['dyn_prot_3_8_1_swap', 'dyn_prot_3_4_1_swap', 'weight_3_swap', 'weight_3_direct',
                      'weight_3_direct_swap']:
        qc.define_sub_circuit("ABC")

        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B")
        qc.define_sub_circuit("C", concurrent_sub_circuits=["A", "B"])

        qc.define_sub_circuit("AB")
        qc.define_sub_circuit("BC")
        qc.define_sub_circuit("AC")

    elif 'direct_ghz' in protocol:
        qc.define_sub_circuit("ABCD")
        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B")
        qc.define_sub_circuit("C")
        qc.define_sub_circuit("D", concurrent_sub_circuits=["A", "B", "C"])

    return qc, supop_qubits


def monolithic(qc: QuantumCircuit, *, operation):
    qc.set_qubit_states({0: ket_p})
    # qc.stabilizer_measurement(operation, nodes=["A"])

    PBAR.update(70) if PBAR is not None else None

    return ["A"]


def plain(qc: QuantumCircuit, *, operation):
    qc.start_sub_circuit("AB")
    qc.create_bell_pair(7, 5)
    qc.start_sub_circuit("CD")
    qc.create_bell_pair(3, 1)
    qc.start_sub_circuit("AC")
    success = qc.single_selection(operation, 6, 2)
    if not success:
        qc.start_sub_circuit("A")
        qc.X(7)
        qc.start_sub_circuit("B")
        qc.X("B-e")

    # qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"])

    PBAR.update(70) if PBAR is not None else None

    return ["B", "A", "D", "C"]


def plain_swap(qc: QuantumCircuit, *, operation):
    qc.start_sub_circuit("AB")
    qc.create_bell_pair("B-e", "A-e")
    qc.SWAP("A-e", "A-e+1")
    qc.start_sub_circuit("CD")
    qc.create_bell_pair("D-e", "C-e")
    qc.SWAP("C-e", "C-e+1")
    qc.start_sub_circuit("AC")
    success = qc.single_selection(CZ_gate, "C-e", "A-e", swap=True)
    if not success:
        qc.start_sub_circuit("A")
        qc.X("A-e+1")
        qc.start_sub_circuit("B")
        qc.X("B-e")

    # qc.stabilizer_measurement(operation, nodes=["C", "D", "A", "B"], swap=True)

    PBAR.update(70) if PBAR is not None else None

    return ["C", "D", "A", "B"]


def dejmps_2_4_1_swap(qc: QuantumCircuit, *, operation):

    level_1 = False
    while not level_1:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB", forced_level=True)
        qc.create_bell_pair("B-e", "A-e")
        qc.SWAP("B-e", "B-e+1", efficient=True)
        qc.SWAP("A-e", "A-e+1", efficient=True)
        success_level_1 = qc.single_selection(CZ_gate, "B-e", "A-e")
        if not success_level_1:
            continue

        PBAR.update(25) if PBAR is not None else None

        level_2 = False
        while not level_2:
            qc.create_bell_pair("B-e", "A-e")
            qc.SWAP("B-e", "B-e+2", efficient=True)
            qc.SWAP("A-e", "A-e+2", efficient=True)
            level_2 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+2", "A-e+2")

        PBAR.update(25) if PBAR is not None else None

        qc.SWAP("B-e+2", "B-e", efficient=True)
        qc.SWAP("A-e+2", "A-e", efficient=True)
        level_1 = qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+1", "A-e+1", create_bell_pair=False)

        PBAR.update(20) if PBAR is not None else None

    # qc.stabilizer_measurement(operation, nodes=["A", "B"])

    PBAR.update(10) if PBAR is not None else None

    return ["A", "B"]


def dejmps_2_6_1_swap(qc: QuantumCircuit, *, operation):

    level_1 = False
    while not level_1:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB", forced_level=True)
        qc.create_bell_pair("B-e", "A-e")
        qc.SWAP("B-e", "B-e+1", efficient=True)
        qc.SWAP("A-e", "A-e+1", efficient=True)
        int_level_1 = qc.single_selection(CZ_gate, "B-e", "A-e")
        if not int_level_1:
            continue

        PBAR.update(20) if PBAR is not None else None

        level_2 = False
        while not level_2:
            qc.create_bell_pair("B-e", "A-e")
            qc.SWAP("B-e", "B-e+2", efficient=True)
            qc.SWAP("A-e", "A-e+2", efficient=True)
            level_2 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+2", "A-e+2")

        PBAR.update(20) if PBAR is not None else None

        qc.SWAP("B-e+2", "B-e", efficient=True)
        qc.SWAP("A-e+2", "A-e", efficient=True)
        int_level_3 = qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+1", "A-e+1", create_bell_pair=False)
        if not int_level_3:
            continue

        PBAR.update(20) if PBAR is not None else None

        level_4 = False
        while not level_4:
            qc.create_bell_pair("B-e", "A-e")
            qc.SWAP("B-e", "B-e+2", efficient=True)
            qc.SWAP("A-e", "A-e+2", efficient=True)
            level_4 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+2", "A-e+2")

        qc.SWAP("B-e+2", "B-e", efficient=True)
        qc.SWAP("A-e+2", "A-e", efficient=True)
        level_1 = qc.single_selection(CiY_gate, "B-e", "A-e", "B-e+1", "A-e+1", create_bell_pair=False,)

        PBAR.update(20) if PBAR is not None else None

    # qc.stabilizer_measurement(operation, nodes=["A", "B"])
    # PBAR.update(20) if PBAR is not None else None

    return ["A", "B"]


def dejmps_2_8_1_swap(qc: QuantumCircuit, *, operation):

    level_1 = False
    while not level_1:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB", forced_level=True)
        qc.create_bell_pair("B-e", "A-e")
        qc.SWAP("B-e", "B-e+1", efficient=True)
        qc.SWAP("A-e", "A-e+1", efficient=True)
        int_level_1 = qc.single_selection(CZ_gate, "B-e", "A-e")
        if not int_level_1:
            continue

        PBAR.update(15) if PBAR is not None else None

        level_2 = False
        while not level_2:
            qc.create_bell_pair("B-e", "A-e")
            qc.SWAP("B-e", "B-e+2", efficient=True)
            qc.SWAP("A-e", "A-e+2", efficient=True)
            level_2 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+2", "A-e+2")

        PBAR.update(15) if PBAR is not None else None

        qc.SWAP("B-e+2", "B-e", efficient=True)
        qc.SWAP("A-e+2", "A-e", efficient=True)
        int_level_3 = qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+1", "A-e+1", create_bell_pair=False)
        if not int_level_3:
            continue

        PBAR.update(15) if PBAR is not None else None

        level_4 = False
        while not level_4:
            qc.create_bell_pair("B-e", "A-e")
            qc.SWAP("B-e", "B-e+2", efficient=True)
            qc.SWAP("A-e", "A-e+2", efficient=True)
            int_level_4 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+2", "A-e+2")
            if not int_level_4:
                continue

            PBAR.update(15) if PBAR is not None else None

            level_5 = False
            while not level_5:
                qc.create_bell_pair("B-e", "A-e")
                qc.SWAP("B-e", "B-e+3", efficient=True)
                qc.SWAP("A-e", "A-e+3", efficient=True)
                level_5 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+3", "A-e+3")

            PBAR.update(15) if PBAR is not None else None

            qc.SWAP("B-e+3", "B-e", efficient=True)
            qc.SWAP("A-e+3", "A-e", efficient=True)
            level_4 = qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+2", "A-e+2", create_bell_pair=False)

        qc.SWAP("B-e+2", "B-e", efficient=True)
        qc.SWAP("A-e+2", "A-e", efficient=True)
        level_1 = qc.single_selection(CiY_gate, "B-e", "A-e", "B-e+1", "A-e+1", create_bell_pair=False)

        PBAR.update(5) if PBAR is not None else None

    # qc.stabilizer_measurement(operation, nodes=["A", "B"])
    # PBAR.update(10) if PBAR is not None else None

    return ["A", "B"]


def bipartite_4_swap(qc: QuantumCircuit, *, operation, tqubit=None):
    level_1 = False
    while not level_1:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB", forced_level=True)
        qc.create_bell_pair("B-e", "A-e")       # [12, 0]
        qc.SWAP("B-e", "B-e+1", efficient=True)
        qc.SWAP("A-e", "A-e+1", efficient=True)   # [13, 1]
        qc.create_bell_pair("B-e", "A-e")
        qc.SWAP("B-e", "B-e+2", efficient=True)
        qc.SWAP("A-e", "A-e+2", efficient=True)   # [14, 2]
        qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+1", "A-e+1", measure=False)   # [12, 0, 13, 1]
        int_level_1 = qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+2", "A-e+2", create_bell_pair=False)    # [13, 1, 14, 2]
        if not int_level_1:
            continue

        PBAR.update(30) if PBAR is not None else None

        int_level_2 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+2", "A-e+2")
        if not int_level_2:
            continue

        PBAR.update(30) if PBAR is not None else None

        qc.SWAP("B-e", "B-e+2", efficient=True)
        qc.SWAP("A-e", "A-e+2", efficient=True)       # [13, 1, 12, 0]
        level_1 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+1", "A-e+1", create_bell_pair=False)
        PBAR.update(20) if PBAR is not None else None

    # qc.stabilizer_measurement(operation, nodes=["A", "B"], swap=True, tqubit=tqubit)
    # PBAR.update(10) if PBAR is not None else None

    return ["A", "B"]


def bipartite_4_to_1(qc: QuantumCircuit, *, operation, tqubit=None):
    succ_1 = False
    while not succ_1:
        qc.start_sub_circuit("AB", forced_level=True)

        qc.create_bell_pair("B-e", "A-e")
        qc.create_bell_pair("B-e+1", "A-e+1")
        qc.create_bell_pair("B-e+2", "A-e+2")
        qc.create_bell_pair("B-e+3", "A-e+3")

        qc.CNOT("B-e+3", "B-e")
        qc.CNOT("A-e+3", "A-e")

        qc.CZ("B-e+2", "B-e+1")
        qc.CZ("A-e+2", "A-e+1")

        qc.CZ("B-e+3", "B-e+2")
        qc.CZ("A-e+3", "A-e+2")

        qc.CZ("B-e+1", "B-e")
        qc.CZ("A-e+1", "A-e")

        outcome1a = qc.measure("A-e+3", basis="X")
        outcome1b = qc.measure("B-e+3", basis="X")
        outcome1 = 0 if outcome1a == outcome1b else 1

        outcome2a = qc.measure("A-e+2", basis="X")
        outcome2b = qc.measure("B-e+2", basis="X")
        outcome2 = 0 if outcome2a == outcome2b else 1

        outcome3a = qc.measure("A-e+1", basis="X")
        outcome3b = qc.measure("B-e+1", basis="X")
        outcome3 = 0 if outcome3a == outcome3b else 1

        succ_1 = True if (outcome1 == 0 and outcome2 == 0 and outcome3 == 0) else False

    return ["A", "B"]


def bipartite_7_to_1(qc: QuantumCircuit, *, operation, tqubit=None):
    succ_1 = False
    while not succ_1:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB", forced_level=True)

        qc.create_bell_pair("B-e", "A-e")
        qc.create_bell_pair("B-e+1", "A-e+1")
        qc.create_bell_pair("B-e+2", "A-e+2")
        qc.create_bell_pair("B-e+3", "A-e+3")
        qc.create_bell_pair("B-e+4", "A-e+4")
        qc.create_bell_pair("B-e+5", "A-e+5")
        qc.create_bell_pair("B-e+6", "A-e+6")

        qc.CNOT("B-e+4", "B-e+3")
        qc.CNOT("A-e+4", "A-e+3")

        qc.CNOT("B-e+2", "B-e")
        qc.CNOT("A-e+2", "A-e")

        qc.CNOT("B-e+4", "B-e+2")
        qc.CNOT("A-e+4", "A-e+2")

        qc.CNOT("B-e+1", "B-e")
        qc.CNOT("A-e+1", "A-e")

        qc.CZ("B-e+5", "B-e+6")
        qc.CZ("A-e+5", "A-e+6")

        qc.CZ("B-e+5", "B-e+1")
        qc.CZ("A-e+5", "A-e+1")

        qc.CZ("B-e+3", "B-e+1")
        qc.CZ("A-e+3", "A-e+1")

        outcome6a = qc.measure("A-e+1", basis="X")
        outcome6b = qc.measure("B-e+1", basis="X")
        outcome6 = 0 if outcome6a == outcome6b else 1
        if outcome6 == 1:
            continue

        qc.CZ("B-e+3", "B-e+2")
        qc.CZ("A-e+3", "A-e+2")

        outcome4a = qc.measure("A-e+3", basis="X")
        outcome4b = qc.measure("B-e+3", basis="X")
        outcome4 = 0 if outcome4a == outcome4b else 1
        if outcome4 == 1:
            continue

        qc.CZ("B-e+4", "B-e+5")
        qc.CZ("A-e+4", "A-e+5")

        outcome2a = qc.measure("A-e+5", basis="X")
        outcome2b = qc.measure("B-e+5", basis="X")
        outcome2 = 0 if outcome2a == outcome2b else 1
        if outcome2 == 1:
            continue

        outcome3a = qc.measure("A-e+4", basis="X")
        outcome3b = qc.measure("B-e+4", basis="X")
        outcome3 = 0 if outcome3a == outcome3b else 1
        if outcome3 == 1:
            continue

        qc.CZ("B-e+6", "B-e+2")
        qc.CZ("A-e+6", "A-e+2")

        outcome1a = qc.measure("A-e+6", basis="X")
        outcome1b = qc.measure("B-e+6", basis="X")
        outcome1 = 0 if outcome1a == outcome1b else 1
        if outcome1 == 1:
            continue

        qc.CZ("B-e+2", "B-e")
        qc.CZ("A-e+2", "A-e")

        outcome5a = qc.measure("A-e+2", basis="X")
        outcome5b = qc.measure("B-e+2", basis="X")
        outcome5 = 0 if outcome5a == outcome5b else 1

        succ_1 = True if (outcome1 == 0 and outcome2 == 0 and outcome3 == 0 and outcome4 == 0 and outcome5 == 0
                          and outcome6 == 0) else False

    return ["A", "B"]


def bipartite_6_swap(qc: QuantumCircuit, *, operation):
    level_1 = False
    while not level_1:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB", forced_level=True)
        qc.create_bell_pair("B-e", "A-e")           # [12, 0]
        qc.SWAP("B-e", "B-e+1", efficient=True)
        qc.SWAP("A-e", "A-e+1", efficient=True)       # [13, 1]
        qc.create_bell_pair("B-e", "A-e")
        qc.SWAP("B-e", "B-e+2", efficient=True)
        qc.SWAP("A-e", "A-e+2", efficient=True)
        qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+1", "A-e+1", measure=False)   # [12, 0, 13, 1]
        qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+2", "A-e+2", create_bell_pair=False, measure=False, reverse_den_mat_add=True)   # [14, 2, 12, 0, 13, 1]
        qc.SWAP("B-e", "B-e+3", efficient=True)
        qc.SWAP("A-e", "A-e+3", efficient=True)   # [14, 2, 15, 3, 13, 1]
        qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+1", "A-e+1", measure=False, reverse_den_mat_add=True)   # [14, 2, 15, 3, 13, 1, 12, 0]
        qc.SWAP("B-e", "B-e+4", efficient=True)
        qc.SWAP("A-e", "A-e+4", efficient=True)   # [14, 2, 15, 3, 13, 1, 16, 4]
        int_level_1 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+4", "A-e+4")
        if not int_level_1:
            continue

        PBAR.update(15) if PBAR is not None else None

        int_level_2 = qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+3", "A-e+3")
        if not int_level_2:
            continue

        PBAR.update(15) if PBAR is not None else None

        qc.SWAP("B-e+3", "B-e", efficient=True)
        qc.SWAP("A-e+3", "A-e", efficient=True)   # [14, 2, 12, 0, 13, 1, 16, 4]
        qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+2", "A-e+2", create_bell_pair=False, measure=False)
        int_level_3 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+1", "A-e+1", create_bell_pair=False)      # [14, 2, 13, 1, 16, 4]
        if not int_level_3:
            continue

        PBAR.update(15) if PBAR is not None else None

        qc.SWAP("B-e+2", "B-e", efficient=True)
        qc.SWAP("A-e+2", "A-e", efficient=True)       # [12, 0, 13, 1, 16, 4]
        int_level_4 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+4", "A-e+4", create_bell_pair=False)      # [13, 1, 16, 4]
        if not int_level_4:
            continue

        PBAR.update(15) if PBAR is not None else None

        qc.SWAP("B-e+4", "B-e", efficient=True)
        qc.SWAP("A-e+4", "A-e", efficient=True)
        meas_outc = qc.measure(["A-e", "B-e"])
        level_1 = meas_outc[0] == meas_outc[1]

        PBAR.update(20) if PBAR is not None else None

    # qc.stabilizer_measurement(operation, nodes=["A", "B"])
    # PBAR.update(20) if PBAR is not None else None

    return ["A", "B"]


def dyn_prot_3_4_1_swap(qc: QuantumCircuit, *, operation, tqubit=None):

    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB")
        qc.create_bell_pair("A-e", "B-e")       # 3, 6
        qc.SWAP("A-e", "A-e+2", efficient=True)
        qc.SWAP("B-e", "B-e+1", efficient=True)   # 4, 7

        PBAR.update(30) if PBAR is not None else None

        qc.start_sub_circuit("AC")
        qc.create_bell_pair("C-e", "A-e")       # 6, 0
        qc.SWAP("C-e", "C-e+1", efficient=True)   # 6, 1
        qc.apply_gate(CNOT_gate, cqubit="A-e+2", tqubit="A-e", electron_is_target=True, reverse=True)   # 6, 1, 4, 7
        measurement_outcome = qc.measure(["A-e"], basis="Z")    # 1, 4, 7
        # BEGIN FUSION CORRECTION:
        time_in_A = qc.nodes["A"].sub_circuit_time
        time_in_C = qc.nodes["C"].sub_circuit_time
        if time_in_C < time_in_A:
            qc._increase_duration(time_in_A - time_in_C, [], involved_nodes=["C"])
        if measurement_outcome[0] == 1:
            qc.X("C-e+1")
        # END FUSION CORRECTION

        PBAR.update(30) if PBAR is not None else None

        level_2 = False
        while not level_2:
            qc.start_sub_circuit("BC")
            qc.create_bell_pair("B-e", "C-e")  # 0, 3
            qc.SWAP("B-e", "B-e+2", efficient=True)
            qc.SWAP("C-e", "C-e+2", efficient=True)  # 2, 5
            level_2 = qc.single_selection(CiY_gate, "B-e", "C-e", "B-e+2", "C-e+2")
        qc.SWAP("B-e+2", "B-e", efficient=True)
        qc.SWAP("C-e+2", "C-e", efficient=True)
        ghz_success = qc.single_selection(CZ_gate, "B-e", "C-e", "B-e+1", "C-e+1", create_bell_pair=False)

        PBAR.update(20) if PBAR is not None else None

    # qc.stabilizer_measurement(operation, nodes=["C", "B", "A"], tqubit=tqubit)
    # PBAR.update(10) if PBAR is not None else None

    return ["C", "B", "A"]


def dyn_prot_3_8_1_swap(qc: QuantumCircuit, *, operation, tqubit=None):

    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB")
        qc.create_bell_pair("A-e", "B-e")       # 4, 8
        qc.SWAP("A-e", "A-e+2", efficient=True)
        qc.SWAP("B-e", "B-e+3", efficient=True)   # 5, 9
        int_level_1 = qc.single_selection(CNOT_gate, "A-e", "B-e", "A-e+2", "B-e+3")
        if not int_level_1:
            continue

        PBAR.update(30) if PBAR is not None else None

        qc.start_sub_circuit("AC")
        level_2 = False
        while not level_2:
            qc.create_bell_pair("C-e", "A-e")       # 8, 0
            qc.SWAP("A-e", "A-e+1", efficient=True)
            qc.SWAP("C-e", "C-e+3", efficient=True)   # 10, 1
            level_2 = qc.single_selection(CiY_gate, "A-e", "C-e", "A-e+1", "C-e+3")
        qc.SWAP("A-e+1", "A-e", efficient=True)       # 8, 1
        qc.apply_gate(CNOT_gate, cqubit="A-e+2", tqubit="A-e", electron_is_target=True, reverse=True)   # 8, 1, 5, 9
        measurement_outcome = qc.measure(["A-e"], basis="Z")    # 1, 5, 9

        qc.start_sub_circuit("ABC", forced_level=True)
        if measurement_outcome != SKIP() and measurement_outcome[0] == 1:
            qc.X("C-e+3")

        PBAR.update(30) if PBAR is not None else None

        qc.start_sub_circuit("BC")
        level_3 = False
        while not level_3:
            qc.create_bell_pair("B-e", "C-e")  # 0, 4
            qc.SWAP("B-e", "B-e+2", efficient=True)
            qc.SWAP("C-e", "C-e+2", efficient=True)  # 2, 6
            int_level_3 = qc.single_selection(CNOT_gate, "B-e", "C-e", "B-e+2", "C-e+2")
            if not int_level_3:
                continue
            level_4 = False
            while not level_4:
                qc.create_bell_pair("B-e", "C-e")  # 0, 4
                qc.SWAP("B-e", "B-e+1", efficient=True)
                qc.SWAP("C-e", "C-e+1", efficient=True)  # 3, 7
                level_4 = qc.single_selection(CNOT_gate, "B-e", "C-e", "B-e+1", "C-e+1")
            qc.SWAP("B-e+1", "B-e", efficient=True)
            qc.SWAP("C-e+1", "C-e", efficient=True)
            level_3 = qc.single_selection(CiY_gate, "B-e", "C-e", "B-e+2", "C-e+2", create_bell_pair=False)

        qc.SWAP("B-e+2", "B-e", efficient=True)
        qc.SWAP("C-e+2", "C-e", efficient=True)
        ghz_success = qc.single_selection(CZ_gate, "B-e", "C-e", "B-e+3", "C-e+3", create_bell_pair=False)

        PBAR.update(20) if PBAR is not None else None

    # qc.stabilizer_measurement(operation, nodes=["C", "B", "A"], tqubit=tqubit)
    # PBAR.update(20) if PBAR is not None else None

    return ["C", "B", "A"]


def modicum(qc: QuantumCircuit, *, operation):

    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AC")
        qc.create_bell_pair("A-e", "C-e")

        qc.start_sub_circuit("BD")
        qc.create_bell_pair("D-e", "B-e")

        PBAR.update(40) if PBAR is not None else None

        qc.start_sub_circuit("AB")
        qc.create_bell_pair("B-e+1", "A-e+1")
        qc.apply_gate(CNOT_gate, cqubit="A-e", tqubit="A-e+1", reverse=True)
        qc.apply_gate(CNOT_gate, cqubit="B-e", tqubit="B-e+1", reverse=True)
        meas_parity_ab = qc.measure(["A-e+1", "B-e+1"], basis="Z")

        qc.start_sub_circuit("CD")
        qc.create_bell_pair("C-e+1", "D-e+1")
        meas_parity_cd = qc.single_selection(CZ_gate, "C-e+1", "D-e+1", "C-e", "D-e", create_bell_pair=False)

        qc.start_sub_circuit("ABCD", forced_level=True)
        ghz_success = (len(set(meas_parity_ab)) == 1) == meas_parity_cd if not qc.cut_off_time_reached else True
        if not ghz_success:
            continue
        if not meas_parity_cd:
            qc.X("B-e")
            qc.X("D-e")

        PBAR.update(40) if PBAR is not None else None

    # mat = qc.get_combined_density_matrix([0, 2, 4, 6])[0]
    # pickle.dump(mat, open("matrix", "wb"))
    # qc.stabilizer_measurement(operation, nodes=["C", "A", "B", "D"], swap=True)
    # PBAR.update(20) if PBAR is not None else None

    return ["C", "A", "B", "D"]


def dejmps(qc: QuantumCircuit, *, operation):
    success = False
    while not success:
        qc.start_sub_circuit("AB")
        qc.create_bell_pair("A-e", "B-e")
        qc.SWAP("A-e", "A-e+1")
        qc.SWAP("B-e", "B-e+1")
        qc.create_bell_pair("A-e", "B-e")
        qc.H("A-e")
        qc.S("A-e")
        qc.H("A-e")
        qc.H("B-e")
        qc.S("B-e")
        qc.H("B-e")
        qc.apply_gate(CZ_gate, "A-e", "A-e+1")
        qc.apply_gate(CZ_gate, "B-e", "B-e+1")
        measurement_outcomes = qc.measure(["A-e", "B-e"], basis="Z")
        if measurement_outcomes[0] == measurement_outcomes[1]:
            success = True

    # qc.stabilizer_measurement(operation, nodes=["A", "B"], swap=True)

    return ["A", "B"]


def modicum_swap(qc: QuantumCircuit, *, operation):

    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AC")
        qc.create_bell_pair("A-e", "C-e")
        qc.SWAP("A-e", "A-e+1", efficient=True)
        qc.SWAP("C-e", "C-e+1", efficient=True)

        qc.start_sub_circuit("BD")
        qc.create_bell_pair("D-e", "B-e")
        qc.SWAP("B-e", "B-e+1", efficient=True)
        qc.SWAP("D-e", "D-e+1", efficient=True)

        PBAR.update(40) if PBAR is not None else None

        qc.start_sub_circuit("AB")
        qc.create_bell_pair("B-e", "A-e")

        # Option I
        qc.H("A-e")
        qc.H("B-e")
        qc.apply_gate(CZ_gate, cqubit="A-e", tqubit="A-e+1")
        qc.apply_gate(CZ_gate, cqubit="B-e", tqubit="B-e+1")
        qc.H("A-e")
        qc.H("B-e")
        # # Option II
        # qc.apply_gate(CNOT_gate, cqubit="A-e+1", tqubit="A-e", electron_is_target=True, reverse=True)
        # qc.apply_gate(CNOT_gate, cqubit="B-e+1", tqubit="B-e", electron_is_target=True, reverse=True)

        meas_parity_ab = qc.measure(["A-e", "B-e"], basis="Z")

        qc.start_sub_circuit("CD")
        qc.create_bell_pair("C-e", "D-e")
        meas_parity_cd = qc.single_selection(CZ_gate, "C-e", "D-e", "C-e+1", "D-e+1", create_bell_pair=False)

        qc.start_sub_circuit("ABCD", forced_level=True)
        ghz_success = (len(set(meas_parity_ab)) == 1) == meas_parity_cd if not qc.cut_off_time_reached else True
        if not ghz_success:
            continue
        if not meas_parity_cd:
            qc.X("B-e+1")
            qc.X("D-e+1")

        PBAR.update(40) if PBAR is not None else None

    # qc.stabilizer_measurement(operation, nodes=["C", "A", "B", "D"], swap=True)

    # PBAR.update(20) if PBAR is not None else None

    return ["C", "A", "B", "D"]


def dyn_prot_4_4_1_auto_swap(qc: QuantumCircuit, *, operation):

    print_ops = False

    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB")
        print("SUBSYSTEM [0, 1]:") if print_ops else None
        qc.create_bell_pair("A-e", "B-e")
        print("CREATE_LINK between qubits [[0, 0], [1, 0]]") if print_ops else None
        qc.SWAP("A-e", "A-e+1")
        print("SWAP qubit [[0, 0]] and [[0, 1]]") if print_ops else None
        qc.SWAP("B-e", "B-e+1")
        print("SWAP qubit [[1, 0]] and [[1, 1]]") if print_ops else None

        qc.start_sub_circuit("CD")
        print("SUBSYSTEM [2, 3]") if print_ops else None
        qc.create_bell_pair("C-e", "D-e")
        print("CREATE_LINK between qubits [[2, 0], [3, 0]]") if print_ops else None
        qc.SWAP("C-e", "C-e+1")
        print("SWAP qubit [[2, 0]] and [[2, 1]]") if print_ops else None
        qc.SWAP("D-e", "D-e+1")
        print("SWAP qubit [[3, 0]] and [[3, 1]]") if print_ops else None

        PBAR.update(40) if PBAR is not None else None
        print("") if print_ops else None

        qc.start_sub_circuit("AC")
        print("SUBSYSTEM [0, 2]") if print_ops else None
        qc.create_bell_pair("C-e", "A-e")
        print("CREATE_LINK between qubits [[2, 0], [0, 0]]") if print_ops else None
        # qc.H("A-e")
        # qc.H("A-e+1")
        # qc.apply_gate(CNOT_gate, cqubit="A-e", tqubit="A-e+1")
        # qc.H("A-e")
        # qc.H("A-e+1")
        qc.apply_gate(CNOT_gate, cqubit="A-e+1", tqubit="A-e", electron_is_target=True)
        meas_parity_a = qc.measure("A-e", basis="Z")
        print("FUSE by measuring out qubits [[0, 0]] and keeping qubits [[0, 1]]") if print_ops else None
        qc.SWAP("C-e", "C-e+1")
        print("SWAP qubit [[2, 0]] and [[2, 1]]") if print_ops else None

        qc.start_sub_circuit("BD")
        print("SUBSYSTEM [1, 3]") if print_ops else None
        qc.create_bell_pair("D-e", "B-e")
        print("CREATE_LINK between qubits [[3, 0], [1, 0]]") if print_ops else None
        # Option I:
        qc.H("B-e")
        qc.apply_gate(CZ_gate, cqubit="B-e", tqubit="B-e+1")
        qc.H("B-e")
        # # Option II:
        # qc.apply_gate(CNOT_gate, cqubit="B-e+1", tqubit="B-e", electron_is_target=True)
        meas_parity_b = qc.measure("B-e", basis="Z")
        print("FUSE by measuring out qubits [[1, 0]] and keeping qubits [[1, 1]]") if print_ops else None
        qc.SWAP("D-e", "D-e+1")
        print("SWAP qubit [[3, 0]] and [[3, 1]]") if print_ops else None

        open_subcircuit = False
        if meas_parity_a[0]:
            qc.start_sub_circuit("ABCD")
            print("SUBSYSTEM [1, 2, 3, 4]") if print_ops else None
            print("CORRECT qubit [[2, 1]]") if print_ops else None
            qc.X("C-e+1")
            open_subcircuit = True
        if meas_parity_b[0]:
            if open_subcircuit is False:
                qc.start_sub_circuit("ABCD")
                print("SUBSYSTEM [1, 2, 3, 4]") if print_ops else None
            print("CORRECT qubit [[3, 1]]") if print_ops else None
            qc.X("D-e+1")

        print("") if print_ops else None

        qc.start_sub_circuit("CD")
        print("SUBSYSTEM [2, 3]") if print_ops else None
        qc.apply_gate(CZ_gate, cqubit="C-e", tqubit="C-e+1")
        qc.apply_gate(CZ_gate, cqubit="D-e", tqubit="D-e+1")
        measurement_outcomes = qc.measure(["C-e", "D-e"], basis="X")
        ghz_post_select = (measurement_outcomes.count(1)) % 2
        print("DISTILL operation [1, 1] by measuring out qubits [[2, 0], [3, 0]] and keeping qubits [[2, 1], [3, 1]]") \
            if print_ops else None

        if ghz_post_select == 1:
            print("Distillation operation was unsuccessful.") if print_ops else None
            continue
        else:
            ghz_success = True

        PBAR.update(40) if PBAR is not None else None

    # qc.stabilizer_measurement(operation,  nodes=["B", "A", "D", "C"], swap=True)
    # PBAR.update(20) if PBAR is not None else None

    return ["B", "A", "D", "C"]


def dyn_prot_4_6_sym_1(qc: QuantumCircuit, *, operation):

    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB")
        qc.create_bell_pair(9, 7)

        PBAR.update(25) if PBAR is not None else None
        qc.start_sub_circuit("CD")
        qc.create_bell_pair(2, 3)

        PBAR.update(25) if PBAR is not None else None

        qc.start_sub_circuit("AC", forced_level=True)
        qc.create_bell_pair(4, 8)
        qc.apply_gate(CNOT_gate, cqubit=9, tqubit=8, reverse=True)    # 8, 4, 7, 9
        qc.apply_gate(CNOT_gate, cqubit=4, tqubit=3, reverse=True)      # 3, 2, 8, 4, 7, 9
        measurement_outcomes = qc.measure([3, 8], basis="Z")           # 2, 4, 7, 9
        # BEGIN FUSION CORRECTION: The correction in node C can only be applied AFTER the measurement in A
        time_between_meas = qc.nodes["A"].sub_circuit_time - qc.nodes["C"].sub_circuit_time
        if time_between_meas > 0:
            qc._increase_duration(time_between_meas, [], involved_nodes=["C"])
        if measurement_outcomes[0] == 1:
            qc.X(4)
        # The correction in node D can only be applied after both measurements in A and C
        time_after_both_meas = max(qc.nodes["A"].sub_circuit_time, qc.nodes["C"].sub_circuit_time)
        # END FUSION CORRECTION
        qc.create_bell_pair(3, 8)
        qc.apply_gate(CZ_gate, cqubit=8, tqubit=9)        # 8, 3, 2, 4, 7, 9
        qc.apply_gate(CZ_gate, cqubit=3, tqubit=4)
        measurement_outcomes_1 = qc.measure([8, 3])       # 2, 4, 7, 9
        ghz_success_1 = measurement_outcomes_1[0] == measurement_outcomes_1[1]

        PBAR.update(25) if PBAR is not None else None

        qc.start_sub_circuit("BD")
        success_bd = False
        while not success_bd:
            qc.create_bell_pair(1, 6)
            qc.create_bell_pair(5, 0)
            success_bd = qc.single_selection(CiY_gate, 5, 0, 6, 1, create_bell_pair=False)
        # BEGIN FUSION CORRECTION
        time_diff_with_meas = time_after_both_meas - qc.nodes["D"].sub_circuit_time
        if time_diff_with_meas > 0:
            qc._increase_duration(time_diff_with_meas, [], involved_nodes=["D"])
        if measurement_outcomes in [[0, 1], [1, 0]]:
            qc.X(2)
        # END FUSION CORRECTION
        qc.apply_gate(CZ_gate, cqubit=6, tqubit=7)        # 6, 1, 2, 4, 7, 9
        qc.apply_gate(CZ_gate, cqubit=1, tqubit=2)
        measurement_outcomes_2 = qc.measure([6, 1])       # 2, 4, 7, 9
        ghz_success_2 = measurement_outcomes_2[0] == measurement_outcomes_2[1]
        if ghz_success_1 and ghz_success_2:
            ghz_success = True
        else:
            ghz_success = False

        PBAR.update(15) if PBAR is not None else None

    # qc.stabilizer_measurement(operation, nodes=["D", "C", "B", "A"], swap=False)
    # PBAR.update(20) if PBAR is not None else None

    return ["D", "C", "B", "A"]


def dyn_prot_4_6_sym_1_swap(qc: QuantumCircuit, *, operation):

    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB")
        qc.create_bell_pair(8, 5)
        qc.SWAP(5, 7, efficient=True)
        qc.SWAP(8, 9, efficient=True)

        PBAR.update(25) if PBAR is not None else None
        qc.start_sub_circuit("CD")
        qc.create_bell_pair(0, 3)
        qc.SWAP(3, 4, efficient=True)
        qc.SWAP(0, 2, efficient=True)

        PBAR.update(25) if PBAR is not None else None

        qc.start_sub_circuit("AC", forced_level=True)
        qc.create_bell_pair(3, 8)
        qc.apply_gate(CNOT_gate, cqubit=9, tqubit=8, electron_is_target=True, reverse=True)    # 8, 3, 7, 9
        qc.apply_gate(CNOT_gate, cqubit=3, tqubit=4)      # 8, 3, 7, 9, 4, 2
        qc.SWAP(3, 4, efficient=False)
        measurement_outcomes = qc.measure([8, 3], basis="Z")           # 7, 9, 4, 2
        # BEGIN FUSION CORRECTION: The correction in node C can only be applied AFTER the measurement in A
        time_between_meas = qc.nodes["A"].sub_circuit_time - qc.nodes["C"].sub_circuit_time
        if time_between_meas > 0:
            qc._increase_duration(time_between_meas, [], involved_nodes=["C"])
        if measurement_outcomes[0] == 1:
            qc.X(4)
        # The correction in node D can only be applied after both measurements in A and C
        time_after_both_meas = max(qc.nodes["A"].sub_circuit_time, qc.nodes["C"].sub_circuit_time)
        # END FUSION CORRECTION
        qc.create_bell_pair(3, 8)
        qc.apply_gate(CZ_gate, cqubit=8, tqubit=9)        # 8, 3, 7, 9, 4, 2
        qc.apply_gate(CZ_gate, cqubit=3, tqubit=4)
        measurement_outcomes_1 = qc.measure([8, 3])       # 7, 9, 4, 2
        ghz_success_1 = measurement_outcomes_1[0] == measurement_outcomes_1[1]

        PBAR.update(25) if PBAR is not None else None

        qc.start_sub_circuit("BD")
        success_bd = False
        while not success_bd:
            qc.create_bell_pair(5, 0)
            qc.SWAP(5, 6, efficient=True)
            qc.SWAP(0, 1, efficient=True)
            qc.create_bell_pair(5, 0)
            success_bd = qc.single_selection(CiY_gate, 5, 0, 6, 1, create_bell_pair=False)
        qc.SWAP(5, 6, efficient=True)
        qc.SWAP(0, 1, efficient=True)
        # BEGIN FUSION CORRECTION
        time_diff_with_meas = time_after_both_meas - qc.nodes["D"].sub_circuit_time
        if time_diff_with_meas > 0:
            qc._increase_duration(time_diff_with_meas, [], involved_nodes=["D"])
        if measurement_outcomes in [[0, 1], [1, 0]]:
            qc.X(2)
        # END FUSION CORRECTION
        qc.apply_gate(CZ_gate, cqubit=5, tqubit=7)        # 0, 5, 7, 9, 4, 2
        qc.apply_gate(CZ_gate, cqubit=0, tqubit=2)
        measurement_outcomes_2 = qc.measure([0, 5])       # 7, 9, 4, 2
        ghz_success_2 = measurement_outcomes_2[0] == measurement_outcomes_2[1]
        if ghz_success_1 and ghz_success_2:
            ghz_success = True
        else:
            ghz_success = False

        PBAR.update(15) if PBAR is not None else None

    # qc.stabilizer_measurement(operation, nodes=["B", "A", "C", "D"], swap=True)
    # PBAR.update(20) if PBAR is not None else None

    return ["B", "A", "C", "D"]


def dyn_prot_4_14_1(qc: QuantumCircuit, *, operation):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        qc.start_sub_circuit("AB")
        success_ab = False
        while not success_ab:
            qc.create_bell_pair(13, 9)
            success_ab = qc.single_selection(CZ_gate, 12, 8)
            if not success_ab:
                continue
            success_ab2 = False
            while not success_ab2:
                qc.create_bell_pair(12, 8)
                success_ab2 = qc.single_selection(CNOT_gate, 11, 7)
            success_ab = qc.single_selection(CiY_gate, 12, 8, 13, 9, create_bell_pair=False)

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair(6, 2)
            success_cd = qc.single_selection(CZ_gate, 5, 1)
            if not success_cd:
                continue
            success_cd2 = False
            while not success_cd2:
                qc.create_bell_pair(5, 1)
                success_cd2 = qc.single_selection(CZ_gate, 4, 0)
            success_cd = qc.single_selection(CNOT_gate, 5, 1, 6, 2, create_bell_pair=False)

        PBAR.update(30) if PBAR is not None else None

        qc.start_sub_circuit("AC")
        success_ac = False
        while not success_ac:
            qc.create_bell_pair(12, 5)
            success_ac = qc.single_selection(CZ_gate, 11, 4)
            if not success_ac:
                continue
            success_ac2 = False
            while not success_ac2:
                qc.create_bell_pair(11, 4)
                success_ac2 = qc.single_selection(CNOT_gate, 10, 3)
            success_ac = qc.single_selection(CiY_gate, 11, 4, 12, 5, create_bell_pair=False)

        qc.start_sub_circuit("BD")
        success_bd = False
        while not success_bd:
            qc.create_bell_pair(8, 1)
            success_bd = qc.single_selection(CZ_gate, 7, 0)

        qc.start_sub_circuit("AC", forced_level=True)
        qc.apply_gate(CNOT_gate, cqubit=13, tqubit=12, reverse=True)    # 5, 12, 9, 13
        qc.apply_gate(CNOT_gate, cqubit=6, tqubit=5, reverse=True)      # 5, 12, 9, 13, 2, 6
        measurement_outcomes = qc.measure([5, 12], basis="Z")           # 9, 13, 2, 6
        success = measurement_outcomes[0] == measurement_outcomes[1]
        qc.start_sub_circuit("AB")
        if not success:
            qc.X(13)
            qc.X(9)
        qc.start_sub_circuit("BD")
        qc.apply_gate(CZ_gate, cqubit=9, tqubit=8, reverse=True)        # 1, 8, 9, 13, 2, 6
        qc.apply_gate(CZ_gate, cqubit=2, tqubit=1, reverse=True)        # 1, 8, 9, 13, 2, 6
        measurement_outcomes2 = qc.measure([1, 8])
        ghz_success = measurement_outcomes2[0] == measurement_outcomes2[1]
        PBAR.update(20) if PBAR is not None else None

    # qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"], swap=False)
    # PBAR.update(20) if PBAR is not None else None

    return ["B", "A", "D", "C"]


def dyn_prot_4_14_1_swap(qc: QuantumCircuit, *, operation):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        qc.start_sub_circuit("AB")
        success_ab = False
        while not success_ab:
            qc.create_bell_pair("A-e", "B-e")
            qc.SWAP("A-e", "A-e+1", efficient=True)
            qc.SWAP("B-e", "B-e+1", efficient=True)
            success_ab = qc.single_selection(CZ_gate, "A-e", "B-e", swap=True)
            if not success_ab:
                continue
            success_ab2 = False
            while not success_ab2:
                qc.create_bell_pair("A-e", "B-e")
                qc.SWAP("A-e", "A-e+2", efficient=True)
                qc.SWAP("B-e", "B-e+2", efficient=True)
                success_ab2 = qc.single_selection(CNOT_gate, "A-e", "B-e", "A-e+2", "B-e+2", swap=True)
            qc.SWAP("A-e", "A-e+2", efficient=True)
            qc.SWAP("B-e", "B-e+2", efficient=True)
            success_ab = qc.single_selection(CiY_gate, "A-e", "B-e", create_bell_pair=False, swap=True)

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair("C-e", "D-e")
            qc.SWAP("C-e", "C-e+1", efficient=True)
            qc.SWAP("D-e", "D-e+1", efficient=True)
            success_cd = qc.single_selection(CZ_gate, "C-e", "D-e", swap=True)
            if not success_cd:
                continue
            success_cd2 = False
            while not success_cd2:
                qc.create_bell_pair("C-e", "D-e")
                qc.SWAP("C-e", "C-e+2", efficient=True)
                qc.SWAP("D-e", "D-e+2", efficient=True)
                success_cd2 = qc.single_selection(CZ_gate, "C-e", "D-e", "C-e+2", "D-e+2", swap=True)
            qc.SWAP("C-e", "C-e+2", efficient=True)
            qc.SWAP("D-e", "D-e+2", efficient=True)
            success_cd = qc.single_selection(CNOT_gate, "C-e", "D-e", create_bell_pair=False, swap=True)

        PBAR.update(30) if PBAR is not None else None

        qc.start_sub_circuit("AC")
        success_ac = False
        while not success_ac:
            qc.create_bell_pair("A-e", "C-e")
            qc.SWAP("C-e", "C-e+2", efficient=True)
            qc.SWAP("A-e", "A-e+2", efficient=True)
            success_ac = qc.single_selection(CZ_gate, "A-e", "C-e", "A-e+2", "C-e+2", swap=True)
            if not success_ac:
                continue
            success_ac2 = False
            while not success_ac2:
                qc.create_bell_pair("A-e", "C-e")
                qc.SWAP("C-e", "C-e+3", efficient=True)
                qc.SWAP("A-e", "A-e+3", efficient=True)
                success_ac2 = qc.single_selection(CNOT_gate, "A-e", "C-e", "A-e+3", "C-e+3", swap=True)
            qc.SWAP("A-e", "A-e+3", efficient=True)
            qc.SWAP("C-e", "C-e+3", efficient=True)
            success_ac = qc.single_selection(CiY_gate, "A-e", "C-e", "A-e+2", "C-e+2", create_bell_pair=False,
                                             swap=True)
            qc.SWAP("A-e", "A-e+2", efficient=True)
            qc.SWAP("C-e", "C-e+2", efficient=True)

        qc.H("C-e")
        qc.H("C-e+1")
        qc.apply_gate(CNOT_gate, cqubit="C-e", tqubit="C-e+1")      # 5, 12, 9, 13, 2, 6
        qc.H("C-e")
        qc.H("C-e+1")
        qc.H("A-e")
        qc.H("A-e+1")
        qc.apply_gate(CNOT_gate, cqubit="A-e", tqubit="A-e+1")    # 5, 12, 9, 13
        qc.H("A-e")
        qc.H("A-e+1")
        measurement_outcomes = qc.measure(["C-e", "A-e"], basis="Z")           # 9, 13, 2, 6
        success = type(measurement_outcomes) == SKIP or measurement_outcomes[0] == measurement_outcomes[1]

        qc.start_sub_circuit("BD")
        success_bd = False
        while not success_bd:
            qc.create_bell_pair("B-e", "D-e")
            qc.SWAP("B-e", "B-e+2", efficient=True)
            qc.SWAP("D-e", "D-e+2", efficient=True)
            success_bd = qc.single_selection(CZ_gate, "B-e", "D-e", "B-e+2", "D-e+2", swap=True)

        qc.SWAP("B-e", "B-e+2", efficient=True)
        qc.SWAP("D-e", "D-e+2", efficient=True)
        qc.apply_gate(CZ_gate, cqubit="B-e", tqubit="B-e+1", reverse=True)  # 5, 12, 9, 13
        qc.apply_gate(CZ_gate, cqubit="D-e", tqubit="D-e+1", reverse=True)  # 5, 12, 9, 13, 2, 6
        measurement_outcomes2 = qc.measure(["D-e", "B-e"], basis="X")
        distillation_success = measurement_outcomes2[0] == measurement_outcomes2[1]
        if not success:
            qc.start_sub_circuit("CD")
            qc.X("C-e+1")
            qc.X("D-e+1")
        ghz_success = type(measurement_outcomes2) == SKIP or distillation_success == success
        PBAR.update(20) if PBAR is not None else None

    # qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"], swap=True)
    # PBAR.update(20) if PBAR is not None else None

    return ["B", "A", "D", "C"]


def dyn_prot_4_22_1(qc: QuantumCircuit, *, operation):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        qc.start_sub_circuit("AB")
        success_ab = False
        while not success_ab:
            qc.create_bell_pair(15, 11)
            success_ab = qc.single_selection(CiY_gate, 14, 10, 15, 11)
            if not success_ab:
                continue
            success_ab2 = False
            while not success_ab2:
                qc.create_bell_pair(14, 10)
                success_ab2 = qc.single_selection(CZ_gate, 13, 9)
            success_ab = qc.single_selection(CNOT_gate, 14, 10, 15, 11, create_bell_pair=False)

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair(6, 2)
            success_cd = qc.single_selection(CiY_gate, 7, 3, 6, 2)
            if not success_cd:
                continue
            success_cd = qc.single_selection(CZ_gate, 7, 3, 6, 2)
            if not success_cd:
                continue
            success_cd2 = False
            while not success_cd2:
                qc.create_bell_pair(7, 3)
                success_cd2 = qc.single_selection(CiY_gate, 5, 1, 7, 3)
            success_cd = qc.single_selection(CNOT_gate, 7, 3, 6, 2, create_bell_pair=False)
            if not success_cd:
                continue
            success_cd2 = False
            while not success_cd2:
                qc.create_bell_pair(7, 3)
                success_cd2 = qc.single_selection(CNOT_gate, 5, 1, 7, 3)
                if not success_cd2:
                    continue
                success_cd3 = False
                while not success_cd3:
                    qc.create_bell_pair(5, 1)
                    success_cd3 = qc.single_selection(CiY_gate, 4, 0, 5, 1)
                success_cd2 = qc.single_selection(CZ_gate, 5, 1, 7, 3, create_bell_pair=False)
            success_cd = qc.single_selection(CZ_gate, 7, 3, 6, 2, create_bell_pair=False)

        PBAR.update(30) if PBAR is not None else None

        qc.start_sub_circuit("AC")
        success_ac = False
        while not success_ac:
            qc.create_bell_pair(7, 14)
            success_ac = qc.single_selection(CiY_gate, 13, 5, 14, 7)
            if not success_ac:
                continue
            success_ac2 = False
            while not success_ac2:
                qc.create_bell_pair(13, 5)
                success_ac2 = qc.single_selection(CZ_gate, 12, 4)
            success_ac = qc.single_selection(CNOT_gate, 13, 5, 14, 7, create_bell_pair=False)

        qc.start_sub_circuit("BD")
        success_bd = False
        while not success_bd:
            qc.create_bell_pair(3, 10)
            success_bd = qc.single_selection(CNOT_gate, 9, 1, 10, 3)
            if not success_bd:
                continue
            success_bd2 = False
            while not success_bd2:
                qc.create_bell_pair(8, 0)
                success_bd2 = qc.single_selection(CZ_gate, 9, 1, 8, 0)
                if not success_bd2:
                    continue
                qc.create_bell_pair(9, 1)
                success_bd2 = qc.single_selection(CZ_gate, 8, 0, 9, 1, create_bell_pair=False)
            success_bd = qc.single_selection(CiY_gate, 9, 1, 10, 3, create_bell_pair=False)

        qc.start_sub_circuit("AB", forced_level=True)
        qc.apply_gate(CNOT_gate, cqubit=15, tqubit=14, reverse=True)    # 14, 7, 11, 15
        # qc.start_sub_circuit("C")
        qc.apply_gate(CNOT_gate, cqubit=11, tqubit=10, reverse=True)      # 10, 3, 14, 7, 11, 15
        # qc.start_sub_circuit("AC")
        # qc._thread_safe_printing = False
        # qc.draw_circuit()
        measurement_outcomes = qc.measure([10, 14], basis="Z")           # 3, 7, 11, 15
        success = type(measurement_outcomes) == SKIP or measurement_outcomes[0] == measurement_outcomes[1]
        qc.start_sub_circuit("AC")
        if not success:
            qc.X(15)
            qc.X(7)
        qc.start_sub_circuit("CD")
        qc.apply_gate(CZ_gate, cqubit=7, tqubit=6, reverse=True)        # 2, 6, 3, 7, 11, 15
        # qc.start_sub_circuit("D")
        qc.apply_gate(CZ_gate, cqubit=3, tqubit=2)
        # qc.start_sub_circuit("BD")
        measurement_outcomes2 = qc.measure([2, 6])      # 3, 7, 11, 15
        ghz_success = type(measurement_outcomes) == SKIP or measurement_outcomes2[1]
        PBAR.update(20) if PBAR is not None else None

    # qc.stabilizer_measurement(operation, nodes=["D", "C", "B", "A"], swap=False)
    # PBAR.update(20) if PBAR is not None else None
    # qc.append_print_lines("\nGHZ fidelity: {}\n".format(qc.ghz_fidelity))

    return ["D", "C", "B", "A"]


def expedient(qc: QuantumCircuit, *, operation):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        # Step 1-2 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AB")
        success_ab = False
        while not success_ab:
            qc.create_bell_pair(11, 8)
            success_ab = qc.double_selection(CZ_gate, 10, 7)
            if not success_ab:
                continue
            success_ab = qc.double_selection(CNOT_gate, 10, 7)

        PBAR.update(20) if PBAR is not None else None

        # Step 1-2 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair(5, 2)
            success_cd = qc.double_selection(CZ_gate, 4, 1)
            if not success_cd:
                continue
            success_cd = qc.double_selection(CNOT_gate, 4, 1)

        PBAR.update(20) if PBAR is not None else None

        # Step 3-5 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit('AC')
        outcome_ac = qc.single_dot(CZ_gate, "A-e+1", "C-e+1")
        qc.start_sub_circuit('BD')
        outcome_bd = qc.single_dot(CZ_gate, "B-e+1", "D-e+1")
        qc.start_sub_circuit("ABCD")
        ghz_success = outcome_bd == outcome_ac
        if not ghz_success:
            continue
        if not outcome_bd:
            qc.X("A-e+1")
            qc.X("B-e+1")

        PBAR.update(20) if PBAR is not None else None

        # Step 6-8 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC", forced_level=True)
        ghz_success_1 = qc.single_dot(CZ_gate, 10, 4)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.single_dot(CZ_gate, 7, 1)
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

        PBAR.update(10) if PBAR is not None else None

    # Step 9 from Table D.1 (Thesis Naomi Nickerson)
    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    # qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"])
    # PBAR.update(20) if PBAR is not None else None

    return ["B", "A", "D", "C"]


def stringent(qc: QuantumCircuit, *, operation):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        # Step 1-8 from Table D.2 (Thesis Naomi Nickerson)
        success_ab = False
        qc.start_sub_circuit("AB")
        while not success_ab:
            qc.create_bell_pair(11, 8)
            success_ab = qc.double_selection(CZ_gate, 10, 7)
            if not success_ab:
                continue
            success_ab = qc.double_selection(CNOT_gate, 10, 7)
            if not success_ab:
                continue

            success_ab = qc.double_dot(CZ_gate, 10, 7)
            if not success_ab:
                continue
            success_ab = qc.double_dot(CNOT_gate, 10, 7)
            if not success_ab:
                continue

        PBAR.update(20) if PBAR is not None else None

        # Step 1-8 from Table D.2 (Thesis Naomi Nickerson)
        success_cd = False
        qc.start_sub_circuit("CD")
        while not success_cd:
            qc.create_bell_pair(5, 2)
            success_cd = qc.double_selection(CZ_gate, 4, 1)
            if not success_cd:
                continue
            success_cd = qc.double_selection(CNOT_gate, 4, 1)
            if not success_cd:
                continue

            success_cd = qc.double_dot(CZ_gate, 4, 1)
            if not success_cd:
                continue
            success_cd = qc.double_dot(CNOT_gate, 4, 1)
            if not success_cd:
                continue

        PBAR.update(20) if PBAR is not None else None

        # Step 9-11 from Table D.2 (Thesis Naomi Nickerson)
        qc.start_sub_circuit('AC')
        outcome_ac, single_selection_success_ac = qc.double_dot(CZ_gate, "A-e+1", "C-e+1")
        qc.start_sub_circuit('BD')
        outcome_bd, single_selection_success_bd = qc.double_dot(CZ_gate, "B-e+1", "D-e+1")
        qc.start_sub_circuit("ABCD", forced_level=True)
        ghz_success = outcome_bd == outcome_ac and single_selection_success_ac and single_selection_success_bd
        if not ghz_success:
            continue
        if not outcome_bd:
            qc.X("A-e+1")
            qc.X("B-e+1")

        PBAR.update(20) if PBAR is not None else None

        # Step 12-14 from Table D.2 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC", forced_level=True)
        ghz_success_1 = qc.double_dot(CZ_gate, 10, 4)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.double_dot(CZ_gate, 7, 1)
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

        PBAR.update(10) if PBAR is not None else None

    # Step 15 from Table D.2 (Thesis Naomi Nickerson)
    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    # qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"])
    # PBAR.update(20) if PBAR is not None else None

    return ["B", "A", "D", "C"]


def expedient_swap(qc: QuantumCircuit, *, operation, tqubit=None):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        qc.start_sub_circuit("AB")
        success_ab = False
        while not success_ab:
            qc.create_bell_pair("A-e", "B-e")
            qc.SWAP("A-e", "A-e+1", efficient=True)
            qc.SWAP("B-e", "B-e+1", efficient=True)
            success_ab = qc.double_selection(CZ_gate, "A-e", "B-e", swap=True)
            if not success_ab:
                continue
            success_ab = qc.double_selection(CNOT_gate, "A-e", "B-e", swap=True)

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair("C-e", "D-e")
            qc.SWAP("C-e", "C-e+1", efficient=True)
            qc.SWAP("D-e", "D-e+1", efficient=True)
            success_cd = qc.double_selection(CZ_gate, "C-e", "D-e", swap=True)
            if not success_cd:
                continue
            success_cd = qc.double_selection(CNOT_gate, "C-e", "D-e", swap=True)

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit('AC')
        outcome_ac = qc.single_dot(CZ_gate, "A-e", "C-e", swap=True)
        qc.start_sub_circuit('BD')
        outcome_bd = qc.single_dot(CZ_gate, "B-e", "D-e", swap=True)
        qc.start_sub_circuit("ABCD", forced_level=True)
        ghz_success = outcome_bd == outcome_ac if not qc.cut_off_time_reached else True
        if not ghz_success:
            continue
        if not outcome_bd:
            qc.X("A-e+1")
            qc.X("B-e+1")

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit('AC', forced_level=True)
        ghz_success_1 = qc.single_dot(CZ_gate, "A-e", "C-e", swap=True)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.single_dot(CZ_gate, "B-e", "D-e", swap=True)
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

    PBAR.update(10) if PBAR is not None else None

    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    # qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"], swap=True, tqubit=tqubit)
    # PBAR.update(20) if PBAR is not None else None

    return ["B", "A", "D", "C"]


def stringent_swap(qc: QuantumCircuit, *, operation):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        qc.start_sub_circuit("AB")
        success_ab = False
        while not success_ab:
            qc.create_bell_pair(9, 6)
            qc.SWAP(9, 10, efficient=True)
            qc.SWAP(6, 7, efficient=True)
            if not qc.double_selection(CZ_gate, 9, 6, swap=True):
                continue
            if not qc.double_selection(CNOT_gate, 9, 6, swap=True):
                continue
            outcome = qc.double_dot(CZ_gate, 9, 6, swap=True)
            if outcome != SKIP() and not all(outcome):
                continue
            outcome_2 = qc.double_dot(CNOT_gate, 9, 6, swap=True)
            if outcome_2 == SKIP() or all(outcome_2):
                success_ab = True

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair(3, 0)
            qc.SWAP(3, 4, efficient=True)
            qc.SWAP(0, 1, efficient=True)
            if not qc.double_selection(CZ_gate, 3, 0, swap=True):
                continue
            if not qc.double_selection(CNOT_gate, 3, 0, swap=True):
                continue
            outcome = qc.double_dot(CZ_gate, 3, 0, swap=True)
            if outcome != SKIP() and not all(outcome):
                continue
            outcome_2 = qc.double_dot(CNOT_gate, 3, 0, swap=True)
            if outcome_2 == SKIP() or all(outcome_2):
                success_cd = True

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit('AC')
        outcome_ac = qc.double_dot(CZ_gate, "A-e", "C-e", swap=True)
        qc.start_sub_circuit('BD')
        outcome_bd = qc.double_dot(CZ_gate, "B-e", "D-e", swap=True)
        qc.start_sub_circuit("ABCD", forced_level=True)
        ghz_success = (outcome_bd[0] == outcome_ac[0] and outcome_ac[1] and outcome_bd[1]
                       if not qc.cut_off_time_reached else True)
        if not ghz_success:
            continue
        if outcome_bd != SKIP() and not outcome_bd[0]:
            qc.X("A-e+1")
            qc.X("B-e+1")

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("AC", forced_level=True)
        ghz_success_1 = qc.double_dot(CZ_gate, 9, 3, swap=True)
        ghz_success_1 = all(ghz_success_1) if ghz_success_1 != SKIP() else True
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.double_dot(CZ_gate, 6, 0, swap=True)
        ghz_success_2 = all(ghz_success_2) if ghz_success_2 != SKIP() else True
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

        PBAR.update(10) if PBAR is not None else None

    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    # qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"], swap=True)
    # PBAR.update(20) if PBAR is not None else None

    return ["B", "A", "D", "C"]


def basic_medium_refined_ghz_block(qc: QuantumCircuit, offset=1, prot='basic'):
    ghz_success = False
    while not ghz_success:
        for round in [1, 2]:
            outcomes = []
            sub_circuits = ["AB", "CD"] if round == 1 else ["AC", "BD"]
            for circuit in sub_circuits:
                bell_success = False
                while not bell_success:
                    qc.start_sub_circuit(circuit)
                    qc.create_bell_pair(f"{circuit[0]}-e", f"{circuit[1]}-e")
                    mem = offset if round == 1 else 3
                    if round == 1 or (round == 2 and prot in ['medium', 'refined']):
                        qc.SWAP(f"{circuit[0]}-e", f"{circuit[0]}-e+{mem}", efficient=True)
                        qc.SWAP(f"{circuit[1]}-e", f"{circuit[1]}-e+{mem}", efficient=True)

                    if prot == 'medium':
                        bell_success = qc.single_selection(CZ_gate, f"{circuit[0]}-e", f"{circuit[1]}-e",
                                                           f"{circuit[0]}-e+{mem}", f"{circuit[1]}-e+{mem}")
                        if type(bell_success) == SKIP:
                            break
                    elif prot == 'refined':
                        qc.single_selection(CZ_gate, f"{circuit[0]}-e", f"{circuit[1]}-e",
                                            f"{circuit[0]}-e+{mem}", f"{circuit[1]}-e+{mem}", measure=False)
                        qc.SWAP(f"{circuit[0]}-e", f"{circuit[0]}-e+{round + 2}", efficient=True)
                        qc.SWAP(f"{circuit[1]}-e", f"{circuit[1]}-e+{round + 2}", efficient=True)
                        bell_result_1 = qc.single_selection(CZ_gate, f"{circuit[0]}-e", f"{circuit[1]}-e",
                                                            f"{circuit[0]}-e+{round + 2}", f"{circuit[1]}-e+{round + 2}")
                        if type(bell_result_1) == SKIP:
                            break
                        if bell_result_1 is False:
                            continue
                        qc.SWAP(f"{circuit[0]}-e", f"{circuit[0]}-e+{round + 2}", efficient=True)
                        qc.SWAP(f"{circuit[1]}-e", f"{circuit[1]}-e+{round + 2}", efficient=True)
                        bell_result_2 = qc.measure([f"{circuit[1]}-e", f"{circuit[0]}-e"])
                        if type(bell_result_2) == SKIP:
                            break
                        if bell_result_2[0] != bell_result_2[1]:
                            continue

                        qc.single_selection(CNOT_gate, f"{circuit[0]}-e", f"{circuit[1]}-e",
                                            f"{circuit[0]}-e+{mem}", f"{circuit[1]}-e+{mem}", measure=False)
                        qc.SWAP(f"{circuit[0]}-e", f"{circuit[0]}-e+{round + 2}", efficient=True)
                        qc.SWAP(f"{circuit[1]}-e", f"{circuit[1]}-e+{round + 2}", efficient=True)
                        bell_result_3 = qc.single_selection(CNOT_gate, f"{circuit[0]}-e", f"{circuit[1]}-e",
                                                            f"{circuit[0]}-e+{round + 2}",
                                                            f"{circuit[1]}-e+{round + 2}")
                        if type(bell_result_3) == SKIP:
                            break
                        if bell_result_3 is False:
                            continue
                        qc.SWAP(f"{circuit[0]}-e", f"{circuit[0]}-e+{round + 2}", efficient=True)
                        qc.SWAP(f"{circuit[1]}-e", f"{circuit[1]}-e+{round + 2}", efficient=True)
                        bell_result_4 = qc.measure([f"{circuit[1]}-e", f"{circuit[0]}-e"])
                        if type(bell_result_4) == SKIP:
                            break
                        bell_success = bell_result_4[0] == bell_result_4[1]
                    else:
                        bell_success = True
                if round == 2:
                    if prot in ['medium', 'refined']:
                        qc.SWAP(f"{circuit[0]}-e", f"{circuit[0]}-e+3", efficient=True)
                        qc.SWAP(f"{circuit[1]}-e", f"{circuit[1]}-e+3", efficient=True)
                    outcomes.append(qc.single_selection(CZ_gate, f"{circuit[0]}-e", f"{circuit[1]}-e",
                                                        f"{circuit[0]}-e+{offset}", f"{circuit[1]}-e+{offset}",
                                                        create_bell_pair=False))
        qc.start_sub_circuit("ABCD", forced_level=True)
        ghz_success = outcomes[0] == outcomes[1] if not qc.cut_off_time_reached else True
        if not ghz_success:
            continue
        if not qc.cut_off_time_reached and not outcomes[1]:
            qc.X(f"A-e+{offset}")
            qc.X(f"B-e+{offset}")


def basic_medium_refined_protocol(qc: QuantumCircuit, version=1, prot='basic'):
    ghz_success_main = False
    while not ghz_success_main:
        basic_medium_refined_ghz_block(qc, offset=1, prot=prot)  # Create a GHZ state between A-e+1, B-e+1, C-e+1 and D-e+1
        if not qc.cut_off_time_reached:
            basic_medium_refined_ghz_block(qc, offset=2, prot=prot)  # Create a GHZ state between A-e+2, B-e+2, C-e+2 and D-e+2
        if not qc.cut_off_time_reached:
            meas_res = []
            qc.start_sub_circuit("ABCD", forced_level=True)
            for i_n, node in enumerate(["B", "A", "D", "C"]):
                if version == 1: # The version number indicates which GHZ state is consumed with the distillation step
                    qc.SWAP(f"{node}-e", f"{node}-e+1", efficient=True)
                    qc.CNOT(f"{node}-e", f"{node}-e+2")
                else:
                    qc.SWAP(f"{node}-e", f"{node}-e+2", efficient=True)
                    qc.CNOT(f"{node}-e", f"{node}-e+1")
                result = qc.measure(f"{node}-e")
                if type(result) != SKIP:
                    result = result[0]
                meas_res.append(result)
            ghz_success_main = meas_res.count(1) % 2 == 0 if SKIP() not in meas_res else True
        else:
            break

    return ["B", "A", "D", "C"]


def basic1_swap(qc: QuantumCircuit, *, operation):
    return basic_medium_refined_protocol(qc, version=1, prot='basic')


def basic2_swap(qc: QuantumCircuit, *, operation):
    return basic_medium_refined_protocol(qc, version=2, prot='basic')


def medium1_swap(qc: QuantumCircuit, *, operation):
    return basic_medium_refined_protocol(qc, version=1, prot='medium')


def medium2_swap(qc: QuantumCircuit, *, operation):
    return basic_medium_refined_protocol(qc, version=2, prot='medium')


def refined1_swap(qc: QuantumCircuit, *, operation):
    return basic_medium_refined_protocol(qc, version=1, prot='refined')


def refined2_swap(qc: QuantumCircuit, *, operation):
    return basic_medium_refined_protocol(qc, version=2, prot='refined')


def minimum4x_40_ghz_block(qc: QuantumCircuit, offset=1):
    ghz_success = False
    while not ghz_success:
        outcomes = []
        for circuit in ["AB", "CD", "AC", "BD"]:
            gate = CNOT_gate if circuit in ["AB", "CD"] else CZ_gate
            mem = offset if circuit in ["AB", "CD"] else 3
            bell_success = False
            while not bell_success:
                qc.start_sub_circuit(circuit)
                qc.create_bell_pair(f"{circuit[0]}-e", f"{circuit[1]}-e")
                qc.SWAP(f"{circuit[0]}-e", f"{circuit[0]}-e+{mem}", efficient=True)
                qc.SWAP(f"{circuit[1]}-e", f"{circuit[1]}-e+{mem}", efficient=True)
                bell_success = qc.single_selection(gate, f"{circuit[0]}-e", f"{circuit[1]}-e",
                                                   f"{circuit[0]}-e+{mem}", f"{circuit[1]}-e+{mem}")
                if type(bell_success) == SKIP:
                    break
            if circuit in ["AC", "BD"]:
                qc.SWAP(f"{circuit[0]}-e", f"{circuit[0]}-e+3", efficient=True)
                qc.SWAP(f"{circuit[1]}-e", f"{circuit[1]}-e+3", efficient=True)
                outcomes.append(qc.single_selection(CZ_gate, f"{circuit[0]}-e", f"{circuit[1]}-e",
                                                    f"{circuit[0]}-e+{offset}", f"{circuit[1]}-e+{offset}",
                                                    create_bell_pair=False))
        qc.start_sub_circuit("ABCD", forced_level=True)
        ghz_success = outcomes[0] == outcomes[1] if not qc.cut_off_time_reached else True
        if not ghz_success:
            continue
        if not qc.cut_off_time_reached and not outcomes[1]:
            qc.X(f"A-e+{offset}")
            qc.X(f"B-e+{offset}")

        for circuit in ["AB", "CD", "AC", "BD"]:
            bell_success = False
            if circuit in ["AB", "AC"]:
                outcomes = []
            while not bell_success:
                qc.start_sub_circuit(circuit)
                qc.create_bell_pair(f"{circuit[0]}-e", f"{circuit[1]}-e")
                qc.SWAP(f"{circuit[0]}-e", f"{circuit[0]}-e+3", efficient=True)
                qc.SWAP(f"{circuit[1]}-e", f"{circuit[1]}-e+3", efficient=True)
                bell_success = qc.single_selection(CNOT_gate, f"{circuit[0]}-e", f"{circuit[1]}-e",
                                                   f"{circuit[0]}-e+3", f"{circuit[1]}-e+3")
                if type(bell_success) == SKIP:
                    break
                if bell_success is False:
                    continue
                bell_success = qc.single_selection(CZ_gate, f"{circuit[0]}-e", f"{circuit[1]}-e",
                                                   f"{circuit[0]}-e+3", f"{circuit[1]}-e+3")
                if type(bell_success) == SKIP:
                    break
            qc.SWAP(f"{circuit[0]}-e", f"{circuit[0]}-e+3", efficient=True)
            qc.SWAP(f"{circuit[1]}-e", f"{circuit[1]}-e+3", efficient=True)
            outcome = qc.single_selection(CZ_gate, f"{circuit[0]}-e", f"{circuit[1]}-e",
                                          f"{circuit[0]}-e+{offset}", f"{circuit[1]}-e+{offset}",
                                          create_bell_pair=False)
            outcomes.append(outcome)
            if circuit in ["CD", "BD"]:
                if SKIP() in outcomes or qc.cut_off_time_reached:
                    ghz_success = True
                    break
                else:
                    ghz_success = outcomes[0] == outcomes[1]
                if not ghz_success:
                    break


def minimum4x_40_swap(qc: QuantumCircuit, version=1):
    ghz_success_main = False
    while not ghz_success_main:
        minimum4x_40_ghz_block(qc, offset=1)  # Create a GHZ state between A-e+1, B-e+1, C-e+1 and D-e+1
        if not qc.cut_off_time_reached:
            minimum4x_40_ghz_block(qc, offset=2)  # Create a GHZ state between A-e+2, B-e+2, C-e+2 and D-e+2
        if not qc.cut_off_time_reached:
            meas_res = []
            qc.start_sub_circuit("ABCD", forced_level=True)
            for i_n, node in enumerate(["B", "A", "D", "C"]):
                if version == 1:
                    qc.SWAP(f"{node}-e", f"{node}-e+1", efficient=True)
                    qc.CNOT(f"{node}-e", f"{node}-e+2")
                else:
                    qc.SWAP(f"{node}-e", f"{node}-e+2", efficient=True)
                    qc.CNOT(f"{node}-e", f"{node}-e+1")
                result = qc.measure(f"{node}-e")
                if type(result) != SKIP:
                    result = result[0]
                meas_res.append(result)
            ghz_success_main = meas_res.count(1) % 2 == 0 if SKIP() not in meas_res else True
        else:
            break

    return ["B", "A", "D", "C"]


def minimum4x_40_1_swap(qc: QuantumCircuit, *, operation):
    return minimum4x_40_swap(qc, version=1)


def minimum4x_40_2_swap(qc: QuantumCircuit, *, operation):
    return minimum4x_40_swap(qc, version=2)


def minimum4x_22_swap(qc: QuantumCircuit, *, operation):
    ghz_success = False
    while not ghz_success:
        outcomes = []
        for circuit in ["AB", "CD", "AC", "BD"]:
            if circuit in ["AB", "CD"]:
                gates = [CNOT_gate, CNOT_gate]
            elif circuit == "AC":
                gates = [CZ_gate, CZ_gate]
            elif circuit == "BD":
                gates = [CNOT_gate, CZ_gate]
            mem = 1 if circuit in ["AB", "CD"] else 2
            bell_success = False
            while not bell_success:
                qc.start_sub_circuit(circuit)
                qc.create_bell_pair(f"{circuit[0]}-e", f"{circuit[1]}-e")
                qc.SWAP(f"{circuit[0]}-e", f"{circuit[0]}-e+{mem}", efficient=True)
                qc.SWAP(f"{circuit[1]}-e", f"{circuit[1]}-e+{mem}", efficient=True)
                bell_success = qc.single_selection(gates[0], f"{circuit[0]}-e", f"{circuit[1]}-e",
                                                   f"{circuit[0]}-e+{mem}", f"{circuit[1]}-e+{mem}")
                if type(bell_success) == SKIP:
                    break
                if bell_success is False:
                    continue
                bell_success = qc.single_selection(gates[1], f"{circuit[0]}-e", f"{circuit[1]}-e",
                                                   f"{circuit[0]}-e+{mem}", f"{circuit[1]}-e+{mem}")
                if type(bell_success) == SKIP:
                    break
            if circuit in ["AC", "BD"]:
                qc.SWAP(f"{circuit[0]}-e", f"{circuit[0]}-e+2", efficient=True)
                qc.SWAP(f"{circuit[1]}-e", f"{circuit[1]}-e+2", efficient=True)
                outcomes.append(qc.single_selection(CZ_gate, f"{circuit[0]}-e", f"{circuit[1]}-e",
                                                    f"{circuit[0]}-e+1", f"{circuit[1]}-e+1",
                                                    create_bell_pair=False))
        qc.start_sub_circuit("ABCD", forced_level=True)
        ghz_success = outcomes[0] == outcomes[1] if not qc.cut_off_time_reached else True
        if not ghz_success:
            continue
        if not qc.cut_off_time_reached and not outcomes[1]:
            qc.X(f"A-e+1")
            qc.X(f"B-e+1")

        outcomes = []
        gates = [CNOT_gate, CZ_gate, CNOT_gate, CZ_gate]
        for circuit in ["AD", "BC"]:
            bell_success = False
            while not bell_success:
                qc.start_sub_circuit(circuit, forced_level=True if circuit == "AD" else False)
                qc.create_bell_pair(f"{circuit[0]}-e", f"{circuit[1]}-e")
                qc.SWAP(f"{circuit[0]}-e", f"{circuit[0]}-e+2", efficient=True)
                qc.SWAP(f"{circuit[1]}-e", f"{circuit[1]}-e+2", efficient=True)
                for gate_number in range(4):
                    bell_success = qc.single_selection(gates[gate_number], f"{circuit[0]}-e", f"{circuit[1]}-e",
                                                       f"{circuit[0]}-e+2", f"{circuit[1]}-e+2")
                    if type(bell_success) == SKIP:
                        break
                    if bell_success is False:
                        break
                if type(bell_success) == SKIP:
                    break
            qc.SWAP(f"{circuit[0]}-e", f"{circuit[0]}-e+2", efficient=True)
            qc.SWAP(f"{circuit[1]}-e", f"{circuit[1]}-e+2", efficient=True)
            outcomes.append(qc.single_selection(CZ_gate, f"{circuit[0]}-e", f"{circuit[1]}-e",
                                                f"{circuit[0]}-e+1", f"{circuit[1]}-e+1",
                                                create_bell_pair=False))
        ghz_success = all(outcomes) if not (SKIP() in outcomes or qc.cut_off_time_reached) else True

    return ["B", "A", "D", "C"]


def direct_ghz(qc: QuantumCircuit, *, operation):
    qc.create_ghz_state_direct([*range(4)])
    return ["A", "B", "C", "D"]


def direct_ghz_swap(qc: QuantumCircuit, *, operation):
    return direct_ghz(qc, operation=operation)


def duo_structure(qc: QuantumCircuit, *, operation):
    qc.start_sub_circuit("AB")
    qc.create_bell_pair(2, 5)
    qc.double_selection(CZ_gate, 1, 4)
    qc.double_selection(CNOT_gate, 1, 4)

    PBAR.update(30) if PBAR is not None else None

    # qc.stabilizer_measurement(operation, nodes=["A", "B"])
    # PBAR.update(20) if PBAR is not None else None

    return ["A", "B"]

# def test_protocol_swap(qc: QuantumCircuit, *, operation):
#
#     print("\nTime step 0")
#     qc.start_sub_circuit("CD")
#
#     print("Subcircuit CD")
#     qc.create_bell_pair("C-e", "D-e")
#     qc.SWAP("C-e", "C-e+1", efficient=True)
#     qc.SWAP("D-e", "D-e+1", efficient=True)
#     qc.create_bell_pair("C-e", "D-e")
#     qc.apply_gate(CNOT_gate, cqubit="C-e", tqubit="C-e+1")
#     qc.apply_gate(CNOT_gate, cqubit="D-e", tqubit="D-e+1")
#     qc.measure(["C-e", "D-e"], basis="X")
#     print(f"Node times: {[qc.nodes[node].sub_circuit_time for node in 'ABCD']}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
#     qc.start_sub_circuit("AB")
#     print("Subcircuit AB")
#     print(f"Node times: {[qc.nodes[node].sub_circuit_time for node in 'ABCD']}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
#     qc.create_bell_pair("A-e", "B-e")
#     qc.SWAP("A-e", "A-e+1", efficient=True)
#     qc.SWAP("B-e", "B-e+1", efficient=True)
#     qc.create_bell_pair("A-e", "B-e")
#     qc.apply_gate(CNOT_gate, cqubit="A-e", tqubit="A-e+1")
#     qc.apply_gate(CNOT_gate, cqubit="B-e", tqubit="B-e+1")
#     qc.measure(["A-e", "B-e"], basis="X")
#
#     print(f"\nNode times: {[qc.nodes[node].sub_circuit_time for node in 'ABCD']}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
#     print("Time step 1")
#     qc.start_sub_circuit("BC")
#     print("Subcircuit BC")
#     print(f"Node times: {[qc.nodes[node].sub_circuit_time for node in 'ABCD']}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
#     qc.create_bell_pair("C-e", "B-e")
#     qc.H("C-e")
#     qc.apply_gate(CZ_gate, cqubit="C-e", tqubit="C-e+1")
#     qc.H("C-e")
#     qc.measure("C-e", basis="Z")
#     qc.H("B-e")
#     qc.apply_gate(CZ_gate, cqubit="B-e", tqubit="B-e+1")
#     qc.H("B-e")
#     qc.measure("B-e", basis="Z")
#
#     print(f"\nNode times: {[qc.nodes[node].sub_circuit_time for node in 'ABCD']}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
#     print("Time step 2")
#     qc.start_sub_circuit("AB")
#     print("Subcircuit AB")
#     print(f"Node times: {[qc.nodes[node].sub_circuit_time for node in 'ABCD']}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
#     qc.create_bell_pair("A-e", "B-e")
#     qc.SWAP("A-e", "A-e+2", efficient=True)
#     qc.SWAP("B-e", "B-e+2", efficient=True)
#     qc.create_bell_pair("A-e", "B-e")
#     qc.apply_gate(CNOT_gate, cqubit="A-e", tqubit="A-e+2")
#     qc.apply_gate(CNOT_gate, cqubit="B-e", tqubit="B-e+2")
#     qc.measure(["A-e", "B-e"], basis="X")
#
#     print(f"\nNode times: {[qc.nodes[node].sub_circuit_time for node in 'ABCD']}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
#     print("Time step 3")
#     qc.start_sub_circuit("AC")
#     print("Subcircuit AC")
#     print(f"Node times: {[qc.nodes[node].sub_circuit_time for node in 'ABCD']}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
#     qc.create_bell_pair("A-e", "C-e")
#     qc.H("A-e")
#     qc.apply_gate(CZ_gate, cqubit="A-e", tqubit="A-e+2")
#     qc.H("A-e")
#     qc.measure("A-e", basis="Z")
#     qc.SWAP("A-e", "A-e+2", efficient=True)
#     print(f"Node times: {[qc.nodes[node].sub_circuit_time for node in 'ABCD']}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
#     qc.start_sub_circuit("BD")
#     print("Subcircuit BD")
#     print(f"Node times: {[qc.nodes[node].sub_circuit_time for node in 'ABCD']}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
#     qc.create_bell_pair("B-e", "D-e")
#     qc.H("B-e")
#     qc.apply_gate(CZ_gate, cqubit="B-e", tqubit="B-e+2")
#     qc.H("B-e")
#     qc.measure("B-e", basis="Z")
#     qc.SWAP("B-e", "B-e+2", efficient=True)
#
#     print(f"\nNode times: {[qc.nodes[node].sub_circuit_time for node in 'ABCD']}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
#     print("Time step 4")
#     qc.start_sub_circuit("ABCD", forced_level=True)
#     print("Subcircuit ABCD")
#     print(f"Node times: {[qc.nodes[node].sub_circuit_time for node in 'ABCD']}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
#     qc.apply_gate(CiY_gate, cqubit="A-e", tqubit="A-e+1")
#     qc.apply_gate(CNOT_gate, cqubit="B-e", tqubit="B-e+1")
#     qc.apply_gate(CNOT_gate, cqubit="C-e", tqubit="C-e+1")
#     qc.apply_gate(CiY_gate, cqubit="D-e", tqubit="D-e+1")
#     qc.measure(["A-e", "B-e", "C-e", "D-e"], basis="X")
#     qc.create_bell_pair("B-e", "C-e")
#     qc.apply_gate(CZ_gate, cqubit="B-e", tqubit="B-e+1")
#     qc.apply_gate(CZ_gate, cqubit="C-e", tqubit="C-e+1")
#     qc.measure(["B-e", "C-e"], basis="X")
#
#     print(f"\n\nNode times: {[qc.nodes[node].sub_circuit_time for node in 'ABCD']}; subcircuit times: { {sc: qc._sub_circuits[sc].total_duration for sc in qc._sub_circuits.keys()} }, total time: {qc.total_duration}.")
#
#     return ["C", "A", "B", "D"]

# def weight_2_4_swap(qc: QuantumCircuit, *, operation):
#     expedient_swap(qc, operation=operation, tqubit=[26, 30, 18, 22])
#
#     return [[30, 26, 22, 18], [28, 24, 20, 16]]
#
#
# def weight_2_4_secondary_swap(qc: QuantumCircuit, *, operation):
#     bipartite_4_swap(qc, operation=operation)


def weight_3_swap(qc: QuantumCircuit, *, operation):
    stab_meas_nodes = dyn_prot_3_8_1_swap(qc, operation=operation, tqubit=[30, 28, 26, 22])
    return stab_meas_nodes


def weight_3_direct_swap(qc: QuantumCircuit, *, operation):
    qc.create_ghz_state_direct([*range(3)])

    return ["C", "B", "A"]

