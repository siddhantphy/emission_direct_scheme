import numpy as np
from .gate import TwoQubitGate, SingleQubitGate

"""
    SINGLE QUBIT GATES
"""
X_gate = SingleQubitGate("X", np.array([[0, 1], [1, 0]]), "X", duration=13e-3, duration_electron=14e-6)
Y_gate = SingleQubitGate("Y", np.array([[0, -1j], [1j, 0]]), "Y", duration=13e-3, duration_electron=14e-6)
Z_gate = SingleQubitGate("Z", np.array([[1, 0], [0, -1]]), "Z", duration=6.5e-3, duration_electron=1e-6)
I_gate = SingleQubitGate("Identity", np.array([[1, 0], [0, 1]]), "I", duration=0)
H_gate = SingleQubitGate("Hadamard", 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]), "H", duration=6.5e-3,
                         duration_electron=1e-6)
S_gate = SingleQubitGate("Phase", np.array([[1, 0], [0, 1j]]), "S")

"""
    TWO-QUBIT GATES
"""
CNOT_gate = TwoQubitGate("CNOT",
                         np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, 1, 0]]),
                         "X",
                         duration=25e-3)
CZ_gate = TwoQubitGate("CPhase",
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, -1]]),
                       "Z",
                       duration=25e-3)
CiY_gate = TwoQubitGate("CiY",
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, -1, 0]]),
                       "iY",
                       duration=25e-3)
NV_two_qubit_gate = TwoQubitGate("NV two-qubit gate",
                                 np.array([[np.cos(np.pi/4), 1 * np.sin(np.pi/4), 0, 0],
                                           [-1 * np.sin(np.pi/4), np.cos(np.pi/4), 0, 0],
                                           [0, 0, np.cos(np.pi/4), -1 * np.sin(np.pi/4)],
                                           [0, 0, 1 * np.sin(np.pi/4), np.cos(np.pi/4)]]),
                                 "NV")

SWAP_gate = TwoQubitGate("Swap",
                         np.array([[1, 0, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1]]),
                         "(X)",
                         control_repr="(X)",
                         duration=0.075)

locals_gates = locals()


def set_duration_of_known_gates(gates_dict):
    for gate, [duration_nuclear, duration_electron] in gates_dict.items():
        if gate in locals_gates and type(locals_gates[gate]) in [SingleQubitGate, TwoQubitGate]:
            if duration_nuclear is not None:
                locals_gates[gate].duration = duration_nuclear
            if duration_electron is not None:
                locals_gates[gate].duration_electron = duration_electron


def set_gate_durations_from_file(filename, noiseless_swap=None):
    if filename is None:
        return
    gates_dict = {}
    with open(filename, 'r') as gate_durations:
        lines = gate_durations.read().split('\n')
        for line in lines:
            line.replace(" ", "")
            if line:
                gate_name, gate_duration = line.split("=")
                if "(" in gate_duration:
                    gate_duration = list(eval(gate_duration))
                else:
                    gate_duration = [float(gate_duration), None]
                gates_dict[gate_name] = gate_duration

    if noiseless_swap and noiseless_swap is True:
        gates_dict["SWAP_gate"] = [float(0), None]
        # for gate in gates_dict.keys():
        #     # If noiseless_swap is True, we set the gate duration times of single-qubit gates
        #     # equal to the electron value
        #     if gates_dict[gate][1] is not None:
        #         gates_dict[gate] = (gates_dict[gate][1], gates_dict[gate][1])

    set_duration_of_known_gates(gates_dict)

    return gates_dict
