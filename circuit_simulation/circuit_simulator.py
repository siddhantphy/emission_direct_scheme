import os
import sys
sys.path.insert(1, os.path.abspath(os.getcwd()))
from circuit_simulation.basic_operations.basic_operations import (
    CT, KP, get_value_by_prob, fidelity, fidelity_elementwise, trace, csr_matrix_equal
)
from circuit_simulation.ghz_states.direct_ghz_states import *
from circuit_simulation.ghz_states.direct_emission_states import *
from circuit_simulation.states.states import *
from circuit_simulation.gates.gates import *
from circuit_simulation.gates.gate import SingleQubitGate
from circuit_simulation.qubit.qubit import Qubit
from circuit_simulation.sub_circuit.sub_quantum_circuit import SubQuantumCircuit
from circuit_simulation.node.node import Node
from scipy import sparse as sp
import hashlib
from circuit_simulation._superoperator.superoperator import SuperoperatorElement
from circuit_simulation.termcolor.termcolor import colored
from circuit_simulation._draw.qasm_to_pdf import create_pdf_from_qasm
from fractions import Fraction as Fr
import math
import random
import cmath
import functools as ft
from circuit_simulation.utilities.decorators import (handle_none_parameters, skip_if_cut_off_reached, SKIP,
                                                     determine_qubit_index)
from copy import copy
import pickle
from pprint import pprint
import time
import cirq
SUM_ACCURACY = 16


class QuantumCircuit:
    """
        QuantumCircuit(num_qubits, init_type=0, noise=False, p_g=0.01, p_m=0.01)

            A QuantumCircuit consists of qubits on which various operations can be applied.
            From this information about the density matrix of the system and others can be
            gathered.

            Parameters
            ----------
            num_qubits : int
                The amount of qubits the system contains.
            init_type : int [0-3], optional, default=0
                Determines how the system is initialised. All these options do NOT include noise.
                The options are:

                0 ->    The system is initialised with all qubits being in the |0> state.
                1 ->    Almost the same as 0, but the first qubit is in the |+> state
                2 ->    The system is initialised with a perfect Bell-pair between all adjacent
                        qubits.
                3 ->    The system is initialised with the first qubit being the |+> state and the
                        rest of the qubits is in the |0> state. On every qubit a CNOT gate is
                        applied with the first qubit being the control qubit.

            noise : bool, optional, default=False
                Will apply noise on every operation that is applied to the QuantumCircuit object,
                unless specified otherwise.
            p_g : float [0-1], optional, default=0.01
                The overall amount of gate noise that will be applied when 'noise' is set to True.
            p_m : float [0-1], optional, default=0.01
                The overall amount of measurement error that will be applied when 'noise' set to
                True. In case p_m_1 is specified, this value holds as the measurement error when a 0-state is
                supposed to be measured.
            p_m_1 : float [0-1], optional, default=None
                The amount of measurement error when a 1-state is supposed to be measured. This can be used in case
                there is a difference in measurement error between an 0-state and an 1-state.
            F_link : float [0-1], optional, default=None
                The overall amount of network noise that will be applied when 'noise is set to True.
            p_dec : float [0-1], optional, default=0
                The overall amount of decoherence in the system. This is only applied when noise is True and
                the value is greater than 0.
            p_link : float [0-1], optional, default=1
                Specifies the success rate of the creation of Bell pairs. Default value is 1, which equals the case
                that a Bell pair creation always instantly succeeds.
            basis_transformation_noise : bool, optional, default = None
                Set to true if the transformation from the computational basis to the X-basis for a
                measurement should be noisy.
            probabilistic : bool, optional, default=False
                In case measurements should be probabilistic of nature, this can be set to True. Measurement
                outcomes will then be determined based on their probabilities if not differently specified
            t_meas : float, optional, default=4
                In case of decoherence, the measurement duration is used to determine the amount of decoherence that
                should be applied for a measurement operation
            t_link : float, optional, default=4
                In case of decoherence, the bell creation duration is used to determine the amount of decoherence that
                should be applied for a measurement operation
            network_noise_type : int, optional, default=0
                The type of network noise that should be used. At this point in time, two variants are
                available:

                0 ->    NV centre specific noise for the creation of a Bell pair
                1 ->    Noise specified by Naomi Nickerson in her master thesis
            no_single_qubit_error : bool, optional, default=False
                When single qubit gates are free of noise, but noise in general is present, this boolean
                is set to True. It prevents the addition of noise when applying a single qubit gate
            thread_safe_printing : bool, optional, default=False
                If working with threads, this can be set to True. This prevents print statements from being
                printed in real-time. Instead the lines will be saved and can at all time be printed all in once
                when running the 'print' method. Print lines are always saved in the _print_lines array until printing


            Attributes
            ----------
            num_qubits : int
                The number of qubits present in the system.
                *** NUMBER IS NOT DEFINITE AND CAN AND WILL BE CHANGED BY SOME METHODS ***
            d : int
                Dimension of the system. This is 2**num_qubits.
            noise: bool, optional, default=False
                If there is general noise present in the system. This will add noise to the gate
                and measurement operations applied to the system.
            basis_transformation_noise : bool, optional, default=False
                Whether the H-gate that is applied to transform the basis in which the qubit is measured should be
                noisy (True) or noiseless (False) in general. If not specified, it will have the same value as the
                'noise' attribute.
            p_g : float [0-1], optional, default=0.01
                The amount of gate noise present in the system. Will only be applied if 'noise' is True.
            p_m : float [0-1], optional, default=0.01
                The amount of measurement noise present in the system. Will only be applied if 'noise' is True.
            _qubit_density_matrix_lookup : dict
                The density matrix of the entire system is split into separate density matrices where ever possible
                (density matrices will be fused when two-qubit gate is applied). This dictionary is used to lookup
                to which density matrix a qubit belongs
            _qubit_array : ndarray
                A list containing the initial state of the qubits.
            _draw_order : list of dict items
                A list containing dict items that specify the operations that should be drawn.
            _user_operation_order : list
                List containing the actions on the circuit applied by the user.
            _effective_measurements : int, default=0
                Integer keeping track of the amount of effectively measured qubits. Used for more clear circuit
                drawings.
            _measured_qubits : list
                List containing the indices of the qubits that have been measured and are therefore not used after.
                Used for more clear circuit drawings.
            _init_parameters : dict
                A dictionary containing the initial parameters of the system, including the '_qubit_array' and
                'density_matrix' attribute. The keys are the names of the attributes.

    """

    def __init__(self, num_qubits, init_type=0, noise=False, basis_transformation_noise=None, p_g=0.001, p_m=0.001,
                 p_m_1=None, F_link=None, decoherence=False, T1n_idle=None, T2n_idle=None, T1e_idle=None,
                 T2e_idle=None, T1n_link=None, T2n_link=None, p_link=1, time_step=1, t_meas=1,
                 t_link=1, probabilistic=False, network_noise_type=0, no_single_qubit_error=False,
                 thread_safe_printing=False, single_qubit_gate_lookup=None, two_qubit_gate_lookup=None,
                 t_pulse=0, n_DD=1, cut_off_time=np.inf, noiseless_swap=False, combine=False,
                 debug=False, bell_pair_type=3, bell_pair_parameters=None, set_number=None, dynamic_direct_states=False, photon_number_resolution=False, alpha_distill=None, only_GHZ=False,shots_emission_direct=None,**kwargs):

        # Basic attributes
        self.num_qubits = num_qubits
        self.d = 2 ** num_qubits
        self.qubits = None
        self.nodes = None
        self.ghz_fidelity = None
        self.bell_pair_type = bell_pair_type
        self._init_type = init_type
        self._qubit_array = num_qubits * [ket_0]
        self._draw_order = []
        self._user_operation_order = []
        self._effective_measurements = 0
        self._measured_qubits = []
        self._uninitialised_qubits = []
        self._qubit_density_matrix_lookup = {}
        self._print_lines = []
        self._thread_safe_printing = thread_safe_printing
        self._fused = False
        self._single_qubit_gate_lookup = single_qubit_gate_lookup if single_qubit_gate_lookup is not None else {}
        self._two_qubit_gate_lookup = two_qubit_gate_lookup if two_qubit_gate_lookup is not None else {}

        # Noise attributes (without decoherence)
        self.noise = noise
        self.p_g = p_g
        self.p_m = p_m
        self.p_m_1 = p_m_1
        self.F_link = F_link
        self.network_noise_type = network_noise_type
        self.no_single_qubit_error = no_single_qubit_error
        self.noiseless_swap = noiseless_swap
        self.basis_transformation_noise = noise if basis_transformation_noise is None else basis_transformation_noise
        self.set_number = set_number

        # Decoherence and duration attributes
        self.decoherence = decoherence
        self.time_step = time_step
        self.T1n_idle = T1n_idle
        self.T2n_idle = T2n_idle
        self.T1e_idle = T1e_idle
        self.T2e_idle = T2e_idle
        self.T1n_link = T1n_link
        self.T2n_link = T2n_link
        self.total_duration = 0
        self.t_link = t_link
        self.t_meas = t_meas
        self.t_pulse = t_pulse
        self.cut_off_time = cut_off_time
        self.cut_off_time_reached = False

        # Probabilistic nature attributes
        self.probabilistic = probabilistic
        self.p_link = p_link
        self.n_DD = n_DD
        self._total_link_attempts = 0
        self._total_succeeded_link = 0

        # Bell pair parameters
        # self.mu = None
        # self.F_prep = None
        # self.labda = None
        # self.p_DE = None
        # self.eta = None
        # self.alpha = None
        # self.ent_prot = None
        self.dynamic_direct_states = dynamic_direct_states
        self.photon_number_resolution = photon_number_resolution
        self.alpha_distill = alpha_distill
        self.only_GHZ = only_GHZ
        self.shots_emission_direct = shots_emission_direct
        self.noisy_bell_state = self._construct_noisy_bell_pair_state(bell_pair_parameters, network_noise_type, pg=self.p_g,only_GHZ=self.only_GHZ)
        

        # Sub circuit attributes
        self._sub_circuits = {}
        self._current_sub_circuit = None
        self._circuit_operations_ended = False

        # Superoperator attributes
        self._superoperator_decomposition = None
        self._error_density_matrix_lookup = {}
        self.combine = combine

        self._init_density_matrix()

        self._init_parameters = self._init_parameters_to_dict()

        if debug:
            print("X gate duration {}".format(X_gate.duration), flush=True)
            pprint({k: getattr(self, k) for k in self.__dir__()
                    if k[0:2] != '__' and type(getattr(self, k)) in [float, int, str, bool]})

    from . import _noise
    from . import _quantum_circuit_init
    from . import _superoperator
    from . import _draw
    from . import _operations
    """
        ---------------------------------------------------------------------------------------------------------
                                                    Init Methods
        ---------------------------------------------------------------------------------------------------------     
    """

    def _construct_noisy_bell_pair_state(self, bell_pair_parameters, network_noise_type=None, pg=None, only_GHZ=False):
        if network_noise_type in range(40, 43) or network_noise_type in range(30, 33):
            # Noisy state becomes a direct GHZ state
            if network_noise_type == 40 or network_noise_type == 30:
                phi_0 = -0.3674 * np.pi
                c1 = 0.9990
                phi_1 = -0.0924 * np.pi
                cerr = 0.0456
                phi_err = -0.0789 * np.pi
            elif network_noise_type == 41:
                phi_0 = 0.7796 * np.pi
                c1 = 0.9996
                phi_1 = -0.2999 * np.pi
                cerr = 0.0299
                phi_err = -0.4909 * np.pi
            ket0 = np.array([[1], [0]])
            ket1 = np.array([[0], [1]])
            weight = 3 if network_noise_type in range(30, 33) else 4
            psi = 1 / cmath.sqrt(2) * (cmath.exp(1j * phi_0) * ft.reduce(np.kron, [ket0] * weight)
                                       + ft.reduce(np.kron, [c1 * cmath.exp(1j * phi_1) * ket1
                                                             + cerr * cmath.exp(1j * phi_err) * ket0] * weight))
            # Normalizing psi:
            psi = psi / cmath.sqrt((np.matrix(psi).H @ np.matrix(psi))[0, 0])
            noisy_density_matrix = sp.csr_matrix(np.matrix(psi) @ np.matrix(psi).H)

            density_matrix_target = sp.lil_matrix((2**weight, 2**weight))
            density_matrix_target[0, 0] = 0.5
            density_matrix_target[0, 2**weight-1] = 0.5
            density_matrix_target[2**weight-1, 0] = 0.5
            density_matrix_target[2**weight-1, 2**weight-1] = 0.5
            if network_noise_type == 40 or network_noise_type == 30:
                self.p_link = 0.7414 * (0.5)**weight
            elif network_noise_type == 41:
                self.p_link = 0.4376 * (0.5)**weight
            self.t_link = 6e-6
            self.F_link = fidelity(noisy_density_matrix, density_matrix_target)
            return noisy_density_matrix

        if network_noise_type in range(50, 55) or network_noise_type in range(60, 65) \
                or network_noise_type in range(70, 75) or network_noise_type in range(80, 85):
            df_ghz_states = import_density_matrix_fidelity_for_direct_schemes(path=None, choice=network_noise_type, dynamic_states=self.dynamic_direct_states, gate_error=self.p_g)
            noisy_density_matrix = df_ghz_states[1]
            self.p_link = df_ghz_states[0][1]
            imported_fidelity = df_ghz_states[0][0]
            self.t_link = 6e-6

            weight = 4 if (network_noise_type in range(50, 55) or network_noise_type in range(70,75)) else 3
            density_matrix_target = sp.lil_matrix((2**weight, 2**weight))
            density_matrix_target[0, 0] = 0.5
            density_matrix_target[0, 2**weight-1] = 0.5
            density_matrix_target[2**weight-1, 0] = 0.5
            density_matrix_target[2**weight-1, 2**weight-1] = 0.5
            self.F_link = fidelity(noisy_density_matrix, density_matrix_target)
            if self.dynamic_direct_states is not True:
                if self.F_link != imported_fidelity:
                    print(f"\n\nCalculated fidelity of {self.F_link} is not equal to the imported fidelity of {imported_fidelity}.")
            if self.dynamic_direct_states:
                print(f"*** GHZ state fidelity is {self.F_link}.***")
            print(f"*** Success probability is {self.p_link}.***")
            return noisy_density_matrix
        
        if network_noise_type == 100:
            # Direct-emission scheme Raw state
            mu = bell_pair_parameters['mu']
            F_prep = bell_pair_parameters['F_prep']
            labda = bell_pair_parameters['lambda']
            p_DE = bell_pair_parameters['p_DE']
            eta = bell_pair_parameters['eta']
            alpha = bell_pair_parameters['alpha']

            weight = 4
            density_matrix_target = sp.lil_matrix((2**weight, 2**weight))
            density_matrix_target[0, 0] = 0.5
            density_matrix_target[0, 2**weight-1] = 0.5
            density_matrix_target[2**weight-1, 0] = 0.5
            density_matrix_target[2**weight-1, 2**weight-1] = 0.5

            noisy_density_matrix = sp.lil_matrix((2**weight, 2**weight), dtype=complex)
            if self.photon_number_resolution is True:
                noisy_density_matrix[0,0] = -1/2*((-1 + alpha)**2*(1 + mu**2))/((-1 + alpha*eta)**2*(-3 + mu**2))
                noisy_density_matrix[0,15] = -(((1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                noisy_density_matrix[2,2] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                noisy_density_matrix[2,8] = -(((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                noisy_density_matrix[3,3] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                noisy_density_matrix[3,6] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                noisy_density_matrix[3,9] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                noisy_density_matrix[3,12] = 0
                noisy_density_matrix[6,3] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                noisy_density_matrix[6,6] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                noisy_density_matrix[6,9] = 0
                noisy_density_matrix[6,12] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                noisy_density_matrix[8,2] = -(((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                noisy_density_matrix[8,8] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                noisy_density_matrix[9,3] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                noisy_density_matrix[9,6] = 0
                noisy_density_matrix[9,9] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                noisy_density_matrix[9,12] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                noisy_density_matrix[10,10] = (alpha**2*(-1 + eta)**2)/(-1 + alpha*eta)**2
                noisy_density_matrix[11,11] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                noisy_density_matrix[11,14] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                noisy_density_matrix[12,3] = 0
                noisy_density_matrix[12,6] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                noisy_density_matrix[12,9] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                noisy_density_matrix[12,12] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                noisy_density_matrix[14,11] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                noisy_density_matrix[14,14] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                noisy_density_matrix[15,0] = -(((1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                noisy_density_matrix[15,15] = -1/2*((-1 + alpha)**2*(1 + mu**2))/((-1 + alpha*eta)**2*(-3 + mu**2))
                self.p_link = np.real((-3*alpha**2*eta**2*(-1 + alpha*eta)**2*(-3 + mu**2))/2)

            elif self.photon_number_resolution is False:
                noisy_density_matrix[0,0] = (-16*(-1 + alpha)**2*(1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[0,15] = (-32*(1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[2,2] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[2,8] = (-8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[3,3] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[3,6] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[3,9] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[3,12] = 0
                noisy_density_matrix[6,3] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[6,6] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[6,9] = 0
                noisy_density_matrix[6,12] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[8,2] = (-8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[8,8] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[9,3] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[9,6] = 0
                noisy_density_matrix[9,9] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[9,12] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[10,10] = (alpha**2*(32*(-3 + mu**2) + eta*(96 - 7*eta + 32*mu**2*(-3 + 2*mu) + eta*mu**2*(54 + (-56 + mu)*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[11,11] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[11,14] = (8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[12,3] = 0
                noisy_density_matrix[12,6] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[12,9] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[12,12] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[14,11] = (8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[14,14] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[15,0] = (-32*(1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                noisy_density_matrix[15,15] = (-16*(-1 + alpha)**2*(1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                self.p_link = np.real((-3*alpha**2*eta**2*(32*(-3 + mu**2) + 32*alpha*eta*(3 - 3*mu**2 + 2*mu**3) + alpha**2*eta**2*(-7 + 54*mu**2 - 56*mu**3 + mu**4)))/64)

            self.t_link = 1e-5
            
            self.F_link = fidelity(noisy_density_matrix, density_matrix_target)
            print("#################################################")
            print(f"*** GHZ state fidelity of Raw state is {self.F_link}.***")
            print(f"*** Success probability of Raw state is {self.p_link}.***")
            print("#################################################")
            return noisy_density_matrix
        
        if network_noise_type == 101:
            # Double-click protocol for direct emission scheme
            mu = bell_pair_parameters['mu']
            F_prep = bell_pair_parameters['F_prep']
            labda = 1 # Ignore the path length differences for double-click protocol
            p_DE = bell_pair_parameters['p_DE']
            eta = bell_pair_parameters['eta']
            # alpha = 1/2 # Optimal for success probability, while fidelity does not change w.r.t. alpha
            alpha = bell_pair_parameters['alpha']
            pg=4*pg/3 # Make the conversion from the definition of mathematica to the definition of the simulator for the depolarizing quantum channel

            weight = 4
            density_matrix_target = sp.lil_matrix((2**weight, 2**weight))
            density_matrix_target[0, 0] = 0.5
            density_matrix_target[0, 2**weight-1] = 0.5
            density_matrix_target[2**weight-1, 0] = 0.5
            density_matrix_target[2**weight-1, 2**weight-1] = 0.5
            
            noisy_density_matrix = sp.lil_matrix((2**weight, 2**weight), dtype=complex)

            if self.photon_number_resolution is True:
                noisy_density_matrix[0,0] = -1/2*((1 + mu**2)*(-4*pg**3*(1 + alpha*(-2 + eta))**2*(-3 + mu**2) + pg**4*(1 + alpha*(-2 + eta))**2*(-3 + mu**2) - 8*(-1 + alpha)**2*(1 + mu**2) + 8*pg*(-1 + alpha)*(-2 + 5*alpha - 3*alpha*eta + (-2 + alpha + alpha*eta)*mu**2) + 4*pg**2*(-5 - mu**2 + alpha*(19 - 9*eta + (-1 + 3*eta)*mu**2 + alpha*(-17 + 3*mu**2 + (-5 + eta)*eta*(-3 + mu**2))))))/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[0,15] = (16*(1 - 2*F_prep)**4*(1 - 2*p_DE)**8*(-1 + pg)**4*(-1 + alpha)**2*mu**4)/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[2,2] = -1/4*(pg*(2 - 2*alpha + (-2 + pg)*pg*(1 + alpha*(-2 + eta)))*(pg + 2*(-1 + alpha) + pg*alpha*(-2 + eta))*(-1 + eta)*(6 - 2*mu**2)*(-3 + mu**2))/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[2,8] = (-4*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*pg*(-1 + alpha)*(pg + 2*(-1 + alpha) + pg*alpha*(-2 + eta))*(-1 + eta)*(-1 + mu)**2*mu**2)/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[3,3] = -1/8*((4 - 4*mu**2)*(-4*pg**3*(1 + alpha*(-2 + eta))**2*(-3 + mu**2) + pg**4*(1 + alpha*(-2 + eta))**2*(-3 + mu**2) + 8*(-1 + alpha)**2*(-1 + mu**2) + 8*pg*(-1 + alpha)*(-2 + 5*alpha - 3*alpha*eta + (2 + alpha*(-3 + eta))*mu**2) + 4*pg**2*(-5 + 3*mu**2 + alpha*(19 - 9*eta + 3*(-3 + eta)*mu**2 + alpha*(-17 + 7*mu**2 + (-5 + eta)*eta*(-3 + mu**2))))))/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[3,6] = (-2*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*(2 - 2*alpha + (-2 + pg)*pg*(1 + alpha*(-2 + eta)))*(-1 + mu)**2*mu**2)/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[3,9] = (-2*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*(2 - 2*alpha + (-2 + pg)*pg*(1 + alpha*(-2 + eta)))*(-1 + mu)**2*mu**2)/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[3,12] = 0
                noisy_density_matrix[6,3] = (-2*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*(2 - 2*alpha + (-2 + pg)*pg*(1 + alpha*(-2 + eta)))*(-1 + mu)**2*mu**2)/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[6,6] = -1/8*((4 - 4*mu**2)*(-4*pg**3*(1 + alpha*(-2 + eta))**2*(-3 + mu**2) + pg**4*(1 + alpha*(-2 + eta))**2*(-3 + mu**2) + 8*(-1 + alpha)**2*(-1 + mu**2) + 8*pg*(-1 + alpha)*(-2 + 5*alpha - 3*alpha*eta + (2 + alpha*(-3 + eta))*mu**2) + 4*pg**2*(-5 + 3*mu**2 + alpha*(19 - 9*eta + 3*(-3 + eta)*mu**2 + alpha*(-17 + 7*mu**2 + (-5 + eta)*eta*(-3 + mu**2))))))/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[6,9] = 0
                noisy_density_matrix[6,12] = (-2*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*(2 - 2*alpha + (-2 + pg)*pg*(1 + alpha*(-2 + eta)))*(-1 + mu)**2*mu**2)/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[8,2] = (-4*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*pg*(-1 + alpha)*(pg + 2*(-1 + alpha) + pg*alpha*(-2 + eta))*(-1 + eta)*(-1 + mu)**2*mu**2)/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[8,8] = -1/4*(pg*(2 - 2*alpha + (-2 + pg)*pg*(1 + alpha*(-2 + eta)))*(pg + 2*(-1 + alpha) + pg*alpha*(-2 + eta))*(-1 + eta)*(6 - 2*mu**2)*(-3 + mu**2))/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[9,3] = (-2*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*(2 - 2*alpha + (-2 + pg)*pg*(1 + alpha*(-2 + eta)))*(-1 + mu)**2*mu**2)/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[9,6] = 0
                noisy_density_matrix[9,9] = -1/8*((4 - 4*mu**2)*(-4*pg**3*(1 + alpha*(-2 + eta))**2*(-3 + mu**2) + pg**4*(1 + alpha*(-2 + eta))**2*(-3 + mu**2) + 8*(-1 + alpha)**2*(-1 + mu**2) + 8*pg*(-1 + alpha)*(-2 + 5*alpha - 3*alpha*eta + (2 + alpha*(-3 + eta))*mu**2) + 4*pg**2*(-5 + 3*mu**2 + alpha*(19 - 9*eta + 3*(-3 + eta)*mu**2 + alpha*(-17 + 7*mu**2 + (-5 + eta)*eta*(-3 + mu**2))))))/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[9,12] = (-2*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*(2 - 2*alpha + (-2 + pg)*pg*(1 + alpha*(-2 + eta)))*(-1 + mu)**2*mu**2)/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[10,10] = (pg**2*(pg + 2*(-1 + alpha) + pg*alpha*(-2 + eta))**2*(-1 + eta)**2*(-3 + mu**2)**2)/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[11,11] = -1/4*(pg*(2 - 2*alpha + (-2 + pg)*pg*(1 + alpha*(-2 + eta)))*(pg + 2*(-1 + alpha) + pg*alpha*(-2 + eta))*(-1 + eta)*(6 - 2*mu**2)*(-3 + mu**2))/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[11,14] = (-4*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*pg*(-1 + alpha)*(pg + 2*(-1 + alpha) + pg*alpha*(-2 + eta))*(-1 + eta)*(-1 + mu)**2*mu**2)/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[12,3] = 0
                noisy_density_matrix[12,6] = (-2*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*(2 - 2*alpha + (-2 + pg)*pg*(1 + alpha*(-2 + eta)))*(-1 + mu)**2*mu**2)/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[12,9] = (-2*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*(2 - 2*alpha + (-2 + pg)*pg*(1 + alpha*(-2 + eta)))*(-1 + mu)**2*mu**2)/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[12,12] = -1/8*((4 - 4*mu**2)*(-4*pg**3*(1 + alpha*(-2 + eta))**2*(-3 + mu**2) + pg**4*(1 + alpha*(-2 + eta))**2*(-3 + mu**2) + 8*(-1 + alpha)**2*(-1 + mu**2) + 8*pg*(-1 + alpha)*(-2 + 5*alpha - 3*alpha*eta + (2 + alpha*(-3 + eta))*mu**2) + 4*pg**2*(-5 + 3*mu**2 + alpha*(19 - 9*eta + 3*(-3 + eta)*mu**2 + alpha*(-17 + 7*mu**2 + (-5 + eta)*eta*(-3 + mu**2))))))/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[14,11] = (-4*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*pg*(-1 + alpha)*(pg + 2*(-1 + alpha) + pg*alpha*(-2 + eta))*(-1 + eta)*(-1 + mu)**2*mu**2)/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[14,14] = -1/4*(pg*(2 - 2*alpha + (-2 + pg)*pg*(1 + alpha*(-2 + eta)))*(pg + 2*(-1 + alpha) + pg*alpha*(-2 + eta))*(-1 + eta)*(6 - 2*mu**2)*(-3 + mu**2))/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[15,0] = (16*(1 - 2*F_prep)**4*(1 - 2*p_DE)**8*(-1 + pg)**4*(-1 + alpha)**2*mu**4)/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                noisy_density_matrix[15,15] = -1/2*((1 + mu**2)*(-4*pg**3*(1 + alpha*(-2 + eta))**2*(-3 + mu**2) + pg**4*(1 + alpha*(-2 + eta))**2*(-3 + mu**2) - 8*(-1 + alpha)**2*(1 + mu**2) + 8*pg*(-1 + alpha)*(-2 + 5*alpha - 3*alpha*eta + (-2 + alpha + alpha*eta)*mu**2) + 4*pg**2*(-5 - mu**2 + alpha*(19 - 9*eta + (-1 + 3*eta)*mu**2 + alpha*(-17 + 3*mu**2 + (-5 + eta)*eta*(-3 + mu**2))))))/(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4))
                self.p_link = np.real((-3*alpha**2*eta**4*(-3 + mu**2)*(4*pg**3*(1 + alpha*(-2 + eta))*(alpha - eta)*eta*(-3 + mu**2)**2 + pg**4*(1 + alpha*(-2 + eta))**2*eta**2*(-3 + mu**2)**2 + 8*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) - 8*pg*(-1 + alpha)*(3 + 6*alpha - 9*eta - 2*(1 + 2*alpha - 3*eta)*mu**2 + (-5 + 6*alpha - eta)*mu**4) + 4*pg**2*(-3 + 9*eta*(1 + eta) + alpha*(6 + 9*(-5 + eta)*eta) + alpha**2*(6 - 9*(-2 + eta)*eta) + 2*(1 - 3*eta*(1 + eta) + alpha*(-2 - 3*(-5 + eta)*eta) + alpha**2*(-2 + 3*(-2 + eta)*eta))*mu**2 + (5 + eta + eta**2 + alpha*(-10 + (-5 + eta)*eta) + alpha**2*(6 - (-2 + eta)*eta))*mu**4)))/(128*(3 - mu**2)))

            elif self.photon_number_resolution is False:
                noisy_density_matrix[0,0] = (-16*(1 + mu**2)*(-256*(-1 + alpha)**2*(1 + mu**2) - 128*pg*(-1 + alpha)*(4*(1 + mu**2) - 2*alpha*(5 + mu**2) + alpha*eta*(3 + mu**2*(-3 + 2*mu))) - 4*pg**3*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + pg**4*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 4*pg**2*(-32*(5 + mu**2) - 16*alpha*(2*(-19 + mu**2) + eta*(9 - 9*mu**2 + 6*mu**3)) + alpha**2*(-544 + 96*mu**2 + 80*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))))))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[0,15] = (16384*(1 - 2*F_prep)**4*(1 - 2*p_DE)**8*(-1 + pg)**4*(-1 + alpha)**2*mu**4)/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[2,2] = (-4*pg*(4*(-3 + mu**2) + eta*(6 - 6*mu**2 + 4*mu**3))*(-128*(-1 + alpha)**2*(-3 + mu**2) + 96*pg*(-1 + alpha)*(6 + 3*alpha*(-4 + eta) + (-2 + alpha*(4 - 3*eta))*mu**2 + 2*alpha*eta*mu**3) + pg**3*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 2*pg**2*(-64*(-3 + mu**2) + alpha*(224*(-3 + mu**2) - 192*alpha*(-3 + mu**2) + 48*eta*(3 + mu**2*(-3 + 2*mu)) - alpha*eta*(80*(3 + mu**2*(-3 + 2*mu)) + eta*(-7 + mu**2*(54 + (-56 + mu)*mu)))))))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[2,8] = (-256*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*pg*(-1 + alpha)*mu**2*(4 - 4*mu + eta*(-1 + mu*(2 + mu)))*(4*pg*(-1 + mu) + 8*(-1 + alpha)*(-1 + mu) + pg*alpha*(8 - 8*mu + eta*(-1 + mu*(2 + mu)))))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[3,3] = (-4*(4 - 4*mu**2)*(256*(-1 + alpha)**2*(-1 + mu**2) - 128*pg*(-1 + alpha)*(4 - 4*mu**2 + alpha*(-10 + 6*mu**2 + eta*(3 + mu**2*(-3 + 2*mu)))) - 4*pg**3*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + pg**4*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 4*pg**2*(32*(-5 + 3*mu**2) - 16*alpha*(-38 + 9*eta - 9*(-2 + eta)*mu**2 + 6*eta*mu**3) + alpha**2*(32*(-17 + 7*mu**2) + 80*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))))))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[3,6] = (-512*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*(-1 + mu)*mu**2*(-((-2 + pg)*pg*(4 + alpha*(-8 + eta))) + 2*(-4*alpha + (-2 + pg)*pg*(2 + alpha*(-4 + eta)))*mu + (-2 + pg)*pg*alpha*eta*mu**2 + 8*(-1 + alpha + mu)))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[3,9] = (-512*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*(-1 + mu)*mu**2*(-((-2 + pg)*pg*(4 + alpha*(-8 + eta))) + 2*(-4*alpha + (-2 + pg)*pg*(2 + alpha*(-4 + eta)))*mu + (-2 + pg)*pg*alpha*eta*mu**2 + 8*(-1 + alpha + mu)))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[3,12] = 0
                noisy_density_matrix[6,3] = (-512*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*(-1 + mu)*mu**2*(-((-2 + pg)*pg*(4 + alpha*(-8 + eta))) + 2*(-4*alpha + (-2 + pg)*pg*(2 + alpha*(-4 + eta)))*mu + (-2 + pg)*pg*alpha*eta*mu**2 + 8*(-1 + alpha + mu)))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[6,6] = (-4*(4 - 4*mu**2)*(256*(-1 + alpha)**2*(-1 + mu**2) - 128*pg*(-1 + alpha)*(4 - 4*mu**2 + alpha*(-10 + 6*mu**2 + eta*(3 + mu**2*(-3 + 2*mu)))) - 4*pg**3*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + pg**4*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 4*pg**2*(32*(-5 + 3*mu**2) - 16*alpha*(-38 + 9*eta - 9*(-2 + eta)*mu**2 + 6*eta*mu**3) + alpha**2*(32*(-17 + 7*mu**2) + 80*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))))))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[6,9] = 0
                noisy_density_matrix[6,12] = (-512*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*(-1 + mu)*mu**2*(-((-2 + pg)*pg*(4 + alpha*(-8 + eta))) + 2*(-4*alpha + (-2 + pg)*pg*(2 + alpha*(-4 + eta)))*mu + (-2 + pg)*pg*alpha*eta*mu**2 + 8*(-1 + alpha + mu)))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[8,2] = (-256*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*pg*(-1 + alpha)*mu**2*(4 - 4*mu + eta*(-1 + mu*(2 + mu)))*(4*pg*(-1 + mu) + 8*(-1 + alpha)*(-1 + mu) + pg*alpha*(8 - 8*mu + eta*(-1 + mu*(2 + mu)))))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[8,8] = (-4*pg*(4*(-3 + mu**2) + eta*(6 - 6*mu**2 + 4*mu**3))*(-128*(-1 + alpha)**2*(-3 + mu**2) + 96*pg*(-1 + alpha)*(6 + 3*alpha*(-4 + eta) + (-2 + alpha*(4 - 3*eta))*mu**2 + 2*alpha*eta*mu**3) + pg**3*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 2*pg**2*(-64*(-3 + mu**2) + alpha*(224*(-3 + mu**2) - 192*alpha*(-3 + mu**2) + 48*eta*(3 + mu**2*(-3 + 2*mu)) - alpha*eta*(80*(3 + mu**2*(-3 + 2*mu)) + eta*(-7 + mu**2*(54 + (-56 + mu)*mu)))))))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[9,3] = (-512*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*(-1 + mu)*mu**2*(-((-2 + pg)*pg*(4 + alpha*(-8 + eta))) + 2*(-4*alpha + (-2 + pg)*pg*(2 + alpha*(-4 + eta)))*mu + (-2 + pg)*pg*alpha*eta*mu**2 + 8*(-1 + alpha + mu)))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[9,6] = 0
                noisy_density_matrix[9,9] = (-4*(4 - 4*mu**2)*(256*(-1 + alpha)**2*(-1 + mu**2) - 128*pg*(-1 + alpha)*(4 - 4*mu**2 + alpha*(-10 + 6*mu**2 + eta*(3 + mu**2*(-3 + 2*mu)))) - 4*pg**3*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + pg**4*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 4*pg**2*(32*(-5 + 3*mu**2) - 16*alpha*(-38 + 9*eta - 9*(-2 + eta)*mu**2 + 6*eta*mu**3) + alpha**2*(32*(-17 + 7*mu**2) + 80*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))))))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[9,12] = (-512*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*(-1 + mu)*mu**2*(-((-2 + pg)*pg*(4 + alpha*(-8 + eta))) + 2*(-4*alpha + (-2 + pg)*pg*(2 + alpha*(-4 + eta)))*mu + (-2 + pg)*pg*alpha*eta*mu**2 + 8*(-1 + alpha + mu)))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[10,10] = -((pg**2*(-32*(-3 + mu**2) - eta*(96 - 7*eta + 32*mu**2*(-3 + 2*mu) + eta*mu**2*(54 + (-56 + mu)*mu)))*(128*(-1 + alpha)**2*(-3 + mu**2) - 64*pg*(-1 + alpha)*(6 + 3*alpha*(-4 + eta) + (-2 + alpha*(4 - 3*eta))*mu**2 + 2*alpha*eta*mu**3) + pg**2*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))))))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu)))))))))
                noisy_density_matrix[11,11] = (-4*pg*(4*(-3 + mu**2) + eta*(6 - 6*mu**2 + 4*mu**3))*(-128*(-1 + alpha)**2*(-3 + mu**2) + 96*pg*(-1 + alpha)*(6 + 3*alpha*(-4 + eta) + (-2 + alpha*(4 - 3*eta))*mu**2 + 2*alpha*eta*mu**3) + pg**3*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 2*pg**2*(-64*(-3 + mu**2) + alpha*(224*(-3 + mu**2) - 192*alpha*(-3 + mu**2) + 48*eta*(3 + mu**2*(-3 + 2*mu)) - alpha*eta*(80*(3 + mu**2*(-3 + 2*mu)) + eta*(-7 + mu**2*(54 + (-56 + mu)*mu)))))))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[11,14] = (-256*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*pg*(-1 + alpha)*mu**2*(4 - 4*mu + eta*(-1 + mu*(2 + mu)))*(4*pg*(-1 + mu) + 8*(-1 + alpha)*(-1 + mu) + pg*alpha*(8 - 8*mu + eta*(-1 + mu*(2 + mu)))))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[12,3] = 0
                noisy_density_matrix[12,6] = (-512*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*(-1 + mu)*mu**2*(-((-2 + pg)*pg*(4 + alpha*(-8 + eta))) + 2*(-4*alpha + (-2 + pg)*pg*(2 + alpha*(-4 + eta)))*mu + (-2 + pg)*pg*alpha*eta*mu**2 + 8*(-1 + alpha + mu)))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[12,9] = (-512*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*(-1 + mu)*mu**2*(-((-2 + pg)*pg*(4 + alpha*(-8 + eta))) + 2*(-4*alpha + (-2 + pg)*pg*(2 + alpha*(-4 + eta)))*mu + (-2 + pg)*pg*alpha*eta*mu**2 + 8*(-1 + alpha + mu)))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[12,12] = (-4*(4 - 4*mu**2)*(256*(-1 + alpha)**2*(-1 + mu**2) - 128*pg*(-1 + alpha)*(4 - 4*mu**2 + alpha*(-10 + 6*mu**2 + eta*(3 + mu**2*(-3 + 2*mu)))) - 4*pg**3*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + pg**4*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 4*pg**2*(32*(-5 + 3*mu**2) - 16*alpha*(-38 + 9*eta - 9*(-2 + eta)*mu**2 + 6*eta*mu**3) + alpha**2*(32*(-17 + 7*mu**2) + 80*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))))))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[14,11] = (-256*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*pg*(-1 + alpha)*mu**2*(4 - 4*mu + eta*(-1 + mu*(2 + mu)))*(4*pg*(-1 + mu) + 8*(-1 + alpha)*(-1 + mu) + pg*alpha*(8 - 8*mu + eta*(-1 + mu*(2 + mu)))))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[14,14] = (-4*pg*(4*(-3 + mu**2) + eta*(6 - 6*mu**2 + 4*mu**3))*(-128*(-1 + alpha)**2*(-3 + mu**2) + 96*pg*(-1 + alpha)*(6 + 3*alpha*(-4 + eta) + (-2 + alpha*(4 - 3*eta))*mu**2 + 2*alpha*eta*mu**3) + pg**3*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 2*pg**2*(-64*(-3 + mu**2) + alpha*(224*(-3 + mu**2) - 192*alpha*(-3 + mu**2) + 48*eta*(3 + mu**2*(-3 + 2*mu)) - alpha*eta*(80*(3 + mu**2*(-3 + 2*mu)) + eta*(-7 + mu**2*(54 + (-56 + mu)*mu)))))))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[15,0] = (16384*(1 - 2*F_prep)**4*(1 - 2*p_DE)**8*(-1 + pg)**4*(-1 + alpha)**2*mu**4)/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                noisy_density_matrix[15,15] = (-16*(1 + mu**2)*(-256*(-1 + alpha)**2*(1 + mu**2) - 128*pg*(-1 + alpha)*(4*(1 + mu**2) - 2*alpha*(5 + mu**2) + alpha*eta*(3 + mu**2*(-3 + 2*mu))) - 4*pg**3*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + pg**4*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 4*pg**2*(-32*(5 + mu**2) - 16*alpha*(2*(-19 + mu**2) + eta*(9 - 9*mu**2 + 6*mu**3)) + alpha**2*(-544 + 96*mu**2 + 80*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))))))/(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu))))))))
                self.p_link = np.real((3*alpha**2*eta**4*(8192*(-1 + alpha)**2*(3 - 2*mu**2 + 3*mu**4) + 128*pg**2*(96*(-1 + 2*alpha*(1 + alpha)) + 144*(1 + alpha*(-5 + 2*alpha))*eta + 3*(7 - 58*(-1 + alpha)*alpha)*eta**2 + (64 - 128*alpha*(1 + alpha) - 192*(1 + alpha*(-5 + 2*alpha))*eta + (-169 + 94*(-1 + alpha)*alpha)*eta**2)*mu**2 + 24*eta*(4 + 7*eta + 2*alpha**2*(4 + eta) - 2*alpha*(10 + eta))*mu**3 + (32*(5 + 2*alpha*(-5 + 3*alpha)) + 48*(1 + alpha*(-5 + 2*alpha))*eta + 3*(17 - 38*(-1 + alpha)*alpha)*eta**2)*mu**4 + 8*eta*(-4 - 7*eta + 2*alpha*(10 - 11*eta + alpha*(-4 + 11*eta)))*mu**5 + (1 - 94*(-1 + alpha)*alpha)*eta**2*mu**6) - 4096*pg*(-1 + alpha)*(6 - 4*mu**2 - 10*mu**4 + 4*alpha*(3 - 2*mu**2 + 3*mu**4) + eta*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu))) + pg**4*eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu))*(32*(-3 + mu**2) - 32*alpha*(4*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))) + alpha**2*(128*(-3 + mu**2) + 64*eta*(3 + mu**2*(-3 + 2*mu)) + eta**2*(-7 + mu**2*(54 + (-56 + mu)*mu)))) + 64*pg**3*eta*(-2*eta*(-3 + mu**2)*(-7 + mu**2*(54 + (-56 + mu)*mu)) + alpha*(-32*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta**2*(3 + mu**2*(-3 + 2*mu))*(-7 + mu**2*(54 + (-56 + mu)*mu)) - 2*eta*(9 + mu**2*(363 + mu*(-408 + mu*(-81 + mu*(72 + 29*mu)))))) + 4*alpha**2*(16*(-3 + mu**2)*(3 + mu**2*(-3 + 2*mu)) + eta*(87 + mu**2*(-47 + mu*(-24 + mu*(57 + mu*(-88 + 47*mu)))))))))/131072)

            self.t_link = 2e-5 + 1e-7
            self.F_link = fidelity(noisy_density_matrix, density_matrix_target)

            print("#################################################")
            print(f"*** GHZ state fidelity of DC direct emission state is {self.F_link}.***")
            print(f"*** Success probability of DC direct emission state is {self.p_link}.***")
            print("#################################################")
            return noisy_density_matrix

        pauli_operators = [cirq.I, cirq.X, cirq.Y, cirq.Z]
        # Define the two-qubit correlated gate noise channel function, will be used for the distillation protocols for direct emission protocol
        def correlated_two_qubit_noise_channel(p_g):
            kraus_ops = [np.sqrt(1 - p_g) * np.eye(4)]  # Identity term
            prob = p_g / 15

            for P_j in pauli_operators:
                for P_k in pauli_operators:
                    if P_j != cirq.I or P_k != cirq.I:  # Skip the (I, I) combination
                        kraus_op = np.sqrt(prob) * np.kron(P_j._unitary_(), P_k._unitary_())
                        kraus_ops.append(kraus_op)

            return kraus_ops
        
        def apply_correlated_two_qubit_noise_channel(p_g, qubits):
            return cirq.KrausChannel(correlated_two_qubit_noise_channel(p_g)).on(*qubits)

        def partial_trace_numpy(rho, keep, dims=None):
            """
            Compute the partial trace of a density matrix.

            Parameters:
            - rho: the full 2^m × 2^m density matrix (NumPy array)
            - keep: list of qubit indices to keep (e.g., [0, 2] keeps qubits 0 and 2)
            - dims: list of subsystem dimensions. Default is [2]*m for qubits.

            Returns:
            - Reduced density matrix after tracing out the other qubits.
            """
            n_qubits = int(np.log2(rho.shape[0]))
            if dims is None:
                dims = [2] * n_qubits

            trace_out = [i for i in range(n_qubits) if i not in keep]

            # reshape rho into 2n indices
            reshaped_rho = rho.reshape([2] * n_qubits * 2)

            # reorder indices to group: keep_in, trace_in, keep_out, trace_out
            keep_in = keep
            trace_in = trace_out
            keep_out = [i + n_qubits for i in keep]
            trace_out = [i + n_qubits for i in trace_out]

            perm = keep_in + trace_in + keep_out + trace_out
            reshaped = reshaped_rho.transpose(perm)

            dim_keep = 2 ** len(keep)
            dim_trace = 2 ** (n_qubits - len(keep))

            reshaped = reshaped.reshape((dim_keep, dim_trace, dim_keep, dim_trace))

            # trace over the traced dimensions
            reduced_rho = np.trace(reshaped, axis1=1, axis2=3)

            return reduced_rho

        if network_noise_type == 102:
            # Bell-pair distillation for direct emission scheme
            mu = bell_pair_parameters['mu']
            F_prep = bell_pair_parameters['F_prep']
            labda = bell_pair_parameters['lambda']
            p_DE = bell_pair_parameters['p_DE']
            eta = bell_pair_parameters['eta']
            alpha = bell_pair_parameters['alpha']
            bell_pair_protocol = bell_pair_parameters['ent_prot']
            if self.alpha_distill is None:
                self.alpha_distill = bell_pair_parameters['alpha'] # Use alpha_distill = alpha if not explicitly stated as the default value

            weight = 4
            target_GHZ_state = sp.lil_matrix((2**weight, 2**weight))
            target_GHZ_state[0, 0] = 0.5
            target_GHZ_state[0, 2**weight-1] = 0.5
            target_GHZ_state[2**weight-1, 0] = 0.5
            target_GHZ_state[2**weight-1, 2**weight-1] = 0.5

            bell_target = sp.lil_matrix((2**2, 2**2), dtype=complex)
            bell_target[0, 0] = 0.5
            bell_target[0, 3] = 0.5
            bell_target[3, 0] = 0.5
            bell_target[3, 3] = 0.5

            raw_state = np.zeros((2**weight, 2**weight), dtype=complex)
            if self.photon_number_resolution is True:
                raw_state[0,0] = -1/2*((-1 + alpha)**2*(1 + mu**2))/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[0,15] = -(((1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                raw_state[2,2] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                raw_state[2,8] = -(((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                raw_state[3,3] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[3,6] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[3,9] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[3,12] = 0
                raw_state[6,3] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[6,6] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[6,9] = 0
                raw_state[6,12] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[8,2] = -(((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                raw_state[8,8] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                raw_state[9,3] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[9,6] = 0
                raw_state[9,9] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[9,12] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[10,10] = (alpha**2*(-1 + eta)**2)/(-1 + alpha*eta)**2
                raw_state[11,11] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                raw_state[11,14] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[12,3] = 0
                raw_state[12,6] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[12,9] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[12,12] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[14,11] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[14,14] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                raw_state[15,0] = -(((1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                raw_state[15,15] = -1/2*((-1 + alpha)**2*(1 + mu**2))/((-1 + alpha*eta)**2*(-3 + mu**2))

                p_link_raw = (-3*alpha**2*eta**2*(-1 + alpha*eta)**2*(-3 + mu**2))/2

            elif self.photon_number_resolution is False:
                raw_state[0,0] = (-16*(-1 + alpha)**2*(1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[0,15] = (-32*(1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[2,2] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[2,8] = (-8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[3,3] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[3,6] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[3,9] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[3,12] = 0
                raw_state[6,3] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[6,6] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[6,9] = 0
                raw_state[6,12] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[8,2] = (-8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[8,8] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[9,3] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[9,6] = 0
                raw_state[9,9] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[9,12] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[10,10] = (alpha**2*(32*(-3 + mu**2) + eta*(96 - 7*eta + 32*mu**2*(-3 + 2*mu) + eta*mu**2*(54 + (-56 + mu)*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[11,11] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[11,14] = (8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[12,3] = 0
                raw_state[12,6] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[12,9] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[12,12] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[14,11] = (8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[14,14] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[15,0] = (-32*(1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[15,15] = (-16*(-1 + alpha)**2*(1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))

                p_link_raw = (-3*alpha**2*eta**2*(32*(-3 + mu**2) + 32*alpha*eta*(3 - 3*mu**2 + 2*mu**3) + alpha**2*eta**2*(-7 + 54*mu**2 - 56*mu**3 + mu**4)))/64

            alpha = self.alpha_distill
            single_click_bell_pair = np.zeros((4,4), dtype=complex)
            if self.photon_number_resolution is True:
                single_click_bell_pair[0,0] = (-1 + alpha)/(-2 + 2*alpha*eta)
                single_click_bell_pair[0,3] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*(-1 + 2*labda)*mu)/(-2 + 2*alpha*eta)
                single_click_bell_pair[2,2] = (alpha*(-1 + eta))/(-1 + alpha*eta)
                single_click_bell_pair[3,0] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*(-1 + 2*labda)*mu)/(-2 + 2*alpha*eta)
                single_click_bell_pair[3,3] = (-1 + alpha)/(-2 + 2*alpha*eta)
                p_link_sc_bell = 2 * alpha * eta * (1 - alpha * eta)
            if self.photon_number_resolution is False:
                single_click_bell_pair[0,0] = (2 - 2*alpha)/(4 + alpha*eta*(-3 + mu))
                single_click_bell_pair[0,3] = (-2*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*(-1 + 2*labda)*np.sqrt(mu))/(4 + alpha*eta*(-3 + mu))
                single_click_bell_pair[2,2] = (alpha*(4 + eta*(-3 + mu)))/(4 + alpha*eta*(-3 + mu))
                single_click_bell_pair[3,0] = (-2*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*(-1 + 2*labda)*np.sqrt(mu))/(4 + alpha*eta*(-3 + mu))
                single_click_bell_pair[3,3] = (2 - 2*alpha)/(4 + alpha*eta*(-3 + mu))
                p_link_sc_bell = alpha * eta * (alpha * eta * (mu - 3) + 4) / 2

            pg=4*pg/3 # Make the conversion from the definition of mathematica to the definition of the simulator for the depolarizing quantum channel
            alpha = 0.5 # Bias to the optimal performance of the double-click bell pairs
            double_click_bell_pair = np.zeros((4,4), dtype=complex)
            if self.photon_number_resolution is True:
                double_click_bell_pair[0,0] = (2 - 2*alpha + (-2 + pg)*pg*(1 + alpha*(-2 + eta)))/(4 + 2*(-2 + pg)*pg*eta + 2*alpha*(-2 + pg*(2 + pg*(-2 + eta)*eta)))
                double_click_bell_pair[0,3] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*mu*(np.sqrt(1-mu) - np.sqrt(1+mu))*(np.sqrt(1-mu) + np.sqrt(1+mu)))/(4 + 2*(-2 + pg)*pg*eta + 2*alpha*(-2 + pg*(2 + pg*(-2 + eta)*eta)))
                double_click_bell_pair[2,2] = (pg*(pg + 2*(-1 + alpha) + pg*alpha*(-2 + eta))*(-1 + eta))/(2 + (-2 + pg)*pg*eta + alpha*(-2 + pg*(2 + pg*(-2 + eta)*eta)))
                double_click_bell_pair[3,0] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*mu*(np.sqrt(1-mu) - np.sqrt(1+mu))*(np.sqrt(1-mu) + np.sqrt(1+mu)))/(4 + 2*(-2 + pg)*pg*eta + 2*alpha*(-2 + pg*(2 + pg*(-2 + eta)*eta)))
                double_click_bell_pair[3,3] = (2 - 2*alpha + (-2 + pg)*pg*(1 + alpha*(-2 + eta)))/(4 + 2*(-2 + pg)*pg*eta + 2*alpha*(-2 + pg*(2 + pg*(-2 + eta)*eta)))
                p_link_dc_bell = alpha*eta**2*(2 - 2*pg*eta + pg**2*eta + alpha*(-2 + 2*pg + pg**2*(-2 + eta)*eta))

            if self.photon_number_resolution is False:
                double_click_bell_pair[0,0] = (2*(8 - 8*alpha + (-2 + pg)*pg*(4 - alpha*(8 + eta*(-3 + mu)))))/(32 + alpha*(-32 + pg*(32 + pg*eta*(8 + eta*(-3 + mu))*(-3 + mu))) - 4*(-2 + pg)*pg*eta*(-3 + mu))
                double_click_bell_pair[0,3] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*mu)/(32 + alpha*(-32 + pg*(32 + pg*eta*(8 + eta*(-3 + mu))*(-3 + mu))) - 4*(-2 + pg)*pg*eta*(-3 + mu))
                double_click_bell_pair[2,2] = (pg*(8 - 8*alpha + pg*(-4 + alpha*(8 + eta*(-3 + mu))))*(4 + eta*(-3 + mu)))/(32 + alpha*(-32 + pg*(32 + pg*eta*(8 + eta*(-3 + mu))*(-3 + mu))) - 4*(-2 + pg)*pg*eta*(-3 + mu))
                double_click_bell_pair[3,0] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*mu)/(32 + alpha*(-32 + pg*(32 + pg*eta*(8 + eta*(-3 + mu))*(-3 + mu))) - 4*(-2 + pg)*pg*eta*(-3 + mu))
                double_click_bell_pair[3,3] = (2*(8 - 8*alpha + (-2 + pg)*pg*(4 - alpha*(8 + eta*(-3 + mu)))))/(32 + alpha*(-32 + pg*(32 + pg*eta*(8 + eta*(-3 + mu))*(-3 + mu))) - 4*(-2 + pg)*pg*eta*(-3 + mu))
                p_link_dc_bell = (alpha*eta**2*(32 + alpha*(-32 + 32*pg + pg**2*eta*(8 + eta*(-3 + mu))*(-3 + mu)) + 8*pg*eta*(-3 + mu) - 4*pg**2*eta*(-3 + mu)))/16
            
            if bell_pair_protocol == "single_click":
                bell_pair_state = single_click_bell_pair
                p_link_bell = p_link_sc_bell
            elif bell_pair_protocol == "double_click":
                bell_pair_state = double_click_bell_pair
                p_link_bell = p_link_dc_bell

            pg=3*pg/4  # Reverse the scaling back to original
            
            # Print for comparison and fidleity improvement
            print("#################################################")
            print(f"*** GHZ state fidelity of the raw state is {fidelity(raw_state, target_GHZ_state)}.***")
            print(f"*** Bell state fidelity is {fidelity(bell_pair_state, bell_target)}.***")
            
            # raw_state is created first and undergoes a SWAP operation to the memory (not modeled, because two copies are considered), but we apply the corresponding gate noise due to this operation.
            rho_emitters_bell_distilled_final = sp.lil_matrix((2**weight, 2**weight), dtype=complex)  # Final density matrix for the emitters
            rho_emitters_bell_distilled_final[:, :] = 0  # Fill the matrix with all zeros
            t_link = 0 # Time for the link generation
            f_link = 0 # Fidelity average
            p_link = 0 # Probability of link generation
            successful_shots = 0 # Number of successful shots

            if self.only_GHZ is True: # If we only want to model and analyse the GHZ state then we repeat the shots, else we repeat the entire stabilizer protocol
                shots = self.shots_emission_direct
            else:
                shots = 1
            for shot in range(shots):
                raw_t_link = 1e-5 # Time for one link generation attempt
                bell_t_link = 1e-5 # Time for one link generation attempt
                time_comm = 0 # Time keeping for the communication qubits
                time_mem = 0 # Time keeping for the memory qubits
                total_time = 0 # Total time keeping for the entire protocol

                t_CX = 0.0005 # Time for the CNOT gate

                qubits_raw = [cirq.LineQubit(i) for i in range(4)]  # Qubits for raw state generation, moved to the memory qubits
                qubits_bell_1 = [cirq.LineQubit(i + 4) for i in range(2)]  # Qubits for first Bell-pair generation, on the communication qubits
                qubits_bell_2 = [cirq.LineQubit(i + 6) for i in range(2)] # Qubits for second Bell-pair generation, on the communication qubits
            # We assume that Bell pairs are generated simultaneously in the two nodes, and the raw state is generated in the first node

                simulator = cirq.DensityMatrixSimulator()
                combined_density_matrix = np.kron(raw_state,np.kron(bell_pair_state,bell_pair_state))
                circuit = cirq.Circuit()

                # Create a Direct Raw state first
                # raw_state_1 is created first and undergoes a SWAP operation to the memory (not modeled, because two copies are considered), but we apply the corresponding gate noise due to this operation.
                attempts_raw = 0 # Number of attempts to create the link
                successes = 0 # Number of successful attempts to create the link, we require one successful event
                while successes < 1:
                    attempts_raw += 1
                    if np.random.rand() < p_link_raw: # If the link generation is successful
                        successes += 1 # Increase the number of successful attempts
                        attempts_raw += 1 # Increase the number of attempts for the first link generation
                        total_time += raw_t_link # Time for the successful link generation 

                # SWAP it to the memory qubits
                time_mem += 3*t_CX # Time for the SWAP operation
                total_time += 3*t_CX # Total time for the SWAP operation added

                # Gate noise on the raw-2 qubits
                circuit.append([cirq.DepolarizingChannel(p=pg).on_each(qubits_raw[i]) for i in range(4)])

                # Decoherence after the SWAP gates, before the CNOT gates
                circuit.append([cirq.PhaseDampingChannel(1-np.exp(-time_mem/self.T2n_idle)).on_each(qubits_raw[i]) for i in range(4)])
                circuit.append([cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_mem/self.T1n_idle)).on_each(qubits_raw[i]) for i in range(4)])

                #Generate the Bell states
                attempts_bell_1 = 0 # Number of attempts to create the link
                successes = 0 # Number of successful attempts to create the link, we require one successful event
                while successes < 1:
                    attempts_bell_1 += 1
                    if np.random.rand() < p_link_bell: # If the link generation is successful
                        successes += 1 # Increase the number of successful attempts
                        attempts_bell_1 += 1 # Increase the number of attempts for the first link generation
                
                attempts_bell_2 = 0 # Number of attempts to create the link
                successes = 0 # Number of successful attempts to create the link, we require one successful event
                while successes < 1:
                    attempts_bell_2 += 1
                    if np.random.rand() < p_link_bell: # If the link generation is successful
                        successes += 1 # Increase the number of successful attempts
                        attempts_bell_2 += 1 # Increase the number of attempts for the first link generation
                
                attempts_bell = attempts_bell_1 if attempts_bell_1 > attempts_bell_2 else attempts_bell_2 # Take the maximum number of attempts for the two links

                # Decoherence on one Bell pair from another which is delayed
                if attempts_bell_1 > attempts_bell_2:
                    effective_attempts = attempts_bell_1 - attempts_bell_2
                    circuit.append([cirq.PhaseDampingChannel(1-np.exp(-(time_mem+effective_attempts*bell_t_link)/self.T2e_idle)).on_each(qubits_bell_2[i]) for i in range(2)])
                    circuit.append([cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-(time_mem+effective_attempts*bell_t_link)/self.T1e_idle)).on_each(qubits_bell_2[i]) for i in range(2)])
                elif attempts_bell_2 > attempts_bell_1:
                    effective_attempts = attempts_bell_2 - attempts_bell_1
                    circuit.append([cirq.PhaseDampingChannel(1-np.exp(-(time_mem+effective_attempts*bell_t_link)/self.T2e_idle)).on_each(qubits_bell_1[i]) for i in range(2)])
                    circuit.append([cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-(time_mem+effective_attempts*bell_t_link)/self.T1e_idle)).on_each(qubits_bell_1[i]) for i in range(2)])
                else:
                    pass

                time_mem += attempts_bell * bell_t_link # Time for the successful link generation
                total_time += attempts_bell * bell_t_link # Total time for the successful link generation added

                # Then decoherence noise due to the second link generation
                circuit.append([cirq.PhaseDampingChannel(1-np.exp(-time_mem/self.T2n_link)).on_each(qubits_raw[i]) for i in range(4)])
                circuit.append([cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_mem/self.T1n_link)).on_each(qubits_raw[i]) for i in range(4)])

                # Apply the 4-CNOT gates in parallel within all the nodes
                circuit.append([cirq.CNOT(qubits_raw[i], qubits_bell_1[i]) for i in range(2)]) # All these CNOT gates are parallel on the architecture
                circuit.append([cirq.CNOT(qubits_raw[i+2], qubits_bell_2[i]) for i in range(2)]) # All these CNOT gates are parallel on the architecture

                time_comm += t_CX # Time for the CNOT gates
                time_mem += t_CX # Time for the CNOT gates
                total_time += t_CX # Total time for the CNOT gates added

                # Apply depolarizing noise to the qubits involved in the CNOT gates
                circuit.append([apply_correlated_two_qubit_noise_channel(pg, [qubits_raw[i], qubits_bell_1[i]]) for i in range(2)])
                circuit.append([apply_correlated_two_qubit_noise_channel(pg, [qubits_raw[i+2], qubits_bell_2[i]]) for i in range(2)])

                # Decoherence after the CNOT gates
                # First on the memory qubits which suffer twice the duration of the two-qubit gates
                circuit.append([cirq.PhaseDampingChannel(1-np.exp(-time_mem/self.T2n_link)).on_each(qubits_raw[i]) for i in range(4)])
                circuit.append([cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_mem/self.T1n_link)).on_each(qubits_raw[i]) for i in range(4)])

                # The other Bell-states suffer this noise only for the duration of the CNOT gates, these are the communication qubits
                circuit.append([cirq.PhaseDampingChannel(1-np.exp(-time_comm/self.T2e_idle)).on_each(qubits_bell_1[i]) for i in range(2)])
                circuit.append([cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_comm/self.T1e_idle)).on_each(qubits_bell_1[i]) for i in range(2)])
                circuit.append([cirq.PhaseDampingChannel(1-np.exp(-time_comm/self.T2e_idle)).on_each(qubits_bell_2[i]) for i in range(2)])
                circuit.append([cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_comm/self.T1e_idle)).on_each(qubits_bell_2[i]) for i in range(2)])

                # Finally, apply the noisy measurement noise on the qubits, here the measurement noise is intrinsically taken to be equal to the gate noise
                circuit.append([cirq.BitFlipChannel(p=pg).on_each(qubits_bell_1[i]) for i in range(2)])
                circuit.append([cirq.BitFlipChannel(p=pg).on_each(qubits_bell_2[i]) for i in range(2)])

                # Add measurements on communication qubits
                for i in range(2):
                    circuit.append(cirq.measure(qubits_bell_1[i], key=f'm{i}'))
                    circuit.append(cirq.measure(qubits_bell_2[i], key=f'm{i+2}'))
                    

                result = simulator.simulate(circuit, initial_state=combined_density_matrix)
                # Extract the final density matrix from the simulation result
                final_density_matrix = result.final_density_matrix

                if ((result.measurements['m0'][0] == 0 and result.measurements['m1'][0] == 0 and result.measurements['m2'][0] == 0 and result.measurements['m3'][0] == 0) or 
                    (result.measurements['m0'][0] == 1 and result.measurements['m1'][0] == 1 and result.measurements['m2'][0] == 1 and result.measurements['m3'][0] == 1) or 
                    (result.measurements['m0'][0] == 0 and result.measurements['m1'][0] == 1 and result.measurements['m2'][0] == 0 and result.measurements['m3'][0] == 1) or
                    (result.measurements['m0'][0] == 1 and result.measurements['m1'][0] == 0 and result.measurements['m2'][0] == 1 and result.measurements['m3'][0] == 0)):
                    post_selected_matrix = final_density_matrix
                    p_distill = np.trace(post_selected_matrix)
                    successful_shots += 1 # Increase the number of successful shots
                    # Normalize the post-selected matrix
                    post_selected_matrix /= p_distill

                    # Partial trace over qubits_2 (qubits 4 to 7)
                    rho_emitters_bell_distilled = partial_trace_numpy(post_selected_matrix, [0,1,2,3], dims=[2] * 8)
                    
                    # Apply the final noisy SWAP operation
                    qubits_raw = [cirq.LineQubit(i) for i in range(4)]

                    # Apply depolarizing noise to the qubits involved in the SWAP gates, beause the measurements are done only on the communication qubits
                    noise_SWAP = [cirq.DepolarizingChannel(p=pg).on_each(qubits_raw[i]) for i in range(4)]

                    time_comm = 3*t_CX # Time for the SWAP operation, after the communication qubits were reset after the measurements
                    total_time += 3*t_CX # Total time for the SWAP operation added

                    # Decoherence after the SWAP gates after the measurement
                    pd_channel_after_SWAP_c = [cirq.PhaseDampingChannel(1-np.exp(-time_comm/self.T2e_idle)).on_each(qubits_raw[i]) for i in range(4)]
                    gad_channel_after_SWAP_c = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_comm/self.T1e_idle)).on_each(qubits_raw[i]) for i in range(4)]

                    circuit = cirq.Circuit(noise_SWAP+pd_channel_after_SWAP_c+gad_channel_after_SWAP_c)
                    rho_emitters_bell_distilled_current = simulator.simulate(circuit, initial_state=rho_emitters_bell_distilled).final_density_matrix

                    rho_emitters_bell_distilled_current = sp.lil_matrix(rho_emitters_bell_distilled_current)

                    current_t_link = total_time
                    t_link += current_t_link # Total time for the link generation

                    current_f_link = fidelity(rho_emitters_bell_distilled_current, target_GHZ_state)
                    f_link += current_f_link # Fidelity average

                    current_p_link = np.real(1/(attempts_raw+attempts_bell) * p_distill)
                    p_link += current_p_link

                    rho_emitters_bell_distilled_final += rho_emitters_bell_distilled_current # Add the current density matrix to the final density matrix
                else:
                    pass

            if successful_shots != 0:
                rho_emitters_bell_distilled_final /= successful_shots # Normalize the final density matrix
                self.t_link = t_link/successful_shots
                self.F_link = f_link/successful_shots
                self.p_link = p_link/successful_shots
            if successful_shots == 0:
                self.t_link = np.inf
                self.F_link = 0
                self.p_link = 0
                rho_emitters_bell_distilled_final = sp.lil_matrix((2**weight, 2**weight), dtype=complex)  # Final density matrix for the emitters if the attempts fail!

            print(f"*** GHZ state fidelity of the Bell-distillation GHZ protocol state is {self.F_link}.***")
            print(f"*** Success rate of the Bell-distillation GHZ protocol state is {self.p_link}.***")
            print("#################################################")

            return rho_emitters_bell_distilled_final


        if network_noise_type == 103:
            # Basic protocol distillation for direct emission scheme
            # We explicitly code the basic protocol inclusive of decoherence effects to the GHZ state and then use that further in the stabilizer protocol
            mu = bell_pair_parameters['mu']
            F_prep = bell_pair_parameters['F_prep']
            labda = bell_pair_parameters['lambda']
            p_DE = bell_pair_parameters['p_DE']
            eta = bell_pair_parameters['eta']
            alpha = bell_pair_parameters['alpha']
            if self.alpha_distill is None:
                self.alpha_distill = bell_pair_parameters['alpha'] # Use alpha_distill = alpha if not explicitly stated as the default value

            weight = 4
            target_GHZ_state = sp.lil_matrix((2**weight, 2**weight))
            target_GHZ_state[0, 0] = 0.5
            target_GHZ_state[0, 2**weight-1] = 0.5
            target_GHZ_state[2**weight-1, 0] = 0.5
            target_GHZ_state[2**weight-1, 2**weight-1] = 0.5
            
            raw_state_1 = np.zeros((2**weight, 2**weight), dtype=complex)
            if self.photon_number_resolution is True:
                raw_state_1[0,0] = -1/2*((-1 + alpha)**2*(1 + mu**2))/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_1[0,15] = -(((1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                raw_state_1[2,2] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                raw_state_1[2,8] = -(((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                raw_state_1[3,3] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_1[3,6] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_1[3,9] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_1[3,12] = 0
                raw_state_1[6,3] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_1[6,6] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_1[6,9] = 0
                raw_state_1[6,12] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_1[8,2] = -(((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                raw_state_1[8,8] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                raw_state_1[9,3] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_1[9,6] = 0
                raw_state_1[9,9] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_1[9,12] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_1[10,10] = (alpha**2*(-1 + eta)**2)/(-1 + alpha*eta)**2
                raw_state_1[11,11] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                raw_state_1[11,14] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_1[12,3] = 0
                raw_state_1[12,6] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_1[12,9] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_1[12,12] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_1[14,11] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_1[14,14] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                raw_state_1[15,0] = -(((1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                raw_state_1[15,15] = -1/2*((-1 + alpha)**2*(1 + mu**2))/((-1 + alpha*eta)**2*(-3 + mu**2))
                p_link_raw_1 = (-3*alpha**2*eta**2*(-1 + alpha*eta)**2*(-3 + mu**2))/2

            elif self.photon_number_resolution is False:
                raw_state_1[0,0] = (-16*(-1 + alpha)**2*(1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[0,15] = (-32*(1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[2,2] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[2,8] = (-8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[3,3] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[3,6] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[3,9] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[3,12] = 0
                raw_state_1[6,3] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[6,6] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[6,9] = 0
                raw_state_1[6,12] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[8,2] = (-8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[8,8] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[9,3] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[9,6] = 0
                raw_state_1[9,9] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[9,12] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[10,10] = (alpha**2*(32*(-3 + mu**2) + eta*(96 - 7*eta + 32*mu**2*(-3 + 2*mu) + eta*mu**2*(54 + (-56 + mu)*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[11,11] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[11,14] = (8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[12,3] = 0
                raw_state_1[12,6] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[12,9] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[12,12] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[14,11] = (8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[14,14] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[15,0] = (-32*(1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_1[15,15] = (-16*(-1 + alpha)**2*(1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                p_link_raw_1 = (-3*alpha**2*eta**2*(32*(-3 + mu**2) + 32*alpha*eta*(3 - 3*mu**2 + 2*mu**3) + alpha**2*eta**2*(-7 + 54*mu**2 - 56*mu**3 + mu**4)))/64
            
            alpha = self.alpha_distill
            raw_state_2 = np.zeros((2**weight, 2**weight), dtype=complex)
            if self.photon_number_resolution is True:
                raw_state_2[0,0] = -1/2*((-1 + alpha)**2*(1 + mu**2))/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_2[0,15] = -(((1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                raw_state_2[2,2] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                raw_state_2[2,8] = -(((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                raw_state_2[3,3] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_2[3,6] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_2[3,9] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_2[3,12] = 0
                raw_state_2[6,3] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_2[6,6] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_2[6,9] = 0
                raw_state_2[6,12] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_2[8,2] = -(((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                raw_state_2[8,8] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                raw_state_2[9,3] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_2[9,6] = 0
                raw_state_2[9,9] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_2[9,12] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_2[10,10] = (alpha**2*(-1 + eta)**2)/(-1 + alpha*eta)**2
                raw_state_2[11,11] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                raw_state_2[11,14] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_2[12,3] = 0
                raw_state_2[12,6] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_2[12,9] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_2[12,12] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_2[14,11] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state_2[14,14] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                raw_state_2[15,0] = -(((1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                raw_state_2[15,15] = -1/2*((-1 + alpha)**2*(1 + mu**2))/((-1 + alpha*eta)**2*(-3 + mu**2))
                p_link_raw_2 = (-3*alpha**2*eta**2*(-1 + alpha*eta)**2*(-3 + mu**2))/2

            elif self.photon_number_resolution is False:
                raw_state_2[0,0] = (-16*(-1 + alpha)**2*(1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[0,15] = (-32*(1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[2,2] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[2,8] = (-8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[3,3] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[3,6] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[3,9] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[3,12] = 0
                raw_state_2[6,3] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[6,6] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[6,9] = 0
                raw_state_2[6,12] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[8,2] = (-8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[8,8] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[9,3] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[9,6] = 0
                raw_state_2[9,9] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[9,12] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[10,10] = (alpha**2*(32*(-3 + mu**2) + eta*(96 - 7*eta + 32*mu**2*(-3 + 2*mu) + eta*mu**2*(54 + (-56 + mu)*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[11,11] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[11,14] = (8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[12,3] = 0
                raw_state_2[12,6] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[12,9] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[12,12] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[14,11] = (8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[14,14] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[15,0] = (-32*(1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state_2[15,15] = (-16*(-1 + alpha)**2*(1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                p_link_raw_2 = (-3*alpha**2*eta**2*(32*(-3 + mu**2) + 32*alpha*eta*(3 - 3*mu**2 + 2*mu**3) + alpha**2*eta**2*(-7 + 54*mu**2 - 56*mu**3 + mu**4)))/64
            
            print("#################################################")
            print(f"*** GHZ state fidelity of the raw state-1 is {fidelity(raw_state_1, target_GHZ_state)}.***")
            print(f"*** GHZ state fidelity of the raw state-2 is {fidelity(raw_state_2, target_GHZ_state)}.***")

            rho_emitters_basic_distilled_final = sp.lil_matrix((2**weight, 2**weight), dtype=complex)  # Final density matrix for the emitters
            rho_emitters_basic_distilled_final[:, :] = 0  # Fill the matrix with all zeros
            t_link = 0 # Time for the link generation
            f_link = 0 # Fidelity average
            p_link = 0 # Probability of link generation
            successful_shots = 0 # Number of successful shots

            if self.only_GHZ is True: # If we only want to model and analyse the GHZ state then we repeat the shots, else we repeat the entire stabilizer protocol
                shots = self.shots_emission_direct
            else:
                shots = 1
            for shot in range(shots):
                raw_t_link = 1e-5 # Time for one link generation attempt
                time_comm = 0 # Time keeping for the communication qubits
                time_mem = 0 # Time keeping for the memory qubits
                total_time = 0 # Total time keeping for the entire protocol

                t_CX = 0.0005 # Time for the CNOT gate

                qubits_2 = [cirq.LineQubit(i) for i in range(4)]  # Qubits for W-state, newly generated on the communication qubits
                qubits_1 = [cirq.LineQubit(i + 4) for i in range(4)]  # Qubits for raw state, sent to the memory qubits, generated before the raw state

                simulator = cirq.DensityMatrixSimulator()
                combined_density_matrix = np.kron(raw_state_2,raw_state_1)

                # Create a Direct Raw state first
                # raw_state_1 is created first and undergoes a SWAP operation to the memory (not modeled, because two copies are considered), but we apply the corresponding gate noise due to this operation.
                attempts_raw_1 = 0 # Number of attempts to create the link
                successes = 0 # Number of successful attempts to create the link, we require one successful event
                while successes < 1:
                    attempts_raw_1 += 1
                    if np.random.rand() < p_link_raw_1: # If the link generation is successful
                        successes += 1 # Increase the number of successful attempts
                        attempts_raw_1 += 1 # Increase the number of attempts for the first link generation
                        total_time += raw_t_link # Time for the successful link generation 

                # SWAP it to the memory qubits
                time_mem += 3*t_CX # Time for the SWAP operation
                total_time += 3*t_CX # Total time for the SWAP operation added

                # Gate noise on the raw-2 qubits
                noise_SWAP = [cirq.DepolarizingChannel(p=pg).on_each(qubits_1[i]) for i in range(4)]

                # Decoherence after the SWAP gates, before the CNOT gates
                pd_channel_after_SWAP = [cirq.PhaseDampingChannel(1-np.exp(-time_mem/self.T2n_idle)).on_each(qubits_1[i]) for i in range(4)]
                gad_channel_after_SWAP = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_mem/self.T1n_idle)).on_each(qubits_1[i]) for i in range(4)]

                #Generate the second Raw state
                attempts_raw_2 = 0 # Number of attempts to create the link
                successes = 0 # Number of successful attempts to create the link, we require one successful event
                while successes < 1:
                    attempts_raw_2 += 1
                    if np.random.rand() < p_link_raw_2: # If the link generation is successful
                        successes += 1 # Increase the number of successful attempts
                        attempts_raw_2 += 1 # Increase the number of attempts for the first link generation
                        total_time += raw_t_link # Time for the successful link generation 

                time_mem += attempts_raw_2 * raw_t_link # Time for the successful link generation
                total_time += attempts_raw_2 * raw_t_link # Total time for the successful link generation added
                time_comm += raw_t_link # Time for the successful link generation on the communication qubits, one attempt time added

                # Then decoherence noise due to the second link generation
                pd_channel_during_link = [cirq.PhaseDampingChannel(1-np.exp(-time_mem/self.T2n_link)).on_each(qubits_1[i]) for i in range(4)]
                gad_channel_during_link = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_mem/self.T1n_link)).on_each(qubits_1[i]) for i in range(4)]

                # Apply the 4-CNOT gates in parallel within all the nodes
                cnots = [cirq.CNOT(qubits_1[i], qubits_2[i]) for i in range(4)] # All these CNOT gates are parallel on the architecture

                time_comm += t_CX # Time for the CNOT gates
                time_mem += t_CX # Time for the CNOT gates
                total_time += t_CX # Total time for the CNOT gates added

                # Apply depolarizing noise to the qubits involved in the CNOT gates
                depolarizing_noise = [apply_correlated_two_qubit_noise_channel(pg, [qubits_1[i], qubits_2[i]]) for i in range(4)]

                # Decoherence after the CNOT gates
                # First on the memory qubits which suffer twice the duration of the two-qubit gates
                pd_channel_after_CNOTs_m = [cirq.PhaseDampingChannel(1-np.exp(-time_mem/self.T2n_link)).on_each(qubits_1[i]) for i in range(4)]
                gad_channel_after_CNOTs_m = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_mem/self.T1n_link)).on_each(qubits_1[i]) for i in range(4)]
                # The other raw state suffers this noise only for the duration of the CNOT gates, these are the communication qubits
                pd_channel_after_CNOTs_c = [cirq.PhaseDampingChannel(1-np.exp(-time_comm/self.T2e_idle)).on_each(qubits_2[i]) for i in range(4)]
                gad_channel_after_CNOTs_c = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_comm/self.T1e_idle)).on_each(qubits_2[i]) for i in range(4)]

                # Finally, apply the noisy measurement noise on the qubits, here the measurement noise is intrinsically taken to be equal to the gate noise
                measurement_noise = [cirq.BitFlipChannel(p=pg).on_each(qubits_2[i]) for i in range(4)]


                circuit = cirq.Circuit(
                    noise_SWAP + pd_channel_after_SWAP + gad_channel_after_SWAP   +
                    pd_channel_during_link + gad_channel_during_link+ cnots+ depolarizing_noise + pd_channel_after_CNOTs_m +
                    gad_channel_after_CNOTs_m + pd_channel_after_CNOTs_c + gad_channel_after_CNOTs_c +
                    measurement_noise)
                
                # Add measurements on communication qubits
                for i in range(4):
                    circuit.append(cirq.measure(qubits_2[i], key=f'm{i}'))

                result = simulator.simulate(circuit, initial_state=combined_density_matrix)
                # Extract the final density matrix from the simulation result
                final_density_matrix = result.final_density_matrix

                if (result.measurements['m0'][0] == 0 and result.measurements['m1'][0] == 0 and result.measurements['m2'][0] == 0 and result.measurements['m3'][0] == 0) or (result.measurements['m0'][0] == 1 and result.measurements['m1'][0] == 1 and result.measurements['m2'][0] == 1 and result.measurements['m3'][0] == 1):
                    # If the measurement results are all |0>, we can proceed with the distillation
                    post_selected_matrix = final_density_matrix
                    p_distill = np.trace(post_selected_matrix)
                    successful_shots += 1 # Increase the number of successful shots
                    # Normalize the post-selected matrix
                    post_selected_matrix /= p_distill

                    # Partial trace over qubits_2 (qubits 4 to 7)
                    rho_emitters_basic_distilled = partial_trace_numpy(post_selected_matrix, [4,5,6,7], dims=[2] * 8)  # Measured qubits traced out
                    rho_emitters_basic_distilled = rho_emitters_basic_distilled/np.trace(rho_emitters_basic_distilled)  # Normalize the density matrix

                    
                    # Apply the final noisy SWAP operation
                    qubits_1 = [cirq.LineQubit(i) for i in range(4)]

                    # Apply depolarizing noise to the qubits involved in the SWAP gates, beause the measurements are done only on the communication qubits
                    noise_SWAP = [cirq.DepolarizingChannel(p=pg).on_each(qubits_1[i]) for i in range(4)]

                    time_comm += 3*t_CX # Time for the SWAP operation
                    total_time += 3*t_CX # Total time for the SWAP operation added

                    # Decoherence after the SWAP gates after the measurement
                    pd_channel_after_SWAP_c = [cirq.PhaseDampingChannel(1-np.exp(-time_comm/self.T2e_idle)).on_each(qubits_1[i]) for i in range(4)]
                    gad_channel_after_SWAP_c = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_comm/self.T1e_idle)).on_each(qubits_1[i]) for i in range(4)]

                    circuit = cirq.Circuit(noise_SWAP+pd_channel_after_SWAP_c+gad_channel_after_SWAP_c)
                    rho_emitters_basic_distilled_current = simulator.simulate(circuit, initial_state=rho_emitters_basic_distilled).final_density_matrix

                    rho_emitters_basic_distilled_current = sp.lil_matrix(rho_emitters_basic_distilled_current)

                    current_t_link = total_time
                    t_link += current_t_link # Total time for the link generation

                    current_f_link = fidelity(rho_emitters_basic_distilled_current, target_GHZ_state)
                    f_link += current_f_link # Fidelity average

                    current_p_link = np.real(1/(attempts_raw_1+attempts_raw_2) * p_distill)
                    p_link += current_p_link

                    rho_emitters_basic_distilled_final += rho_emitters_basic_distilled_current # Add the current density matrix to the final density matrix
                else:
                    pass

            if successful_shots != 0:
                rho_emitters_basic_distilled_final /= successful_shots # Normalize the final density matrix
                self.t_link = t_link/successful_shots
                self.F_link = f_link/successful_shots
                self.p_link = p_link/successful_shots
            if successful_shots == 0:
                self.t_link = np.inf
                self.F_link = 0
                self.p_link = 0
                rho_emitters_basic_distilled_final = sp.lil_matrix((2**weight, 2**weight), dtype=complex)  # Final density matrix for the emitters if the attempts fail!

            print(f"*** GHZ state fidelity of the GHZ Basic protocol state is {self.F_link}.***")
            print(f"*** Success rate of the GHZ Basic protocol state is {self.p_link}.***")
            print("#################################################")

            return rho_emitters_basic_distilled_final

            
        if network_noise_type == 104:
            # W state distillation for direct emission scheme
            mu = bell_pair_parameters['mu']
            F_prep = bell_pair_parameters['F_prep']
            labda = bell_pair_parameters['lambda']
            p_DE = bell_pair_parameters['p_DE']
            eta = bell_pair_parameters['eta']
            alpha = bell_pair_parameters['alpha']
            if self.alpha_distill is None:
                self.alpha_distill = bell_pair_parameters['alpha'] # Use alpha_distill = alpha if not explicitly stated as the default value

            # Target GHZ state
            weight = 4
            target_GHZ_state = sp.lil_matrix((2**weight, 2**weight))
            target_GHZ_state[0, 0] = 0.5
            target_GHZ_state[0, 2**weight-1] = 0.5
            target_GHZ_state[2**weight-1, 0] = 0.5
            target_GHZ_state[2**weight-1, 2**weight-1] = 0.5

            # Target W-State
            target_w_state = sp.lil_matrix((2**weight, 2**weight), dtype=complex)
            indices = [1,2,4,8]
            for i in indices:
                for j in indices:
                    target_w_state[i, j] = 0.25

            raw_state = np.zeros((2**weight, 2**weight), dtype=complex)
            if self.photon_number_resolution is True:
                raw_state[0,0] = -1/2*((-1 + alpha)**2*(1 + mu**2))/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[0,15] = -(((1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                raw_state[2,2] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                raw_state[2,8] = -(((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                raw_state[3,3] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[3,6] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[3,9] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[3,12] = 0
                raw_state[6,3] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[6,6] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[6,9] = 0
                raw_state[6,12] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[8,2] = -(((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                raw_state[8,8] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                raw_state[9,3] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[9,6] = 0
                raw_state[9,9] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[9,12] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[10,10] = (alpha**2*(-1 + eta)**2)/(-1 + alpha*eta)**2
                raw_state[11,11] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                raw_state[11,14] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[12,3] = 0
                raw_state[12,6] = -1/2*((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[12,9] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[12,12] = ((-1 + alpha)**2*(-1 + mu**2))/(2*(-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[14,11] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*(-1 + eta)*(-1 + mu)*mu)/((-1 + alpha*eta)**2*(-3 + mu**2))
                raw_state[14,14] = ((-1 + alpha)*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**2)
                raw_state[15,0] = -(((1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/((-1 + alpha*eta)**2*(-3 + mu**2)))
                raw_state[15,15] = -1/2*((-1 + alpha)**2*(1 + mu**2))/((-1 + alpha*eta)**2*(-3 + mu**2))
                p_link_raw = (-3*alpha**2*eta**2*(-1 + alpha*eta)**2*(-3 + mu**2))/2

            elif self.photon_number_resolution is False:
                raw_state[0,0] = (-16*(-1 + alpha)**2*(1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[0,15] = (-32*(1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[2,2] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[2,8] = (-8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[3,3] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[3,6] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[3,9] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[3,12] = 0
                raw_state[6,3] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[6,6] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[6,9] = 0
                raw_state[6,12] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[8,2] = (-8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[8,8] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[9,3] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[9,6] = 0
                raw_state[9,9] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[9,12] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[10,10] = (alpha**2*(32*(-3 + mu**2) + eta*(96 - 7*eta + 32*mu**2*(-3 + 2*mu) + eta*mu**2*(54 + (-56 + mu)*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[11,11] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[11,14] = (8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[12,3] = 0
                raw_state[12,6] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[12,9] = (16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*(-1 + mu)*mu)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[12,12] = (16*(-1 + alpha)**2*(-1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[14,11] = (8*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha*mu*(4 - 4*mu + eta*(-1 + mu*(2 + mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[14,14] = (-8*(-1 + alpha)*alpha*(2*(-3 + mu**2) + eta*(3 + mu**2*(-3 + 2*mu))))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[15,0] = (-32*(1 - 2*F_prep)**4*(1 - 2*p_DE)**4*(-1 + alpha)**2*mu**2)/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))
                raw_state[15,15] = (-16*(-1 + alpha)**2*(1 + mu**2))/(32*(-3 + mu**2) + alpha*eta*(32*(3 + mu**2*(-3 + 2*mu)) + alpha*eta*(-7 + mu**2*(54 + (-56 + mu)*mu))))

                p_link_raw = (-3*alpha**2*eta**2*(32*(-3 + mu**2) + 32*alpha*eta*(3 - 3*mu**2 + 2*mu**3) + alpha**2*eta**2*(-7 + 54*mu**2 - 56*mu**3 + mu**4)))/64
            
            alpha = self.alpha_distill
            w_state = np.zeros((2**weight, 2**weight), dtype=complex)
            if self.photon_number_resolution is True:
                w_state[1,1] = (-1 + alpha)**3/(4*(-1 + alpha*eta)**3)
                w_state[1,2] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(4*(-1 + alpha*eta)**3)
                w_state[1,4] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(4*(-1 + alpha*eta)**3)
                w_state[1,8] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(4*(-1 + alpha*eta)**3)
                w_state[2,1] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(4*(-1 + alpha*eta)**3)
                w_state[2,2] = (-1 + alpha)**3/(4*(-1 + alpha*eta)**3)
                w_state[2,4] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(4*(-1 + alpha*eta)**3)
                w_state[2,8] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(4*(-1 + alpha*eta)**3)
                w_state[3,3] = ((-1 + alpha)**2*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**3)
                w_state[3,5] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[3,6] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[3,9] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[3,10] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[4,1] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(4*(-1 + alpha*eta)**3)
                w_state[4,2] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(4*(-1 + alpha*eta)**3)
                w_state[4,4] = (-1 + alpha)**3/(4*(-1 + alpha*eta)**3)
                w_state[4,8] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(4*(-1 + alpha*eta)**3)
                w_state[5,3] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[5,5] = ((-1 + alpha)**2*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**3)
                w_state[5,6] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[5,9] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[5,12] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[6,3] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[6,5] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[6,6] = ((-1 + alpha)**2*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**3)
                w_state[6,10] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[6,12] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[7,7] = (3*(-1 + alpha)*alpha**2*(-1 + eta)**2)/(4*(-1 + alpha*eta)**3)
                w_state[7,11] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*(-1 + eta)**2*mu)/(4*(-1 + alpha*eta)**3)
                w_state[7,13] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*(-1 + eta)**2*mu)/(4*(-1 + alpha*eta)**3)
                w_state[7,14] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*(-1 + eta)**2*mu)/(4*(-1 + alpha*eta)**3)
                w_state[8,1] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(4*(-1 + alpha*eta)**3)
                w_state[8,2] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(4*(-1 + alpha*eta)**3)
                w_state[8,4] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(4*(-1 + alpha*eta)**3)
                w_state[8,8] = (-1 + alpha)**3/(4*(-1 + alpha*eta)**3)
                w_state[9,3] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[9,5] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[9,9] = ((-1 + alpha)**2*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**3)
                w_state[9,10] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[9,12] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[10,3] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[10,6] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[10,9] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[10,10] = ((-1 + alpha)**2*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**3)
                w_state[10,12] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[11,7] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*(-1 + eta)**2*mu)/(4*(-1 + alpha*eta)**3)
                w_state[11,11] = (3*(-1 + alpha)*alpha**2*(-1 + eta)**2)/(4*(-1 + alpha*eta)**3)
                w_state[11,13] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*(-1 + eta)**2*mu)/(4*(-1 + alpha*eta)**3)
                w_state[11,14] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*(-1 + eta)**2*mu)/(4*(-1 + alpha*eta)**3)
                w_state[12,5] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[12,6] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[12,9] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[12,10] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(4*(-1 + alpha*eta)**3)
                w_state[12,12] = ((-1 + alpha)**2*alpha*(-1 + eta))/(2*(-1 + alpha*eta)**3)
                w_state[13,7] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*(-1 + eta)**2*mu)/(4*(-1 + alpha*eta)**3)
                w_state[13,11] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*(-1 + eta)**2*mu)/(4*(-1 + alpha*eta)**3)
                w_state[13,13] = (3*(-1 + alpha)*alpha**2*(-1 + eta)**2)/(4*(-1 + alpha*eta)**3)
                w_state[13,14] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*(-1 + eta)**2*mu)/(4*(-1 + alpha*eta)**3)
                w_state[14,7] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*(-1 + eta)**2*mu)/(4*(-1 + alpha*eta)**3)
                w_state[14,11] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*(-1 + eta)**2*mu)/(4*(-1 + alpha*eta)**3)
                w_state[14,13] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*(-1 + eta)**2*mu)/(4*(-1 + alpha*eta)**3)
                w_state[14,14] = (3*(-1 + alpha)*alpha**2*(-1 + eta)**2)/(4*(-1 + alpha*eta)**3)
                w_state[15,15] = (alpha**3*(-1 + eta)**3)/(-1 + alpha*eta)**3
                p_link_w = -4*alpha*eta*(-1 + alpha*eta)**3

            elif self.photon_number_resolution is False:
                w_state[1,1] = (-64*(-1 + alpha)**3)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[1,2] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[1,4] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[1,8] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[2,1] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[2,2] = (-64*(-1 + alpha)**3)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[2,4] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[2,8] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[3,3] = (-128*(-1 + alpha)**2*alpha*(-1 + eta))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[3,5] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[3,6] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[3,9] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[3,10] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[4,1] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[4,2] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[4,4] = (-64*(-1 + alpha)**3)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[4,8] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[5,3] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[5,5] = (-128*(-1 + alpha)**2*alpha*(-1 + eta))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[5,6] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[5,9] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[5,12] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[6,3] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[6,5] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[6,6] = (-128*(-1 + alpha)**2*alpha*(-1 + eta))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[6,10] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[6,12] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[7,7] = (-4*(-1 + alpha)*alpha**2*(48 + eta*(-96 + eta*(49 + mu**2*(3 + 2*mu)))))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[7,11] = (-4*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*mu*(16 + eta*(-32 + eta*(17 + mu*(2 + 3*mu)))))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[7,13] = (-4*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*mu*(16 + eta*(-32 + eta*(17 + mu*(2 + 3*mu)))))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[7,14] = (-4*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*mu*(16 + eta*(-32 + eta*(17 + mu*(2 + 3*mu)))))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[8,1] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[8,2] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[8,4] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**3*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[8,8] = (-64*(-1 + alpha)**3)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[9,3] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[9,5] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[9,9] = (-128*(-1 + alpha)**2*alpha*(-1 + eta))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[9,10] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[9,12] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[10,3] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[10,6] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[10,9] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[10,10] = (-128*(-1 + alpha)**2*alpha*(-1 + eta))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[10,12] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[11,7] = (-4*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*mu*(16 + eta*(-32 + eta*(17 + mu*(2 + 3*mu)))))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[11,11] = (-4*(-1 + alpha)*alpha**2*(48 + eta*(-96 + eta*(49 + mu**2*(3 + 2*mu)))))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[11,13] = (-4*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*mu*(16 + eta*(-32 + eta*(17 + mu*(2 + 3*mu)))))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[11,14] = (-4*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*mu*(16 + eta*(-32 + eta*(17 + mu*(2 + 3*mu)))))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[12,5] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[12,6] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[12,9] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[12,10] = (-64*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)**2*alpha*(-1 + eta)*mu)/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[12,12] = (-128*(-1 + alpha)**2*alpha*(-1 + eta))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[13,7] = (-4*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*mu*(16 + eta*(-32 + eta*(17 + mu*(2 + 3*mu)))))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[13,11] = (-4*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*mu*(16 + eta*(-32 + eta*(17 + mu*(2 + 3*mu)))))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[13,13] = (-4*(-1 + alpha)*alpha**2*(48 + eta*(-96 + eta*(49 + mu**2*(3 + 2*mu)))))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[13,14] = (-4*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*mu*(16 + eta*(-32 + eta*(17 + mu*(2 + 3*mu)))))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[14,7] = (-4*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*mu*(16 + eta*(-32 + eta*(17 + mu*(2 + 3*mu)))))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[14,11] = (-4*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*mu*(16 + eta*(-32 + eta*(17 + mu*(2 + 3*mu)))))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[14,13] = (-4*(1 - 2*F_prep)**2*(1 - 2*p_DE)**2*(-1 + alpha)*alpha**2*mu*(16 + eta*(-32 + eta*(17 + mu*(2 + 3*mu)))))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[14,14] = (-4*(-1 + alpha)*alpha**2*(48 + eta*(-96 + eta*(49 + mu**2*(3 + 2*mu)))))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))
                w_state[15,15] = (alpha**3*(256 + eta*(-768 + eta*(784 - 271*eta + 6*(8 - 7*eta)*mu**2 + 8*(4 - 3*eta)*mu**3 + 9*eta*mu**4))))/(256 + alpha*eta*(-768 + alpha*eta*(16*(49 + mu**2*(3 + 2*mu)) + alpha*eta*(-271 + 3*mu**2*(-14 + mu*(-8 + 3*mu))))))

                p_link_w = (alpha*eta*(256 - 768*alpha*eta + 16*alpha**2*eta**2*(49 + 3*mu**2 + 2*mu**3) + alpha**3*eta**3*(-271 - 42*mu**2 - 24*mu**3 + 9*mu**4)))/64

            print("#################################################")
            print(f"*** GHZ state fidelity of the raw state is {fidelity(raw_state, target_GHZ_state)}.***")
            print(f"*** GHZ state fidelity of the W-state is {fidelity(w_state, target_w_state)}.***")

            # raw_state_1 is created first and undergoes a SWAP operation to the memory (not modeled, because separately considered), but we apply the corresponding gate noise due to this operation.

            rho_emitters_W_distilled_final = sp.lil_matrix((2**weight, 2**weight), dtype=complex)  # Final density matrix for the emitters
            rho_emitters_W_distilled_final[:, :] = 0  # Fill the matrix with all zeros
            t_link = 0 # Time for the link generation
            f_link = 0 # Fidelity average
            p_link = 0 # Probability of link generation
            successful_shots = 0 # Number of successful shots

            if self.only_GHZ is True: # If we only want to model and analyse the GHZ state then we repeat the shots, else we repeat the entire stabilizer protocol
                shots = self.shots_emission_direct
            else:
                shots = 1
            for shot in range(shots):
                raw_t_link = 1e-5 # Time for one link generation attempt
                w_t_link = 1e-5 # Time for one link generation attempt
                time_comm = 0 # Time keeping for the communication qubits
                time_mem = 0 # Time keeping for the memory qubits
                total_time = 0 # Total time keeping for the entire protocol

                t_CX = 0.0005 # Time for the CNOT gate
                t_mH = t_CX # Time for the Hadamard gate on the memory qubits

                qubits_w = [cirq.LineQubit(i) for i in range(4)]  # Qubits for W-state, newly generated on the communication qubits
                qubits_raw = [cirq.LineQubit(i + 4) for i in range(4)]  # Qubits for raw state, sent to the memory qubits, generated before the raw state

                simulator = cirq.DensityMatrixSimulator()
                combined_density_matrix = np.kron(w_state,raw_state)

                # Create a Direct Raw state first
                # raw_state_1 is created first and undergoes a SWAP operation to the memory (not modeled, because two copies are considered), but we apply the corresponding gate noise due to this operation.
                attempts_raw = 0 # Number of attempts to create the link
                successes = 0 # Number of successful attempts to create the link, we require one successful event
                while successes < 1:
                    attempts_raw += 1
                    if np.random.rand() < p_link_raw: # If the link generation is successful
                        successes += 1 # Increase the number of successful attempts
                        attempts_raw += 1 # Increase the number of attempts for the first link generation
                        total_time += raw_t_link # Time for the successful link generation 

                # SWAP it to the memory qubits
                time_mem += 3*t_CX # Time for the SWAP operation
                total_time += 3*t_CX # Total time for the SWAP operation added

                # Gate noise on the raw-2 qubits
                noise_SWAP = [cirq.DepolarizingChannel(p=pg).on_each(qubits_raw[i]) for i in range(4)]

                # Decoherence after the SWAP gates, before the CNOT gates
                pd_channel_after_SWAP = [cirq.PhaseDampingChannel(1-np.exp(-time_mem/self.T2n_idle)).on_each(qubits_raw[i]) for i in range(4)]
                gad_channel_after_SWAP = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_mem/self.T1n_idle)).on_each(qubits_raw[i]) for i in range(4)]

                #Generate the second Raw state
                attempts_w = 0 # Number of attempts to create the link
                successes = 0 # Number of successful attempts to create the link, we require one successful event
                while successes < 1:
                    attempts_w += 1
                    if np.random.rand() < p_link_w: # If the link generation is successful
                        successes += 1 # Increase the number of successful attempts
                        attempts_w += 1 # Increase the number of attempts for the first link generation
                        total_time += w_t_link # Time for the successful link generation 

                time_mem += attempts_w * w_t_link # Time for the successful link generation
                total_time += attempts_w * w_t_link # Total time for the successful link generation added
                time_comm += w_t_link # Time for the successful link generation on the communication qubits, one attempt time added

                # Then decoherence noise due to the second link generation
                pd_channel_during_link = [cirq.PhaseDampingChannel(1-np.exp(-time_mem/self.T2n_link)).on_each(qubits_raw[i]) for i in range(4)]
                gad_channel_during_link = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_mem/self.T1n_link)).on_each(qubits_raw[i]) for i in range(4)]

                # Apply the hadamard on the w-state qubits
                hadamards = [cirq.H(qubits_w[i]) for i in range(4)]
                time_mem += t_mH # Time for the Hadamard
                time_comm += t_mH # Time for the Hadamard
                total_time += t_mH # Total time for the Hadamard added

                hadamard_depolarising_noise = [cirq.DepolarizingChannel(p=pg).on_each(qubits_w[i]) for i in range(4)]

                # Decoherence after the Hadamard gates
                pd_channel_after_hadamard_m = [cirq.PhaseDampingChannel(1-np.exp(-time_mem/self.T2n_idle)).on_each(qubits_raw[i]) for i in range(4)]
                gad_channel_after_hadamard_m = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_mem/self.T1n_idle)).on_each(qubits_raw[i]) for i in range(4)]

                pd_channel_after_hadamard_c = [cirq.PhaseDampingChannel(1-np.exp(-time_comm/self.T2e_idle)).on_each(qubits_w[i]) for i in range(4)]
                gad_channel_after_hadamard_c = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_comm/self.T1e_idle)).on_each(qubits_w[i]) for i in range(4)]

                # Apply the 4-CNOT gates in parallel within all the nodes
                cnots = [cirq.CNOT(qubits_raw[i], qubits_w[i]) for i in range(4)] # All these CNOT gates are parallel on the architecture

                time_comm += t_CX # Time for the CNOT gates
                time_mem += t_CX # Time for the CNOT gates
                total_time += t_CX # Total time for the CNOT gates added

                # Apply depolarizing noise to the qubits involved in the CNOT gates
                depolarizing_noise = [apply_correlated_two_qubit_noise_channel(pg, [qubits_raw[i], qubits_w[i]]) for i in range(4)]

                # Decoherence after the CNOT gates
                # First on the memory qubits which suffer twice the duration of the two-qubit gates
                pd_channel_after_CNOTs_m = [cirq.PhaseDampingChannel(1-np.exp(-time_mem/self.T2n_link)).on_each(qubits_raw[i]) for i in range(4)]
                gad_channel_after_CNOTs_m = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_mem/self.T1n_link)).on_each(qubits_raw[i]) for i in range(4)]
                # The other raw state suffers this noise only for the duration of the CNOT gates, these are the communication qubits
                pd_channel_after_CNOTs_c = [cirq.PhaseDampingChannel(1-np.exp(-time_comm/self.T2e_idle)).on_each(qubits_w[i]) for i in range(4)]
                gad_channel_after_CNOTs_c = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_comm/self.T1e_idle)).on_each(qubits_w[i]) for i in range(4)]

                # Finally, apply the noisy measurement noise on the qubits, here the measurement noise is intrinsically taken to be equal to the gate noise
                measurement_noise = [cirq.BitFlipChannel(p=pg).on_each(qubits_w[i]) for i in range(4)]


                circuit = cirq.Circuit(noise_SWAP + pd_channel_after_SWAP + gad_channel_after_SWAP+ pd_channel_during_link + gad_channel_during_link + hadamards + hadamard_depolarising_noise + pd_channel_after_hadamard_m + gad_channel_after_hadamard_m  + pd_channel_after_hadamard_c + gad_channel_after_hadamard_c + cnots 
                                       + depolarizing_noise + pd_channel_after_CNOTs_m + gad_channel_after_CNOTs_m + pd_channel_after_CNOTs_c + gad_channel_after_CNOTs_c + measurement_noise) 
                
                # Add measurements on communication qubits
                for i in range(4):
                    circuit.append(cirq.measure(qubits_w[i], key=f'm{i}'))

                result = simulator.simulate(circuit, initial_state=combined_density_matrix)
                # Extract the final density matrix from the simulation result
                final_density_matrix = result.final_density_matrix

                if (result.measurements['m0'][0] == 0 and result.measurements['m1'][0] == 0 and result.measurements['m2'][0] == 0 and result.measurements['m3'][0] == 0) or (result.measurements['m0'][0] == 1 and result.measurements['m1'][0] == 1 and result.measurements['m2'][0] == 1 and result.measurements['m3'][0] == 1):
                    # If the measurement results are all |0>, we can proceed with the distillation
                    post_selected_matrix = final_density_matrix
                    p_distill = np.trace(post_selected_matrix)
                    successful_shots += 1 # Increase the number of successful shots
                    # Normalize the post-selected matrix
                    post_selected_matrix /= p_distill

                    # Partial trace over qubits_2 (qubits 4 to 7)
                    rho_emitters_w_distilled = partial_trace_numpy(post_selected_matrix, [4,5,6,7], dims=[2] * 8)  # Measured qubits traced out
                    rho_emitters_w_distilled = rho_emitters_w_distilled/np.trace(rho_emitters_w_distilled)  # Normalize the density matrix

                    # Define Pauli matrices
                    I = np.eye(2)
                    Z = np.array([[1, 0], [0, -1]])
                    IZII = np.kron(I, np.kron(Z, np.kron(I, I)))

                    # Correct the density matrices with Pauli corrections
                    rho_emitters_w_corrected = IZII @ rho_emitters_w_distilled @ IZII.T
                    
                    # Apply the final noisy SWAP operation
                    qubits_raw = [cirq.LineQubit(i) for i in range(4)]

                    # Apply depolarizing noise to the qubits involved in the SWAP gates, beause the measurements are done only on the communication qubits
                    noise_SWAP = [cirq.DepolarizingChannel(p=pg).on_each(qubits_raw[i]) for i in range(4)]

                    time_comm += 3*t_CX # Time for the SWAP operation
                    total_time += 3*t_CX # Total time for the SWAP operation added

                    # Decoherence after the SWAP gates after the measurement
                    pd_channel_after_SWAP_c = [cirq.PhaseDampingChannel(1-np.exp(-time_comm/self.T2e_idle)).on_each(qubits_raw[i]) for i in range(4)]
                    gad_channel_after_SWAP_c = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_comm/self.T1e_idle)).on_each(qubits_raw[i]) for i in range(4)]

                    circuit = cirq.Circuit(noise_SWAP+pd_channel_after_SWAP_c+gad_channel_after_SWAP_c)
                    rho_emitters_W_distilled_current = simulator.simulate(circuit, initial_state=rho_emitters_w_corrected).final_density_matrix

                    rho_emitters_W_distilled_current = sp.lil_matrix(rho_emitters_W_distilled_current)

                    current_t_link = total_time
                    t_link += current_t_link # Total time for the link generation

                    current_f_link = fidelity(rho_emitters_W_distilled_current, target_GHZ_state)
                    f_link += current_f_link # Fidelity average

                    current_p_link = np.real(1/(attempts_raw+attempts_w) * p_distill)
                    p_link += current_p_link

                    rho_emitters_W_distilled_final += rho_emitters_W_distilled_current # Add the current density matrix to the final density matrix
                else:
                    pass

            if successful_shots != 0:
                rho_emitters_W_distilled_final /= successful_shots # Normalize the final density matrix
                self.t_link = t_link/successful_shots
                self.F_link = f_link/successful_shots
                self.p_link = p_link/successful_shots
            if successful_shots == 0:
                self.t_link = np.inf
                self.F_link = 0
                self.p_link = 0
                rho_emitters_W_distilled_final = sp.lil_matrix((2**weight, 2**weight), dtype=complex)  # Final density matrix for the emitters if the attempts fail!


            print(f"*** GHZ state fidelity of the W-State protocol state is {self.F_link}.***")
            print(f"*** Success rate of the W-State protocol state is {self.p_link}.***")
            print("#################################################")

            return rho_emitters_W_distilled_final
        
        # k=11 Bell-pair fusion scheme for comparison (k=11 scheme)
        # Figure of this scheme is appened in `results.ipynb`
        if network_noise_type == 105:
            mu = bell_pair_parameters['mu']
            F_prep = bell_pair_parameters['F_prep']
            labda = bell_pair_parameters['lambda']
            p_DE = bell_pair_parameters['p_DE']
            eta = bell_pair_parameters['eta']
            alpha = bell_pair_parameters['alpha']
            bell_pair_protocol = bell_pair_parameters['ent_prot']

            weight = 4
            density_matrix_target = sp.lil_matrix((2**weight, 2**weight))
            density_matrix_target[0, 0] = 0.5
            density_matrix_target[0, 2**weight-1] = 0.5
            density_matrix_target[2**weight-1, 0] = 0.5
            density_matrix_target[2**weight-1, 2**weight-1] = 0.5

            # This protocol uses all the elementary links as the double-click Bell-pairs only
            double_click_bell_pair = np.zeros((4,4), dtype=complex)
            if self.photon_number_resolution is True:
                double_click_bell_pair[0,0] = (2 - 2*alpha + (-2 + pg)*pg*(1 + alpha*(-2 + eta)))/(4 + 2*(-2 + pg)*pg*eta + 2*alpha*(-2 + pg*(2 + pg*(-2 + eta)*eta)))
                double_click_bell_pair[0,3] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*mu*(np.sqrt(1-mu) - np.sqrt(1+mu))*(np.sqrt(1-mu) + np.sqrt(1+mu)))/(4 + 2*(-2 + pg)*pg*eta + 2*alpha*(-2 + pg*(2 + pg*(-2 + eta)*eta)))
                double_click_bell_pair[2,2] = (pg*(pg + 2*(-1 + alpha) + pg*alpha*(-2 + eta))*(-1 + eta))/(2 + (-2 + pg)*pg*eta + alpha*(-2 + pg*(2 + pg*(-2 + eta)*eta)))
                double_click_bell_pair[3,0] = ((1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*mu*(np.sqrt(1-mu) - np.sqrt(1+mu))*(np.sqrt(1-mu) + np.sqrt(1+mu)))/(4 + 2*(-2 + pg)*pg*eta + 2*alpha*(-2 + pg*(2 + pg*(-2 + eta)*eta)))
                double_click_bell_pair[3,3] = (2 - 2*alpha + (-2 + pg)*pg*(1 + alpha*(-2 + eta)))/(4 + 2*(-2 + pg)*pg*eta + 2*alpha*(-2 + pg*(2 + pg*(-2 + eta)*eta)))
                p_link_dc_bell = alpha*eta**2*(2 - 2*pg*eta + pg**2*eta + alpha*(-2 + 2*pg + pg**2*(-2 + eta)*eta))

            if self.photon_number_resolution is False:
                double_click_bell_pair[0,0] = (2*(8 - 8*alpha + (-2 + pg)*pg*(4 - alpha*(8 + eta*(-3 + mu)))))/(32 + alpha*(-32 + pg*(32 + pg*eta*(8 + eta*(-3 + mu))*(-3 + mu))) - 4*(-2 + pg)*pg*eta*(-3 + mu))
                double_click_bell_pair[0,3] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*mu)/(32 + alpha*(-32 + pg*(32 + pg*eta*(8 + eta*(-3 + mu))*(-3 + mu))) - 4*(-2 + pg)*pg*eta*(-3 + mu))
                double_click_bell_pair[2,2] = (pg*(8 - 8*alpha + pg*(-4 + alpha*(8 + eta*(-3 + mu))))*(4 + eta*(-3 + mu)))/(32 + alpha*(-32 + pg*(32 + pg*eta*(8 + eta*(-3 + mu))*(-3 + mu))) - 4*(-2 + pg)*pg*eta*(-3 + mu))
                double_click_bell_pair[3,0] = (-16*(1 - 2*F_prep)**2*(1 - 2*p_DE)**4*(-1 + pg)**2*(-1 + alpha)*mu)/(32 + alpha*(-32 + pg*(32 + pg*eta*(8 + eta*(-3 + mu))*(-3 + mu))) - 4*(-2 + pg)*pg*eta*(-3 + mu))
                double_click_bell_pair[3,3] = (2*(8 - 8*alpha + (-2 + pg)*pg*(4 - alpha*(8 + eta*(-3 + mu)))))/(32 + alpha*(-32 + pg*(32 + pg*eta*(8 + eta*(-3 + mu))*(-3 + mu))) - 4*(-2 + pg)*pg*eta*(-3 + mu))
                p_link_dc_bell = (alpha*eta**2*(32 + alpha*(-32 + 32*pg + pg**2*eta*(8 + eta*(-3 + mu))*(-3 + mu)) + 8*pg*eta*(-3 + mu) - 4*pg**2*eta*(-3 + mu)))/16
        
        

            # Create a circuit for the Bell pair fusion protocol
            bell_dc_t_link = 1e-5  # Time for one link generation attempt
            two_qubit_times = 0.0005  # Time for the two-qubit gates (CNOT, CZ, etc.)
            t_CX = two_qubit_times
            t_CZ = two_qubit_times
            t_CiY = two_qubit_times
            t_mH = two_qubit_times

            # Function to distill over two nodes
            def create_bell_link_and_distill_over_two_nodes(time_tracking):
                # These are all local variables of the function
                time_mem = time_tracking[0] # Time keeping for the memory qubits
                time_comm = time_tracking[1] # Time keeping for the communication qubits
                total_time = time_tracking[2] # Total time keeping for the entire protocol

                # First we only model the creating two Bell-pairs and distilling one from another
                qubits_2 = [cirq.LineQubit(i) for i in range(2)]  # Qubits for W-state, newly generated on the communication qubits
                qubits_1 = [cirq.LineQubit(i + 2) for i in range(2)]  # Qubits for raw state, sent to the memory qubits, generated before the raw state

                simulator = cirq.DensityMatrixSimulator()
                combined_density_matrix = np.kron(double_click_bell_pair, double_click_bell_pair)

                # Create a Direct DC Bell state first
                # bell_state_1 is created first and undergoes a SWAP operation to the memory (not modeled, because two copies are considered), but we apply the corresponding gate noise due to this operation.
                attempts_bell_1 = 0 # Number of attempts to create the link
                successes = 0 # Number of successful attempts to create the link, we require one successful event
                while successes < 1:
                    attempts_bell_1 += 1
                    if np.random.rand() < p_link_dc_bell: # If the link generation is successful
                        successes += 1 # Increase the number of successful attempts
                        attempts_bell_1 += 1 # Increase the number of attempts for the first link generation
                        total_time += bell_dc_t_link # Time for the successful link generation 

                # SWAP it to the memory qubits
                time_mem += 3*t_CX # Time for the SWAP operation
                total_time += 3*t_CX # Total time for the SWAP operation added

                # Gate noise on the raw-2 qubits
                noise_SWAP = [cirq.DepolarizingChannel(p=pg).on_each(qubits_1[i]) for i in range(2)]

                # Decoherence after the SWAP gates, before the CNOT gates
                pd_channel_after_SWAP = [cirq.PhaseDampingChannel(1-np.exp(-time_mem/self.T2n_idle)).on_each(qubits_1[i]) for i in range(2)]
                gad_channel_after_SWAP = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_mem/self.T1n_idle)).on_each(qubits_1[i]) for i in range(2)]

                #Generate the second Raw state
                attempts_bell_2 = 0 # Number of attempts to create the link
                successes = 0 # Number of successful attempts to create the link, we require one successful event
                while successes < 1:
                    attempts_bell_2 += 1
                    if np.random.rand() < p_link_dc_bell: # If the link generation is successful
                        successes += 1 # Increase the number of successful attempts
                        attempts_bell_2 += 1 # Increase the number of attempts for the first link generation
                        total_time += bell_dc_t_link # Time for the successful link generation 

                time_mem += attempts_bell_2 * bell_dc_t_link # Time for the successful link generation
                total_time += attempts_bell_2 * bell_dc_t_link # Total time for the successful link generation added
                time_comm += bell_dc_t_link # Time for the successful link generation on the communication qubits, one attempt time added

                # Then decoherence noise due to the second link generation
                pd_channel_during_link = [cirq.PhaseDampingChannel(1-np.exp(-time_mem/self.T2n_link)).on_each(qubits_1[i]) for i in range(2)]
                gad_channel_during_link = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_mem/self.T1n_link)).on_each(qubits_1[i]) for i in range(2)]

                # Apply the 4-CNOT gates in parallel within all the nodes
                cnots = [cirq.CNOT(qubits_1[i], qubits_2[i]) for i in range(2)] # All these CNOT gates are parallel on the architecture

                time_comm += t_CX # Time for the CNOT gates
                time_mem += t_CX # Time for the CNOT gates
                total_time += t_CX # Total time for the CNOT gates added

                # Apply depolarizing noise to the qubits involved in the CNOT gates
                depolarizing_noise = [apply_correlated_two_qubit_noise_channel(pg, [qubits_1[i], qubits_2[i]]) for i in range(2)]

                # Decoherence after the CNOT gates
                # First on the memory qubits which suffer twice the duration of the two-qubit gates
                pd_channel_after_CNOTs_m = [cirq.PhaseDampingChannel(1-np.exp(-time_mem/self.T2n_link)).on_each(qubits_1[i]) for i in range(2)]
                gad_channel_after_CNOTs_m = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_mem/self.T1n_link)).on_each(qubits_1[i]) for i in range(2)]
                # The other raw state suffers this noise only for the duration of the CNOT gates, these are the communication qubits
                pd_channel_after_CNOTs_c = [cirq.PhaseDampingChannel(1-np.exp(-time_comm/self.T2e_idle)).on_each(qubits_2[i]) for i in range(2)]
                gad_channel_after_CNOTs_c = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_comm/self.T1e_idle)).on_each(qubits_2[i]) for i in range(2)]

                # Finally, apply the noisy measurement noise on the qubits, here the measurement noise is intrinsically taken to be equal to the gate noise
                measurement_noise = [cirq.BitFlipChannel(p=pg).on_each(qubits_2[i]) for i in range(2)]


                circuit = cirq.Circuit(
                    noise_SWAP + pd_channel_after_SWAP + gad_channel_after_SWAP   +
                    pd_channel_during_link + gad_channel_during_link+ cnots+ depolarizing_noise + pd_channel_after_CNOTs_m +
                    gad_channel_after_CNOTs_m + pd_channel_after_CNOTs_c + gad_channel_after_CNOTs_c +
                    measurement_noise)
                
                # Add measurements on communication qubits
                for i in range(2):
                    circuit.append(cirq.measure(qubits_2[i], key=f'm{i}'))

                result = simulator.simulate(circuit, initial_state=combined_density_matrix)
                # Extract the final density matrix from the simulation result
                final_density_matrix = result.final_density_matrix

                if (result.measurements['m0'][0] == 0 and result.measurements['m1'][0] == 0) or (result.measurements['m0'][0] == 1 and result.measurements['m1'][0] == 1):
                    # If the measurement results are all |0>, we can proceed with the distillation
                    post_selected_matrix = final_density_matrix
                    p_distill = np.trace(post_selected_matrix)
                    # Normalize the post-selected matrix
                    post_selected_matrix /= p_distill

                    # Partial trace over qubits_2 (qubits 4 to 7)
                    rho_emitters_distilled = partial_trace_numpy(post_selected_matrix, [2,3], dims=[2] * 4)  # Measured qubits traced out
                    rho_emitters_distilled = rho_emitters_distilled/np.trace(rho_emitters_distilled)  # Normalize the density matrix

                    
                    # Apply the final noisy SWAP operation
                    qubits_1 = [cirq.LineQubit(i) for i in range(2)]

                    # Apply depolarizing noise to the qubits involved in the SWAP gates, beause the measurements are done only on the communication qubits
                    noise_SWAP = [cirq.DepolarizingChannel(p=pg).on_each(qubits_1[i]) for i in range(2)]

                    time_comm += 3*t_CX # Time for the SWAP operation
                    total_time += 3*t_CX # Total time for the SWAP operation added

                    # Decoherence after the SWAP gates after the measurement
                    pd_channel_after_SWAP_c = [cirq.PhaseDampingChannel(1-np.exp(-time_comm/self.T2e_idle)).on_each(qubits_1[i]) for i in range(2)]
                    gad_channel_after_SWAP_c = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_comm/self.T1e_idle)).on_each(qubits_1[i]) for i in range(2)]

                    circuit = cirq.Circuit(noise_SWAP+pd_channel_after_SWAP_c+gad_channel_after_SWAP_c)
                    rho_bell_distilled = simulator.simulate(circuit, initial_state=rho_emitters_distilled).final_density_matrix
                    return [rho_bell_distilled, [time_comm, time_mem, total_time]] # Return the distilled density matrix and the time taken for the protocol
            
                else:
                    return None
                
            def distill_with_CiYiY(base_state,time_tracking):
                # Module to perform the distillation with the CiYiY gates as required in this fusion-distillation protocol
                # These are all local variables of the function below
                # These are all local variables of the function
                time_mem = time_tracking[0] # Time keeping for the memory qubits
                time_comm = time_tracking[1] # Time keeping for the communication qubits
                total_time = time_tracking[2] # Total time keeping for the entire protocol

                qubits_2 = [cirq.LineQubit(i) for i in range(2)]  # Qubits for W-state, newly generated on the communication qubits
                qubits_1 = [cirq.LineQubit(i + 2) for i in range(2)]  # Qubits for raw state, sent to the memory qubits, generated before the raw state
                simulator = cirq.DensityMatrixSimulator()

                combined_density_matrix = np.kron(base_state, double_click_bell_pair)
                # Create a Direct DC Bell state first
                # bell_state_1 is created first and undergoes a SWAP operation to the memory (not modeled, because two copies are considered), but we apply the corresponding gate noise due to this operation.
                attempts_bell = 0 # Number of attempts to create the link
                successes = 0
                # Number of successful attempts to create the link, we require one successful event
                while successes < 1:
                    attempts_bell += 1
                    if np.random.rand() < p_link_dc_bell:
                        successes += 1
                        attempts_bell += 1
                        total_time += bell_dc_t_link
                # SWAP it to the memory qubits
                time_mem += 3*t_CX # Time for the SWAP operation
                total_time += 3*t_CX
                # Gate noise on the raw-2 qubits
                noise_SWAP = [cirq.DepolarizingChannel(p=pg).on_each(qubits_1[i]) for i in range(2)]
                # Decoherence after the SWAP gates, before the CNOT gates
                pd_channel_after_SWAP = [cirq.PhaseDampingChannel(1-np.exp(-time_mem/self.T2n_idle)).on_each(qubits_1[i]) for i in range(2)]
                gad_channel_after_SWAP = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_mem/self.T1n_idle)).on_each(qubits_1[i]) for i in range(2)]
                
                # Define iY gate
                iY = cirq.Y**0.5  # Because iY = exp(iπ/2 Y), which is equivalent to Y**0.5 up to a global phase
                ciY = cirq.ControlledGate(iY)

                ciYciYs = [ciY.on(qubits_1[i], qubits_2[i]) for i in range(2)]

                time_comm += t_CiY # Time for the CNOT gates
                time_mem += t_CiY # Time for the CNOT gates
                total_time += t_CiY # Total time for the CNOT gates added

                # Apply depolarizing noise to the qubits involved in the CNOT gates
                depolarizing_noise = [apply_correlated_two_qubit_noise_channel(pg, [qubits_1[i], qubits_2[i]]) for i in range(2)]

                # Decoherence after the CNOT gates
                # First on the memory qubits which suffer twice the duration of the two-qubit gates
                pd_channel_after_CNOTs_m = [cirq.PhaseDampingChannel(1-np.exp(-time_mem/self.T2n_link)).on_each(qubits_1[i]) for i in range(2)]
                gad_channel_after_CNOTs_m = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_mem/self.T1n_link)).on_each(qubits_1[i]) for i in range(2)]
                # The other raw state suffers this noise only for the duration of the CNOT gates, these are the communication qubits
                pd_channel_after_CNOTs_c = [cirq.PhaseDampingChannel(1-np.exp(-time_comm/self.T2e_idle)).on_each(qubits_2[i]) for i in range(2)]
                gad_channel_after_CNOTs_c = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_comm/self.T1e_idle)).on_each(qubits_2[i]) for i in range(2)]

                # Finally, apply the noisy measurement noise on the qubits, here the measurement noise is intrinsically taken to be equal to the gate noise
                measurement_noise = [cirq.BitFlipChannel(p=pg).on_each(qubits_2[i]) for i in range(2)]


                circuit = cirq.Circuit(
                    noise_SWAP + pd_channel_after_SWAP + gad_channel_after_SWAP   +
                    ciYciYs + depolarizing_noise + pd_channel_after_CNOTs_m +
                    gad_channel_after_CNOTs_m + pd_channel_after_CNOTs_c + gad_channel_after_CNOTs_c +
                    measurement_noise)
                
                # Add measurements on communication qubits
                for i in range(2):
                    circuit.append(cirq.measure(qubits_2[i], key=f'm{i}'))

                result = simulator.simulate(circuit, initial_state=combined_density_matrix)
                # Extract the final density matrix from the simulation result
                final_density_matrix = result.final_density_matrix

                if (result.measurements['m0'][0] == 0 and result.measurements['m1'][0] == 0) or (result.measurements['m0'][0] == 1 and result.measurements['m1'][0] == 1):
                    # If the measurement results are all |0>, we can proceed with the distillation
                    post_selected_matrix = final_density_matrix
                    p_distill = np.trace(post_selected_matrix)
                    # Normalize the post-selected matrix
                    post_selected_matrix /= p_distill

                    # Partial trace over qubits_2 (qubits 4 to 7)
                    rho_emitters_distilled = partial_trace_numpy(post_selected_matrix, [0,1], dims=[2] * 4)  # Measured qubits traced out
                    rho_emitters_distilled = rho_emitters_distilled/np.trace(rho_emitters_distilled)  # Normalize the density matrix

                    
                    # Apply the final noisy SWAP operation
                    qubits_1 = [cirq.LineQubit(i) for i in range(2)]

                    # Apply depolarizing noise to the qubits involved in the SWAP gates, beause the measurements are done only on the communication qubits
                    noise_SWAP = [cirq.DepolarizingChannel(p=pg).on_each(qubits_1[i]) for i in range(2)]

                    time_comm += 3*t_CX # Time for the SWAP operation
                    total_time += 3*t_CX # Total time for the SWAP operation added

                    # Decoherence after the SWAP gates after the measurement
                    pd_channel_after_SWAP_c = [cirq.PhaseDampingChannel(1-np.exp(-time_comm/self.T2e_idle)).on_each(qubits_1[i]) for i in range(2)]
                    gad_channel_after_SWAP_c = [cirq.GeneralizedAmplitudeDampingChannel(0.5, 1-np.exp(-time_comm/self.T1e_idle)).on_each(qubits_1[i]) for i in range(2)]

                    circuit = cirq.Circuit(noise_SWAP+pd_channel_after_SWAP_c+gad_channel_after_SWAP_c)
                    rho_bell_distilled = simulator.simulate(circuit, initial_state=rho_emitters_distilled).final_density_matrix
                    return [rho_bell_distilled, [time_comm, time_mem, total_time]]
            
                else:
                    return None

            
            rho_fusion_protocol_final = sp.lil_matrix((2**weight, 2**weight), dtype=complex)  # Final density matrix for the emitters
            rho_fusion_protocol_final[:, :] = 0  # Fill the matrix with all zeros
            t_link = 0 # Time for the link generation
            f_link = 0 # Fidelity average
            p_link = 0 # Probability of link generation
            successful_shots = 0 # Number of successful shots

            if self.only_GHZ is True: # If we only want to model and analyse the GHZ state then we repeat the shots, else we repeat the entire stabilizer protocol
                shots = self.shots_emission_direct
            else:
                shots = 1
            for shot in range(shots):
                # Protocol success/fail flag
                protocol_failed = False
                raw_t_link = 1e-5 # Time for one link generation attempt
                time_comm = 0 # Time keeping for the communication qubits
                time_mem = 0 # Time keeping for the memory qubits
                total_time = 0 # Total time keeping for the entire protocol

                # Create the pair AB and distill with other AB
                bell_AB = create_bell_link_and_distill_over_two_nodes([0,0,0])
                if bell_AB is None:
                    protocol_failed = True
                    continue
                bell_AB = bell_AB[0] # Extract the distilled density matrix
                time_comm += bell_AB[1][0] # Time for the communication qubits
                time_mem += bell_AB[1][1] # Time for the memory qubits
                total_time += bell_AB[1][2] # Total time for the entire protocol

                # At the same time create the elementary link CD
                bell_CD = create_bell_link_and_distill_over_two_nodes([0,0,0])
                if bell_CD is None:
                    protocol_failed = True
                    continue
                bell_CD = bell_CD[0] # Extract the distilled density matrix
                time_comm += bell_CD[1][0] # Time for the communication qubits
                time_mem += bell_CD[1][1] # Time for the memory qubits
                total_time += bell_CD[1][2] # Total time for the entire protocol

                
                # Then distill AB with another AB
                distilled_bell_AB = distill_with_CiYiY(bell_AB, [time_mem, time_comm, total_time])
                if distilled_bell_AB is None:
                    protocol_failed = True
                    continue
                distilled_bell_AB = distilled_bell_AB[0]

                # In parallel, distill CD with another CD
                distilled_bell_CD = distill_with_CiYiY(bell_CD, [time_mem, time_comm, total_time])
                if distilled_bell_CD is None:
                    protocol_failed = True
                    continue
                distilled_bell_CD = distilled_bell_CD[0]

                time_comm += distilled_bell_AB[1][0]
                time_mem += distilled_bell_AB[1][1]
                total_time += distilled_bell_AB[1][2]

                # SWAP A and C to memory qubits to create link AC

                bell_AC = create_bell_link_and_distill_over_two_nodes([time_mem, time_comm, total_time])
                if bell_AC is None:
                    protocol_failed = True
                    continue
                bell_AC = bell_AC[0]
                time_comm += bell_AC[1][0]
                time_mem += bell_AC[1][1]
                total_time += bell_AC[1][2]

                # Apply depolarizing noise for SWAP on qubits of A and C

                # Apply decoherence due to the SWAP gate

                # Then distill AC with another AC
                distilled_bell_AC = distill_with_CiYiY(bell_AC, [time_mem, time_comm, total_time])
                if distilled_bell_AC is None:
                    protocol_failed = True
                    continue
                distilled_bell_AC = distilled_bell_AC[0]
                time_comm += distilled_bell_AC[1][0]
                time_mem += distilled_bell_AC[1][1]
                total_time += distilled_bell_AC[1][2]

                # Apply decohrence on AB and CD due to new AC generation

                # Fuse at C to create ACD

                # Fuse at A to create ABCD

                # Create BD with iYiY distillation

                # Distill ABCD with BD via ZZ

                rho_fusion_protocol_final += distilled_bell_AB # Add the current density matrix to the final density matrix

                
                current_t_link = total_time
                t_link += current_t_link # Total time for the link generation
                current_f_link = fidelity(distilled_bell_AB, density_matrix_target)
                f_link += current_f_link # Fidelity average
                current_p_link = np.real(1/(attempts_bell_1+attempts_bell_2) * p_distill)
                p_link += current_p_link
                successful_shots += 1 # Increase the number of successful shots
                

            if successful_shots != 0:
                rho_fusion_protocol_final /= successful_shots # Normalize the final density matrix
                self.t_link = t_link/successful_shots
                self.F_link = f_link/successful_shots
                self.p_link = p_link/successful_shots
            if successful_shots == 0:
                self.t_link = np.inf
                self.F_link = 0
                self.p_link = 0
                rho_fusion_protocol_final = sp.lil_matrix((2**weight, 2**weight), dtype=complex)  # Final density matrix for the emitters if the attempts fail!


            print(rf"*** GHZ state fidelity of the GHZ Fusion ($k=11$) protocol state is {self.F_link}.***")
            print(rf"*** Success rate of the GHZ Fusion ($k=11$) protocol state is {self.p_link}.***")

            return rho_fusion_protocol_final
        

        if network_noise_type in range(10, 22):
            data = np.load('circuit_simulation/states/non_emission_based_99_fidelity_Bell_states.npy', allow_pickle=True)
            non_emission_based_i = network_noise_type - 10
            density_matrix_target = sp.lil_matrix((4, 4))
            density_matrix_target[0, 0] = 0.5
            density_matrix_target[0, 3] = 0.5
            density_matrix_target[3, 0] = 0.5
            density_matrix_target[3, 3] = 0.5
            noisy_density_matrix = sp.lil_matrix(data[non_emission_based_i]['super_simulation']['density_matrix'])
            self.p_link = data[non_emission_based_i]['super_simulation']['p_success']
            self.t_link = data[non_emission_based_i]['super_simulation']['eff_time']
            self.F_link = fidelity(noisy_density_matrix, density_matrix_target)
            return noisy_density_matrix
        
        elif network_noise_type == 99: # emission based scheme with gate error included
            mu = bell_pair_parameters['mu']
            F_prep = bell_pair_parameters['F_prep']
            labda = bell_pair_parameters['lambda']
            p_DE = bell_pair_parameters['p_DE']
            eta = bell_pair_parameters['eta']

            if bell_pair_parameters['ent_prot'] == 'single_click':
                alpha = bell_pair_parameters['alpha']
                p_link = 0.5 * alpha * eta * (4 + alpha * eta * (-3 + mu))
                noisy_density_matrix = sp.lil_matrix((4, 4), dtype=complex)
                noisy_density_matrix[0, 0] = (-18 * (-1 + alpha) + pg * (-18 + 8 * pg - 2 * pg * alpha * (8 + eta * (-3 + mu)) + 3 * alpha * (10 + eta * (-3 + mu)))) / (9 * (4 + alpha * eta * (-3 + mu)))
                noisy_density_matrix[0, 1] = 0
                noisy_density_matrix[0, 2] = 0
                noisy_density_matrix[0, 3] = -(2 * (1 - 2 * F_prep)**2 * (-1 + p_DE)**2 * (9 + 2 * pg * (-9 + 4 * pg)) * (-1 + alpha) * (-1 + 2 * labda) * cmath.sqrt(mu)) / (9 * (4 + alpha * eta * (-3 + mu)))

                noisy_density_matrix[1, 0]= 0
                noisy_density_matrix[1, 1] =(-9 * pg * (-2 + alpha * (6 + eta * (-3 + mu))) + 2 * pg**2 * (-4 + alpha * (8 + eta * (-3 + mu))) + 9 * alpha * (4 + eta * (-3 + mu))) / (9 * (4 + alpha * eta * (-3 + mu)))
                noisy_density_matrix[1, 2] = 0
                noisy_density_matrix[1, 3] = 0

                noisy_density_matrix[2, 0] = 0
                noisy_density_matrix[2, 1] = 0
                noisy_density_matrix[2, 2] = (2 * pg * (9 - 9 * alpha + pg * (-4 + alpha * (8 + eta * (-3 + mu))))) / (9 * (4 + alpha * eta * (-3 + mu)))
                noisy_density_matrix[2, 3] = 0

                noisy_density_matrix[3, 0] = -(2 * (1 - 2 * F_prep)**2 * (-1 + p_DE)**2 * (9 + 2 * pg * (-9 + 4 * pg)) * (-1 + alpha) * (-1 + 2 * labda) * cmath.sqrt(mu)) / (9 * (4 + alpha * eta * (-3 + mu)))
                noisy_density_matrix[3, 1] = 0
                noisy_density_matrix[3, 2] = 0
                noisy_density_matrix[3, 3] = -((2 * (9 * (-1 + alpha) + pg * (9 + pg * (-4 + alpha * (8 + eta * (-3 + mu))) - 3 * alpha * (7 + eta * (-3 + mu))))) / (9 * (4 + alpha * eta * (-3 + mu))))

            elif bell_pair_parameters['ent_prot'] == 'double_click':
                alpha = 1/2
                p_link = (1/36) * eta**2 * (18 + pg * (24 + eta * (12 + pg * eta * (-3 + mu)) * (-3 + mu)))
                noisy_density_matrix = sp.lil_matrix((4, 4), dtype=complex)
                labda = 0

                noisy_density_matrix[0, 0] = -((-162 + 27 * pg**2 * (-16 + eta * (-3 + mu)) - 54 * pg * (-3 + eta * (-3 + mu)) - 27 * cmath.sqrt(1 - pg) * pg**(3/2) * cmath.sqrt(-((-1 + pg) * pg**3)) * (8 + eta * (-3 + mu)) + 27 * cmath.sqrt(-((-1 + pg) * pg)) * cmath.sqrt(-((-1 + pg) * pg**3)) * (8 + eta * (-3 + mu)) + pg**4 * (8 + eta * (-3 + mu)) * (-27 + 4 * eta * (-3 + mu)) - 6 * pg**3 * (-88 + eta * (1 + eta * (-3 + mu)) * (-3 + mu)))) / (36 * (9 + 1/2 * pg * (24 + eta * (12 + pg * eta * (-3 + mu))) * (-3 + mu)))
                noisy_density_matrix[0, 1] = 0
                noisy_density_matrix[0, 2] = 0
                noisy_density_matrix[0, 3] = ((1 - 2 * F_prep)**2 * (-1 + p_DE)**4 * (-3 + 2 * pg) * (-3 + 4 * pg)**3 * mu) / (9 * (18 + pg * (24 + eta * (12 + pg * eta * (-3 + mu))) * (-3 + mu)))

                noisy_density_matrix[1, 0] = 0
                noisy_density_matrix[1, 1]  = (54 * cmath.sqrt(1 - pg) * pg**(3/2) * cmath.sqrt(-((-1 + pg) * pg**3)) - 54 * cmath.sqrt(-((-1 + pg) * pg)) * cmath.sqrt(-((-1 + pg) * pg**3)) + 27 * pg * (11 + 2 * eta * (-3 + mu)) + 9 * pg**2 * (-22 + eta * (1 + eta * (-3 + mu)) * (-3 + mu)) - 3 * pg**3 * (20 + 3 * eta * (6 + eta * (-3 + mu)) * (-3 + mu)) + 2 * pg**4 * (27 + eta * (8 + eta * (-3 + mu)) * (-3 + mu))) / (9 * (18 + pg * (24 + eta * (12 + pg * eta * (-3 + mu))) * (-3 + mu)))
                noisy_density_matrix[1, 2]  = 0
                noisy_density_matrix[1, 3]  = 0

                noisy_density_matrix[2, 0]  = 0
                noisy_density_matrix[2, 1]  = 0
                noisy_density_matrix[2, 2]  = (pg * (81 + 6 * pg**2 * (8 - 3 * eta * (-3 + mu)) + 9 * pg * (-4 + 3 * eta * (-3 + mu)) + 2 * pg**3 * eta * (8 + eta * (-3 + mu)) * (-3 + mu))) / (9 * (18 + pg * (24 + eta * (12 + pg * eta * (-3 + mu))) * (-3 + mu)))
                noisy_density_matrix[2, 3]  = 0

                noisy_density_matrix[3, 0]  = ((1 - 2 * F_prep)**2 * (-1 + p_DE)**4 * (-3 + 2 * pg) * (-3 + 4 * pg)**3 * mu) / (9 * (18 + pg * (24 + eta * (12 + pg * eta * (-3 + mu)) * (-3 + mu))))
                noisy_density_matrix[3, 1]  = 0
                noisy_density_matrix[3, 2]  = 0
                noisy_density_matrix[3, 3]  = (162 + 54 * pg * (-3 + eta * (-3 + mu)) + 27 * cmath.sqrt(1 - pg) * pg**(3/2) * cmath.sqrt(-((-1 + pg) * pg**3)) * (8 + eta * (-3 + mu)) - 27 * cmath.sqrt(-((-1 + pg) * pg)) * cmath.sqrt(-((-1 + pg) * pg**3)) * (8 + eta * (-3 + mu)) + 9 * pg**2 * (64 + eta * (-3 + mu)) + 6 * pg**3 * (8 + eta * (-3 + mu)) * (-11 + 2 * eta * (-3 + mu)) -  pg**4 * (8 + eta * (-3 + mu)) * (-27 + 4 * eta * (-3 + mu))) / (36 * (9 + 1/2 * pg * (24 + eta * (12 + pg * eta * (-3 + mu)) * (-3 + mu))))

            else:
                raise ValueError(f"'ent_prot' should be either 'single_click' or 'double_click', and not "
                                 f"{bell_pair_parameters['ent_prot']}.")

            self.F_link = None
            self.p_link = p_link
            self.mu = mu
            self.F_prep = F_prep
            self.labda = labda
            self.p_DE = p_DE
            self.eta = eta
            self.alpha = alpha
            self.ent_prot = bell_pair_parameters['ent_prot']

            return noisy_density_matrix

        elif bell_pair_parameters is None:
            return None

        else:
            mu = bell_pair_parameters['mu']
            F_prep = bell_pair_parameters['F_prep']
            labda = bell_pair_parameters['lambda']
            p_DE = bell_pair_parameters['p_DE']
            eta = bell_pair_parameters['eta']

            phi = math.sqrt(mu) * ((2 * F_prep - 1) ** 2) * (2 * labda - 1) * ((1 - p_DE) ** 2)

            if bell_pair_parameters['ent_prot'] == 'single_click':
                alpha = bell_pair_parameters['alpha']
                p_link = (2 * eta * (1 - eta) + (1 + mu)/2 * eta**2) * alpha**2 + 2 * eta * alpha * (1 - alpha)
                coeff_psi_p = 1 / p_link * (1 + phi) * eta * alpha * (1 - alpha)
                coeff_psi_m = 1 / p_link * (1 - phi) * eta * alpha * (1 - alpha)
                coeff_00 = 1 / p_link * (2 * eta * (1 - eta) + (1 + mu)/2 * eta ** 2) * alpha ** 2
            elif bell_pair_parameters['ent_prot'] == 'double_click':
                alpha = 1/2
                p_link = 2 * alpha * (1 - alpha) * eta ** 2
                coeff_psi_p = 1 / p_link * (1 + phi ** 2) * alpha * (1 - alpha) * eta ** 2
                coeff_psi_m = 1 / p_link * (1 - phi ** 2) * alpha * (1 - alpha) * eta ** 2
                coeff_00 = 0
            else:
                raise ValueError(f"'ent_prot' should be either 'single_click' or 'double_click', and not "
                                 f"{bell_pair_parameters['ent_prot']}.")

            noisy_density_matrix = sp.lil_matrix((4, 4), dtype=complex)
            noisy_density_matrix[0, 0] = coeff_00
            noisy_density_matrix[1, 1] = (coeff_psi_p + coeff_psi_m) / 2
            noisy_density_matrix[2, 2] = (coeff_psi_p + coeff_psi_m) / 2
            noisy_density_matrix[1, 2] = (coeff_psi_p - coeff_psi_m) / 2
            noisy_density_matrix[2, 1] = (coeff_psi_p - coeff_psi_m) / 2

            self.F_link = None
            self.p_link = p_link
            self.mu = mu
            self.F_prep = F_prep
            self.labda = labda
            self.p_DE = p_DE
            self.eta = eta
            self.alpha = alpha
            self.ent_prot = bell_pair_parameters['ent_prot']

            return noisy_density_matrix

    def _init_density_matrix(self):
        """ Realises init_type option 0. See class description for more info. """
        init_type = self._init_type
        if init_type == 0:
            self._quantum_circuit_init.quantum_circuit_init.init_density_matrix(self)
        elif init_type == 1:
            self._init_density_matrix_first_qubit_ket_p()
        elif init_type == 2:
            self._init_density_matrix_maximally_entangled_state()
        elif init_type > 3:
            self._init_density_matrix_maximally_entangled_state(amount_qubits=init_type)

    def _init_density_matrix_first_qubit_ket_p(self):
        """ Realises init_type option 1. See class description for more info. """

        return self._quantum_circuit_init.quantum_circuit_init.init_density_matrix_first_qubit_ket_p(self)

    def _init_density_matrix_maximally_entangled_state(self, amount_qubits=8, draw=True):
        """ Realises init_type option 2. See class description for more info. """

        return self._quantum_circuit_init.quantum_circuit_init \
            .init_density_matrix_maximally_entangled_state(self, amount_qubits, draw)

    def _init_density_matrix_ket_p_and_CNOTS(self):
        """ Realises init_type option 3. See class description for more info. """

        return self._quantum_circuit_init.quantum_circuit_init.init_density_matrix_ket_p_and_CNOTS(self)

    def _init_parameters_to_dict(self):
        return self._quantum_circuit_init.quantum_circuit_init.init_parameters_to_dict(self)
    """
        ---------------------------------------------------------------------------------------------------------
                                                Separated Density Matrices Methods
        ---------------------------------------------------------------------------------------------------------
    """
    def _correct_lookup_for_addition(self, new_density_matrix, amount_qubits=1, position='top'):
        """
            Method corrects the qubit_density_matrix_lookup dictionary for the addition of a top or bottom qubit.

            Parameters
            ----------
            amount_qubits : int
                Amount of qubits that is added to the top (or bottom) of the system.
            position : str['top', 'bottom'], optional, default='top'
                String value that indicates if the qubit is added to the top or the bottom of the system
        """
        if position.lower() == 'top':
            position = 0
        elif position.lower() == 'bottom':
            position = -1
        else:
            raise ValueError("position argument can only be 'top' or 'bottom'.")

        new_lookup_dict = {}
        for qubit, (density_matrix, qubits) in sorted(self._qubit_density_matrix_lookup.items()):
            new_lookup_dict[qubit+amount_qubits] = (density_matrix, [q + amount_qubits for q in qubits])
        self._qubit_density_matrix_lookup = new_lookup_dict

        qubit_indices = [i for i in range(amount_qubits)]
        for qubit_num in range(amount_qubits):
            self._qubit_density_matrix_lookup[qubit_num] = (new_density_matrix, qubit_indices)

    def _correct_lookup_for_two_qubit_gate(self, cqubit, tqubit):
        """
            Method corrects the qubit_density_matrix_lookup dictionary when a two-qubit gate is applied.
            Due to two-qubit gates, the density matrices of the involved qubits should be fused (if not already).

            Parameters
            ----------
            cqubit : int
                Qubit number of the control qubit
            tqubit : int
                Qubit number of the control qubit
        """
        cqubit_density_matrix, c_qubits = self._qubit_density_matrix_lookup[cqubit]
        tqubit_density_matrix, t_qubits = self._qubit_density_matrix_lookup[tqubit]
        fused_density_matrix = KP(cqubit_density_matrix, tqubit_density_matrix)
        fused_qubits = c_qubits + t_qubits

        for qubit in fused_qubits:
            self._qubit_density_matrix_lookup[qubit] = (fused_density_matrix, fused_qubits)

    def _get_qubit_relative_objects(self, qubit):
        """
            Method returns for the given qubit the following relative objects:
             - relative density matrix,
             - qubits order that is present in the density matrix,
             - the qubit index for the density matrix
             - the amount of qubits that is present in the density matrix

            Parameters
            ----------
            qubit : int
                Qubit number of the qubit that the relative objects are requested for
        """
        density_matrix, qubits = self._qubit_density_matrix_lookup[qubit]
        relative_qubit_index = qubits.index(qubit)
        relative_num_qubits = len(qubits)

        return density_matrix, qubits, relative_qubit_index, relative_num_qubits

    def _correct_lookup_for_measurement_top(self):
        """
            Method corrects the qubit_density_matrix_lookup dictionary for the (destructive) measurement of the top
            qubit

            **NOTE: Qubits involved in the same density matrix should all point to the same density matrix object
            in memory and the same involved qubits list object in memory. This is why the qubits list is adapted in the
            qubits[:] way, this ensures that the same memory address is used.**
        """
        new_lookup_dict = {}
        _, qubits_old = self._qubit_density_matrix_lookup[0]
        del qubits_old[-1]
        for qubit, (density_matrix, qubits) in sorted(self._qubit_density_matrix_lookup.items()):
            if qubit == 0:
                qubits_old = qubits
                continue
            if qubits_old is qubits:
                qubits = qubits_old
            else:
                qubits[:] = [i - 1 for i in qubits]
                qubits_old = qubits

            new_lookup_dict[qubit - 1] = density_matrix, qubits

        self._qubit_density_matrix_lookup = new_lookup_dict

    def _correct_lookup_for_measurement_any(self, qubit, qubits, density_matrix_measured, new_density_matrix):
        """
            Corrects the lookup table, where for each qubit the corresponding density matrix can be found,
            for the measurement of a qubit. In case of a measurement, the qubit that is measured will separate from
            the density matrix it was involved in and will get the new density matrix that corresponds to the state
            that has been measured on the qubit.

            Parameters
            ----------
            qubit : int
                The qubit index of the qubit that has been measured
            qubits : list
                List of qubit indices of the qubits, including the measured qubit, that span the density matrix
                before the measurement.
            density_matrix_measured : sp.csr_matrix
                Density of the new state of the measured qubit
            new_density_matrix : sp.csr_matrix
                Density matrix of the resulting system after the measurement (system without the measured qubit)
        """
        self._qubit_density_matrix_lookup[qubit] = (density_matrix_measured, [qubit])
        qubits.remove(qubit)
        for q in qubits:
            self._qubit_density_matrix_lookup[q] = (new_density_matrix, qubits)

    def _correct_lookup_for_circuit_fusion(self, lookup_other):
        """
            Correct the qubit density matrix look-up table for the fusion of two QuantumCircuit objects

            Parameters
            ----------
            lookup_other : dict
                Lookup table of the other QuantumCircuit object that is fused with the current QuantumCircuit object.
        """
        num_qubits_other = len(lookup_other)
        new_lookup = lookup_other
        prev_qubits = None
        for qubit, (density_matrix, qubits) in sorted(self._qubit_density_matrix_lookup.items()):
            if prev_qubits is not qubits:
                qubits[:] = [i + num_qubits_other for i in qubits]
                prev_qubits = qubits
            new_lookup[qubit + num_qubits_other] = (density_matrix, qubits)
        self._qubit_density_matrix_lookup = new_lookup

    def _set_density_matrix(self, qubit, new_density_matrix):
        """
            Method sets the density matrix for the given qubit and all qubits that are involved in the same density
            matrix

            *** NOTE: density matrices have to be set with this method in order to guarantee proper functioning of the
            program. It ensures that qubits involved in the same density matrix will point to the same density matrix
            object in memory (such that when the matrix changes, it changes for each involved qubit) ***

            Parameters
            ----------
            qubit : int
                Qubit number for which the density matrix should be set
            new_density_matrix : csr_matrix
                The new density matrix that should be set
        """
        _, qubits, _, _ = self._get_qubit_relative_objects(qubit)
        for qubit in qubits:
            self._qubit_density_matrix_lookup[qubit] = (new_density_matrix, qubits)

    def _lookup_sanity_check(self):
        for qubit in range(self.num_qubits):
            dm, qubits, _, _ = self._get_qubit_relative_objects(qubit)
            sanity_dm = all([dm is self._qubit_density_matrix_lookup[qubit_2][0] for qubit_2 in qubits])
            sanity_qubits = all([qubits is self._qubit_density_matrix_lookup[qubit_2][1] for qubit_2 in qubits])

            if not sanity_dm or not sanity_qubits:
                raise ValueError("Density matrix is not sane. Memory addresses differ")

        return True

    @determine_qubit_index(parameter_positions=[1])
    def _reset_density_matrices(self, qubits, state=None):
        """
            Method resets the density matrices of the given qubits. If the qubit is in a density matrix with a qubit
            not given, it will also reset the density matrix of this qubit

            Parameters
            ----------
            qubits : list
                List of qubits of which the density matrix should be reset.
            state : State
                Single qubit state with which the qubits should be reset. If None, |0> is used.
        """
        skip_qubits = []
        if not isinstance(qubits, list):
            qubits = [qubits]
        for qubit in qubits:
            # Skip qubits that have already been reset
            if qubit not in skip_qubits:
                state = state if state else ket_0
                _, matrix_qubits, _, _ = self._get_qubit_relative_objects(qubit)
                # Loop over the qubits that are in the same density matrix as the qubit
                for matrix_qubit in matrix_qubits:
                    if matrix_qubit not in skip_qubits:
                        self._qubit_density_matrix_lookup[matrix_qubit] = (CT(state), [matrix_qubit])
                        # Add the qubit to the skip list
                        skip_qubits.append(matrix_qubit)
        self._update_uninitialised_qubit_register(qubits, "add")

    def get_combined_density_matrix(self, qubits):
        """
            Returns the combined density matrix of the qubits requested and returns a list of the qubits that span
            this combined density matrix. The list of qubits is given in the exact order of how the qubits are
            situated in the density matrix.

            Parameters
            ----------
            qubits : list
                List of qubits of which the combined density matrix is requested

            Returns
            -------
            combined_density_matrix : sp.csr_matrix
                Combined density matrix of the qubits requested
            spanning_qubits : list
                List of qubits spanning the density matrix. The qubits are in the exact order of appareance in the
                density matrix
        """
        density_matrices = []
        skip_qubits = []
        for qubit in qubits:
            if qubit not in skip_qubits:
                density_matrix, involved_qubits, _, _ = self._get_qubit_relative_objects(qubit)
                density_matrices.append(density_matrix)
                skip_qubits.extend(involved_qubits)
        return KP(*density_matrices), skip_qubits

    def total_density_matrix(self):
        """
            Get the total density matrix of the system and the order of the qubits that span it.
        """
        density_matrices = []
        skip_qubits = []
        for qubit, (density_matrix, qubits) in sorted(self._qubit_density_matrix_lookup.items()):
            if qubit not in skip_qubits:
                density_matrices.append(density_matrix)
                skip_qubits.extend(qubits)
        return KP(*density_matrices), skip_qubits

    def export_density_matrix(self, location, density_matrix=None):
        if density_matrix is None:
            density_matrix = self.total_density_matrix()

        with open(location, 'wb') as file_location:
            pickle.dump(density_matrix, file_location)

    """
        ---------------------------------------------------------------------------------------------------------
                                                SubQuantumCircuit Methods
        ---------------------------------------------------------------------------------------------------------
    """
    def define_sub_circuit(self, name, qubits=None, waiting_qubits=None, concurrent_sub_circuits=None,
                           involved_nodes=None):
        """
            Define a sub circuit for the QuantumCircuit object. Sub circuits can be used to emulate concurrent
            circuits. This can be useful when working with decoherence or to obtain the concurrent circuit drawing
            for example. Note that circuits will not actually run in parallel when simulated, this remains in serial on
            the back-end.

            Parameters
            ----------
            name : str
                Unique name for the sub circuit in order to separate it from the others
            qubits : list
                List of qubit indices that are involved in the sub circuit
            waiting_qubits : list, optional, default=None
                List of qubit indices that are waiting whenever the other concurrent sub circuit takes longer to
                calculate (useful when working with decoherence)
            concurrent_sub_circuits : list, optional, default=None
                List containing the concurrent SubQuantumCircuit objects. Please only specify this parameter for the
                last concurrent sub circuit object created, since otherwise the others cannot be found.
            involved_nodes : list
                list of str containing the names of the nodes that are involved in the sub-circuit. If not provided or None,
                this is deduced from the name of the sub_circuit (example: sub circuit name "AB" will translate to
                involved nodes "A" and "B".)
        """
        concurrent_sub_circuit_objects = []
        if waiting_qubits is None:
            waiting_qubits = qubits
        if concurrent_sub_circuits is not None:
            if type(concurrent_sub_circuits) in [str, int]:
                concurrent_sub_circuits = [concurrent_sub_circuits]
            concurrent_sub_circuit_objects = [self._sub_circuits[sub_name] for sub_name in concurrent_sub_circuits]
        if involved_nodes is None:
            involved_nodes = list(name)
        if not all(node_name in self.nodes for node_name in involved_nodes):
            raise ValueError("involved_nodes either contains nodes that do not exist or it could not be derived from "
                             "the name of the sub circuit. involved_nodes list for sub circuit '{}' contained: {}"
                             .format(name, involved_nodes))
        if qubits is None:
            qubits = []
            for node in involved_nodes:
                copy_qubits = copy(self.nodes[node].qubits)
                qubits.extend(copy_qubits)

        sub_circuit = SubQuantumCircuit(name, qubits, waiting_qubits, concurrent_sub_circuit_objects, involved_nodes)

        if concurrent_sub_circuit_objects is not None:
            for sub_circuit_object in concurrent_sub_circuit_objects:
                copy_csco = copy(concurrent_sub_circuit_objects)
                copy_csco.remove(sub_circuit_object)
                decreased_concurrent_objects = [sub_circuit] + copy_csco

                sub_circuit_object.add_concurrent_sub_circuits(decreased_concurrent_objects)

        self._sub_circuits[name] = sub_circuit

    def start_sub_circuit(self, name, forced_level=False):
        """
            Sets the provided sub circuit (here referred to as: 'started sub circuit') as current sub circuit and will
            mark the previous sub circuit (here referred to as: 'current sub circuit') as 'ran' if present. Method will
            first add the maximum duration of the concurrent sub circuits, of which the 'current sub circuit' is part,
            to the total duration of the QuantumCircuit object if the 'started sub circuit' is not part these concurrent
            sub circuits.

            Parameters
            ----------
            name : str
                Name of the sub circuit that should be marked as current sub circuit.
            forced_level: bool
                Force the method to level the drawing and duration of the total circuit. This means that the drawing
                each qubit path will be leveled and the maximum duration of the sub circuits will be added to the
                total duration of the circuit. Usually this will only happen when all concurrent sub circuits have
                been evaluated by the circuit simulator.
        """
        if name not in self._sub_circuits.keys():
            raise ValueError('Provided sub circuit name is not an existing sub circuit.')

        # Add the maximum duration of the concurrent sub circuits to the total duration of the QuantumCircuit object
        started_sub_circuit = self._sub_circuits[name]
        started_sub_circuit.set_ran(False)
        self.end_current_sub_circuit(forced_level=forced_level)

        started_sub_circuit.reset()
        self._current_sub_circuit = started_sub_circuit if not self.cut_off_time_reached else None

    def end_current_sub_circuit(self, total=False, duration=None, sub_circuit=None, forced_level=False,
                                apply_decoherence=False, debug_print=False):
        """
            Method can be used to mark the current sub circuit as 'ran'. This method is only needed when no new sub
            circuit is started.

            Parameters
            ----------
            total : bool
                If set to True, the operations to the main circuit are marked as finished. This is necessary when
                working with the cut-off time. This is thus ONLY set to True at the very end of the operations that are
                applied to the main circuit. The boolean '_circuit_operations_ended' is used in order to prevent
                methods from being skipped when not used specifically as an operation to the main circuit.
        """
        # Add duration of the current sub circuit to the total duration if sub circuit present
        current_sub_circuit = self._current_sub_circuit if sub_circuit is None else None
        if current_sub_circuit is not None and not current_sub_circuit.ran:
            self._update_sub_circuit_duration_with_node_duration()
            current_sub_circuit.set_ran()

            if current_sub_circuit.all_ran or forced_level:
                if debug_print:
                    print("\n\n\n\n\n")
                    print('End sub circuit:')
                    print(f"Total duration so far: {self.total_duration}.")
                    total_duration_list = {}
                    for sc in current_sub_circuit.concurrent_sub_circuits + [current_sub_circuit]:
                        total_duration_list[sc._name] = sc.total_duration
                    print(total_duration_list)
                added_dur = max([sc.total_duration for sc in current_sub_circuit.concurrent_sub_circuits
                                 + [current_sub_circuit]])
                self._draw_order.append(["LEVEL", added_dur, current_sub_circuit])
                self.total_duration += added_dur
                if round(self.total_duration - self.cut_off_time, SUM_ACCURACY) > 0:
                # if self.total_duration > self.cut_off_time:
                    self.cut_off_time_reached = True
                    if debug_print:
                        print(f"Total cut-off time exceeded: {self.total_duration - self.cut_off_time}.")
                if debug_print:
                    print("\n\n\n\n\n")
                self._apply_waiting_time_to_fastest_sub_circuits()
                self._draw_order.append(["LEVEL", None, None])
                # Reset all the sub_circuits when all ran or when a forced level is requested
                [sub_circuit.reset() for sub_circuit in current_sub_circuit.concurrent_sub_circuits
                 + [current_sub_circuit]]

                self._current_sub_circuit = None

        # First apply left over waiting time of all qubits in the form of decoherence
        if apply_decoherence and self.noise:
            self._N_decoherence(decoherence=self.decoherence)

        if duration is not None and sub_circuit is not None:
            self._draw_order.append(["LEVEL", duration, sub_circuit]) if sub_circuit is not None else None

        # Used in case cut-off can be reached, this frees the methods that otherwise will be skipped due to cut-off
        # reached.
        if total:
            self._circuit_operations_ended = True

    def define_node(self, name, qubits, electron_qubits=None, data_qubits=None, amount_data_qubits=1):
        """
            Defines a node for the QuantumCircuit object. This is especially useful when working with a networked
            architecture. For now it is assumed that one uses an NV-center as a node.

            Parameters
            ----------
            name : str
                Unique name for the defined node
            qubits : list
                List of qubit indices that are part of the node
            electron_qubits : list or int
                Sub list of qubits that should be marked as the electron qubits
            data_qubits : list
                List of the data qubits in the node. If not defined, the first 'amount_data_qubits' qubits in the
                array are marked as the data qubits.
            amount_data_qubits : int
                The amount of data qubits present in the node.
        """
        qubits = sorted(qubits, reverse=True)
        if self.nodes is None:
            self.nodes = {}
        if self.qubits is None:
            self.qubits = {}

        if data_qubits is None:
            data_qubits = [qubit for qubit in qubits[:amount_data_qubits]]
        elif type(data_qubits) == int:
            data_qubits = [data_qubits]

        if electron_qubits is None:
            electron_qubits = [qubits[-1]]
        elif type(electron_qubits) == int:
            electron_qubits = [electron_qubits]

        node = Node(name, qubits, self, electron_qubits, data_qubits)
        self.nodes.update({name: node})
        for qubit in qubits:
            qubit_type = 'e' if qubit in electron_qubits else 'n'
            is_data_qubit = qubit in data_qubits
            T1_idle = self.T1n_idle if qubit_type == 'n' else self.T1e_idle
            T2_idle = self.T2n_idle if qubit_type == 'n' else self.T2e_idle
            T1_link = self.T1n_link if qubit_type == 'n' else None
            T2_link = self.T2n_link if qubit_type == 'n' else None
            q = Qubit(self, qubit, qubit_type, node=name, T1_idle=T1_idle, T2_idle=T2_idle, T1_link=T1_link,
                      T2_link=T2_link, is_data_qubit=is_data_qubit)
            self.qubits[qubit] = q

    def get_node_qubits(self, qubits):
        """
            Returns the qubits of a node of which the supplied qubit is part of.

            Parameters
            ----------
            qubits : list
                Qubit index of the qubit of which the node qubits should be returned
        """
        if self.nodes is None:
            return []

        if type(qubits) == int:
            qubits = [qubits]

        node_qubits = []
        for node in self.nodes.values():
            if any(node.qubit_in_node(qubit) for qubit in qubits):
                node_qubits.extend(node.qubits)
        return node_qubits

    @property
    def data_qubits(self):
        if self.qubits is None:
            return []
        return [qubit.index for qubit in self.qubits.values() if qubit.is_data_qubit]

    def get_node_name_from_qubit(self, qubit):
        """
            Returns the name of the node that the supplied qubit is part of.

            Parameters
            ----------
            qubit : int
                Qubit index of the qubit of which the name of the node should be returned
        """
        if self.nodes is None:
            return
        if self.qubits is not None:
            if qubit in self.qubits:
                return self.qubits[qubit].node

    def _apply_waiting_time_to_fastest_sub_circuits(self):
        """
            Applies waiting time to the qubits that have been waiting for a slowest concurrent sub circuit to finish.
            Also adds waiting time to the qubits that are not part of the current concurrent sub circuits,
            but are initialised and therefore waiting as well.
        """
        if not self.decoherence:
            return

        all_sub_circuits = self._current_sub_circuit.concurrent_sub_circuits + [self._current_sub_circuit]
        longest_duration = max(sc.total_duration for sc in all_sub_circuits)

        for sub_circuit in all_sub_circuits:
            if (longest_duration - sub_circuit.total_duration) > 0:
                if sub_circuit.waiting_qubits is not None:
                    waiting_qubits = sub_circuit.waiting_qubits
                else:
                    waiting_qubits = [qubit for qubit in sub_circuit.qubits if qubit not in self._uninitialised_qubits]

                self._increase_qubit_duration(longest_duration - sub_circuit.total_duration,
                                              included_qubits=waiting_qubits)

                self._check_if_cut_off_time_is_reached()

        # Apply waiting time to all node qubits that are not part of the current concurrent sub circuits and initialised
        left_over_qubits = list(set([qubit for node in self.nodes.values() for qubit in node.qubits])
                                .difference(self._current_sub_circuit.get_all_concurrent_qubits +
                                            self._uninitialised_qubits))
        if not left_over_qubits:
            return
        self._increase_qubit_duration(longest_duration, included_qubits=left_over_qubits)

    def correct_for_failed_ghz_check(self, success_dict):
        """
            Method is used in the Expedient and Stringent protocols. When the GHZ check step fails (step 8 in table
            D.1 and step 14 in table D.2 of thesis Naomi Nickerson), the time of the shortest sub_circuit that failed
            should be used to add to the total duration of the circuit and to the waiting qubits as decoherence. This
            method ensures this. THIS METHOD CAN BE USED WHEN THE SLOWER SUB CIRCUIT IS CUT-OFF WHEN THE PARITY CHECK
            FAILS. AT THIS POINT IN TIME IT IS ASSUMED THAT THE SUB CIRCUITS WAIT FOR THE OTHER TO REACH THE PARITY
            CHECK AND SIMULTANEOUSLY PERFORM THE PARITY CHECK.

            The waiting qubit of the sub circuits that took longer than the circuit that failed the first should be
            reset to the time of the waiting qubits of this first failed sub circuit. This simplification can be
            justified, knowing that the decoherence that is a result of previous sub circuits has already been added
            to the qubits.
        """
        # Find shortest sub circuit that failed, from this point the circuit will start over, so any longer duration
        # should be forgotten
        shortest_duration, shortest_failed_sc = min([(self._sub_circuits[sc_name].total_duration,
                                                      self._sub_circuits[sc_name]) for sc_name, success
                                                     in success_dict.items() if not success])
        data_qubit_shortest = [self.qubits[qubit_index] for qubit_index in self.get_node_qubits(
            shortest_failed_sc.qubits[0]) if self.qubits[qubit_index].is_data_qubit][0]

        all_sub_circuits = self._current_sub_circuit.concurrent_sub_circuits + [self._current_sub_circuit]

        for sub_circuit in all_sub_circuits:
            if (sub_circuit.total_duration - shortest_duration) > 0:
                sub_circuit._total_duration = shortest_duration
                # Correct waiting time on data qubits of longer circuit (other qubits will be reinitialised at this
                # point in time and and therefore be neglected)
                for qubit_index in sub_circuit.waiting_qubits:
                    qubit = self.qubits[qubit_index]
                    if qubit.is_data_qubit:
                        # Usage of waiting time of data qubit of shortest failed sub-circuit for sub circuits that
                        # took longer is justified, since previous decoherence on the data qubits is applied after
                        # each end of concurrent sub-circuits. So the decoherence times used here are only of this
                        # build up during the current sub-circuit
                        qubit._waiting_time_idle = data_qubit_shortest.waiting_time_idle
                        qubit._waiting_time_link = data_qubit_shortest.waiting_time_link

    def _increase_duration(self, amount, excluded_qubits, included_qubits=None, kind='idle', involved_nodes=None,
                           check=True, print_time_progression=False):
        """
            Increases the total duration of the QuantumCircuit if no current sub circuit is present, else it updates
            the total duration of the current sub circuit. If qubits are specified, their idle times (idle or link) are
            updated depending on the value set for 'kind' parameter.

            Parameters
            ----------
            amount : float
                Amount of time with which the duration should be increased
            excluded_qubits : list
                List of qubit indices that are excluded from idle time addition (usually the qubits involved in the
                operation)
            included_qubits : list
                List of qubit indices of which the idle time should be increased. If not specified, the program will
                determine this dynamically (preferred).
            kind : str
                Type of waiting time that should be added to the qubits (choose from: 'idle' or 'link')
        """
        if amount == 0:
            return

        if self._current_sub_circuit is None:
            self.total_duration += amount
            if print_time_progression:
                print(f"[INCREASE DURATION] Amount: {amount}.")

        else:
            if involved_nodes is None:
                involved_nodes = list(set([self.get_node_name_from_qubit(qubit) for qubit in excluded_qubits]))

            for node in involved_nodes:
                self.nodes[node].increase_sub_circuit_time(amount)
            if print_time_progression:
                print(f"[INCREASE DURATION] Involved nodes: {involved_nodes}, node times: {[self.nodes[node].sub_circuit_time for node in 'ABC']}.")

        if self.qubits is not None:
            self._increase_qubit_duration(amount, excluded_qubits, included_qubits, kind, involved_nodes)

        if check:
            self._check_if_cut_off_time_is_reached()

    def _check_if_cut_off_time_is_reached(self):
        """
            Checks whether the cut-off time for the circuit duration is reached. When not all concurrent sub circuits
            are finished, but a sub circuit already reaches the cut-off time then only for this sub circuit it is
            marked that the cut-off time is reached.
        """
        if self._current_sub_circuit:
            longest_duration = min([self.nodes[node].sub_circuit_time
                                    for node in self._current_sub_circuit.involved_nodes])
        else:
            longest_duration = 0

        if round(self.total_duration + longest_duration - self.cut_off_time, SUM_ACCURACY) > 0:
        # if self.total_duration + longest_duration > self.cut_off_time:
            if self._current_sub_circuit is not None:
                if self._current_sub_circuit.all_ran:
                    self.cut_off_time_reached = True
                else:
                    self._current_sub_circuit.set_cut_off_time_reached()
            if round(self.total_duration - self.cut_off_time, SUM_ACCURACY) > 0:
            # if self.total_duration > self.cut_off_time:
                self.cut_off_time_reached = True

    def _increase_qubit_duration(self, amount, excluded_qubits=None, included_qubits=None, kind='idle',
                                 involved_nodes=None):
        """
            Increase the idle time of the given qubit objects. This is used to determine the amount of decoherence that
            a qubit is supposed to experience.

            Parameters
            ----------
            amount : float
                Amount of idle time with which the given qubit objects should be increased
            excluded_qubits : list
                List of qubit indices of which the idle time should NOT be increased
            included_qubits : list
                List of qubit indices of which the idle time should be increased. If not specified, the program will
                determine this dynamically (preferred).
            kind : str
                String indicating the kind of waiting time that is supposed to be added (options are 'idle' or 'link')
        """
        excluded_qubits_copy = copy(excluded_qubits) if excluded_qubits is not None else []
        if not included_qubits:
            if self._current_sub_circuit is not None:
                # If excluded qubits are in the same node, it's a local operation. Decoherence only on local qubits
                if involved_nodes:
                    involved_qubits = [qubit for node in involved_nodes for qubit in self.nodes[node].qubits]
                else:
                    involved_qubits = self._current_sub_circuit.qubits
            else:
                involved_qubits = [i for i in self.qubits.keys()]

            excluded_qubits_copy.extend(self._uninitialised_qubits)
            # apply waiting time to the qubits not taking part in the operation.
            included_qubits = sorted(list(set(involved_qubits).difference(excluded_qubits_copy)))

        for qubit in included_qubits:
            current_qubit = self.qubits[qubit]
            current_qubit.increase_waiting_time(amount, waiting_type=kind)

    def _update_sub_circuit_duration_with_node_duration(self):
        max_duration = max(self.nodes[node].sub_circuit_time for node in self._current_sub_circuit.involved_nodes)

        # Apply decoherence to the initialised qubits of a node that has a shorter duration than the max_duration
        for node in self._current_sub_circuit.involved_nodes:
            current_node = self.nodes[node]
            if current_node.sub_circuit_time < max_duration:
                amount = max_duration - current_node.sub_circuit_time
                initialised_node_qubits = [qubit for qubit in current_node.qubits
                                           if qubit not in self._uninitialised_qubits]

                self._increase_duration(amount, [], included_qubits=initialised_node_qubits, involved_nodes=[node])

        self._current_sub_circuit.increase_duration(max_duration)

        # reset the sub_circuit_time for the nodes, since the sub circuit time is updated
        [self.nodes[node].reset_sub_circuit_time() for node in self._current_sub_circuit.involved_nodes]

    def _update_uninitialised_qubit_register(self, qubits, update_type):
        """
            Updates the qubit uninitialised qubit register. This register is used in the dynamic process of which
            qubits should obtain decoherence.

            Parameters
            ----------
            qubits : list
                List of qubit indices that should be removed/added/swapped
            update_type : str
                Type of update action (options: 'remove', 'add' or 'swap')
        """
        if update_type.lower() not in ["remove", "add", 'swap']:
            raise ValueError("Type can only be 'remove', 'add' or 'swap'.")

        if update_type.lower() == 'remove':
            self._uninitialised_qubits = [qubit for qubit in self._uninitialised_qubits if qubit not in qubits]
            # When a qubit is initialised, the pulse sequence should be reset
            [self.qubits[qubit].reset_sequence_time() for qubit in qubits if self.qubits is not None]
        if update_type.lower() == 'add':
            self._uninitialised_qubits.extend(qubits)
            self._uninitialised_qubits = list(set(self._uninitialised_qubits))
        if update_type.lower() == 'swap':
            # Get the initialisation state of the qubits that are swapped
            qubit_1_state = qubits[0] in self._uninitialised_qubits
            qubit_2_state = qubits[1] in self._uninitialised_qubits

            # If the initialisation state of the qubits that are swapped is different, then this should be swapped too
            if qubit_1_state != qubit_2_state:
                uninitialised_qubit = [qubit_1_state, qubit_2_state].index(True)
                index = self._uninitialised_qubits.index(qubits[uninitialised_qubit])
                self._uninitialised_qubits[index] = qubits[uninitialised_qubit ^ 1]
    """ 
        ---------------------------------------------------------------------------------------------------------
                                                Setter and getter Methods
        ---------------------------------------------------------------------------------------------------------
    """
    @handle_none_parameters
    def set_qubit_states(self, qubit_dict, p_prep=0, noise=None, user_operation=True):
        """
        qc.set_qubit_states(dict)

            Sets the initial state of the specified qubits in the dict according to the specified state.

            *** METHOD SHOULD ONLY BE USED IN THE INITIALISATION PHASE OF THE CIRCUIT. SHOULD NOT BE USED
            AFTER OPERATIONS HAVE BEEN APPLIED TO THE CIRCUIT IN ORDER TO PREVENT ERRORS. ***

            Parameters
            ----------
            qubit_dict : dict
                Dictionary with the keys being the number of the qubits to be modified (first qubit is 0)
                and the value being the state the qubit should be in
            user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.

            Example
            -------
            qc.set_qubit_state({0 : ket_1}) --> This sets the first qubit to the ket_1 state
        """
        if user_operation:
            self._user_operation_order.append({"set_qubit_states": [qubit_dict]})

        for tqubit, state in qubit_dict.items():
            _, _, _, rel_num_qubits = self._get_qubit_relative_objects(tqubit)
            if rel_num_qubits > 1 or tqubit >= self.num_qubits:
                raise ValueError("Qubit is not suitable to set state for.")

            if noise and p_prep > 0:
                state = self._N_preparation(state, p_prep)

            self._qubit_array[tqubit] = state
            self._qubit_density_matrix_lookup[tqubit] = (CT(state), [tqubit])
            self._update_uninitialised_qubit_register([tqubit], "remove")

    def get_begin_states(self):
        """ Returns the initial state vector of the qubits """
        return KP(*self._qubit_array)

    def create_bell_pairs_circuit(self, qubits, user_operation=True):
        """
        qc.create_bell_pair(qubits)

            Creates Bell pairs between the specified qubits.

            *** THIS WILL ONLY WORK PROPERLY WHEN THE SPECIFIED QUBITS ARE IN NO WAY ENTANGLED AND THE
            STATE OF THE QUBITS IS |0> ***

            Parameters
            ----------
            qubits : list
                List containing tuples with the pairs of qubits that should form a Bell pair
            user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.

            Example
            -------
            qc.create_bell_pairs([(0, 1), (2, 3), (4,5)]) --> Creates Bell pairs between qubit 0 and 1,
            between qubit 2 and 3 and between qubit 4 and 5.
        """
        if user_operation:
            self._user_operation_order.append({"create_bell_pairs_circuit": [qubits]})

        for qubit1, qubit2 in qubits:
            self.H(qubit1, noise=False, draw=False, user_operation=False)
            self.CNOT(qubit1, qubit2, noise=False, draw=False, user_operation=False)
            self._add_draw_operation("#", (qubit1, qubit2))

    @handle_none_parameters
    def create_bell_pairs_top(self, N, new_qubit=False, noise=None, F_link=None, network_noise_type=None,
                              probabilistic=None, p_link=None, t_link=None, user_operation=True):
        """
        qc.create_bell_pair(N, F_link=0.1)

            This appends noisy Bell pairs on the top of the system. The noise is based on network noise
            modeled as (paper: https://www.nature.com/articles/ncomms2773.pdf)

                rho_raw = (1 - 4/3*F_link) |psi><psi| + F_link/3 * I,

            in which |psi> is a perfect Bell state.

            *** THIS METHOD APPENDS THE QUBITS TO THE TOP OF THE SYSTEM. THIS MEANS THAT THE AMOUNT OF
            QUBITS IN THE SYSTEM WILL GROW WITH '2N' AND THE INDICES OF THE EXISTING QUBITS INCREASE WITH 2N AS WELL,
            WHICH IS IMPORTANT FOR FUTURE OPERATIONS ***

            Parameters
            ----------
            N : int
                Number of noisy Bell pairs that should be added to the top of the system.
            new_qubit: bool, optional, default=False
                If the creation of the Bell pair adds a new qubit to the drawing scheme (True) or reuses the top qubit
                (False) (this can be done in case the top qubit has been measured)
            noise : bool, optional, default=None
                Can be specified to force the creation of the Bell pairs noisy (True) or noiseless (False).
                If not specified (None), it will take the general noise parameter of the QuantumCircuit object.
            F_link : float [0-1], optional, default=0.1
                The amount of network noise present
            user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.
            network_noise_type : int, optional, default=None
                Type of network noise that should be used. If not specified, the network noise type known for the
                QuantumCircuit object is used
            # bell_state_type : int [1-4], optional, default=3
            #     Choose the Bell state type which should be created, types are:
            #         1 : |00> + |11>
            #         2 : |00> - |11>
            #         3 : |01> + |10>
            #         4 : |01> - |10>
            probabilistic : bool, optional, default=None
                In case of a probabilistic, the method will keep trying to create the bell state untill success. When
                decoherence is present, this adds decoherence after each try. If not specified, the value kwnown for
                the QuantumCircuit object is used
            p_link : float [0-1], optional, default=None
                The success rate of the bell state creation when probabilistic. If not specified, the value known for
                the QuantumCircuit object is used.
            t_link : float, optional, defualt=None,
                The duration of a Bell pair creation relative to the time-step. If not specified, the value known for
                the QuantumCircuit object is used.

            Example
            -------
            qc.create_bell_pairs([(0, 1), (2, 3), (4,5)]) --> Creates Bell pairs between qubit 0 and 1,
            between qubit 2 and 3 and between qubit 4 and 5.
        """
        if user_operation:
            self._user_operation_order.append({"create_bell_pairs_top": [N, new_qubit, noise, F_link]})

        for i in range(0, 2 * N, 2):
            times = 1
            while probabilistic and random.random() > p_link:
                times += 1

            # print("\nBell Pair creation took {} time{}".format(times, "s" if times > 1 else ""))

            self.num_qubits += 2
            self.d = 2 ** self.num_qubits
            density_matrix = self._get_bell_state_by_type(self.bell_pair_type)

            if noise:
                density_matrix = self._N_network(density_matrix, F_link, network_noise_type)

            self._correct_lookup_for_addition(amount_qubits=2, new_density_matrix=density_matrix)

            self._update_uninitialised_qubit_register([i, i+1], update_type='remove')

            # Drawing the Bell Pair
            if new_qubit:
                self._qubit_array.insert(0, ket_0)
                self._qubit_array.insert(0, ket_0)
                self._correct_drawing_for_n_top_qubit_additions(n=2)
            else:
                self._effective_measurements -= 2
            self._add_draw_operation("#", (0, 1), noise)

            if noise and self.p_dec > 0:
                times_total = times * int(math.ceil(t_link / self.time_step))
                self._N_decoherence([i, i + 1], times=times_total)
                self._increase_duration(t_link)

    def get_cut_off_corrected_link_time(self, attempts, t_link):
        fixed_links = self.n_DD if self.t_pulse else 1

        link_time = attempts * t_link
        pulse_time = math.floor(attempts / (2 * fixed_links)) * self.t_pulse
        longest_duration = (max([self.nodes[node].sub_circuit_time
                                for node in self._current_sub_circuit.involved_nodes]) if self._current_sub_circuit
                            is not None else 0)
        link_plus_longest = link_time + longest_duration + pulse_time

        if self.total_duration + link_plus_longest > self.cut_off_time:
            time_till_cut_off = self.cut_off_time - self.total_duration
            attempts = time_till_cut_off // (t_link + self.t_pulse / (2*fixed_links))
            pulse_time = math.floor(attempts / (2 * fixed_links)) * self.t_pulse
            return time_till_cut_off - pulse_time, attempts, pulse_time

        return link_time, attempts, pulse_time

    @determine_qubit_index(parameter_positions=[1, 2])
    @skip_if_cut_off_reached
    @handle_none_parameters
    def create_bell_pair(self, qubit1, qubit2, noise=None, F_link=None, network_noise_type=None, bell_pair_type=None,
                         probabilistic=None, p_link=None, t_link=None, decoherence=None,
                         attempts=1, user_operation=True, noisy_bell_state=None, print_time_progression=False):
        """
            Creates a Bell pair between the supplied qubits. No actual circuit is applied, the requested Bell state is
            created between the qubits by appointing the corresponding density matrix to the qubits.

            Method is only able to create Bell pairs in this fashion if the qubits supplied have a single qubit
            density matrix or if the two qubits are spanning a two qubit density matrix.

            Parameters
            ----------
            qubit1 : int, str
                Qubit index of one of the qubits involved in the Bell pair. Qubit will be the second qubit in the
                density matrix
            qubit2 : int, str
                Qubit index of one of the qubits involved in the Bell pair. Qubit will be the first qubit in the
                density matrix
            noise : bool
                Applies noise to the operation if True. If not specified, the global noise parameter is used.
            network_noise_type :
                The noise channel that should be used for the noisy operation.
            # bell_state_type : int
            #     The type of Bell state that is created. Types can be found at the '_get_bell_state_by_type' method
            probabilistic : bool
                Determines if the creation of the Bell pair is probabilistic. If not specified, the global
                probabilistic variable is used.
            p_link : float
                The success rate of the Bell pair creation attempt in case the creation is probabilistic. If not
                specified the global p_link value is used.
            t_link : float
                The time it takes to do a Bell pair creation attempt. If not specified, the global
                t_link value will be used.
            decoherence : bool
                Applies decoherence to the qubits that wait on the operation to finish. If not specified, the global
                decoherence value will be used.
            attempts : int
                How many attempts it should take to successfully create a Bell pair. Make sure to set probabilistic
                to False when one wants a fixed number of attempts.
            user_operation : bool
                If True, the operation will be logged as an user operation applied to the circuit.
        """
        if user_operation:
            self._user_operation_order.append({"create_bell_pair": [qubit1, qubit2, noise, F_link, network_noise_type,
                                                                    bell_pair_type]})

        if print_time_progression:
            sub_circuit_durations = [self.nodes[node].sub_circuit_time
                                     for node in [self.get_node_name_from_qubit(qubit) for qubit in [qubit1, qubit2]]]
            print(f"Create Bell pair sub circuit durations: {sub_circuit_durations}.")

        # We have to make sure that both nodes start with creating the Bell pair at the same time: it can happen
        # that one of the two nodes has to wait for a bit before the other node is ready.
        sc_dur_diff = self.nodes[self.get_node_name_from_qubit(qubit1)].sub_circuit_time - \
                      self.nodes[self.get_node_name_from_qubit(qubit2)].sub_circuit_time

        if sc_dur_diff > 0:
            self._increase_duration(sc_dur_diff, [qubit2], kind='idle', print_time_progression=print_time_progression)
        else:
            self._increase_duration(-1*sc_dur_diff, [qubit1], kind='idle', print_time_progression=print_time_progression)

        if print_time_progression:
            sub_circuit_durations = [self.nodes[node].sub_circuit_time
                                     for node in [self.get_node_name_from_qubit(qubit) for qubit in [qubit1, qubit2]]]
            print(f"New create Bell pair sub circuit durations: {sub_circuit_durations}.")

        if noise and decoherence:
            self._N_decoherence([qubit1, qubit2])

        self._total_link_attempts += attempts
        while probabilistic and random.random() > p_link:
            attempts += 1
            self._total_link_attempts += 1

        _, qubits_1, _, num_qubits_1 = self._get_qubit_relative_objects(qubit1)
        _, qubits_2, _, num_qubits_2 = self._get_qubit_relative_objects(qubit2)
        # qb_inits = set(self.qubits.keys()).difference(self.data_qubits).difference(self._uninitialised_qubits)

        # If qubits are not single qubit states or in a state with different qubits the matrix should be reset
        if ((num_qubits_1 > 2 or num_qubits_2 > 2) or
           ((num_qubits_1 > 1 or num_qubits_2 > 1) and not all(qubit in qubits_1 for qubit in qubits_2))):
            # reset_qubits = qubits_1 + qubits_2
            # self._reset_density_matrices(reset_qubits)
            self.measure([qubit1, qubit2], outcome=0, p_m=0.0, basis="Z", probabilistic=False, noise=False)

        if noise and network_noise_type in [3] + [*range(10, 22)] + [99]:
            new_density_matrix = noisy_bell_state
        else:
            new_density_matrix = self._get_bell_state_by_type(bell_pair_type)
            if noise:
                new_density_matrix = self._N_network(new_density_matrix, F_link, network_noise_type)

        qubits = [qubit2, qubit1]
        self._qubit_density_matrix_lookup.update({qubit1: (new_density_matrix, qubits),
                                                  qubit2: (new_density_matrix, qubits)})

        self._update_uninitialised_qubit_register([qubit1, qubit2], update_type="remove")

        duration, attempts, pulse_time = self.get_cut_off_corrected_link_time(attempts, t_link)
        self._increase_duration(duration, [qubit1, qubit2], kind='link')
        # Add duration of pi pulses to the initialised qubit if link took longer than two times the inter pulse delay
        self._add_t_pulse_while_link(pulse_time, [qubit1, qubit2])

        self._total_succeeded_link += 1
        self._add_draw_operation("#{}".format(attempts), (qubit1, qubit2), noise)

        # These gates make sure that we get a Bell pair of the for |00> + |11> (plus noise) again, since this is the
        # state we need for carrying out the non-local operations.
        if network_noise_type != 99: # we only want the gate error case for the Phi+ scenario! Change this if want to generalize.
            if self.bell_pair_type in [2, 3]:
                self.X(qubit2)
            if self.bell_pair_type in [1, 2]:
                self.Z(qubit2)
            # print(self.get_combined_density_matrix([qubit1, qubit2])[0].todense())

        return attempts

    # This cannot be used anymore now that the weight of the GHZ state is unspecified:
    # @determine_qubit_index(parameter_positions=[1, 2, 3, 4])
    @skip_if_cut_off_reached
    @handle_none_parameters
    def create_ghz_state_direct(self, qubits, noise=None, F_link=None, network_noise_type=None, bell_pair_type=None,
                                probabilistic=None, p_link=None, t_link=None, decoherence=None, attempts=1,
                                user_operation=True, noisy_bell_state=None, print_time_progression=False):
        """
            Creates a Bell pair between the supplied qubits. No actual circuit is applied, the requested Bell state is
            created between the qubits by appointing the corresponding density matrix to the qubits.

            Method is only able to create Bell pairs in this fashion if the qubits supplied have a single qubit
            density matrix or if the two qubits are spanning a two qubit density matrix.

            Parameters
            ----------
            qubits : list of int
                Qubit indices of the qubits involved in the entangled state created. The first qubit will be the last
                in the density matrix
            noise : bool
                Applies noise to the operation if True. If not specified, the global noise parameter is used.
            network_noise_type :
                The noise channel that should be used for the noisy operation.
            # bell_state_type : int
            #     The type of Bell state that is created. Types can be found at the '_get_bell_state_by_type' method
            probabilistic : bool
                Determines if the creation of the Bell pair is probabilistic. If not specified, the global
                probabilistic variable is used.
            p_link : float
                The success rate of the Bell pair creation attempt in case the creation is probabilistic. If not
                specified the global p_link value is used.
            t_link : float
                The time it takes to do a Bell pair creation attempt. If not specified, the global
                t_link value will be used.
            decoherence : bool
                Applies decoherence to the qubits that wait on the operation to finish. If not specified, the global
                decoherence value will be used.
            attempts : int
                How many attempts it should take to successfully create a Bell pair. Make sure to set probabilistic
                to False when one wants a fixed number of attempts.
            user_operation : bool
                If True, the operation will be logged as an user operation applied to the circuit.
        """
        if isinstance(qubits, int):
            qubits = [qubits]

        if user_operation:
            self._user_operation_order.append({"create_ghz_state_direct": qubits + [noise, F_link, network_noise_type,
                                                                                    bell_pair_type]})

        if print_time_progression:
            sub_circuit_durations = [self.nodes[node].sub_circuit_time
                                     for node in [self.get_node_name_from_qubit(qubit) for qubit in qubits]]
            print(f"Create direct GHZ state sub circuit durations: {sub_circuit_durations}.")

        # We have to make sure that all nodes start with creating the direct GHZ state at the same time: it can happen
        # that one of the nodes has to wait for a bit before the other nodes are ready.
        sub_circuit_times = [None] * len(qubits)
        for i_q, qubit in enumerate(qubits):
            sub_circuit_times[i_q] = self.nodes[self.get_node_name_from_qubit(qubit)].sub_circuit_time
        max_sub_circuit_time = max(sub_circuit_times)
        for i_q, qubit in enumerate(qubits):
            waiting_time = max_sub_circuit_time - sub_circuit_times[i_q]
            if waiting_time > 0:
                self._increase_duration(waiting_time, [qubit], kind='idle',
                                        print_time_progression=print_time_progression)

        if print_time_progression:
            sub_circuit_durations = [self.nodes[node].sub_circuit_time
                                     for node in [self.get_node_name_from_qubit(qubit) for qubit in qubits]]
            print(f"New create direct GHZ state sub circuit durations: {sub_circuit_durations}.")

        if noise and decoherence:
            self._N_decoherence(qubits)

        self._total_link_attempts += attempts
        while probabilistic and random.random() > p_link:
            attempts += 1
            self._total_link_attempts += 1

        qubit_objects = []
        num_qubits_objects = []
        for qubit in qubits:
            _, qubit_ob, _, num_qub = self._get_qubit_relative_objects(qubit)
            qubit_objects.append(qubit_ob)
            num_qubits_objects.append(num_qub)
        # qb_inits = set(self.qubits.keys()).difference(self.data_qubits).difference(self._uninitialised_qubits)

        ## THIS PART HAS TO BE ADDED FOR DIRECT GHZ STATES
        # # If qubits are not single qubit states or in a state with different qubits the matrix should be reset
        # if ((num_qubits_1 > 2 or num_qubits_2 > 2) or
        #    ((num_qubits_1 > 1 or num_qubits_2 > 1) and not all(qubit in qubits_1 for qubit in qubits_2))):
        #     # reset_qubits = qubits_1 + qubits_2
        #     # self._reset_density_matrices(reset_qubits)
        #     self.measure([qubit1, qubit2], outcome=0, p_m=0.0, basis="Z", probabilistic=False, noise=False)

        if noise and ((len(qubits) == 4 and bell_pair_type == 40) or (len(qubits) == 3 and bell_pair_type == 30)):
            new_density_matrix = noisy_bell_state
        else:
            raise ValueError("For direct GHZ state generation, 'bell_pair_type' has to match the type of entangled "
                             "state that is generated.")

        qubits_reversed = qubits
        qubits_reversed.reverse()
        for qubit in qubits:
            self._qubit_density_matrix_lookup.update({qubit: (new_density_matrix, qubits_reversed)})

        self._update_uninitialised_qubit_register(qubits, update_type="remove")

        duration, attempts, pulse_time = self.get_cut_off_corrected_link_time(attempts, t_link)
        self._increase_duration(duration, qubits, kind='link')
        # Add duration of pi pulses to the initialised qubit if link took longer than two times the inter pulse delay
        self._add_t_pulse_while_link(pulse_time, qubits)

        self._total_succeeded_link += 1
        for qubit in qubits:
            self._add_draw_operation("#{}".format(attempts), qubit, noise)

        return attempts


    def _add_t_pulse_while_link(self, pulse_time, link_qubits):
        if pulse_time == 0:
            return

        node_qubits = self.get_node_qubits(link_qubits)
        # If there are any initialised qubits that are being decoupled, then additional pulses were necessary
        if any(not self.qubits[qubit].equal_to_0_or_1_state() and qubit not in self._uninitialised_qubits
               for qubit in node_qubits):
            self._increase_duration(pulse_time, link_qubits)

    @staticmethod
    def _get_bell_state_by_type(bell_state_type=3):
        """
            Returns a Bell state density matrix based on the type provided. types are:
                    0 : 1/2(|00> + |11>)
                    1 : 1/2(|00> - |11>)
                    2 : 1/2(|01> - |10>)
                    3 : 1/2(|01> + |10>)
        """
        rho = sp.lil_matrix((4, 4))
        if bell_state_type == 0:
            rho[0, 0], rho[0, 3], rho[3, 0], rho[3, 3] = 1 / 2, 1 / 2, 1 / 2, 1 / 2
        elif bell_state_type == 1:
            rho[0, 0], rho[0, 3], rho[3, 0], rho[3, 3] = 1 / 2, -1 / 2, -1 / 2, 1 / 2
        elif bell_state_type == 2:
            rho[1, 1], rho[1, 2], rho[2, 1], rho[2, 2] = 1 / 2, -1 / 2, -1 / 2, 1 / 2
        elif bell_state_type == 3:
            rho[1, 1], rho[1, 2], rho[2, 1], rho[2, 2] = 1 / 2, 1 / 2, 1 / 2, 1 / 2
        elif bell_state_type == 40:
            rho[0, 0], rho[0, 15], rho[15, 0], rho[15, 15] = 1 / 2, 1 / 2, 1 / 2, 1 / 2
        else:
            print(bell_state_type)
            raise ValueError("A non-valid Bell state type was requested. Known types are 0, 1, 2, 3, and 40.")
        return sp.csr_matrix(rho)

    def add_top_qubit(self, qubit_state=ket_0, p_prep=0, user_operation=True):
        """
        qc.add_top_qubit(qubit_state=ket_0)

            Method appends a qubit with a given state to the top of the system.
            *** THE METHOD APPENDS A QUBIT, WHICH MEANS THAT THE AMOUNT OF QUBITS IN THE SYSTEM WILL
            GROW WITH 1 AND THE INDICES OF THE EXISTING QUBITS WILL INCREASE WITH 1 AS WELL***

            Parameters
            ----------
            qubit_state : array, optional, default=ket_0
                Qubit state, a normalised vector of dimension 2x1
            user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.
        """
        if user_operation:
            self._user_operation_order.append({"add_top_qubit": [qubit_state]})
        if self.noise and p_prep > 0:
            qubit_state = self._N_preparation(state=qubit_state, p_prep=p_prep)

        self._qubit_array.insert(0, qubit_state)
        self.num_qubits += 1
        self.d = 2 ** self.num_qubits
        self._correct_drawing_for_n_top_qubit_additions()

        self._correct_lookup_for_addition(CT(qubit_state))

    """
        ---------------------------------------------------------------------------------------------------------
                                                General Gate Application
        ---------------------------------------------------------------------------------------------------------     
    """
    @handle_none_parameters(excluded_parameters=['cqubit'])
    @determine_qubit_index(parameter_positions=[2, 3])
    @skip_if_cut_off_reached
    def apply_gate(self, gate, tqubit, cqubit=None, *, noise=None, conj=False, p_g=None, draw=True, decoherence=None,
                   reverse=False, electron_is_target=False, user_operation=True):
        """
            General method to apply a two- or single-qubit gate to the circuit.

            Parameters
            ----------
            gate : TwoQubitGate, SingleQubitGate
                TwoQubitGate object or SingleQubitGate object that should be applied to the system
            tqubit : int, str
                Qubit index of the target qubit
            cqubit : int, str, optional
                Qubit index of control qubit, if applicable
            noise : bool
                Specifies is noise is present for this operation. If not specified, the global noise variable is used
            conj : bool
                If True, the conjugate of the supplied gate is applied (if known)
            p_g : float
                Specifies the error probability of the gate error. If not specified, the global noise variable is used
            draw : bool
                Whether the gate operation should show in the circuit drawing
            decoherence : bool
                If True, the duration of the gate operation will be added to the qubits that are known to be waiting
                on this operation to finish. If not specified, the global decoherence variable is used.
            reverse : bool
                Reverses the order how density matrices of the qubits are fused. Normally the cqubit density matrix
                is fused on top of the tqubit density matrix. This parameter is typically used rearrange qubits in the
                density matrix, such that qubit measurement is faster eventually.
            electron_is_target : bool
                When working with decoherence in NV centers, the case that the electron qubit is the control qubit
                needs to be handled differently. This is ensured when this boolean is set to True.
            user_operation : bool
                If True, the system will log this as a by the user applied operation on the circuit.
        """
        apply_noiseless_swap = True if (gate.name == "Swap" and self.noiseless_swap is True) else False
        if self.qubits is not None:
            tqubit_obj = self.qubits[tqubit]
            refocus_time_t = self._determine_additional_waiting_pulse_sequence(tqubit_obj) \
                if (tqubit_obj not in self._uninitialised_qubits and tqubit_obj.sequence_time > 0.5e-3) else 0
            if cqubit:
                cqubit_obj = self.qubits[cqubit]
                refocus_time_c = self._determine_additional_waiting_pulse_sequence(cqubit_obj) \
                    if (cqubit_obj not in self._uninitialised_qubits and cqubit_obj.sequence_time > 0.5e-3) else 0
            else:
                refocus_time_c = 0
            refocus_time = max(refocus_time_t, refocus_time_c)
        else:
            refocus_time = 0
        # Skip gate if not within cut-off
        full_duration = 0 if apply_noiseless_swap else gate.duration + refocus_time
        if not self._check_if_operation_within_cut_off(full_duration, tqubit):
            return

        if user_operation:
            self._user_operation_order.append({"apply_gate": [gate, tqubit, cqubit, noise, conj, p_g, draw]})

        # If pulse sequence is taken into account, the SWAP gate must wait for the right point in the sequence
        if not(apply_noiseless_swap):
            self._wait_for_refocus([tqubit, cqubit])

        qubits = [tqubit] if cqubit is None else [tqubit, cqubit]

        if noise and decoherence:
            self._N_decoherence(qubits)

        if electron_is_target:
            draw = self._operations.gate_operations.handle_electron_is_target_qubit(self, tqubit, cqubit, noise=noise,
                                                                                    decoherence=decoherence,
                                                                                    draw=draw, gate=gate)

        if apply_noiseless_swap:
            noise = False

        if type(gate) == SingleQubitGate:
            noise = noise and not self.no_single_qubit_error
            new_density_matrix = self._apply_1_qubit_gate(gate, tqubit, conj=conj, noise=noise, p_g=p_g)
        elif type(gate) == TwoQubitGate:
            new_density_matrix = self._apply_2_qubit_gate(gate, cqubit, tqubit, noise=noise, p_g=p_g, reverse=reverse)
        else:
            raise ValueError("Gate object was not recognised. Please create an gate object to apply this gate.")

        self._set_density_matrix(tqubit, new_density_matrix)
        gate_duration = gate.duration if not self.qubits or self.qubits[tqubit].qubit_type != 'e' else \
            gate.duration_electron
        gate_duration = 0 if apply_noiseless_swap else gate_duration
        self._increase_duration(gate_duration, qubits)

        if draw:
            self._add_draw_operation(gate, qubits, noise)

        if electron_is_target:
            self._operations.gate_operations.handle_electron_is_target_qubit(self, tqubit, cqubit, noise=noise,
                                                                             decoherence=decoherence, draw=draw)

    def _check_if_operation_within_cut_off(self, duration, tqubit=None, nodes=None):
        if tqubit is None and nodes is None:
            raise ValueError("Either the tqubit or nodes needs to be specified!")

        # Never skip when circuit_operation_ended is True or if cut_off_time is infinity
        if self._circuit_operations_ended or self.cut_off_time == np.inf or self.nodes is None:
            return True
        # Get total duration of the node on which the gate is applied
        if tqubit is not None:
            node = self.nodes[self.get_node_name_from_qubit(tqubit)]
        else:
            _, node = max([(self.nodes[n].sub_circuit_time, self.nodes[n]) for n in nodes], key=lambda tup: tup[0])
        total_time = self.total_duration + node.sub_circuit_time

        if nodes is None:
            nodes = [node.name]

        # If the gate duration exceeds the cut-off time, increase the time as decoherence. Set cut-off time reached
        if round(total_time + duration - self.cut_off_time, SUM_ACCURACY) > 0:
        # if total_time + duration > self.cut_off_time:
            time_till_cut_off = self.cut_off_time - total_time
            self.cut_off_time_reached = True
            self._increase_duration(time_till_cut_off, [], involved_nodes=nodes, check=True)
            return False

        return True

    """
        ---------------------------------------------------------------------------------------------------------
                                                One-Qubit Gate Methods
        ---------------------------------------------------------------------------------------------------------     
    """
    def _apply_1_qubit_gate(self, gate, tqubit, conj=False, noise=None, p_g=None):
        """
            qc.apply_1_qubit_gate(gate, tqubit, noise=None, p_g=None, draw=True)

                Applies a single-qubit gate to the specified target qubit. This will update the density
                matrix of the system accordingly.

                Parameters
                ----------
                gate : ndarray
                    Array of dimension 2x2, examples are the well-known pauli matrices (X, Y, Z)
                tqubit : int
                    Integer that indicates the target qubit. Note that the qubit counting starts at
                    0.
                noise : bool, optional, default=None
                    Determines if the gate is noisy. When the QuantumCircuit object is initialised
                    with the 'noise' parameter to True, this parameter will also evaluate to True if
                    not specified otherwise.
                p_g : float [0-1], optional, default=None
                    Specifies the amount of gate noise if present. If the QuantumCircuit object is
                    initialised with a 'p_g' parameter, this will be used if not specified otherwise
                draw : bool, optional, default=True
                    If true, the specified gate will appear when the circuit is visualised.
                user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.
        """
        tqubit_density_matrix, _, relative_tqubit_index, relative_num_qubits = self._get_qubit_relative_objects(tqubit)

        one_qubit_gate = self._create_1_qubit_gate(gate,
                                                   relative_tqubit_index,
                                                   num_qubits=relative_num_qubits,
                                                   conj=conj)
        new_density_matrix = one_qubit_gate.dot(CT(tqubit_density_matrix, one_qubit_gate))

        if noise and not self.no_single_qubit_error:
            new_density_matrix = self._N_depolarising_channel(p_g, relative_tqubit_index, new_density_matrix,
                                                              relative_num_qubits)

        return new_density_matrix

    @handle_none_parameters
    def _create_1_qubit_gate(self, gate, tqubit, *, num_qubits=None, conj=False, lookup=True):
        """
            Private method that is used to create the single-qubit gate matrix used in for example the
            apply_1_qubit_gate method.

            Parameters
            ----------
            gate : ndarray
                Array of dimension 2x2, examples are the well-known pauli matrices (X, Y, Z)
            tqubit : int
                Integer that indicates the target qubit. Note that the qubit counting starts at
                0.
            num_qubits : int, optional, default=None
                Determines the size of the resulting one-qubit gate matrix. If not specified, the
                num_qubits known for the entire QuantumCircuit object is used

            Returns
            -------
            1_qubit_gate : sparse matrix with dimensions equal to the density_matirx attribute
                Returns a matrix with dimensions equal to the dimensions of the density matrix of
                the system.
        """
        return self._operations.gate_operations.create_1_qubit_gate(self, gate, tqubit, num_qubits, conj, lookup)

    def X(self, tqubit, times=1, noise=None, p_g=None, draw=True, user_operation=True):
        """ Applies the pauli X gate to the specified target qubit. See apply_1_qubit_gate for more info """

        for _ in range(times):
            self.apply_gate(X_gate, tqubit, noise=noise, p_g=p_g, draw=draw, user_operation=user_operation)

    def Z(self, tqubit, times=1, noise=None, p_g=None, draw=True, user_operation=True):
        """ Applies the pauli Z gate to the specified target qubit. See apply_1_qubit_gate for more info """

        for _ in range(times):
            self.apply_gate(Z_gate, tqubit, noise=noise, p_g=p_g, draw=draw, user_operation=user_operation)

    def Y(self, tqubit, times=1, noise=None, p_g=None, draw=True, user_operation=True):
        """ Applies the pauli Y gate to the specified target qubit. See apply_1_qubit_gate for more info """

        for _ in range(times):
            self.apply_gate(Y_gate, tqubit, noise=noise, p_g=p_g, draw=draw, user_operation=user_operation)

    def H(self, tqubit, times=1, noise=None, p_g=None, draw=True, user_operation=True):
        """ Applies the Hadamard gate to the specified target qubit. See apply_1_qubit_gate for more info """

        for _ in range(times):
            self.apply_gate(H_gate, tqubit, noise=noise, p_g=p_g, draw=draw, user_operation=user_operation)

    def S(self, tqubit, conj=False, times=1, noise=None, p_g=None, draw=True, user_operation=True):

        for _ in range(times):
            self.apply_gate(S_gate, tqubit, conj=conj, noise=noise, p_g=p_g, draw=draw,
                            user_operation=user_operation)

    def Rx(self, tqubit, theta, times=1, noise=None, p_g=None, draw=True, user_operation=True):
        """ Applies a rotation gate around the x-axis to the specified target qubit with the specified angle.

            Parameters
            ----------
            theta : float (radians)
                Angle of rotation that should be applied. Value should be specified in radians
        """
        R_gate = SingleQubitGate("Rotation gate",
                                 np.array([[np.cos(theta/2), -1j * np.sin(theta/2)],
                                           [-1j * np.sin(theta/2), np.cos(theta/2)]]),
                                 "Rx({})".format(str(Fr(theta/np.pi)) + "\u03C0"))

        for _ in range(times):
            self.apply_gate(R_gate, tqubit, noise=noise, p_g=p_g, draw=draw, user_operation=user_operation)

    def Ry(self, tqubit, theta, times=1, noise=None, p_g=None, draw=True, user_operation=True):
        """ Applies a rotation gate around the y-axis to the specified target qubit with the specified angle.

            Parameters
            ----------
            theta : float (radians)
                Angle of rotation that should be applied. Value should be specified in radians
        """
        R_gate = SingleQubitGate("Rotation gate",
                                 np.array([[np.cos(theta / 2), -1 * np.sin(theta / 2)],
                                           [1 * np.sin(theta / 2), np.cos(theta / 2)]]),
                                 "Ry({})".format(str(Fr(theta/np.pi)) + "\u03C0"))

        for _ in range(times):
            self.apply_gate(R_gate, tqubit, noise=noise, p_g=p_g, draw=draw, user_operation=user_operation)

    def Rz(self, tqubit, theta, times=1, noise=None, p_g=None, draw=True, user_operation=True):
        """ Applies a rotation gate around the x axis to the specified target qubit with the specified angle.

            Parameters
            ----------
            theta : float (radians)
                Angle of rotation that should be applied. Value should be specified in radians

        """
        R_gate = SingleQubitGate("Rotation gate",
                                 np.array([np.exp(-1j * theta / 2), 0],
                                          [0, np.exp(1j * theta / 2)]),
                                 "Rz({})".format(str(Fr(theta/np.pi)) + "\u03C0"))

        for _ in range(times):
            self.apply_gate(R_gate, tqubit, noise=noise, p_g=p_g, draw=draw, user_operation=user_operation)

    """
        ---------------------------------------------------------------------------------------------------------
                                                Two-Qubit Gate Methods
        ---------------------------------------------------------------------------------------------------------     
    """
    def _apply_2_qubit_gate(self, gate, cqubit, tqubit, noise=None, p_g=None, reverse=False):
        """
            Applies a two qubit gate according to the specified control and target qubits. This will update the density
            matrix of the system accordingly.

            Parameters
            ----------
            gate : TwoQubitGate class
                Gate class object, predefined Gate objects are available such as the X, Y and Z gates
            cqubit : int
                Integer that indicates the control qubit. Note that the qubit counting starts at 0
            tqubit : int
                Integer that indicates the target qubit. Note that the qubit counting starts at 0.
            noise : bool, optional, default=None
                Determines if the gate is noisy. When the QuantumCircuit object is initialised
                with the 'noise' parameter to True, this parameter will also evaluate to True if
                not specified otherwise.
            p_g : float [0-1], optional, default=None
                Specifies the amount of gate noise if present. If the QuantumCircuit object is
                initialised with a 'p_g' parameter, this will be used if not specified otherwise
            draw : bool, optional, default=True
                If true, the specified gate will appear when the circuit is visualised.
            gate_2 : array, optional, default=None
                Array of dimension 2x2. This parameter can be used to specify a gate that is applied to the
                target qubit for the case that the control qubit is in the |0> state.
            user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.
        """
        cqubit_density_matrix, _ = self._qubit_density_matrix_lookup[cqubit]
        tqubit_density_matrix, _ = self._qubit_density_matrix_lookup[tqubit]

        # Check if cqubit and tqubit belong to the same density matrix. If not they should fuse
        if not cqubit_density_matrix is tqubit_density_matrix:
            if not reverse:
                self._correct_lookup_for_two_qubit_gate(cqubit, tqubit)
            else:
                self._correct_lookup_for_two_qubit_gate(tqubit, cqubit)

        # Since density matrices are fused if not equal, it is only necessary to get the (new) density matrix from
        # the lookup table by either one of the qubit indices
        density_matrix, qubits, rel_cqubit, rel_num_qubits = self._get_qubit_relative_objects(cqubit)
        rel_tqubit = qubits.index(tqubit)

        two_qubit_gate = self._create_2_qubit_gate(gate,
                                                   rel_cqubit,
                                                   rel_tqubit,
                                                   num_qubits=rel_num_qubits)

        new_density_matrix = two_qubit_gate.dot(CT(density_matrix, two_qubit_gate))

        if noise:
            times = 3 if gate.name.lower() == 'swap' else 1
            new_density_matrix = self._N_two_qubit_gate(p_g, rel_cqubit, rel_tqubit, new_density_matrix,
                                                        num_qubits=rel_num_qubits, times=times)

        return new_density_matrix

    @handle_none_parameters
    def _create_2_qubit_gate(self, gate, cqubit, tqubit, num_qubits=None):
        """
        Create a controlled gate matrix for the density matrix according to the control and target qubits given.
        This is done by
                1.  first taking the Kronecker Product the identity matrix as many times as there are qubits
                    present in the system.
                2.  Then for the two sub gates formed on the place of the control qubit the identity matrix
                    is replaced for a |0><0| and |1><1| matrix respectively.
                3.  Then for the gate_2 the identity matrix on the target qubit index is replaced with the wanted gate.

        So for creating a CNOT gate with the control on the 2nd qubit and target on the first qubit on a system with 3
        qubits one will get:

                1. I#I#I + I#I#I + I#I#I + I#I#I
                2. I#|0><0|#I + I#|1><1|#I + 0#|0><1|#I + 0#|1><0|#I
                3. I#|0><0|#I + X_t#|1><1|#I + 0#|0><1|#I + 0#|1><0|#I

        (In which '#' is the Kronecker Product, and '0' is the zero matrix)
        (https://quantumcomputing.stackexchange.com/questions/4252/
        how-to-derive-the-cnot-matrix-for-a-3-qbit-system-where-the-control-target-qbi and
        https://quantumcomputing.stackexchange.com/questions/9181/swap-gate-on-2-qubits-in-3-entangled-qubit-system)

        The 'create_component_2_qubit_gate' method defined within creates one of the 4 components that is shown in
        step 3 above. Thus 'first_part = create_component_2_qubit_gate(CT(ket_0), zero_state_matrix)' creates the first
        component namely I#|0><0|#I in case of the CNOT mentioned.

        Parameters
        ----------
        gate : TwoQubitGate object
            TwoQubitGate object representing a 2-qubit gate
        cqubit : int
            Integer that indicates the control qubit. Note that the qubit counting starts at 0.
        tqubit : int
            Integer that indicates the target qubit. Note that the qubit counting starts at 0.
        num_qubits : int, optional, default=None
            Determines the size of the resulting two-qubit gate matrix. If not specified, the
            num_qubits known for the entire QuantumCircuit object is used

        """
        return self._operations.gate_operations.create_2_qubit_gate(self, gate, cqubit, tqubit, num_qubits)

    def CNOT(self, cqubit, tqubit, noise=None, p_g=None, draw=True, user_operation=True):
        """ Applies the CNOT gate to the specified target qubit. See apply_2_qubit_gate for more info """

        self.apply_gate(CNOT_gate, tqubit, cqubit, noise=noise, p_g=p_g, draw=draw, user_operation=user_operation)

    def CZ(self, cqubit, tqubit, noise=None, p_g=None, draw=True, user_operation=True):
        """ Applies the CZ gate to the specified target qubit. See apply_2_qubit_gate for more info """

        self.apply_gate(CZ_gate, tqubit, cqubit, noise=noise, p_g=p_g, draw=draw, user_operation=user_operation)

    @determine_qubit_index(parameter_positions=[1, 2])
    @skip_if_cut_off_reached
    @handle_none_parameters
    def SWAP(self, cqubit, tqubit, noise=None, p_g=None, draw=True, efficient=True, user_operation=True):
        """
            Applies the SWAP gate to specified qubits. The efficient parameter is used, when no actual circuit has
            to be applied, but the qubits can be swapped by swapping the qubit indices in the qubit density matrix
            lookup table.
        """
        if cqubit == tqubit:
            return
        # Skip gate if not within cut-off
        gate_duration = 0 if self.noiseless_swap else SWAP_gate.duration
        if not self._check_if_operation_within_cut_off(gate_duration, tqubit):
            return
        # if self.noiseless_swap:
        #     noise = False
        if efficient:
            if user_operation:
                noise_fn = noise if self.noiseless_swap is False else False
                p_g_fn = p_g if self.noiseless_swap is False else 0
                self._user_operation_order.append({"SWAP": [cqubit, tqubit, noise_fn, p_g_fn, draw]})

            self._operations.gate_operations.efficient_SWAP(self, cqubit, tqubit, noise, p_g, draw)
        else:
            self.apply_gate(SWAP_gate, tqubit, cqubit, noise=noise, p_g=p_g, draw=draw, user_operation=user_operation)

    def two_qubit_gate_NV(self, cqubit, tqubit, noise=None, p_g=None, draw=True, user_operation=True):
        """ Applies the two-qubit gate that is specific to the actual NV center"""

        self.apply_gate(NV_two_qubit_gate, tqubit, cqubit, noise, p_g, draw, user_operation=user_operation)

    def CNOT_NV(self, cqubit, tqubit, noise=None, p_g=None, draw=True, user_operation=True):

        self.Z(cqubit, noise=noise, p_g=p_g, draw=draw, user_operation=user_operation)
        self.S(cqubit, noise=noise, p_g=p_g, draw=draw, user_operation=user_operation)
        self.Ry(tqubit, np.pi/2, noise=noise, p_g=p_g, draw=draw, user_operation=user_operation)
        self.S(tqubit, noise=noise, p_g=p_g, draw=draw, user_operation=user_operation)
        self.two_qubit_gate_NV(cqubit, tqubit, noise=noise, p_g=p_g, draw=draw, user_operation=user_operation)
        self.S(tqubit, conj=True, noise=noise, p_g=p_g, draw=draw, user_operation=user_operation)

    @handle_none_parameters
    def _determine_additional_waiting_pulse_sequence(self, qubit: Qubit, n_DD=None,
                                                     t_link=None, t_pulse=None, offset=0):
        """
            Returns the link waiting time and the idle waiting time based on the sequence parameters present for the
            system. The pulse sequence is used to keep the nuclear qubit more coherent, but therefore only at certain
            places in the pulse sequence, the states can be swapped. Consider the following pulse sequence containing 8
            pulses:

            n - pi - n | n - pi - n | n - pi - n | n - pi - n | n - pi - n | n - pi - n |

            Only at the '|' signs the state of the qubit can be swapped. 'n' is the predetermined n_DD
            that can be made before a pulse (pi) is applied. By the amount of link attempts it to took create a Bell
            pair it is thus determined how much of the time is link waiting time (qubits in node experiencing more
            decoherence due to bell pair creation attempts) and how much is idle time which the qubits experience
            after the Bell pair is created but it must be waited before the pulse refocuses.

            Parameters
            ----------
            attempts_till_success : int
                Amount of Bell pair creation attempts it took to create a Bell pair.
            n_DD : int
                Amount of Bell pair creation attempts before a pulse of the pulse sequence is applied ('n' in the
                sequence shown above).
            t_link : float
                Time it takes to do one Bell pair creation attempt.
            t_pulse : float
                The duration of the pulse ('pi' in the sequence shown above).
        """
        if qubit.qubit_type == 'e':
            return 0

        sequence_time = qubit.sequence_time + offset
        n = n_DD * t_link
        full_sequence = (2 * n) + t_pulse

        waiting_time = full_sequence - (sequence_time % full_sequence) if full_sequence != 0 else 0

        return waiting_time

    def _wait_for_refocus(self, qubits):
        if self.t_pulse > 0:
            for qubit in qubits:
                if qubit is None:
                    return
                qubit_obj = self.qubits[qubit]
                # Check if qubit is initialised, not in |0> or |1> (with noise) and if sequence time is not 0
                if qubit not in self._uninitialised_qubits and qubit_obj.sequence_time > 0.5e-3:
                    time_till_swap = self._determine_additional_waiting_pulse_sequence(qubit_obj)
                    self._increase_duration(time_till_swap, [], involved_nodes=[qubit_obj.node])
                qubit_obj.reset_sequence_time()


    """
        ---------------------------------------------------------------------------------------------------------
                                            Protocol gate sequences
        ---------------------------------------------------------------------------------------------------------  
    """
    @determine_qubit_index(parameter_positions=[2, 3, 4, 5])
    @skip_if_cut_off_reached
    def single_selection(self, operation, bell_qubit_1, bell_qubit_2, target_qubit_1=None, target_qubit_2=None,
                         measure=True, noise=None, F_link=None, p_m=None, p_g=None, swap=False, create_bell_pair=True,
                         reverse_den_mat_add=False, user_operation=True):
        """ Single selection as specified by Naomi Nickerson in https://www.nature.com/articles/ncomms2773.pdf """
        if target_qubit_1 is None:
            target_qubit_1 = bell_qubit_1 + 1
        if target_qubit_2 is None:
            target_qubit_2 = bell_qubit_2 + 1

        if create_bell_pair:
            self.create_bell_pair(bell_qubit_1, bell_qubit_2, noise=noise, F_link=F_link, user_operation=user_operation)
        self.apply_gate(operation, cqubit=bell_qubit_1, tqubit=target_qubit_1, noise=noise, p_g=p_g,
                        user_operation=user_operation, reverse=reverse_den_mat_add)
        self.apply_gate(operation, cqubit=bell_qubit_2, tqubit=target_qubit_2, noise=noise, p_g=p_g,
                        user_operation=user_operation, reverse=reverse_den_mat_add)
        if measure:
            measurement_outcomes = self.measure([bell_qubit_2, bell_qubit_1], noise=noise, p_m=p_m,
                                                user_operation=user_operation)
            # If loop necessary for proper cut-off handling
            if type(measurement_outcomes) == SKIP:
                return measurement_outcomes
            return measurement_outcomes[0] == measurement_outcomes[1]
        elif swap:
            self.SWAP(bell_qubit_1, bell_qubit_1 + 2, efficient=True)
            self.SWAP(bell_qubit_2, bell_qubit_2 + 2, efficient=True)
            return True

    @determine_qubit_index(parameter_positions=[2, 3, 4, 5, 6, 7, 8, 9])
    @skip_if_cut_off_reached
    def double_selection(self, operation, bell_qubit_1, bell_qubit_2=None, bell_qubit_3=None, bell_qubit_4=None,
                         target_qubit_1=None, target_qubit_2=None, target_qubit_3=None, target_qubit_4=None,
                         noise=None, F_link=None, p_m=None, p_g=None, swap=False, user_operation=True):
        """ Double selection as specified by Naomi Nickerson in https://www.nature.com/articles/ncomms2773.pdf """
        if bell_qubit_3 is None:
            bell_qubit_3 = bell_qubit_1 - 1 if not swap else bell_qubit_1
        if bell_qubit_4 is None:
            bell_qubit_4 = bell_qubit_2 - 1 if not swap else bell_qubit_2
        if target_qubit_3 is None and swap:
            target_qubit_3 = bell_qubit_3 + 2
        if target_qubit_4 is None and swap:
            target_qubit_4 = bell_qubit_4 + 2

        self.single_selection(operation, bell_qubit_1, bell_qubit_2, target_qubit_1, target_qubit_2, measure=False,
                              noise=noise, F_link=F_link, p_m=p_m, p_g=p_g, swap=swap, user_operation=user_operation)
        # Not the swap version this time, since the swapping is done in this method itself
        self.single_selection(CZ_gate, bell_qubit_3, bell_qubit_4, target_qubit_3, target_qubit_4, measure=False,
                              noise=noise, F_link=F_link, p_m=p_m, p_g=p_g, user_operation=user_operation)

        if not swap:
            measurement_zip = zip([bell_qubit_4, bell_qubit_2], [bell_qubit_3, bell_qubit_1])
        else:
            measurement_zip = zip([bell_qubit_2, bell_qubit_2], [bell_qubit_1, bell_qubit_1])

        parity = []
        for i, (qubit_1, qubit_2) in enumerate(measurement_zip):
            if i == 1 and swap:
                self.SWAP(bell_qubit_1, bell_qubit_1 + 2, efficient=True)
                self.SWAP(bell_qubit_2, bell_qubit_2 + 2, efficient=True)
            measurement_outcomes = self.measure([qubit_1, qubit_2], noise=noise, p_m=p_m,
                                                user_operation=user_operation)
            # If loop necessary for proper cut-off handling
            if type(measurement_outcomes) == SKIP:
                return measurement_outcomes
            parity.append(measurement_outcomes[0] == measurement_outcomes[1])
        return all(parity)

    @determine_qubit_index(parameter_positions=[2, 3, 4, 5, 6, 7, 8, 9])
    @skip_if_cut_off_reached
    def single_dot(self, operation, bell_qubit_1, bell_qubit_2, bell_qubit_3=None, bell_qubit_4=None,
                   target_qubit_1=None, target_qubit_2=None, target_qubit_3=None, target_qubit_4=None, measure=True,
                   noise=None, F_link=None, p_m=None, p_g=None, swap=False, user_operation=True):
        """ single dot as specified by Naomi Nickerson in https://www.nature.com/articles/ncomms2773.pdf """
        if bell_qubit_3 is None:
            bell_qubit_3 = bell_qubit_1 - 1 if not swap else bell_qubit_1
        if bell_qubit_4 is None:
            bell_qubit_4 = bell_qubit_2 - 1 if not swap else bell_qubit_2
        if target_qubit_1 is None:
            target_qubit_1 = bell_qubit_1 + 1
        if target_qubit_2 is None:
            target_qubit_2 = bell_qubit_2 + 1

        success = False
        single_selection_success = False
        while not single_selection_success:
            self.create_bell_pair(bell_qubit_1, bell_qubit_2, noise=noise, F_link=F_link, user_operation=user_operation)
            if swap:
                self.SWAP(bell_qubit_1, bell_qubit_1 + 2, efficient=True)
                self.SWAP(bell_qubit_2, bell_qubit_2 + 2, efficient=True)
                target_qubit_3 = bell_qubit_3 + 2 if target_qubit_3 is None else target_qubit_3
                target_qubit_4 = bell_qubit_4 + 2 if target_qubit_4 is None else target_qubit_4
            single_selection_success = self.single_selection(CNOT_gate, bell_qubit_3, bell_qubit_4, target_qubit_3,
                                                             target_qubit_4, noise=noise, F_link=F_link, p_m=p_m, p_g=p_g,
                                                             swap=swap, user_operation=user_operation)
            if not single_selection_success:
                continue
            single_selection_success = self.single_selection(CZ_gate, bell_qubit_3, bell_qubit_4,  target_qubit_3,
                                                             target_qubit_4, noise=noise, F_link=F_link, p_m=p_m, p_g=p_g,
                                                             swap=swap, user_operation=user_operation)

        if swap:
            self.SWAP(bell_qubit_1, bell_qubit_1 + 2, efficient=True)
            self.SWAP(bell_qubit_2, bell_qubit_2 + 2, efficient=True)
        self.apply_gate(operation, cqubit=bell_qubit_1, tqubit=target_qubit_1, noise=noise, p_g=p_g,
                        user_operation=user_operation)
        self.apply_gate(operation, cqubit=bell_qubit_2, tqubit=target_qubit_2, noise=noise, p_g=p_g,
                        user_operation=user_operation)
        if measure:
            measurement_outcomes = self.measure([bell_qubit_2, bell_qubit_1], noise=noise, p_m=p_m,
                                                user_operation=user_operation)
            # If loop necessary for proper cut-off handling
            if type(measurement_outcomes) == SKIP:
                return measurement_outcomes
            success = measurement_outcomes[0] == measurement_outcomes[1]

        elif swap:
            self.SWAP(bell_qubit_1, bell_qubit_1 + 2, efficient=True)
            self.SWAP(bell_qubit_2, bell_qubit_2 + 2, efficient=True)

        return success

    @determine_qubit_index(parameter_positions=[2, 3, 4, 5, 6, 7, 8, 9])
    @skip_if_cut_off_reached
    def double_dot(self, operation, bell_qubit_1, bell_qubit_2, bell_qubit_3=None, bell_qubit_4=None,
                   target_qubit_1=None, target_qubit_2=None, target_qubit_3=None, target_qubit_4=None, noise=None,
                   F_link=None, p_m=None, p_g=None, swap=False, user_operation=True):
        """ double dot as specified by Naomi Nickerson in https://www.nature.com/articles/ncomms2773.pdf """
        if bell_qubit_3 is None:
            bell_qubit_3 = bell_qubit_1 - 1 if not swap else bell_qubit_1
        if bell_qubit_4 is None:
            bell_qubit_4 = bell_qubit_2 - 1 if not swap else bell_qubit_2
        if target_qubit_3 is None and swap:
            target_qubit_3 = bell_qubit_3 + 2
        if target_qubit_4 is None and swap:
            target_qubit_4 = bell_qubit_4 + 2

        self.single_dot(operation, bell_qubit_1, bell_qubit_2, target_qubit_1, target_qubit_2, measure=False,
                        noise=noise, F_link=F_link, p_m=p_m, p_g=p_g, swap=swap, user_operation=user_operation)
        single_selection_success = self.single_selection(CZ_gate, bell_qubit_3, bell_qubit_4, target_qubit_3,
                                                         target_qubit_4, noise=noise, F_link=F_link, p_m=p_m, p_g=p_g, swap=swap,
                                                         user_operation=user_operation)
        if swap:
            self.SWAP(bell_qubit_1, bell_qubit_1 + 2, efficient=True)
            self.SWAP(bell_qubit_2, bell_qubit_2 + 2, efficient=True)

        measurement_outcomes = self.measure([bell_qubit_2, bell_qubit_1], noise=noise, p_m=p_m,
                                            user_operation=user_operation)
        # If loop necessary for proper cut-off handling
        if type(measurement_outcomes) == SKIP:
            return measurement_outcomes
        return measurement_outcomes[0] == measurement_outcomes[1], single_selection_success

    @skip_if_cut_off_reached
    def stabilizer_measurement(self, operation, cqubit=None, tqubit=None, nodes: list = None, swap=False,
                               electron_qubit=None):
        # Function is here, such that user parameters are not overwritten in the loop
        def node_measurement(node, operation, cqubit, tqubit, swap, electron_qubit):
            if cqubit is None:
                # control qubit is the qubit in the node that is initialised apart from the data qubits
                cqubit = int(set(self.qubits.keys()).intersection(self.nodes[node].qubits)
                             .difference(self._uninitialised_qubits).difference(self.data_qubits).pop())
            ghz_qubit = cqubit
            if tqubit is None:
                data_qubits = self.nodes[node].data_qubits
            else:
                data_qubits = [tqubit] if type(tqubit) != list else tqubit

            if swap:
                electron_qubit = self.nodes[node].electron_qubits[0] if electron_qubit is None else electron_qubit

            self.start_sub_circuit(node)
            for i, data_qubit in enumerate(data_qubits):
                cqubit = ghz_qubit if swap else cqubit
                if swap:
                    self.SWAP(electron_qubit, cqubit, efficient=True) if i == 0 else None
                    cqubit = electron_qubit
                self.apply_gate(operation, cqubit=cqubit, tqubit=data_qubit)
            self.measure(cqubit, probabilistic=False)

        # Main code of the method
        self.end_current_sub_circuit(forced_level=True)

        if nodes is None:
            nodes = [self.get_node_name_from_qubit(cqubit)]
        if tqubit is None:
            tqubits = [None for _ in range(len(nodes))]
        elif type(tqubit) == int:
            tqubits = [tqubit for _ in range(len(nodes))]
        elif type(tqubit) == list:
            tqubits = tqubit
        else:
            raise ValueError("The target qubit must be either None, int or list. It was {}".format(type(tqubit)))

        # Check if stabilizer measurement within cut-off time
        if self.nodes and self.cut_off_time < np.inf:
            # The max amount of data qubits in a node indicates the amount op gates necessary to perform stabilizer
            num_op = max([len(node.data_qubits) for node in self.nodes.values() if node.data_qubits is not None])
            swap_dur = SWAP_gate.duration if swap else 0
            initialised_qubits = set(self.qubits.keys()).difference(self._uninitialised_qubits)
            # # Refocus time control qubits
            # refocus_time_cq = (max([self._determine_additional_waiting_pulse_sequence(self.qubits[q])
            #                    for q in initialised_qubits.difference(self.data_qubits)]) if self.t_pulse
            #                    else 0)
            # # Refocus time target qubits (estimation, may differ eventually after waiting on target qubit)
            # refocus_time_tq = (max([self._determine_additional_waiting_pulse_sequence(self.qubits[q])
            #                    for q in initialised_qubits.intersection(self.data_qubits)]) if self.t_pulse
            #                    else 0)
            # duration = (operation.duration * num_op + self.measurement_duration + swap_dur + refocus_time_cq +
            #             refocus_time_tq)

            durations_per_node = {}
            for i, node in enumerate(nodes):
                g_qubit = int(set(self.qubits.keys()).intersection(self.nodes[node].qubits)
                              .difference(self._uninitialised_qubits).difference(self.data_qubits).pop())
                tqubit = [qubit for qubit in tqubits if qubit in self.nodes[node].qubits] if self.nodes and all(tqubits) \
                    else tqubits[i]
                if tqubit is None:
                    d_qubits = self.nodes[node].data_qubits
                else:
                    d_qubits = [tqubit] if type(tqubit) != list else tqubit
                # e_qubit = self.nodes[node].electron_qubits[0] if electron_qubit is None \
                #     else electron_qubit
                g_qubit_obj = self.qubits[g_qubit]
                # SWAP gate is not necessary if the GHZ state qubit already sits on the electron qubit:
                swap_loop = False if (g_qubit_obj._qubit_type == "e" or g_qubit in self.nodes[node]._electron_qubits) \
                    else swap
                refocus_time_g = self._determine_additional_waiting_pulse_sequence(g_qubit_obj) \
                    if (g_qubit_obj not in self._uninitialised_qubits and g_qubit_obj.sequence_time > 0.5e-3) else 0
                time = refocus_time_g + swap_dur if swap_loop else 0
                for idq, data_qubit in enumerate(d_qubits):
                    d_qubit_obj = self.qubits[d_qubits[idq]]
                    refocus_time_d = self._determine_additional_waiting_pulse_sequence(d_qubit_obj, offset=time) \
                        if (d_qubit_obj not in self._uninitialised_qubits and d_qubit_obj.sequence_time > 0.5e-3) else 0
                    refocus_time = max(refocus_time_g, refocus_time_d) if (idq == 0 and not swap_loop) else refocus_time_d
                    time += refocus_time + operation.duration
                time += self.t_meas
                durations_per_node[node] = time

            duration = max(durations_per_node.values())
            # duration = (operation.duration * num_op + self.t_meas + swap_dur)
            #             + refocus_time_cq + refocus_time_tq)
            # print("\n\n\n\n\nStabilizer measurement duration estimations:")
            # print(f"Total duration so far: {self.total_duration}.")
            # print(f"Pure operation duration: {duration}.")
            # print(f"Durations per node: {durations_per_node}.")
            # print(f"New total duration after operation (estimation): {self.total_duration + max(durations_per_node.values())}.")
            # print(f"Cut off time: {self.cut_off_time}.")
            # print("\n\n\n\n\n")
            if not self._check_if_operation_within_cut_off(duration, nodes=nodes):
                return

        # Cut-off holds for the GHZ creation, stabilizer measurement should always be fully performed if reached
        self._circuit_operations_ended = True
        self.get_state_fidelity() if len(self.nodes) > 1 else None

        # cqubit_indices = [None] * len(nodes)
        # node_order = [None] * len(nodes)
        # for i, node in enumerate(nodes):
        #     cq = int(set(self.qubits.keys()).intersection(self.nodes[node].qubits)
        #              .difference(self._uninitialised_qubits).difference(self.data_qubits).pop())
        #     cqubit_comb_mat = self._qubit_density_matrix_lookup[cq][1]
        #     cqubit_index = cqubit_comb_mat.index(cq)
        #     node_order[i] = node
        #     cqubit_indices[i] = cqubit_index
        #
        # nodes = [x for _, x in sorted(zip(cqubit_indices, node_order))]

        for i, node in enumerate(nodes):
            tqubit = [qubit for qubit in tqubits if qubit in self.nodes[node].qubits] if self.nodes and all(tqubits) \
                else tqubits[i]
            node_measurement(node, operation, cqubit, tqubit, swap, electron_qubit)

        self.cut_off_time_reached = False

    """
        ---------------------------------------------------------------------------------------------------------
                                            Gate Noise Methods
        ---------------------------------------------------------------------------------------------------------  
    """

    def _N_depolarising_channel(self, p_g, tqubit, density_matrix, num_qubits, times=1, SWAP=False):
        """
            Private method to apply noise to the single qubit gates. This is done according to the equation

                N(rho) = (1-p_g) * rho + p_g/3 SUM_A [A * rho * A^], --> A in {X, Y, Z}

            in which '#' is the Kronecker product and ^ is the dagger (Hermitian conjugate).

            Parameters
            ----------
            p_g : float [0-1]
                Indicates the amount of gate noise applied
            tqubit: int
                Integer that indicates the target qubit. Note that the qubit counting starts at 0.
            density_matrix : csr_matrix
                Density matrix to which the noise should be applied to.
            num_qubits : int
                Number of qubits of which the density matrix is composed.
        """
        return self._noise.noise_maps.N_depolarising_channel(self, p_g, tqubit, density_matrix, num_qubits, times, SWAP)

    def _N_two_qubit_gate(self, p_g, cqubit, tqubit, density_matrix, num_qubits, times=1):
        """
            Private method to apply noise to the single qubit gates. This is done according to the equation

                N(rho) = (1-p_g)*rho + p_g/15 SUM_A SUM_B [(A # B) rho (A # B)^], --> {A, B} in {X, Y, Z, I}

            in which '#' is the Kronecker product and ^ is the dagger (Hermitian conjugate).

            Parameters
            ----------
            p_g : float [0-1]
                Indicates the amount of gate noise applied
            cqubit: int
                Integer that indicates the control qubit. Note that the qubit counting starts at 0.
            tqubit: int
                Integer that indicates the target qubit. Note that the qubit counting starts at 0.
            density_matrix : csr_matrix
                Density matrix to which the noise should be applied to.
            num_qubits : int
                Number of qubits of which the density matrix is composed.
        """
        return self._noise.noise_maps.N_two_qubit_gate(self, p_g, cqubit, tqubit, density_matrix, num_qubits, times)

    def _N_network(self, density_matrix, F_link, network_noise_type):
        """
            Parameters
            ----------
            density_matrix : sparse matrix
                Density matrix of the ideal Bell-pair.
            F_link : float [0-1]
                Amount of network noise present in the system.
            network_noise_type: int {0, 1}
                Type of network noise that is requested
        """
        return self._noise.noise_maps.N_network(density_matrix, F_link, network_noise_type)

    def _N_preparation(self, state, p_prep):
        return self._noise.noise_maps.N_preparation(state, p_prep)

    def _N_decoherence(self, qubits=None, sub_circuit=None, sub_circuit_concurrent=False, decoherence=True):
        self._noise.decoherence.N_decoherence(self, qubits, sub_circuit, sub_circuit_concurrent, decoherence)

    def _N_amplitude_damping_channel(self, tqubit, density_matrix, num_qubits, waiting_time, T, p=1/2):
        return self._noise.noise_maps.N_amplitude_damping_channel(self, tqubit, density_matrix, num_qubits,
                                                                  waiting_time, T, p)

    def _N_phase_damping_channel(self, tqubit, density_matrix, num_qubits, waiting_time, T, alpha=1):
        return self._noise.noise_maps.N_phase_damping_channel(self, tqubit, density_matrix, num_qubits, waiting_time,
                                                              T, alpha)

    def _N_combined_amplitude_phase_damping_channel(self, tqubit, density_matrix, num_qubits, waiting_time, T_a, T_p):
        return self._noise.noise_maps.N_combined_amplitude_phase_damping_channel(self, tqubit, density_matrix,
                                                                                 num_qubits, waiting_time, T_a, T_p)

    def _N_dephasing_channel(self, tqubit, density_matrix, num_qubits, p):
        return self._noise.noise_maps.N_dephasing_channel(self, tqubit, density_matrix, num_qubits, p)

    """
        ---------------------------------------------------------------------------------------------------------
                                                Measurement Methods
        ---------------------------------------------------------------------------------------------------------   
    """
    @handle_none_parameters
    def measure_first_N_qubits(self, N, measure=0, uneven_parity=False, noise=None, p_m=None, basis="X",
                               basis_transformation_noise=None, probabilistic=None, user_operation=True,
                               decoherence=None):
        """
            Method measures the first N qubits, given by the user, all in the 0 or 1 state.
            This will thus result in an even parity measurement. To also be able to enforce uneven
            parity measurements this should still be built!
            The density matrix of the system will be changed according to the measurement outcomes.

            *** MEASURED QUBITS WILL BE ERASED FROM THE SYSTEM AFTER MEASUREMENT, THIS WILL THUS
            DECREASE THE AMOUNT OF QUBITS IN THE SYSTEM WITH 'N' AS WELL. THE QUBIT INDICES WILL THEREFORE ALSO
            INCREASE WITH 'N', WHICH IS IMPORTANT FOR FUTURE OPERATIONS ***

            Parameters
            ----------
            N : int
                Specifies the first n qubits that should be measured.
            measure : int [0 or 1], optional, default=0
                The measurement outcome for the qubits, either 0 or 1.
            noise : bool, optional, default=None
                 Whether or not the measurement contains noise.
            p_m : float [0-1], optional, default=None
                The amount of measurement noise that is present (if noise is present).
            basis : str ["X" or "Z"], optional, default="X"
                Whether the measurement should be done in the X-basis or in the computational basis (Z-basis)
            basis_transformation_noise : bool, optional, default=False
                Whether the H-gate that is applied to transform the basis in which the qubit is measured should be
                noisy (True) or noiseless (False)
            probabilistic : bool, optional, default=False
                Whether the measurement should be probabilistic. In case of an uneven parity in the outcome of the
                measurements, the method will return False else it returns True
            user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.
        """
        if user_operation:
            self._user_operation_order.append({"measure_first_N_qubits": [N, measure, noise, p_m, basis,
                                                                          basis_transformation_noise]})

        measurement_outcomes = []

        for qubit in range(N):
            if basis == "X":
                # Do not let the method draw itself, since the qubit will not be removed from the circuit drawing
                self.H(0, noise=basis_transformation_noise, draw=False, user_operation=False)

            qubit_density_matrix, _ = self._qubit_density_matrix_lookup[qubit]

            if probabilistic:
                prob_0, density_matrix_0 = self._measurement_first_qubit(qubit_density_matrix, measure=0, noise=noise,
                                                                         p_m=p_m)
                prob_1, density_matrix_1 = self._measurement_first_qubit(qubit_density_matrix, measure=1, noise=noise,
                                                                         p_m=p_m)

                density_matrices = [density_matrix_0, density_matrix_1]
                outcome = get_value_by_prob([0, 1], [prob_0, prob_1])
                new_density_matrix = density_matrices[outcome]
            else:
                outcome = measure
                if uneven_parity and qubit == 0:
                    outcome = abs(measure - 1)

                new_density_matrix = self._measurement_first_qubit(qubit_density_matrix, outcome, noise=noise,
                                                                   p_m=p_m)[1]

            self._set_density_matrix(0, new_density_matrix)
            self._correct_lookup_for_measurement_top()
            self._update_uninitialised_qubit_register([qubit], update_type="add")
            measurement_outcomes.append(outcome)
            # Remove the measured qubit from the system characteristics and add the operation to the draw_list
            self.num_qubits -= 1
            self.d = 2 ** self.num_qubits
            self._add_draw_operation("M_{}:{}".format(basis, outcome), qubit, noise)

            if noise and decoherence:
                self._effective_measurements += (1+qubit)
                times = int(math.ceil(self.t_meas/self.time_step))
                self._N_decoherence([], times=times)
                self._increase_duration(self.t_meas, qubit)
                self._effective_measurements -= (1+qubit)

        self._effective_measurements += N
        measurement_outcomes = iter(measurement_outcomes)
        parity_outcome = [True if i == j else False for i, j in zip(measurement_outcomes, measurement_outcomes)]
        return all(parity_outcome)

    def _measurement_first_qubit(self, density_matrix, measure=0, noise=None, p_m=0., no_normalisation=False):
        """
            Private method that is used to measure the first qubit (qubit 0) in the system and removing it
            afterwards. If a 0 is measured, the upper left quarter of the density matrix 'survives'
            and if a 1 is measured the lower right quarter of the density matrix 'survives'.
            Noise is applied according to the equation

                rho_noisy = (1-p_m) * rho_p-correct + p_m * rho_p-incorrect,

            where 'rho_p-correct' is the density matrix that should result after the measurement and
            'rho_p-incorrect' is the density matrix that results when the opposite measurement outcome
            is measured.

            Parameters
            ----------
            density_matrix : csr_matrix
                Density matrix to which the top qubit should be measured.
            measure : int [0 or 1], optional, default=0
                The measurement outcome for the qubit, either 0 or 1.
            noise : bool, optional, default=None
                 Whether or not the measurement contains noise.
            p_m : float [0-1], optional, default=0.
                The amount of measurement noise that is present (if noise is present).
        """
        return self._operations.measurement_operations.measurement_first_qubit(density_matrix, measure, noise, p_m,
                                                                               no_normalisation=no_normalisation)

    @determine_qubit_index(parameter_positions=[1])
    @skip_if_cut_off_reached
    @handle_none_parameters
    def measure(self, measure_qubits, outcome=0, uneven_parity=False, basis="X", noise=None, p_m=None, p_m_1=None,
                probabilistic=None, basis_transformation_noise=None, decoherence=None,
                user_operation=True):
        """
            Measurement that can be applied to any qubit.

            Parameters
            ----------
            qubit : int
                Indicates the qubit to be measured (qubit count starts at 0)
            outcome : int [0 or 1], optional, default=None
                The measurement outcome for the qubit, either 0 or 1. If None, the method will choose
                randomly according to the probability of the outcome.
            basis : str ["X" or "Z"], optional, default="X"
                Whether the qubit is measured in the X-basis or in the computational basis (Z-basis)
            basis_transformation_noise : bool, optional, default=False
                Whether the H-gate that is applied to transform the basis in which the qubit is measured should be
                noisy (True) or noiseless (False)
            user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.
        """
        if user_operation:
            self._user_operation_order.append({"measure": [measure_qubits, outcome, basis]})

        if type(measure_qubits) == int:
            measure_qubits = [measure_qubits]

        measurement_outcomes = []

        for i, qubit in enumerate(measure_qubits):
            if noise and decoherence:
                self._N_decoherence([qubit])

            if basis == "X":
                self.H(qubit, noise=basis_transformation_noise if noise else False, user_operation=False, draw=False)

            density_matrix, qubits, rel_qubit, rel_num_qubits = self._get_qubit_relative_objects(qubit)
            start_time = time.time()

            # If no specific measurement outcome is given it is chosen by the hand of the probability
            if probabilistic:
                if rel_qubit == 0:
                    prob_0, density_matrix_0 = self._measurement_first_qubit(density_matrix, measure=0, noise=noise,
                                                                             p_m=p_m)
                    prob_1, density_matrix_1 = self._measurement_first_qubit(density_matrix, measure=1, noise=noise,
                                                                             p_m=p_m_1 if p_m_1 is not None else p_m)
                else:
                    # self.append_print_lines("\nWarning: The measurement of a qubit that is not the first qubit of the "
                    #                         "density matrix is slow. The order of the density matrix is: {}. You want "
                    #                         "to measure qubit {}.".format(qubits, qubit))
                    density_matrix_0 = self._measure_arbitrary_qubit(rel_qubit, density_matrix, outcome=0)
                    density_matrix_1 = self._measure_arbitrary_qubit(rel_qubit, density_matrix, outcome=1)

                    if noise:
                        # Keep p_m_1 on None, such that the if loop below is evaluated correctly
                        p_meas_1 = p_m_1 if p_m_1 is not None else p_m
                        density_matrix_0_int = (1-p_m) * density_matrix_0 + p_m * density_matrix_1
                        density_matrix_1_int = (1-p_meas_1) * density_matrix_1 + p_meas_1 * density_matrix_0
                    else:
                        density_matrix_0_int = density_matrix_0
                        density_matrix_1_int = density_matrix_1

                    prob_0 = trace(density_matrix_0_int)
                    density_matrix_0 = density_matrix_0_int / prob_0

                    prob_1 = trace(density_matrix_1_int)
                    density_matrix_1 = density_matrix_1_int / prob_1

                probs = [prob_0, prob_1]
                # if round(sum(probs), 10) != 1 and p_m_1 is None:
                #     print(prob_0, prob_1, p_m)
                #     raise ValueError("Probabilities do not sum to 1. Sum is {}".format(round(sum(probs), 10)))

                density_matrices = [density_matrix_0, density_matrix_1]
                outcome_new = get_value_by_prob([0, 1], [prob_0, prob_1])

                new_density_matrix = density_matrices[outcome_new]
            else:
                outcome_new = outcome
                if uneven_parity and i == 0:
                    outcome_new = outcome ^ 1

                if rel_qubit == 0:
                    prob, new_density_matrix = self._measurement_first_qubit(density_matrix, measure=outcome_new,
                                                                             noise=noise, p_m=p_m)
                else:
                    # self.append_print_lines("\nWarning: The measurement of a qubit that is not the first qubit of the "
                    #                         "density matrix is slow. The order of the density matrix is: {}. You want "
                    #                         "to measure qubit {}.".format(qubits, qubit))
                    new_density_matrix = self._measure_arbitrary_qubit(rel_qubit, density_matrix,
                                                                             outcome=outcome_new)

                    if noise:
                        wrong_density_matrix = self._measure_arbitrary_qubit(rel_qubit, density_matrix,
                                                                                outcome=outcome_new ^ 1)
                        new_density_matrix = (1 - p_m) * new_density_matrix + p_m * wrong_density_matrix

                    prob = trace(new_density_matrix)
                    new_density_matrix = new_density_matrix / trace(new_density_matrix)

                probs = [prob, prob]
                # if prob == 0:
                #     raise ValueError("Measuring a state with 0 probability cannot be dealt with. Please write"
                #                      " a valid circuit.")

            if basis == "X":
                density_matrix_measured = CT(ket_p) if outcome_new == 0 else CT(ket_m)
                self._correct_lookup_for_measurement_any(qubit, qubits, density_matrix_measured, new_density_matrix)
            else:
                density_matrix_measured = CT(ket_0) if outcome_new == 0 else CT(ket_1)
                self._correct_lookup_for_measurement_any(qubit, qubits, density_matrix_measured, new_density_matrix)

            measurement_outcomes.append(outcome_new)
            self._update_uninitialised_qubit_register([qubit], update_type="add")
            self._add_draw_operation("M_{}:{}-{:3.4f}%".format(basis, outcome_new, probs[outcome_new]*100), qubit,
                                     noise)

            # Please note that the decoherence is implemented after the H gate. When the H gate should be taken into
            # account for decoherence small implementation alteration is necessary.
            self._increase_duration(self.t_meas, [qubit])

            # if rel_qubit > 0:
            #     print(f"Qubit index measured: {rel_qubit}, size object: {len(qubits)}, calc_time: {time.time() - start_time}.")


        return measurement_outcomes

    def _measure_arbitrary_qubit(self, qubit, density_matrix, outcome, keep_qubit=False):
        """
            Method returns the probability and new density matrix for the given measurement outcome of the given qubit.

            *** THIS METHOD IS VERY SLOW FOR LARGER SYSTEMS, SINCE IT DETERMINES THE SYSTEM STATE AFTER
            THE MEASUREMENT BY DIAGONALISING THE DENSITY MATRIX ***

            To explain the approach taken, consider that:
                    |a_1|   |b_1|   |c_1|   |a_1 b_1 c_1|                        |a_1 b_1 c_1 a_1 b_1 c_1 ... |
                    |   | * |   | * |   | = |a_1 b_1 c_2|  ---> density matrix:  |a_1 b_1 c_1 a_1 b_1 c_2 ... |
                    |a_2|   |b_2|   |c_2|   |a_1 b_2 c_1|                        |a_1 b_1 c_1 a_1 b_2 c_1 ... |
                                            |    ...    |                        |          ...               |

            When the second qubit (with the elements b_1 and b_2) is measured and the outcome is a 1, it means
            that b_1 is 0 and b_2 is 1. This thus means that all elements of the density matrix that are built up
            out of b_1 elements are 0 and only the elements not containing b_1 elements survive. This way a new
            density matrix can be constructed of which the trace is equal to the probability of this outcome occurring.
            Pattern of the elements across the density matrix can be compared with a chess pattern, where the square
            dimension reduce by a factor of 2 with the qubit number.

            Parameters
            ----------
            qubit : int
                qubit for which the measurement outcome probability should be measured
            density_matrix : csr_matrix
                Density matrix to which the qubit belongs
            outcome : int [0,1]
                Outcome for which the probability and resulting density matrix should be calculated
        """
        return self._operations.measurement_operations._measure_arbitrary_qubit(qubit, density_matrix,
                                                                                outcome, keep_qubit)

    """
        ---------------------------------------------------------------------------------------------------------
                                                Superoperator Methods
        ---------------------------------------------------------------------------------------------------------     
    """

    def get_superoperator(self, qubits, proj_type, *, stabilizer_protocol=False, save_noiseless_density_matrix=False,
                          combine=True, most_likely=True, print_to_console=True, file_name_noiseless=None,
                          file_name_measerror=None, no_color=False, csv_file_name=None,
                          use_exact_path=False, idle_data_qubit=False, protocol_name=None, return_dataframe=True):
        """
            Returns the superoperator for the system. The superoperator is determined by taking the fidelities
            of the density matrix of the system [rho_real] and the density matrices obtained with any possible
            combination of error on the 4 data qubits in a noiseless version of the system
            [(ABCD) rho_ideal (ABCD)^]. Thus in equation form

            F[rho_real, (ABCD) * rho_ideal * (ABCD)^], {A, B, C, D} in {X, Y, Z, I}

            The fidelity is equal to the probability of this specific error, the combination of (ABCD), happening.

            Parameters
            __________
            qubits : list
                List of qubits of which the superoperator should be calculated. Only for these qubits it will be
                checked if certain errors occured on them. This is necessary to specify in case the circuit contains
                ancilla qubits that should not be evaluated. **The index of the qubits should be the index of the
                resulting density matrix, thus in case of measurements this can differ from the initial indices!!**
            proj_type : str, options: "X" or "Z"
                Specifies the type of stabilizer for which the superoperator should be calculated. This value is
                necessary for the postprocessing of the superoperator results if 'combine' is set to True and used if
                stabilizer_protocol is set to True.
            stabilizer_protocol : bool, optional, default=False
                If the superoperator is calculated for a stabilizer measurement protocol (for example Stringent or
                Expedient).
            save_noiseless_density_matrix : bool, optional, default=True
                Whether or not the calculated noiseless (ideal) version of the circuit should be saved.
                This saved matrix will a next time be used for speedup if the same system is analysed with this method.
            combine : bool, optional, default=True
                Combines the error configuration on the data qubits that are equal up to permutation. This effectively
                means that for example [I, I, I, X] and [X, I, I, I] will be combined to one term [I, I, I, X] with the
                probabilities summed.
            most_likely : bool, optional, default=True
                Will choose the most likely configuration of degenerate configurations. This effectively means that the
                configuration with the highest amount of identity operators will be chosen. Only works if 'combine' is
                also set to True.
            print_to_console : bool, optional, default=True
                Whether the result should be printed in a clear overview to the console.
            file_name_noiseless : str, optional, default=None
                qasm_file name of the noiseless variant of the density matrix of the noisy system. Use this option if
                density matrix has been named manually and this one should be used for the calculations.
            file_name_measerror : str, optional, default=None
                qasm_file name of the noiseless variant with measurement error of the density matrix of the noisy
                system. Use this option if density matrix has been named manually and this one should be used for the
                calculations.
            no_color : bool, optional, default=False
                Indicates if the output of the superoperator to the console should not contain color, when for example
                the used console does not support color codes.
            to_csv : bool, optional, default=False
                Whether the results of the superoperator should be saved to a csv file.
            csv_file_name : str, optional, default=None
                The file name that should be used for the csv file. If not supplied, the system will use generic naming
                and the file will be saved to the 'oopsc/superoperator/csv_files' folder.
            use_exact_path : bool, optional, default=False
                If True, the csv_file_name string will be treated as an exact path to the file and can thus be saved
                anywhere.
        """
        noiseless_density_matrix = self._get_noiseless_density_matrix(stabilizer_protocol=stabilizer_protocol,
                                                                      proj_type=proj_type,
                                                                      save=save_noiseless_density_matrix,
                                                                      file_name=file_name_noiseless,
                                                                      qubits=qubits,
                                                                      idle_data_qubit=idle_data_qubit)
        measerror_density_matrix = self._get_noiseless_density_matrix(measure_error=True,
                                                                      stabilizer_protocol=stabilizer_protocol,
                                                                      proj_type=proj_type,
                                                                      save=save_noiseless_density_matrix,
                                                                      file_name=file_name_measerror,
                                                                      qubits=qubits,
                                                                      idle_data_qubit=idle_data_qubit)
        superoperator = []

        # Get all combinations of gates ([X, Y, Z, I]) possible on the given qubits
        total_density_matrix, qubits_matrix = self.get_combined_density_matrix(qubits)
        superoperator_decomposition = self._create_superoperator_decomposition(qubits, qubits_matrix)

        for kraus_operator, error_matrix in superoperator_decomposition.items():
            error_density_matrix, me_error_density_matrix = self._get_error_density_matrices(kraus_operator,
                                                                                             stabilizer_protocol,
                                                                                             noiseless_density_matrix,
                                                                                             measerror_density_matrix,
                                                                                             error_matrix)
            fid_no_me = fidelity_elementwise(error_density_matrix, total_density_matrix)
            fid_me = fidelity_elementwise(me_error_density_matrix, total_density_matrix)

            if fid_me > 1e-12:
                superoperator.append(SuperoperatorElement(fid_me, True, list(kraus_operator), me_error_density_matrix))
            if fid_no_me > 1e-12:
                superoperator.append(SuperoperatorElement(fid_no_me, False, list(kraus_operator), error_density_matrix))

        # Possible post-processing options for the superoperator
        if self.combine and not idle_data_qubit and not self.cut_off_time_reached:
            superoperator = self._fuse_equal_config_up_to_permutation(superoperator)
        if combine and most_likely:
            superoperator = self._remove_not_likely_configurations(superoperator)

        if print_to_console:
            self._print_superoperator(superoperator, no_color)

        if return_dataframe:
            superoperator_dict = {proj_type: superoperator}
            superoperator_dataframe = self._superoperator_to_dataframe(superoperator_dict,
                                                                       file_name=csv_file_name,
                                                                       use_exact_path=use_exact_path,
                                                                       protocol_name=protocol_name,
                                                                       qubit_order=qubits)
            return superoperator, superoperator_dataframe
        else:
            return superoperator

    @staticmethod
    def _return_QC_object(num_qubits, init):
        return QuantumCircuit(num_qubits, init)

    def _get_noiseless_density_matrix(self, stabilizer_protocol, proj_type, measure_error=False, save=True,
                                      file_name=None, qubits=None, idle_data_qubit=None):
        """
            Private method to calculate the noiseless variant of the density matrix.
            It traverses the operations on the system by the hand of the '_user_operation_order' attribute. If the
            noiseless matrix is present in the 'saved_density_matrices' folder, the method will use this instead
            of recalculating the circuits. When no file name is given, the noiseless density matrix is searched for
            based on the user operations applied to the noisy circuit (see method '_absolute_file_path_from_circuit').

            Parameters
            ----------
            stabilizer_protocol : bool
                If the noiseless density matrix is one of a stabilizer measurement protocol (for example Stringent or
                Expedient). This leads to a speed-up, since the noiseless density matrix can be assumed equal to the
                noiseless density matrix of a stabilizer measurement in a monolithic architecture.
            proj_type : str, options: "X" or "Z"
                Specifies the type of stabilizer for which the superoperator should be calculated.
            measure_error: bool, optional, default=False
                Specifies if the measurement outcome should be opposite of the ideal circuit.
            save : bool
                Whether or not the calculated noiseless version of the circuit should be saved.
                This saved matrix will a next time be used if the same system is analysed wth this method.
            file_name : str
                File name of the density matrix qasm_file that should be used as noiseless density matrix. Note that
                specifying this with an existing qasm_file name will directly return this density matrix.

            Returns
            -------
            noiseless_density_matrix : sparse matrix
                The density matrix of the current system, but without noise
        """
        return self._superoperator.superoperator_methods.get_noiseless_density_matrix(self,
                                                                                      stabilizer_protocol,
                                                                                      proj_type,
                                                                                      measure_error,
                                                                                      save,
                                                                                      file_name,
                                                                                      qubits=qubits,
                                                                                      idle_data_qubit=idle_data_qubit)

    def _file_name_from_circuit(self, measure_error=False, general_name="circuit", extension=""):
        """
            Returns the file name of the Quantum Circuit based on the initial parameters and the user operations
            applied to the circuit.

            Parameters
            ----------
            measure_error : bool, optional, default=False
                This variable is used for the case of density matrix naming for the noiseless density matrices.
                This ensures explicit naming of a density matrix containing a measurement error. For more info see
                the 'get_superoperator' and '_get_noiseless_density_matrix'.
            general_name : str, optional, default="circuit"
                To specify the file name more, one can add a custom start of the file name. Default is 'circuit'.
            extension : str, optional, default=""
                Use this argument if the file name needs a specific type of extension. By default, it will NOT append
                an extension.
        """
        # Create an hash id, based on the operation and there order on the system and use this for the filename
        init_params_id = str(self._init_parameters)
        user_operation_id = "".join(["{}{}".format(list(d.keys())[0], list(d.values())[0])
                              for d in self._user_operation_order])
        total_id = init_params_id + user_operation_id
        hash_id = hashlib.sha1(total_id.encode("UTF-8")).hexdigest()[:10]
        file_name = "{}{}_{}{}".format(general_name, ("_me" if measure_error else ""), hash_id, extension)

        return file_name

    def _absolute_file_path_from_circuit(self, measure_error, kind="dm"):
        """
            Returns a file path to a file based on what kind of object needs to be saved. The kind of files that
            are supported, including their standard directory can be found below in the parameters section.

            Parameters
            ----------
            measure_error : bool
                True if the ideal density matrix containing a measurement error should be returned.
            kind : str, optional, default="dm"
                Kind of file of which the absolute file path should be obtained. In this moment in time the options are
                    * "dm"
                        Density matrix file. Directory will be the 'saved_density_matrix' folder.
                    * "qasm"
                        Qasm file. Directory will be the 'latex_circuit' folder.
                    * "os"
                        Superoperator file. Directory will be the 'oopsc/superoperator/csv_files/' folder.

            Returns
            -------
            file_name : str
                Returns the file_name of the ideal (or ideal up to measurement error if parameter 'measure_error' is set
                to True) density matrix of the noisy QuantumCircuit object.
        """
        if kind == "dm":
            file_name = self._file_name_from_circuit(measure_error, general_name="density_matrix", extension=".npz")
            file_path = os.path.join(os.path.dirname(__file__), "_superoperator", "saved_density_matrices", file_name)
        elif kind == "qasm":
            file_name = self._file_name_from_circuit(measure_error, extension=".qasm")
            file_path = os.path.join(os.path.dirname(__file__), "_draw", file_name)
        elif kind == "so":
            file_name = self._file_name_from_circuit(measure_error, general_name="superoperator", extension=".csv")
            file_path = os.path.join(SuperoperatorElement.file_path(), "csv_files", file_name)
        else:
            file_name = self._file_name_from_circuit(measure_error, extension=".npz")
            file_path = os.path.join(os.getcwd(), file_name)
            self._print_lines.append("\nkind: '{}' was not recognized. Please see method documentation for supported kinds. "
                  "File path is now: '{}'".format(kind, file_path))

        return file_path

    def _create_superoperator_decomposition(self, qubits, qubits_matrix):
        """
            Method returns a list containing all the possible combinations of Pauli matrix gates
            that can be applied to the specified qubits.

            Parameters
            ----------
            qubits : list
                A list of the qubit indices for which all the possible combinations of Pauli matrix gates
                should be returned.

            Returns
            -------
            all_gate_combinations : list
                list of all the qubit gate arrangements that are possible for the specified qubits.

            Examples
            --------
            self._all_single_qubit_gate_possibilities([0, 1]), then the method will return

            [[X, X], [X, Y], [X, Z], [X, I], [Y, X], [Y, Y], [Y, Z] ....]

            in which, in general, A -> {"A": single_qubit_A_gate_object} where A in {X, Y, Z, I}.
        """
        return self._superoperator.superoperator_methods.create_superoperator_decomposition(self, qubits, qubits_matrix)

    def _get_error_density_matrices(self, kraus_operator, stabilizer_protocol, noiseless_density_matrix,
                                    measerror_density_matrix, error_matrix):
        return self._superoperator.superoperator_methods.get_error_density_matrices(self, kraus_operator,
                                                                                    stabilizer_protocol,
                                                                                    noiseless_density_matrix,
                                                                                    measerror_density_matrix,
                                                                                    error_matrix)

    def _fuse_equal_config_up_to_permutation(self, superoperator):
        """
            Post-processing method for the superoperator which fuses similar Pauli-error configurations inside the
            superoperator up to permutation. This is done by sorting the error configurations and comparing them after.
            If equal, the probabilities will be summed and saved as one new entry.

            Parameters
            ----------
            superoperator : list
                Superoperator obtained in the 'get_superoperator' method. Containing all the probabilities of the
                possible Pauli-error configurations on the data qubits.
            proj_type : str ['Z' or 'X']
                The stabilizer type of the to be analysed superoperator. This is necessary in order to determine the
                degenerate configurations, for example [I,I,Z,Z] and [Z,Z,I,I] that on first sight look as if they have
                to be treated equally, but in fact they are degenerate and the probabilities should not be summed (since
                this will cause the total probability to exceed 1).

            Returns
            -------
            sorted_superoperator : list
                New superoperator that now contains only one entry per similar Pauli-error configurations up to
                permutations. The new probability of this one entry is the summed probability of all the similar
                configurations that were fused.

            Example
            -------
            The superoperator contains, among others, the configurations [X,I,I,I], [I,X,I,I], [I,I,X,I] and [I,I,I,X].
            These Pauli-error configurations on the data qubits are similar up to permutations. The method will
            eventually end up making one entry, namely [I,I,I,X], in the returned new superoperator. The according
            probability will be equal to the sum of the probabilities of the 4 configurations.
        """
        return self._superoperator.superoperator_methods.fuse_equal_config_up_to_permutation(superoperator)

    def _fuse_config_cut_off_time_reached(self, superoperator):
        return self._superoperator.superoperator_methods.fuse_config_cut_off_time_reached(superoperator)

    def _remove_not_likely_configurations(self, superoperator):
        """
            Post-processing method for the superoperator which removes the degenerate configurations of the
            superoperator based on the fact that the Pauli-error configuration with the most 'I' operations is the most
            likely to have occurred.

            Parameters
            ----------
            superoperator : list
                Superoperator obtained in the 'get_superoperator' method. Containing all the probabilities of the
                possible Pauli-error configurations on the data qubits.

            Returns
            -------
            sorted_superoperator : list
                Returns the superopertor with the not-likely degenerate configurations entries removed. Note that is a
                full removal, thus the probability is removed from the list (and not summed as in the 'fuse'
                post-processing).

            Example
            -------
            Consider the superoperator with, among others, the degenerate entries [Z,Z,Z,X] and [I,I,I,X]. In this
            method, it is assumed that the configuration [I,I,I,X] is more likely to have occurred than the other and
            therefore only this configuration is kept in the returned superoperator. Effectively, this means that the
            [Z,Z,Z,X] is removed from the superoperator together with the according probability.
        """
        return self._superoperator.superoperator_methods.remove_not_likely_configurations(superoperator)

    def _print_superoperator(self, superoperator, no_color):
        """ Prints the superoperator in a clear way to the console """
        self._superoperator.superoperator_methods.print_superoperator(self, superoperator, no_color)

    def _superoperator_to_dataframe(self, superoperator_dict, file_name=None, use_exact_path=False,
                                    protocol_name=None, qubit_order=None, **kwargs):
        """
            Save the obtained superoperator results to a csv file format that is suitable with the superoperator
            format that is used in the (distributed) surface code simulations.

            superoperator_dict : dictionary with keys str (stabilizer types than have been analysed, options in
                {"X", "Z"}) and values lists containing SuperoperatorElement objects
            file_name : str, optional, default=None
                User specified file name that should be used to save the csv file with. The file will always be stored
                in the 'csv_files' directory, so the string should NOT contain any '/'. These will be removed.
        """
        return self._superoperator.superoperator_methods.superoperator_to_dataframe(self, superoperator_dict,
                                                                                    file_name, use_exact_path,
                                                                                    protocol_name, qubit_order,
                                                                                    **kwargs)

    def get_state_fidelity(self, qubits=None, compare_matrix=None, set_ghz_fidelity=True):
        return self._superoperator.superoperator_methods.get_state_fidelity(self, qubits, compare_matrix,
                                                                            set_ghz_fidelity)

    """
        ----------------------------------------------------------------------------------------------------------
                                            Circuit drawing Methods
        ----------------------------------------------------------------------------------------------------------     
    """

    def draw_circuit(self, no_color=False, color_nodes=False):
        """ Draws the circuit that corresponds to the operation that have been applied on the system,
        up until the moment of calling. """
        legenda = "\n--- Circuit ---\n\n #: Bell-pair, o: control qubit " \
                  "(with target qubit at same level), [X,Y,Z,H]: gates, M: measurement,"\
                  " {}: noisy operation (gate/measurement)\n".format("~" if no_color else colored("~", 'red'))
        init = self._draw_init(no_color)
        self._draw_operations(init, no_color)
        init[-1] += "\n\n"
        if not no_color and color_nodes and self.nodes:
            self._color_qubit_lines(init)
        self._print_lines.append(legenda)
        self._print_lines.extend(init)
        if not self._thread_safe_printing:
            self.print()

    def draw_circuit_latex(self, meas_error=False):
        qasm_file_name = self._create_qasm_file(meas_error)
        create_pdf_from_qasm(qasm_file_name, qasm_file_name.replace(".qasm", ".tex"))

    def _draw_init(self, no_color):
        """ Returns an array containing the visual representation of the initial state of the qubits. """
        return self._draw.draw_circuit.draw_init(self, no_color)

    def _draw_operations(self, init, no_color):
        """ Adds the visual representation of the operations applied on the qubits """
        self._draw.draw_circuit.draw_operations(self, init, no_color)

    def _color_qubit_lines(self, init):
        self._draw.draw_circuit.color_qubit_lines(self, init)

    def _create_qasm_file(self, meas_error):
        """
            Method constructs a qasm file based on the 'self._draw_order' list. It returns the file path to the
            constructed qasm file.

            Parameters
            ----------
            meas_error : bool
                Specify if there has been introduced a measurement error on purpose to the QuantumCircuit object.
                This is needed to create the proper file name.
        """
        return self._draw.draw_circuit_latex.create_qasm_file(self, meas_error)

    def _add_draw_operation(self, operation, qubits, noise=False, sub_circuit=None, sub_circuit_concurrent=False):
        """
            Adds an operation to the draw order list.

            Notes
            -----
            **Note** :
                Since measurements and additions of qubits change the qubit indices dynamically, this will be
                accounted for in this method when adding a draw operation. The '_effective_measurement' attribute keeps
                track of how many qubits have effectively been measured, which means they have not been reinitialised
                after measurement (by creating a Bell-pair at the top or adding a top qubit). The '_measured_qubits'
                attribute contains all the qubits that have been measured and are not used anymore after (in means of
                the drawing scheme).

            **2nd Note** :
                Please consider that the drawing of the circuit can differ from reality due to this dynamic
                way of changing the qubit indices with measurement and/or qubit addition operations. THIS EFFECTIVELY
                MEANS THAT THE CIRCUIT REPRESENTATION MAY NOT ALWAYS PROPERLY REPRESENT THE APPLIED CIRCUIT WHEN USING
                MEASUREMENTS AND QUBIT ADDITIONS.
        """
        self._draw.draw_circuit.add_draw_operation(self, operation, qubits, noise, _current_sub_circuit=sub_circuit,
                                                   sub_circuit_concurrent=sub_circuit_concurrent)

    def level_circuit_drawing(self):
        return self._draw_order.append(['LEVEL', None, None])

    def _correct_drawing_for_n_top_qubit_additions(self, n=1):
        """
            Corrects the self._draw_order list for addition of n top qubits.

            When a qubit gets added to the top of the stack, it gets the index 0. This means that the indices of the
            already existing qubits increase by 1. This should be corrected for in the self._draw_order list, since
            the qubit references used the 'old' qubit index.

            *** Note that for the actual qubit operations that already have been applied to the system the addition of
            a top qubit is not of importance, but after addition the user should know this index change for future
            operations ***

            Parameters
            ----------
            n : int, optional, default=1
                Amount of added top qubits that should be corrected for.
        """
        self._draw.draw_circuit.correct_drawing_for_n_top_qubit_additions(self, n)

    def correct_drawing_for_circuit_fusion(self, other_draw_order, num_qubits_other):
        self._draw.draw_circuit.correct_drawing_for_circuit_fusion(self, other_draw_order, num_qubits_other)

    def save_density_matrix(self, filename=None):
        if filename is None:
            filename = self._absolute_file_path_from_circuit(measure_error=False, kind='dm')

        sp.save_npz(filename, self.total_density_matrix())

        self._print_lines.append("\nFile successfully saved at: {}".format(filename))

    def fuse_circuits(self, other):
        if type(other) != QuantumCircuit:
            raise ValueError("Other should be of type QuantumCircuit, not {}".format(type(other)))

        if self.noise and self.p_dec > 0:
            duration_difference = self.total_duration - other.total_duration
            if duration_difference < 0:
                times = int(math.ceil(abs(duration_difference)/self.time_step))
                self._N_decoherence([], times)
            elif duration_difference > 0:
                times = int(math.ceil(abs(duration_difference)/other.time_step))
                other._N_decoherence([], times)

        self._fused = True
        self.num_qubits = self.num_qubits + other.num_qubits
        self.d = 2 ** self.num_qubits
        self._correct_lookup_for_circuit_fusion(other._qubit_density_matrix_lookup)
        self._correct_drawing_for_circuit_fusion(other._draw_order, len(other._qubit_array))
        self._effective_measurements = other._effective_measurements + self._effective_measurements
        self._measured_qubits = other._measured_qubits + self._measured_qubits
        self._print_lines = other._print_lines + self._print_lines
        self._qubit_array = other._qubit_array + self._qubit_array

    def reset(self):
        self._qubit_array = self.num_qubits * [ket_0]
        self._draw_order = []
        self._user_operation_order = []
        self._effective_measurements = 0
        self._measured_qubits = []
        self._uninitialised_qubits = []
        self._qubit_density_matrix_lookup = {}
        self._print_lines = []
        self._fused = False
        self.ghz_fidelity = None

        # Decoherence and duration attributes
        self.total_duration = 0
        self.cut_off_time_reached = False

        # Probabilistic nature attributes
        self._total_link_attempts = 0
        self._total_succeeded_link = 0

        # Sub circuit attributes
        self._current_sub_circuit = None
        self._circuit_operations_ended = False

        for sub_circuit in self._sub_circuits.values():
            sub_circuit.reset()

        if self.qubits is not None:
            for qubit in self.qubits.values():
                qubit.reset_waiting_time()
                qubit.reset_sequence_time()

        if self.nodes is not None:
            for node in self.nodes.values():
                node.reset_all_times()

        self._init_density_matrix()

    def __repr__(self):
        return "\nQuantumCircuit object containing {} qubits\n".format(self.num_qubits)

    def __copy__(self):
        new_circuit = QuantumCircuit(self.num_qubits)
        new_circuit.density_matrix = self.density_matrix.copy()
        new_circuit.noise = self.noise
        new_circuit.p_g = self.p_g
        new_circuit.p_m = self.p_m
        new_circuit.F_link = self.F_link
        new_circuit._user_operation_order = self._user_operation_order.copy()
        new_circuit._measured_qubits = self._measured_qubits.copy()
        new_circuit._effective_measurements = self._effective_measurements
        new_circuit._draw_order = self._draw_order.copy()
        new_circuit._qubit_array = self._qubit_array.copy()
        new_circuit._init_type = self._init_type

        return new_circuit

    def copy(self):
        return self.__copy__()

    def append_print_lines(self, line):
        self._print_lines.append(line)

    @property
    def print_lines(self):
        return self._print_lines

    def print(self, empty_print_lines=True):
        if self._print_lines:
            print(*self._print_lines)
        if empty_print_lines:
            self._print_lines.clear()

