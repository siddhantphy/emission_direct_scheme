from qulacs import QuantumCircuit, QuantumState, DensityMatrix
from qulacs.state import make_superposition, make_mixture, tensor_product, partial_trace
from qulacs.gate import H, CNOT, CZ, BitFlipNoise
import numpy as np
import scipy.linalg as LA
from scipy import sparse as sp
from scipy.linalg import sqrtm

def create_rho_raw(alpha : float, F_prep : float, mu : float, eta : float):
    """
    Creates a raw quantum state after considering photon emission and loss probabilities.

    Parameters
    ----------
    alpha : float
        Bright state parameter
    F_prep : float
        Preparation fidelity
    mu : float
        Photon indistinguishability
    eta : float
        Photon transmission rate

    Returns
    -------
    DensityMatrix
        The raw quantum state.
    """
    
    success_prob = 1/96 * alpha**2 * eta**2 * (-24 * (-3 + mu**2) - 24 * alpha * eta * (3 - 3 * mu**2 + 2 * mu**3) + alpha**2 * eta**2 * (3 - 46 * mu**2 + 47 * mu**3))
    
    elem[0] = 12 * (-1 + alpha)**2 * (-1 + mu**2)
    elem[1] = -12 * (1 - 2 * F_prep)**2 * (-1 + alpha)**2 * (-1 + mu) * mu
    elem[2] = -24 * (1 - 2 * F_prep)**4 * (-1 + alpha)**2 * mu**2
    elem[3] = -6 * (1 - 2 * F_prep)**2 * (-1 + alpha) * alpha * mu * (4 - 4 * mu + eta * (-1 + 2 * mu + mu**2))
    elem[4] = 6 * (-1 + alpha) * alpha * (2 * (-3 + mu**2) + eta * (3 - 3 * mu**2 + 2 * mu**3))
    elem[5] = alpha**2 * (-24 * (-3 + mu**2) - 24 * eta * (3 - 3 * mu**2 + 2 * mu**3) + eta**2 * (3 - 46 * mu**2 + 47 * mu**3))
    
    elem = [e / -24 * (-3 + mu**2) - 24 * alpha * eta * (3 - 3 * mu**2 + 2 * mu**3) + alpha**2 * eta**2 * (3 - 46 * mu**2 + 47 * mu**3) for e in elem]
    
    mat = np.zeros((4, 4), dtype=complex)
    mat[3, 3]   = -elem[0]
    mat[6, 6]   = -elem[0]
    mat[9, 9]   = -elem[0]
    mat[12, 12] = -elem[0]
    mat[5, 5]   = elem[0] # |0101><0101|
    mat[10, 10] = elem[0] # |1010><1010|
    mat[3, 6]   = elem[1]
    mat[3, 9]   = elem[1]
    mat[6, 3]   = elem[1]
    mat[6, 12]  = elem[1]
    mat[9, 3]   = elem[1]
    mat[9, 12]  = elem[1]
    mat[12, 6]  = elem[1]
    mat[12, 9]  = elem[1]
    mat[5, 10]  = elem[2] # |0101><1010|
    mat[10, 5]  = elem[2] # |1010><0101|
    # below elements are eliminated by basic protocol
    mat[7, 13]  = elem[3]
    mat[13, 7]  = elem[3]
    mat[11, 14] = elem[3]
    mat[14, 11] = elem[3]                
    mat[7, 7]   = elem[4] # |0111><0111|
    mat[11, 11] = elem[4] # |1011><1011|
    mat[13, 13] = elem[4] # |1101><1101|
    mat[14, 14] = elem[4] # |1110><1110|
    mat[15, 15] = elem[5] # |1111><1111|
    
    rho_X = DensityMatrix(4)
    rho_X.load(mat)
    return (success_prob, rho_X)

def apply_basic_protocol(**kwargs):
    """
    Applies the basic protocol by constructing a quantum circuit, 
    applying gates with noise, and performing the corresponding measurement.
    
    Parameters
    ----------
    p_g : float
        Gate noise parameter
    p_m : float
        Measurement noise parameter

    Returns
    -------
    tuple
        A tuple containing the success probability and the resulting quantum state.
    """
    
    p_g = kwargs["p_g"]
    p_m = kwargs["p_m"] 

    # Create a quantum circuit with 8 qubits
    circuit = QuantumCircuit(8)

    # Add CNOT gates with depolarizing noise
    circuit.add_noise_gate(CNOT(0, 4), "Depolarizing", self.p_g)
    circuit.add_noise_gate(CNOT(1, 5), "Depolarizing", self.p_g)
    circuit.add_noise_gate(CNOT(2, 6), "Depolarizing", self.p_g)
    circuit.add_noise_gate(CNOT(3, 7), "Depolarizing", self.p_g)

    # Add BitFlipNoise gates to ancillary qubits (4, 5, 6, 7)
    circuit.add_gate(BitFlipNoise(4, self.p_m))
    circuit.add_gate(BitFlipNoise(5, self.p_m))
    circuit.add_gate(BitFlipNoise(6, self.p_m))
    circuit.add_gate(BitFlipNoise(7, self.p_m))

    # Measure all ancillary qubits (4, 5, 6, 7) by state |1>
    circuit.add_P1_gate(4)
    circuit.add_P1_gate(5)
    circuit.add_P1_gate(6)
    circuit.add_P1_gate(7)

    # Duplicate the raw state and apply the circuit
    rho_raw = self.rho_raw.copy()
    rho_basic = tensor_product(rho_raw, rho_raw)
    circuit.update_quantum_state(rho_basic)

    # Perform partial trace on the ancillary qubits (4, 5, 6, 7)
    rho_basic_traced = partial_trace(rho_basic, [4, 5, 6, 7])

    # Calculate the success probability
    basic_prob = rho_basic_traced.get_squared_norm()

    # Normalize the resulting state
    rho_basic_traced.normalize(rho_basic_traced.get_squared_norm())

    return (basic_prob, rho_basic_traced)

def apply_medium_protocol(rho_basic, p_g, p_m):
    """
    Applies the medium protocol by creating a quantum circuit with controlled-Z gates and Hadamard gates, 
    followed by a measurement of the qubits.

    Returns
    -------
    tuple
        A tuple containing the success probability and the resulting quantum state.
    """
    # Create a quantum circuit with 8 qubits
    circuit = QuantumCircuit(8)

    # Add controlled-Z gates with depolarizing noise
    circuit.add_noise_gate(CZ(0, 4), "Depolarizing", self.p_g)
    circuit.add_noise_gate(CZ(1, 5), "Depolarizing", self.p_g)
    circuit.add_noise_gate(CZ(2, 6), "Depolarizing", self.p_g)
    circuit.add_noise_gate(CZ(3, 7), "Depolarizing", self.p_g)

    # Add Hadamard gates with depolarizing noise
    circuit.add_noise_gate(H(4), "Depolarizing", self.p_g)
    circuit.add_noise_gate(H(5), "Depolarizing", self.p_g)
    circuit.add_noise_gate(H(6), "Depolarizing", self.p_g)
    circuit.add_noise_gate(H(7), "Depolarizing", self.p_g)

    # Apply the measurement corresponding to all qubits in state |1>
    circuit.add_P1_gate(4)
    circuit.add_P1_gate(5)
    circuit.add_P1_gate(6)
    circuit.add_P1_gate(7)

    # Apply the basic protocol and use the resulting state as input to the medium protocol
    rho_medium = tensor_product(rho_basic, rho_basic)

    # Update the quantum state using the constructed circuit
    circuit.update_quantum_state(rho_medium)

    # Perform partial trace on ancillary qubits (4, 5, 6, 7)
    rho_medium_traced = partial_trace(rho_medium, [4, 5, 6, 7])

    # Calculate the success probability
    medium_prob = rho_medium_traced.get_squared_norm()

    # Normalize the resulting state
    rho_medium_traced.normalize(rho_medium_traced.get_squared_norm())

    return (medium_prob, rho_medium_traced)

# Simulations

def generate_dynamic_direct_emission_state(choice: int = 110, alpha: float = 0.04, F_prep: float = 0.999, mu : float = 0.95, eta: float = 0.4472, p_g: float = 0.001, p_m: float = 0.001):
    """
    Generate a dynamic direct emission state based on the specified choice and noise parameters.

    Args:
        choice (int, optional): An identifier for selecting the emission scheme. 
                                - 100: Returns the raw protocol state.
                                - 101: Returns the basic protocol state.
                                - 102: Returns the medium protocol state.
        p_n (float, optional): Noise parameter for imperfect state. Default is 0.0106.
        p_emi (float, optional): Photon emission rate. Default is 0.04.
        eta (float, optional): Efficiency factor for photon loss. Default is 0.4472.
        p_g (float, optional): Gate noise parameter. Default is 0.01.
        p_m (float, optional): Measurement noise parameter. Default is 0.01.

    Returns:
        tuple: A tuple containing:
            - prob_succ (float): The success probability of the protocol based on the selected choice.
            - sparse_density_matrix (scipy.sparse.csr_matrix): The sparse density matrix of the selected emission state.
    
    Raises:
        ValueError: If an invalid choice is provided (other than 100, 101, or 102).
    """

    # Apply the raw, basic, and medium protocols
    (raw_prob, raw_state) = create_raw_state(alpha, F_prep, mu, eta)
    (basic_prob, basic_state)   = apply_basic_protocol(rho_raw, p_g, p_m)
    (medium_prob, medium_state) = apply_medium_protocol(rho_basic, p_g, p_m)
    # Define Pauli matrices
    I = np.eye(2)  # Identity matrix
    X = np.array([[0, 1], [1, 0]])  # Pauli-X matrix
    # Define the 4-qubit Pauli operator IXIX (tensor product)
    IXIX = np.kron(np.kron(I, X), np.kron(I, X)) # To rotate the GHZ state to align with circuit-simulator's target GHZ state

    # Return the corresponding state based on the 'choice' argument
    if choice == 100:
        print(f"*** Chosen Scheme: Wt.4, direct emission with raw protocol! ***")
        return (raw_prob, sp.csr_matrix(IXIX @ raw_state.get_matrix() @ IXIX))
    elif choice == 101:
        print(f"*** Chosen Scheme: Wt.4, direct emission with basic protocol! ***")
        return (raw_prob * basic_prob, sp.csr_matrix(IXIX @ basic_state.get_matrix() @ IXIX))
    elif choice == 102:
        print(f"*** Chosen Scheme: Wt.4, direct emission with medium protocol! ***")
        return (raw_prob * basic_prob * medium_prob, sp.csr_matrix(IXIX @ medium_state.get_matrix() @ IXIX))
    else:
        raise ValueError("Invalid choice. Please select 100, 101, or 102 for the direct emission schemes or other choices for other schemes.")
