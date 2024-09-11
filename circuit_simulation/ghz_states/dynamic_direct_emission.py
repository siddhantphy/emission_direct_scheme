from qulacs import QuantumCircuit, QuantumState, DensityMatrix
from qulacs.state import make_superposition, make_mixture, tensor_product, partial_trace
from qulacs.gate import H, CNOT, CZ
import numpy as np
import scipy.linalg as LA
from scipy import sparse as sp
from scipy.linalg import sqrtm

class Protocol:
    """
    A class that simulates a quantum communication protocol incorporating various noise models 
    such as state noise, photon emission, photon loss, gate noise, and measurement noise.

    Attributes
    ----------
    p_n : float
        Probability of state noise.
    p_emi : float
        Photon emission probability.
    p_loss : float
        Photon loss probability.
    p_g : float
        Gate noise probability.
    p_m : float
        Measurement noise probability (bit flip).
    rho_imperfect : DensityMatrix
        The imperfect quantum state considering state noise.
    rho_raw : DensityMatrix
        The raw quantum state after considering photon emission and loss.
    id_to_measurement : dict
        A dictionary mapping measurement IDs to corresponding measurement functions.
    """

    def __init__(self, p_n=0.0106, p_emi=0.04, p_loss=0.9954, p_g=0.01, p_m=0.01):
        """
        Initializes the Protocol class with noise parameters and prepares the imperfect and raw states.

        Parameters
        ----------
        p_n : float, optional
            State noise probability (default is 0.0106).
        p_emi : float, optional
            Photon emission probability (default is 0.04).
        p_loss : float, optional
            Photon loss probability (default is 0.9954).
        p_g : float, optional
            Gate noise probability (default is 0.01).
        p_m : float, optional
            Measurement noise probability (default is 0.01).
        """
        self.p_n = p_n  # State noise probability
        self.p_emi = p_emi  # Photon emission probability
        self.p_loss = p_loss  # Photon loss probability
        self.p_g = p_g  # Gate noise probability
        self.p_m = p_m  # Measurement noise probability (bit flip)

        # Create the imperfect and raw quantum states considering noise
        self.rho_imperfect = create_rho_imperfect(p_n)
        self.rho_raw = create_rho_raw(self.rho_imperfect, p_emi, p_loss)

        # Mapping of measurement ID to corresponding measurement function
        self.id_to_measurement = {
            0: self.measure0000,
            1: self.measure1000,
            2: self.measure1100,
            3: self.measure1110,
            4: self.measure1111
        }

    def apply_raw_protocol(self):
        """
        Returns the raw quantum state after incorporating photon emission and loss.

        Returns
        -------
        DensityMatrix
            The raw quantum state (rho_raw).
        """
        return self.rho_raw
    
    def apply_basic_protocol(self):
        """
        Applies the basic protocol by measuring different qubit states and combining the results 
        with measurement noise taken into account.

        Returns
        -------
        tuple
            A tuple containing the success probability and the resulting quantum state.
        """
        p_m = self.p_m

        # Perform the basic protocol with different measurement IDs
        (basic_prob0, basic0) = self.apply_basic_protocol_(m_id=0)
        (basic_prob1, basic1) = self.apply_basic_protocol_(m_id=1)
        (basic_prob2, basic2) = self.apply_basic_protocol_(m_id=2)
        (basic_prob3, basic3) = self.apply_basic_protocol_(m_id=3)
        (basic_prob4, basic4) = self.apply_basic_protocol_(m_id=4)

        # Compute the overall probability with measurement noise
        basic_prob = p_m**4 * basic_prob0 \
                    + 4 * p_m**3 * (1 - p_m) * basic_prob1 \
                    + 6 * p_m**2 * (1 - p_m)**2 * basic_prob2 \
                    + 4 * p_m * (1 - p_m)**3 * basic_prob3 \
                    + (1 - p_m)**4 * basic_prob4

        # Combine the resulting states into a mixture considering measurement noise
        basic = \
            make_mixture(1, make_mixture( \
                    1, make_mixture(p_m**4, basic0, 4 * p_m**3 * (1 - p_m), basic1), \
                    1, make_mixture(6 * p_m**2 * (1 - p_m)**2, basic2, 4 * p_m * (1 - p_m)**3, basic3)
                ), \
                (1 - p_m)**4, basic4)
        
        return (basic_prob, basic)

    def apply_basic_protocol_(self, m_id):
        """
        Applies the basic protocol for a specific measurement ID by constructing a quantum circuit, 
        applying gates with noise, and performing the corresponding measurement.

        Parameters
        ----------
        m_id : int
            The measurement ID to determine which qubit state to measure.

        Returns
        -------
        tuple
            A tuple containing the success probability and the resulting quantum state.
        """
        # Create a quantum circuit with 8 qubits
        circuit = QuantumCircuit(8)

        # Add CNOT gates with depolarizing noise
        circuit.add_noise_gate(CNOT(0, 4), "Depolarizing", self.p_g)
        circuit.add_noise_gate(CNOT(1, 5), "Depolarizing", self.p_g)
        circuit.add_noise_gate(CNOT(2, 6), "Depolarizing", self.p_g)
        circuit.add_noise_gate(CNOT(3, 7), "Depolarizing", self.p_g)

        # Apply the corresponding measurement operation based on measurement ID
        self.id_to_measurement[m_id](circuit)

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

    def apply_medium_protocol(self):
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
        self.measure1111(circuit)

        # Apply the basic protocol and use the resulting state as input to the medium protocol
        (_, rho_basic) = self.apply_basic_protocol()
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
    
    # Measurement functions for specific qubit states
    def measure1111(self, circuit):
        """ 
        Measures all ancillary qubits (4, 5, 6, 7) in state |1>. 
        """
        circuit.add_P1_gate(4)
        circuit.add_P1_gate(5)
        circuit.add_P1_gate(6)
        circuit.add_P1_gate(7)
    
    def measure1110(self, circuit):
        """ 
        Measures ancillary qubits 4, 5, 6 in state |1> and qubit 7 in state |0>. 
        """
        circuit.add_P1_gate(4)
        circuit.add_P1_gate(5)
        circuit.add_P1_gate(6)
        circuit.add_P0_gate(7)
    
    def measure1100(self, circuit):
        """ 
        Measures ancillary qubits 4, 5 in state |1> and qubits 6, 7 in state |0>. 
        """
        circuit.add_P1_gate(4)
        circuit.add_P1_gate(5)
        circuit.add_P0_gate(6)
        circuit.add_P0_gate(7)
    
    def measure1000(self, circuit):
        """ 
        Measures ancillary qubit 4 in state |1> and qubits 5, 6, 7 in state |0>. 
        """
        circuit.add_P1_gate(4)
        circuit.add_P0_gate(5)
        circuit.add_P0_gate(6)
        circuit.add_P0_gate(7)
    
    def measure0000(self, circuit):
        """ 
        Measures all ancillary qubits (4, 5, 6, 7) in state |0>. 
        """
        circuit.add_P0_gate(4)
        circuit.add_P0_gate(5)
        circuit.add_P0_gate(6)
        circuit.add_P0_gate(7)

    def success_rate(self):
        """
        Calculates the success rate of the protocol based on photon emission and loss probabilities.

        Returns
        -------
        float
            The success rate of the protocol.
        """
        eta = 1 - self.p_loss  # Photon survival probability
        return 0.5 * eta ** 2 * self.p_emi ** 2 * (1 - self.p_emi * eta) ** 2


def print_state_vector(vec: np.ndarray, eps: float = 1e-10) -> None:
    """
    Prints the non-zero components of a state vector in the computational basis.

    Parameters
    ----------
    vec : np.ndarray
        The state vector to be printed.
    eps : float, optional
        Threshold below which components are considered zero (default is 1e-10).
    """
    n = int(np.log2(len(vec) + 1e-10))  # Number of qubits
    for ind in range(2**n):
        if np.abs(vec[ind]) < eps:
            continue
        s = bin(ind)[2:].zfill(n)  # Convert index to binary string
        print(f"{vec[ind]:.3f} |{s}>")


def print_density_matrix(dm: np.ndarray, eps: float = 1e-10) -> None:
    """
    Prints the non-zero eigenstates of a density matrix.

    Parameters
    ----------
    dm : np.ndarray
        The density matrix to be printed.
    eps : float, optional
        Threshold below which eigenvalues are considered zero (default is 1e-10).
    """
    ee, ev = np.linalg.eigh(dm)  # Eigenvalues and eigenvectors
    for ei, eval in enumerate(ee):
        if eval < eps:
            continue
        print(f"prob = {eval}")
        print_state_vector(ev[:, ei])  # Print the corresponding eigenstate


# Prepare GHZ state and other quantum states for the protocol
ghz_state = DensityMatrix(4)
state1010 = QuantumState(4)
state0101 = QuantumState(4)
state1010.set_computational_basis(0b1010)
state0101.set_computational_basis(0b0101)
ghz_vector = make_superposition(1, state1010, 1, state0101)
ghz_state = make_mixture(1.0, ghz_vector, 0, ghz_vector)
ghz_state.normalize(ghz_state.get_squared_norm())


def create_rho_imperfect(p_n):
    """
    Creates an imperfect GHZ state by adding state noise.

    Parameters
    ----------
    p_n : float
        Probability of state noise.

    Returns
    -------
    DensityMatrix
        The imperfect GHZ state.
    """
    mat = (1 - p_n) * ghz_state.get_matrix() + p_n / 16 * np.eye(16)
    state = DensityMatrix(4)
    state.load(mat)
    return state

"""
## Applying photon loss to create `rho_raw`
ρ_raw = α * ρ_imperfect + Σ β_i * |φ_i⟩⟨φ_i|
      = ℕ [p_0Y_0² * |Ψ⟩⟨Ψ| + p_1Y_1² Σ |φ_i⟩⟨φ_i| + 2 * p_2Y_2² * |φ_5⟩⟨φ_5|]
where
{φ_i} = {|1110⟩, |1101⟩, |1011⟩, |0111⟩, |1111⟩}
p_mY_m² = (1/4) * η² * (1-η)^m * a^(4-2m) * b^(4+2m)
"""

def create_rho_raw(rho_imperfect, p_emi, p_loss):
    """
    Creates a raw quantum state after considering photon emission and loss probabilities.

    Parameters
    ----------
    rho_imperfect : DensityMatrix
        The imperfect quantum state.
    p_emi : float
        Photon emission probability.
    p_loss : float
        Photon loss probability.

    Returns
    -------
    DensityMatrix
        The raw quantum state.
    """
    a = np.sqrt(1 - p_emi)
    b = np.sqrt(p_emi)
    eta = 1 - p_loss
    pY2 = [((eta ** 2) * ((1 - eta) ** m) / 4) * (a ** (4 - 2*m)) * (b ** (4 + 2*m)) for m in [0, 1, 2]]

    # Define quantum states for different computational basis states
    state1110 = DensityMatrix(4)
    state1110.set_computational_basis(0b1110)
    state1101 = DensityMatrix(4)
    state1101.set_computational_basis(0b1101)
    state1011 = DensityMatrix(4)
    state1011.set_computational_basis(0b1011)
    state0111 = DensityMatrix(4)
    state0111.set_computational_basis(0b0111)
    state1111 = DensityMatrix(4)

    # Create a mixture of states
    tmp1 = make_mixture(1,
        make_mixture(1,
            make_mixture(1, state1110, 1, state1101),
            1, state1011),
        1, state0111)
    state1111.set_computational_basis(0b1111)
    tmp2 = make_mixture(1, state1111, 1, state1111)

    # Create the raw state by mixing the imperfect state with other quantum states
    rho_X = make_mixture(pY2[0], rho_imperfect, 1.0, make_mixture(pY2[1], tmp1, pY2[2], tmp2))
    rho_X.normalize(rho_X.get_squared_norm())
    return rho_X


def sqrtmh(A):
    """
    Calculates the matrix square root of a Hermitian matrix.

    Parameters
    ----------
    A : np.ndarray
        The input Hermitian matrix.

    Returns
    -------
    np.ndarray
        The matrix square root of A.
    """
    vals, vecs = LA.eigh(A)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T.conjugate()


def fidelity(rho1, rho2):
    """
    Computes the fidelity between two quantum states represented as density matrices.

    Parameters
    ----------
    rho1 : DensityMatrix
        The first quantum state.
    rho2 : DensityMatrix
        The second quantum state.

    Returns
    -------
    float
        The fidelity between the two quantum states.
    """
    mat1 = sqrtmh(rho1.get_matrix())
    return np.abs(np.trace(sqrtmh(np.dot(np.dot(mat1, rho2.get_matrix()), mat1))))



# Simulations

def generate_dynamic_direct_emission_state(choice: int = 110, p_n: float = 0.0106, p_emi: float = 0.04, eta: float = 0.4472, p_g: float = 0.001, p_m: float = 0.001):
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
    protocol = Protocol(p_n=p_n, p_emi=p_emi, p_loss=1-eta, p_g=p_g, p_m=p_m)

    # Apply the raw, basic, and medium protocols
    raw = protocol.apply_raw_protocol()
    (basic_prob, basic) = protocol.apply_basic_protocol()
    (medium_prob, medium) = protocol.apply_medium_protocol()
    # Define Pauli matrices
    I = np.eye(2)  # Identity matrix
    X = np.array([[0, 1], [1, 0]])  # Pauli-X matrix
    # Define the 4-qubit Pauli operator IXIX (tensor product)
    IXIX = np.kron(np.kron(I, X), np.kron(I, X)) # To rotate the GHZ state to align with circuit-simulator's target GHZ state

    # Return the corresponding state based on the 'choice' argument
    if choice == 100:
        print(f"*** Chosen Scheme: Wt.4, direct emission with raw protocol! ***")
        return (protocol.success_rate(), sp.csr_matrix(IXIX @ raw.get_matrix() @ IXIX))
    elif choice == 101:
        print(f"*** Chosen Scheme: Wt.4, direct emission with basic protocol! ***")
        return (protocol.success_rate() * basic_prob, sp.csr_matrix(IXIX @ basic.get_matrix() @ IXIX))
    elif choice == 102:
        print(f"*** Chosen Scheme: Wt.4, direct emission with medium protocol! ***")
        return (protocol.success_rate() * basic_prob * medium_prob, sp.csr_matrix(IXIX @ medium.get_matrix() @ IXIX))
    else:
        raise ValueError("Invalid choice. Please select 100, 101, or 102 for the direct emission schemes or other choices for other schemes.")
