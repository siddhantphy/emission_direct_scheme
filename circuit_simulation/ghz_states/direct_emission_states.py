import numpy as np
import os
import scipy.sparse as sp
from circuit_simulation.ghz_states.dynamic_direct_emission import generate_dynamic_direct_emission_state

def import_direct_emission_states(choice: int, dynamic_states: bool, path: str, p_g: float, p_m: float, eta: float, p_n: float, p_emi: float):
    """
    Import or generate direct emission states based on dynamic state selection.

    Args:
        choice (int): An identifier for the specific emission state to be used.
        dynamic_states (bool): If True, generates the state dynamically. If False, imports the state from a file.
        path (str): The file path to import states from, if `dynamic_states` is True.
        p_g (float): Gate noise parameter.
        p_m (float): Measurement noise parameter.
        eta (float): Efficiency factor for photon loss.
        p_n (float): Noise parameter for imperfect states.
        p_emi (float): Photon emission rate.

    Returns:
        tuple: A tuple containing:
            - prob_succ (float): The success probability of generating or importing the emission state.
            - sparse_density_matrix: The sparse density matrix representing the emission state.
    """
    if dynamic_states:
        prob_succ, sparse_density_matrix = generate_dynamic_direct_emission_state(choice, p_n, p_emi, eta, p_g, p_m)
        return (prob_succ, sparse_density_matrix)
    
    if not dynamic_states:
        prob_succ, sparse_density_matrix = import_direct_emission_states_from_file(path=path, choice=choice)
        return (prob_succ, sparse_density_matrix)


def import_direct_emission_states_from_file(path: str, choice: int):
    """
    Imports and applies the IXIX Pauli operator to a 4-qubit density matrix
    from a specified file based on the choice parameter, and converts the result to a sparse matrix.

    Parameters
    ----------
    path : str
        The file path to the directory containing the GHZ state files. If None, a default path is used.
    choice : int
        An integer indicating which density matrix file to import and process:
        - 100: Use "ghz_raw.csv"
        - 101: Use "ghz_basic.csv"
        - 102: Use "ghz_medium.csv"

    Returns
    -------
    None
        The function prints the success probability and the resulting sparse density matrix.

    Notes
    -----
    This function assumes that the density matrices are stored in CSV files within the specified directory.
    The function applies the Pauli operator IXIX to the density matrix before converting it to a sparse matrix.
    """
    
    # Set default path if no path is provided
    cwd = os.getcwd()
    path = f'{cwd}/circuit_simulation/ghz_states/direct_emission_states/' if path is None else path

    # Define Pauli matrices
    I = np.eye(2)  # Identity matrix
    X = np.array([[0, 1], [1, 0]])  # Pauli-X matrix

    # Define the 4-qubit Pauli operator IXIX (tensor product)
    IXIX = np.kron(np.kron(I, X), np.kron(I, X))

    # Select the density matrix based on the choice parameter
    if choice == 100:
        density_matrix = IXIX @ np.loadtxt(path + "ghz_raw.csv", delimiter=',') @ IXIX
        prob_succ = 1.6921e-08
    elif choice == 101:
        density_matrix = IXIX @ np.loadtxt(path + "ghz_basic.csv", delimiter=',') @ IXIX
        prob_succ = 1.6921e-08 * 3.5908e-01
    elif choice == 102:
        density_matrix = IXIX @ np.loadtxt(path + "ghz_medium.csv", delimiter=',') @ IXIX
        prob_succ = 1.6921e-08 * 4.4794e-02
    else:
        raise ValueError("Invalid choice. Please select 100, 101, or 102.")
    
    # Convert the density matrix to a sparse matrix format (CSR)
    sparse_density_matrix = sp.csr_matrix(density_matrix)
    
    # Return the success probability and the sparse density matrix
    return prob_succ,sparse_density_matrix