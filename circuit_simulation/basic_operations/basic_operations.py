import numpy as np
from scipy import sparse as sp
from scipy.linalg import sqrtm
import random
import circuit_simulation.states.states as s
import circuit_simulation.gates.gates as g
from circuit_simulation.states.state import State
from circuit_simulation.gates.gate import SingleQubitGate, TwoQubitGate


def state_repr(state):
    """ Returns the visual representation of the given state if known """
    if np.array_equal(state, s.ket_0.vector):
        return "|0>"
    if np.array_equal(state, s.ket_1.vector):
        return "|1>"
    if np.array_equal(state, s.ket_p.vector):
        return "|+>"
    if np.array_equal(state, s.ket_m.vector):
        return "|->"
    return "|?>"


def gate_name(gate):
    """ Returns the (visual) representation of the given gate if known """
    if np.array_equal(gate, g.X_gate.matrix):
        return "X"
    if np.array_equal(gate, g.Y_gate.matrix):
        return "Y"
    if np.array_equal(gate, g.Z_gate.matrix):
        return "Z"
    if np.array_equal(gate, g.I_gate.matrix):
        return "I"
    if np.array_equal(gate, g.H_gate.matrix):
        return "H"
    if np.array_equal(gate, g.S_gate.matrix):
        return "S"
    return "?"


def get_value_by_prob(array, p):
    """ Returns, bases on the given weights 'p', a value out of the given array """
    # if a 0 probability is in a list of 2 weights, get the other value
    if len(p) == 2 and 0 in p:
        return array[p.index(max(p))]

    # Normalise the weights
    p = [i/sum(p) for i in p]
    r = random.random()
    index = 0
    while r > 0 and index < len(p):
        r -= p[index]
        index += 1
    return array[index - 1]


def KP(*args):
    """ Returns the Kronecker product of the given arguments in the exact order """
    result = None
    for state in args:
        if state is None:
            continue
        if type(state) == State:
            state = state.sp_vector
        if type(state) in [SingleQubitGate, TwoQubitGate]:
            state = state.sp_matrix
        if not sp.issparse(state):
            state = sp.csr_matrix(state)
        if result is None:
            result = state
            continue
        result = sp.kron(result, state, format='csr')
    return result


def CT(state1, state2=None):
    """ returns the dot prodcut of the two passed states, where the second state will be the conjugate transpose """
    if type(state1) == State:
        state1 = state1.sp_vector
    elif not sp.issparse(state1):
        state1 = sp.csr_matrix(state1)

    if state2 is not None:
        if type(state2) == State:
            state2 = state2.sp_vector
        elif not sp.issparse(state2):
            state2 = sp.csr_matrix(state2)
    else:
        state2 = state1

    result = state1.dot(state2.conj().T)
    return result


def trace(sparse_matrix):
    """ Returns the trace of a matrix"""
    if not sp.issparse(sparse_matrix):
        sp.csr_matrix(sparse_matrix)

    result = sparse_matrix.diagonal().sum()

    if isinstance(result, complex):
        result = result.real

    return result


def N_dim_ket_0_or_1_density_matrix(N, ket=0):
    """ Returns an N-qubit version of the ket_0 or ket_1 density matrix """
    dim = 2**N
    rho = sp.lil_matrix((dim, dim))
    if ket == 1:
        rho[dim, dim] = 1
    else:
        rho[0, 0] = 1
    return rho


def fidelity(rho, sigma):
    """ Calculates the fidelity of two density matrices according to the 'classical' method """
    if not sp.issparse(rho):
        rho = sp.csr_matrix(rho)
    if not sp.issparse(sigma):
        sigma = sp.csr_matrix(sigma)

    # Ensure matrices are positive semi-definite by adding a small regularization term
    epsilon = 1e-10
    rho = rho + epsilon * sp.eye(rho.shape[0], format='csr')
    sigma = sigma + epsilon * sp.eye(sigma.shape[0], format='csr')

    try:
        rho_root = sp.csr_matrix(sqrtm(rho.toarray()))
        resulting_matrix = sqrtm((rho_root * sigma * rho_root).toarray())
        return trace(resulting_matrix)
    except Exception as e:
        print(f"Error calculating fidelity: {e}")
        return 0.0  # Return a default value in case of failure


def fidelity_elementwise(rho, sigma):
    """ Calculates the fidelity using the element wise multiplication method """
    if not sp.issparse(rho):
        rho = sp.csr_matrix(rho)
    if not sp.issparse(sigma):
        sigma = sp.csr_matrix(sigma)

    resulting_matrix = rho * sigma * rho
    return trace(resulting_matrix)


def csr_matrix_equal(a1, a2):
    # Sort indices, such that equality does not fail because of this
    a1.sort_indices()
    a2.sort_indices()
    return (np.array_equal(a1.indptr, a2.indptr) and
            np.array_equal(a1.indices, a2.indices) and
            np.array_equal(a1.data, a2.data))


def trace_distance(rho, sigma):
    """ Calculates the trace distance of two density matrices """
    if not sp.issparse(rho):
        rho = sp.csr_matrix(rho)
    if not sp.issparse(sigma):
        sigma = sp.csr_matrix(sigma)

    rho_m_sigma = rho - sigma
    resulting_matrix = sqrtm((rho_m_sigma.transpose() * rho_m_sigma).toarray())
    return 0.5 * trace(resulting_matrix)