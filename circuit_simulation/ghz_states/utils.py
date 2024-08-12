import numpy as np

def get_common_para(status):
    if status == 'cur':
        eta = 0.5
        Dlt = 16
        sgm = 0.46
        C = 30
        E = 0.9
        ka = 200
        P = 5
    elif status == 'nf':
        eta = 0.9
        Dlt = 122
        sgm = 0.32
        C = 40
        E = 0.95
        ka = 200
        P = 20
    else:
        raise ValueError('Status input error.')
    
    return eta, Dlt, sgm, C, E, ka, P


def my_distr(nSmp, wt):
    nSgm = 4
    sigma = 1
    ed = nSgm * sigma
    d = 2 * ed / nSmp
    x = np.linspace(-ed + d, ed, nSmp)
    pp = d / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 0.5 * d) / sigma) ** 2)
    pp = pp / np.sum(pp)
    
    PP = pp
    for i in range(wt - 1):
        PP = np.kron(PP, pp)
    
    return PP, d


def convert(N, nSmp, nWt):
    num = np.zeros((nWt, 1))
    
    for i in range(nWt, 0, -1):
        num[i - 1] = N % nSmp
        N = np.floor(N / nSmp)
    
    return num


def def_operators_ref(nWt):
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])
    id = np.eye(2)
    
    HH = np.zeros((2 ** (nWt + 1), 2 ** (nWt + 1), nWt))
    XX = np.zeros((2 ** (nWt + 1), 2 ** (nWt + 1), nWt))
    YY = np.zeros((2 ** (nWt + 1), 2 ** (nWt + 1), nWt), dtype=complex)
    ZZ = np.zeros((2 ** (nWt + 1), 2 ** (nWt + 1), nWt))
    
    for i in range(nWt):
        H = id
        X = id
        Y = id
        Z = id
        
        for j2 in range(nWt):
            if j2 != i:
                H = np.kron(H, id)
                X = np.kron(X, id)
                Y = np.kron(Y, id)
                Z = np.kron(Z, id)
            else:
                H = np.kron(H, h)
                X = np.kron(X, x)
                Y = np.kron(Y, y)
                Z = np.kron(Z, z)
        
        HH[:, :, i] = H
        XX[:, :, i] = X
        YY[:, :, i] = Y
        ZZ[:, :, i] = Z
    
    return HH, XX, YY, ZZ


def cal_reflect_den(r0_g, r1_g, nWt, eta, err, HH, XX, YY, ZZ):
    psi = np.array([1, 1]) / np.sqrt(2)
    for i in range(nWt):
        psi = np.kron(psi, [1, 0])
    
    dm = np.outer(psi, psi)
    
    for i in range(nWt):
        dm = scatter_i(dm, r0_g[:,i], r1_g[:,i], nWt, eta, err, i, HH[:, :, i], XX[:, :, i], YY[:, :, i], ZZ[:, :, i])
    
    i = 0
    P_a, rho = measure(dm, nWt, err)
    return P_a, rho


def scatter_i(dm, r0, r1, nWt, eta, err, i, H, X, Y, Z):
    t0 = np.sqrt(eta) * r0
    t1 = np.sqrt(eta) * r1
    
    scA = 1
    for j3 in range(nWt):
        if j3 != i:
            scA = np.kron(scA, [1, 1])
        else:
            scA = np.kron(scA, np.concatenate((t0, t1), axis = 0))
    
    scE = np.diag( np.concatenate((scA, np.ones(2 ** nWt)), axis = 0) )
    dm = scE @ dm @ scE.conj()
    
    dm = H.T @ dm @ H
    dm = (1 - err) * dm + 0.25 * err * (dm + X @ dm @ X + Y @ dm @ Y + Z @ dm @ Z)
    
    scL = np.diag( np.concatenate((np.ones(2 ** nWt), scA), axis = 0) )
    dm = scL @ dm @ scL.conj()
    
    dm = H.T @ dm @ H

    dm = (1 - err) * dm + 0.25 * err * (dm + X @ dm @ X + Y @ dm @ Y + Z @ dm @ Z)
    
    return dm


def measure(dm, wt, err):
    
    M = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    M = np.kron(M, np.eye(2 ** wt))
    dm = M.T @ dm @ M
    
    
    
    dm_p = dm[:2 ** wt, :2 ** wt]
    dm_m = dm[2 ** wt:, 2 ** wt:]

    X=np.kron(np.array([[0, 1], [1, 0]]), np.eye(2 ** (wt-1)))
    Y=np.kron(np.array([[0, -1j], [1j, 0]]), np.eye(2 ** (wt-1)))
    Z=np.kron(np.array([[1, 0], [0, -1]]), np.eye(2 ** (wt-1)))
    dm_m = Z @ dm_m @ Z
    dm_m = (1 - err) * dm_m + 0.25 * err * (dm_m + X @ dm_m @ X + Y @ dm_m @ Y + Z @ dm_m @ Z)

    P_p = np.trace(dm_p)
    P_m = np.trace(dm_m)
    P_a = P_p + P_m
    
    dm = 1 / P_a * (dm_p + dm_m)
    
    return P_a, dm



def cal_carving_sgl_den(t0_g, t1_g, wt, nPh, eta, err, XX, YY, ZZ):
    # Target states
    psi0 = np.array([1, 0])
    psi1 = np.array([0, 1])
    
    if wt == 4:
        Psi0011 = np.kron(np.kron(psi0, psi0), np.kron(psi1, psi1))
        Psi1100 = np.kron(np.kron(psi1, psi1), np.kron(psi0, psi0))
        tar1 = (Psi0011 + Psi1100) / np.sqrt(2)
        tar2 = (Psi0011 - Psi1100) / np.sqrt(2)
    elif wt == 3:
        Psi001 = np.kron(np.kron(psi0, psi0), psi1)
        Psi110 = np.kron(np.kron(psi1, psi1), psi0)
        tar1 = (Psi001 + Psi110) / np.sqrt(2)
        tar2 = (Psi001 - Psi110) / np.sqrt(2)
    else:
        raise ValueError('Wrong number of Weight')
    
    # Not gate
    Not = np.flipud(np.eye(2 ** wt))
    z = np.array([[1, 0], [0, -1]])
    Z1 = np.kron(z, np.eye(2 ** (wt - 1)))
    Not12 = np.kron(np.flipud(np.eye(2 ** 2)), np.eye(2 ** (wt - 2)))
    
    t_v = np.vstack((t0_g, t1_g))
    
    nWt1 = 2
    nWt2 = wt - nWt1
    
    trnsm1 = np.kron(np.kron(t_v[:, 0], t_v[:, 1]), np.ones(2 ** nWt2))
    
    if wt == 4:
        trnsm2 = np.kron(np.ones(2 ** nWt1), np.kron(t_v[:, 2], t_v[:, 3]))
    elif wt == 3:
        trnsm2 = np.kron(np.ones(2 ** nWt1), t_v[:, 2])
    
    Coeff_plus = np.sqrt(eta) * (trnsm1 + trnsm2) / 2
    Coeff_minus = np.sqrt(eta) * (trnsm1 - trnsm2) / 2
    NCoeff_plus = np.dot(Not, Coeff_plus)
    NCoeff_minus = np.dot(Not, Coeff_minus)

    Coeff_plus=np.outer(Coeff_plus.T, Coeff_plus.conj())
    Coeff_minus=np.outer(Coeff_minus.T, Coeff_minus.conj())
    NCoeff_plus=np.outer(NCoeff_plus.T, NCoeff_plus.conj())
    NCoeff_minus=np.outer(NCoeff_minus.T, NCoeff_minus.conj())

    
    psi = np.ones(2 ** wt) / np.sqrt(2 ** wt)
    dm = np.outer(psi, psi)
    
    for i in range(1, nPh + 1):
        if i % 2 == 1:
            dm1 =  Coeff_plus * dm 
            dm2 =  Coeff_minus * dm
        else:
            dm1 =  NCoeff_plus * dm 
            dm2 =  NCoeff_minus * dm
            
        
        dm = np.dstack((dm1, dm2))
        Coeff_plus=np.dstack((Coeff_plus, Coeff_plus))
        Coeff_minus=np.dstack((Coeff_minus, Coeff_minus))
        NCoeff_plus=np.dstack((NCoeff_plus, NCoeff_plus))
        NCoeff_minus=np.dstack((NCoeff_minus, NCoeff_minus))

        
        if i < nPh:
            for j3 in range(wt):
                X = XX[:, :, j3]
                Y = YY[:, :, j3]
                Z = ZZ[:, :, j3]
                for k in range(2 ** i):
                    dm_a = dm[:, :, k]
                    dm_a = (1 - err) * dm_a + 0.25 * err * (dm_a + np.dot(X, np.dot(dm_a, X)) + np.dot(Y, np.dot(dm_a, Y)) + np.dot(Z, np.dot(dm_a, Z)))
                    dm[:, :, k] = dm_a
    

    P_a = 0
    Den_a = np.zeros((2 ** wt, 2 ** wt), dtype=complex)
    
    for j3 in range(2 ** nPh):
        P_j = np.trace(dm[:, :, j3])
        P_a += P_j
        
        F_j_values = np.array([np.dot(tar1.conj().T, np.dot(dm[:, :, j3], tar1)), np.dot(tar2.conj().T, np.dot(dm[:, :, j3], tar2))])
        S_j = np.argmax(F_j_values) + 1
        
        if S_j == 1:
            Den_a += dm[:, :, j3]
        else:
            Den_a += np.dot(Z1, np.dot(dm[:, :, j3], Z1))
    
    Den_a /= P_a
    Den_a = np.dot(Not12, np.dot(Den_a, Not12))
    
    for j3 in range(2):
        X = XX[:, :, j3]
        Y = YY[:, :, j3]
        Z = ZZ[:, :, j3]
        Den_a = (1 - err) * Den_a + 0.25 * err * (Den_a + np.dot(X, np.dot(Den_a, X)) + np.dot(Y, np.dot(Den_a, Y)) + np.dot(Z, np.dot(Den_a, Z)))
    
    return P_a, Den_a


def def_operators_car(wt):
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])
    id = np.array([[1, 0], [0, 1]])
    
    XX = np.zeros((2 ** wt, 2 ** wt, wt), dtype=complex)
    YY = np.zeros((2 ** wt, 2 ** wt, wt), dtype=complex)
    ZZ = np.zeros((2 ** wt, 2 ** wt, wt), dtype=complex)
    
    for i in range(wt):
        X = 1  # Difference
        Y = 1
        Z = 1
        
        for j2 in range(wt):
            if j2 != i:
                X = np.kron(X, id)
                Y = np.kron(Y, id)
                Z = np.kron(Z, id)
            else:
                X = np.kron(X, x)
                Y = np.kron(Y, y)
                Z = np.kron(Z, z)
        
        XX[:, :, i] = X
        YY[:, :, i] = Y
        ZZ[:, :, i] = Z
    
    return XX, YY, ZZ

