import numpy as np
from .utils import my_distr, convert, def_operators_car, get_common_para, cal_carving_sgl_den

def carving(wt, st, me, err):
    nSmp = 12
    PP, d = my_distr(nSmp, wt)
    XX, YY, ZZ = def_operators_car(wt)
    eta, Dlt, sgm, C, E, ka, P = get_common_para(st)
    
    if me == 'wg':
        def scCeff(dlt, omg):
            return (-2 * (omg + dlt) + 1j) / (-2 * (omg + dlt) + 1j * (1 + P))
        
        dlt1_0 = 0
        omg_0 = 0
        nPh = 4
    elif me == 'cav':
        kc = 2 * ka / E - 2 * ka
        def scCeff(dlt, omg):
            return E / (1 + 2j * omg / (2 * ka + kc) + 4 * C / (1 + 2j * (omg + dlt)))
        dlt1_0 = -4 * C * Dlt * (2 * ka + kc) / (1 + 4 * Dlt ** 2)
        omg_0 = -dlt1_0
        nPh = 2
    else:
        raise ValueError('Invalid media')
    
    P_suc = 0
    rho = np.zeros((2**wt, 2**wt), dtype=complex)
    
    for j in range(nSmp**wt):
        num = convert(j, nSmp, wt)
        dlt0 = dlt1_0 + Dlt
        omg_g = omg_0 + (num - (nSmp - 1) / 2) * d * sgm
        t1_g = scCeff(dlt1_0, omg_g).T
        t0_g = scCeff(dlt0, omg_g).T

        P_i, rho_i = cal_carving_sgl_den(t0_g, t1_g, wt, nPh, eta, err, XX, YY, ZZ)
        P_suc += P_i * PP[j]
        rho += rho_i * PP[j]
    
    P_dk = 1e-6
    dm_I = 2**(-wt) * np.eye(2**wt)
    rho = (P_suc * rho + P_dk * dm_I) / (P_suc + P_dk)
    P_suc += P_dk
    
    return rho, P_suc


def run_carving(wt: int, st: str, me: str, err: float):
    rho, P = carving(wt, st, me, err)
    return [["fake idelity", P.real], rho]