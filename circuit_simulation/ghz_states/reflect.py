import numpy as np

from .utils import my_distr, convert, def_operators_ref, cal_reflect_den, get_common_para


def reflect(wt, st, err):
    nSmp = 12
    PP, d = my_distr(nSmp, wt)
    
    HH, XX, YY, ZZ = def_operators_ref(wt)
    
    eta, Dlt, sgm, C, E, ka, _ = get_common_para(st)
    kc = ka / E - ka
    
    def scCeff(dlt, omg):
        return 1 - 2 * E / (1 + 2j * (omg / (ka + kc)) + 4 * C / (1 + 2j * (omg + dlt)))

    
    if st == 'cur':
        dlt1_0 = -266.7
        omg_0 = 283.0
    elif st == 'nf':
        dlt1_0 = -98.5
        omg_0 = 130.7
    
    P = 0
    rho = np.zeros((2 ** wt, 2 ** wt), dtype=complex)
    
    for j in range(nSmp ** wt):
        num = convert(j, nSmp, wt)
        dlt0 = dlt1_0 + Dlt
        omg_g = omg_0 + (num - (nSmp - 1) / 2) * d * sgm
        
        r1_g = scCeff(dlt1_0, omg_g).T
        r0_g = scCeff(dlt0, omg_g).T
        
        P_i, rho_i = cal_reflect_den(r0_g, r1_g, wt, eta, err, HH, XX, YY, ZZ)
        
        P += P_i * PP[j]
        rho += rho_i * PP[j]
    
    P_dk = 1e-6
    dm_I = 2 ** (-wt) * np.eye(2 ** wt)
    rho = (P * rho + P_dk * dm_I) / (P + P_dk)
    P = P + P_dk
    
    return rho, P

def run_reflection(wt: int, st: str, err: float):
    rho, P = reflect(wt, st, err)
    return [["fake idelity", P.real], rho]