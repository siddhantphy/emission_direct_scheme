'''
2020 Mark Shui Hu, QuTech

www.github.com/watermarkhu/oop_surface_code
_____________________________________________

'''
import copy

import numpy as np
from scipy import optimize
from .sim import get_data


def get_fit_func(modified_ansatz=False):
    if modified_ansatz:
        return fit_func_m
    else:
        return fit_func


def fit_func(PL, pthres, A, B, C, D, nu, mu):
    p, L = PL
    x = (p - pthres) * L ** (1 / nu)
    return A + B * x + C * x ** 2


def fit_func_m(PL, pthres, A, B, C, D, nu, mu):
    p, L = PL
    x = (p - pthres) * L ** (1 / nu)
    return A + B * x + C * x ** 2 + D * L ** (-1 / mu)


def fit_thresholds(data, modified_ansatz=False, latts=[], probs=[], P_store=1, print_results=True, limit_bounds=True,
                   return_chi_squared_red=False):
    fitL, fitp, fitN, fitt = get_data(data, latts, probs, P_store)
    '''
    Initial parameters for fitting function
    '''
    if limit_bounds:
        g_T, T_m, T_M = (min(fitp) + max(fitp))/2, min(fitp), max(fitp)
    else:
        g_T, T_m, T_M = (min(fitp) + max(fitp)) / 2, 0, max(max(fitp), 0.02)
    g_A, A_m, A_M = 0, -np.inf, np.inf
    g_B, B_m, B_M = 0, -np.inf, np.inf
    g_C, C_m, C_M = 0, -np.inf, np.inf
    gnu, num, nuM = 0.974, 0.8, 1.2

    D_m, D_M = -2, 2
    mum, muM = 0, 3

    g_D, gmu = 1.65, 0.71
    # odd:
    # g_D, gmu = -0.053, 2.1

    par_guess = [g_T, g_A, g_B, g_C, g_D, gnu, gmu]
    bound = [(T_m, A_m, B_m, C_m, D_m, num, mum), (T_M, A_M, B_M, C_M, D_M, nuM, muM)]

    '''
    Fitting data
    '''
    ffit = get_fit_func(modified_ansatz)

    new_error_approach = 1

    log_succ = [t/N for t, N in zip(fitt, fitN)]
    standard_error_log_succ = [np.sqrt(t/N*(1-t/N)/N) for t, N in zip(fitt, fitN)]

    if new_error_approach == 1:
        par, pcov = optimize.curve_fit(
            ffit,
            (fitp, fitL),
            log_succ,
            par_guess,
            bounds=bound,
            sigma=standard_error_log_succ,
            absolute_sigma=False
        )
        perr = np.sqrt(np.diag(pcov))
        red_chi_squared = np.sum(np.divide(np.square(np.subtract(ffit((fitp, fitL), *par), log_succ)), np.square(standard_error_log_succ))) / (len(log_succ)-len(perr))
        print(f"Reduced chi squared is {red_chi_squared}.")
        new_standard_error_log_succ = [np.sqrt(red_chi_squared)*std for std in standard_error_log_succ]
        # for i, (p, L, t, N) in enumerate(zip(fitp, fitL, fitt, fitN)):
        #     if p == 0.00105 and L == 6:
        #         print(f"({p}, {L}): observed success probability is {t/N}")
        #         print(f"({p}, {L}): binomial standard deviation is {standard_error_log_succ[i]}")
        #         print(f"({p}, {L}): rescaled standard deviation is {new_standard_error_log_succ[i]}")
        # print(par, perr[0])


        # par, pcov = optimize.curve_fit(
        #     ffit,
        #     (fitp, fitL),
        #     log_succ,
        #     par_guess,
        #     bounds=bound,
        #     sigma=new_standard_error_log_succ,
        #     absolute_sigma=False
        # )
        # perr = np.sqrt(np.diag(pcov))
        # sigma_squared = np.sum(np.divide(np.square(np.subtract(ffit((fitp, fitL), *par), log_succ)), np.square(new_standard_error_log_succ))) / (len(log_succ)-len(perr))
        # chisq = np.sum(np.square(np.divide(log_succ - ffit((fitp, fitL), *par), new_standard_error_log_succ)))
        # chisq_red = chisq / (len(log_succ) - len(perr))
        # print(f"Reduced chi-squared for fit equals {chisq_red}.")
        # print(sigma_squared)
        # print(par, perr[0])
        # exit(0)


    elif new_error_approach == 2:
        par_arr = []
        for i in range(100):
            delta = [np.random.normal(0., stand_err) for stand_err in standard_error_log_succ]
            log_succ_delta = [log_succ[i] + delta[i] for i in range(len(log_succ))]
            par_ran, pcov_ran = optimize.curve_fit(ffit,
                                                   (fitp, fitL),
                                                   log_succ_delta,
                                                   par_guess,
                                                   bounds=bound)
            par_arr.append(par_ran)
        par_arr = np.array(par_arr)
        par = np.mean(par_arr, 0)
        perr = 1 * np.std(par_arr, 0)

    elif new_error_approach == 3:
        par, pcov = optimize.curve_fit(
            ffit,
            (fitp, fitL),
            [t / N for t, N in zip(fitt, fitN)],
            par_guess,
            bounds=bound,
        )
        residuals = np.subtract(ffit((fitp, fitL), *par), log_succ)
        stand_dev_res = np.std(residuals)
        par_arr = []
        for i in range(100):
            log_succ_delta = [log_succ[i] + np.random.normal(0., stand_dev_res) for i in range(len(log_succ))]
            par_ran, pcov_ran = optimize.curve_fit(ffit,
                                                   (fitp, fitL),
                                                   log_succ_delta,
                                                   par_guess,
                                                   bounds=bound)
            par_arr.append(par_ran)
        par_arr = np.array(par_arr)
        par = np.mean(par_arr, 0)
        perr = 1 * np.std(par_arr, 0)

    else:
        par, pcov = optimize.curve_fit(
            ffit,
            (fitp, fitL),
            [t / N for t, N in zip(fitt, fitN)],
            par_guess,
            bounds=bound,
            sigma=[max(fitN) / n for n in fitN],
        )
        perr = np.sqrt(np.diag(pcov))

    if print_results:
        print("Least squared fitting on dataset results:")
        print("Threshold =", par[0], "+-", perr[0])
        print("A=", par[1], "B=", par[2], "C=", par[3])
        print("D=", par[4], "nu=", par[5], "mu=", par[6])
        print("")

    # sigmasq = np.sum(np.square(log_succ - ffit((fitp, fitL), *par))) / (len(log_succ)-len(perr))
    chisq = np.sum(np.square(np.divide(log_succ - ffit((fitp, fitL), *par), standard_error_log_succ)))
    chisq_red = chisq / (len(log_succ)-len(perr))
    print(f"Reduced chi-squared for fit equals {chisq_red}.")
    # Q = gammaincc(0.5 * NDF, 0.5 * chi2)

    if return_chi_squared_red:
        return (fitL, fitp, fitN, fitt), par, perr[0], chisq_red
    else:
        return (fitL, fitp, fitN, fitt), par, perr[0]


def zoomed_in_fit_thresholds(data, modified_ansatz=False, latts=[], probs=[], P_store=1, print_results=True,
                             limit_bounds=True, return_chi_squared_red=False, zoomed_in_fit=True, zoom_number=3):
    par = None
    chi_squared_red = None
    p_g_values = sorted(list(set([p_g for _, p_g in data.index])))
    if zoomed_in_fit and len(p_g_values) <= 2 * zoom_number:
        zoomed_in_fit = False
    maximum_iterations = 4 if (zoomed_in_fit and return_chi_squared_red) else (3 if zoomed_in_fit else 1)
    for i in range(maximum_iterations):
        data_copy = data if i == maximum_iterations - 1 else copy.deepcopy(data)
        if par is not None:
            if return_chi_squared_red and i == maximum_iterations - 2 and chi_squared_red > 10 and zoom_number >= 3:
                zoom_number = zoom_number - 1
            p_g_thres = par[0]
            # if len(p_g_values) < 2 * zoom_number:
            #     zoom_number = int(len(p_g_values) / 2)
            p_g_closest = min(p_g_values, key=lambda x: abs(x - p_g_thres))
            p_g_index = p_g_values.index(p_g_closest)
            corr = 1 if p_g_thres < p_g_values[p_g_index] else 0
            if p_g_index + zoom_number + 1 - corr >= len(p_g_values):
                p_g_to_keep = p_g_values[(-1 * zoom_number * 2):]
            elif p_g_index - zoom_number + 1 - corr < 0:
                p_g_to_keep = p_g_values[:(zoom_number * 2)]
            else:
                p_g_to_keep = p_g_values[(p_g_index - zoom_number + 1 - corr):(p_g_index + zoom_number + 1 - corr)]
            for index in data_copy.index:
                if index[1] not in p_g_to_keep:
                    data_copy.drop(index, inplace=True)

        outcome = fit_thresholds(data=data_copy, modified_ansatz=modified_ansatz, latts=latts, probs=probs,
                                 P_store=P_store, print_results=print_results, limit_bounds=limit_bounds,
                                 return_chi_squared_red=return_chi_squared_red)
        if i == maximum_iterations - 1:
            return outcome
        elif return_chi_squared_red:
            _, par, _, chi_squared_red = outcome
        else:
            _, par, _ = outcome


