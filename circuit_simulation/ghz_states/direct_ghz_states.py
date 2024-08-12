# import numpy as np
import pandas as pd
import scipy.sparse as sp
import openpyxl
import os
from os import listdir
from os.path import isfile, join
import sys

from .carving import run_carving
from .reflect import run_reflection


def import_density_matrix_fidelity_for_direct_schemes(path: str, choice: int, dynamic_states: bool, gate_error: float):
    # path to the folder having the xlsx files
    # choice to make a choice of the scheme

    if dynamic_states == False:
        cwd = os.getcwd()
        path = f'{cwd}/circuit_simulation/ghz_states/scattering_ghz_states/' if path is None else path

        files = [f for f in listdir(path) if isfile(join(path, f))]
        xlsx_files = [f for f in files if f[-4:] == "xlsx"]
        cur_wt4_file = [path + f for f in xlsx_files if "W4" in f and "cur" in f][0]
        cur_wt3_file = [path + f for f in xlsx_files if "W3" in f and "cur" in f][0]
        nf_wt4_file = [path + f for f in xlsx_files if "W4" in f and "nf" in f][0]
        nf_wt3_file = [path + f for f in xlsx_files if "W3" in f and "nf" in f][0]

        cur_wt4_data = pd.ExcelFile(cur_wt4_file)
        sheet_names = cur_wt4_data.sheet_names
        # print(sheet_names)
        # each data set below stores [[scheme Fidelity, Scheme avg success Probability], Scheme Density Matrix]
        cur_wt4_ref_data = [
            [1 - cur_wt4_data.parse(sheet_names[-1]).loc[0, 'inF_avg'], cur_wt4_data.parse(sheet_names[-1]).loc[0, 'P_avg']],
            sp.csr_matrix(cur_wt4_data.parse(sheet_names[0], header=None))]
        cur_wt4_carving_cav_coh_data = [
            [1 - cur_wt4_data.parse(sheet_names[-1]).loc[2, 'inF_avg'], cur_wt4_data.parse(sheet_names[-1]).loc[2, 'P_avg']],
            sp.csr_matrix(cur_wt4_data.parse(sheet_names[2], header=None).to_numpy())]
        cur_wt4_carving_cav_sgl_data = [
            [1 - cur_wt4_data.parse(sheet_names[-1]).loc[1, 'inF_avg'], cur_wt4_data.parse(sheet_names[-1]).loc[1, 'P_avg']],
            sp.csr_matrix(cur_wt4_data.parse(sheet_names[1], header=None).to_numpy())]
        cur_wt4_carving_wg_coh_data = [
            [1 - cur_wt4_data.parse(sheet_names[-1]).loc[4, 'inF_avg'], cur_wt4_data.parse(sheet_names[-1]).loc[4, 'P_avg']],
            sp.csr_matrix(cur_wt4_data.parse(sheet_names[4], header=None).to_numpy())]
        cur_wt4_carving_wg_sgl_data = [
            [1 - cur_wt4_data.parse(sheet_names[-1]).loc[3, 'inF_avg'], cur_wt4_data.parse(sheet_names[-1]).loc[3, 'P_avg']],
            sp.csr_matrix(cur_wt4_data.parse(sheet_names[3], header=None).to_numpy())]

        cur_wt3_data = pd.ExcelFile(cur_wt3_file)
        cur_wt3_ref_data = [
            [1 - cur_wt3_data.parse(sheet_names[-1]).loc[0, 'inF_avg'], cur_wt3_data.parse(sheet_names[-1]).loc[0, 'P_avg']],
            sp.csr_matrix(cur_wt3_data.parse(sheet_names[0], header=None).to_numpy())]
        cur_wt3_carving_cav_coh_data = [
            [1 - cur_wt3_data.parse(sheet_names[-1]).loc[2, 'inF_avg'], cur_wt3_data.parse(sheet_names[-1]).loc[2, 'P_avg']],
            sp.csr_matrix(cur_wt3_data.parse(sheet_names[2], header=None).to_numpy())]
        cur_wt3_carving_cav_sgl_data = [
            [1 - cur_wt3_data.parse(sheet_names[-1]).loc[1, 'inF_avg'], cur_wt3_data.parse(sheet_names[-1]).loc[1, 'P_avg']],
            sp.csr_matrix(cur_wt3_data.parse(sheet_names[1], header=None).to_numpy())]
        cur_wt3_carving_wg_coh_data = [
            [1 - cur_wt3_data.parse(sheet_names[-1]).loc[4, 'inF_avg'], cur_wt3_data.parse(sheet_names[-1]).loc[4, 'P_avg']],
            sp.csr_matrix(cur_wt3_data.parse(sheet_names[4], header=None).to_numpy())]
        cur_wt3_carving_wg_sgl_data = [
            [1 - cur_wt3_data.parse(sheet_names[-1]).loc[3, 'inF_avg'], cur_wt3_data.parse(sheet_names[-1]).loc[3, 'P_avg']],
            sp.csr_matrix(cur_wt3_data.parse(sheet_names[3], header=None).to_numpy())]

        nf_wt4_data = pd.ExcelFile(nf_wt4_file)
        nf_wt4_ref_data = [
            [1 - nf_wt4_data.parse(sheet_names[-1]).loc[0, 'inF_avg'], nf_wt4_data.parse(sheet_names[-1]).loc[0, 'P_avg']],
            sp.csr_matrix(nf_wt4_data.parse(sheet_names[0], header=None).to_numpy())]
        nf_wt4_carving_cav_coh_data = [
            [1 - nf_wt4_data.parse(sheet_names[-1]).loc[2, 'inF_avg'], nf_wt4_data.parse(sheet_names[-1]).loc[2, 'P_avg']],
            sp.csr_matrix(nf_wt4_data.parse(sheet_names[2], header=None).to_numpy())]
        nf_wt4_carving_cav_sgl_data = [
            [1 - nf_wt4_data.parse(sheet_names[-1]).loc[1, 'inF_avg'], nf_wt4_data.parse(sheet_names[-1]).loc[1, 'P_avg']],
            sp.csr_matrix(nf_wt4_data.parse(sheet_names[1], header=None).to_numpy())]
        nf_wt4_carving_wg_coh_data = [
            [1 - nf_wt4_data.parse(sheet_names[-1]).loc[4, 'inF_avg'], nf_wt4_data.parse(sheet_names[-1]).loc[4, 'P_avg']],
            sp.csr_matrix(nf_wt4_data.parse(sheet_names[4], header=None).to_numpy())]
        nf_wt4_carving_wg_sgl_data = [
            [1 - nf_wt4_data.parse(sheet_names[-1]).loc[3, 'inF_avg'], nf_wt4_data.parse(sheet_names[-1]).loc[3, 'P_avg']],
            sp.csr_matrix(nf_wt4_data.parse(sheet_names[3], header=None).to_numpy())]

        nf_wt3_data = pd.ExcelFile(nf_wt3_file)
        nf_wt3_ref_data = [
            [1 - nf_wt3_data.parse(sheet_names[-1]).loc[0, 'inF_avg'], nf_wt3_data.parse(sheet_names[-1]).loc[0, 'P_avg']],
            sp.csr_matrix(nf_wt3_data.parse(sheet_names[0], header=None).to_numpy())]
        nf_wt3_carving_cav_coh_data = [
            [1 - nf_wt3_data.parse(sheet_names[-1]).loc[2, 'inF_avg'], nf_wt3_data.parse(sheet_names[-1]).loc[2, 'P_avg']],
            sp.csr_matrix(nf_wt3_data.parse(sheet_names[2], header=None).to_numpy())]
        nf_wt3_carving_cav_sgl_data = [
            [1 - nf_wt3_data.parse(sheet_names[-1]).loc[1, 'inF_avg'], nf_wt3_data.parse(sheet_names[-1]).loc[1, 'P_avg']],
            sp.csr_matrix(nf_wt3_data.parse(sheet_names[1], header=None).to_numpy())]
        nf_wt3_carving_wg_coh_data = [
            [1 - nf_wt3_data.parse(sheet_names[-1]).loc[4, 'inF_avg'], nf_wt3_data.parse(sheet_names[-1]).loc[4, 'P_avg']],
            sp.csr_matrix(nf_wt3_data.parse(sheet_names[4], header=None).to_numpy())]
        nf_wt3_carving_wg_sgl_data = [
            [1 - nf_wt3_data.parse(sheet_names[-1]).loc[3, 'inF_avg'], nf_wt3_data.parse(sheet_names[-1]).loc[3, 'P_avg']],
            sp.csr_matrix(nf_wt3_data.parse(sheet_names[3], header=None).to_numpy())]

        choices = {50: cur_wt4_ref_data, 51: cur_wt4_carving_cav_coh_data, 52: cur_wt4_carving_cav_sgl_data,
                53: cur_wt4_carving_wg_coh_data, 54: cur_wt4_carving_wg_sgl_data,
                60: cur_wt3_ref_data, 61: cur_wt3_carving_cav_coh_data, 62: cur_wt3_carving_cav_sgl_data,
                63: cur_wt3_carving_wg_coh_data, 64: cur_wt3_carving_wg_sgl_data,
                70: nf_wt4_ref_data, 71: nf_wt4_carving_cav_coh_data, 72: nf_wt4_carving_cav_sgl_data,
                73: nf_wt4_carving_wg_coh_data, 74: nf_wt4_carving_wg_sgl_data,
                80: nf_wt3_ref_data, 81: nf_wt3_carving_cav_coh_data, 82: nf_wt3_carving_cav_sgl_data,
                83: nf_wt3_carving_wg_coh_data, 84: nf_wt3_carving_wg_sgl_data}

        return choices[choice]
    
    elif dynamic_states == True:
        if choice == 50:
            weight = 4
            scheme = "reflection"
            parameters = "cur"
            print(f"*** Chosen Scheme: Wt.{weight}, {scheme} with {parameters} parameters! ***")
            return run_reflection(weight, parameters, gate_error)
        elif choice == 52:
            weight = 4
            scheme = "carving"
            parameters = "cur"
            medium = "cav"
            print(f"*** Chosen Scheme: Wt.{weight}, {medium} {scheme}  with {parameters} parameters, SGL! ***")
            return run_carving(weight, parameters, medium, gate_error)
        elif choice == 54:
            weight = 4
            scheme = "carving"
            parameters = "cur"
            medium = "wg"
            print(f"*** Chosen Scheme: Wt.{weight}, {medium} {scheme}  with {parameters} parameters, SGL! ***")
            return run_carving(weight, parameters, medium, gate_error)
        elif choice == 60:
            weight = 3
            scheme = "reflection"
            parameters = "cur"
            print(f"*** Chosen Scheme: Wt.{weight}, {scheme} with {parameters} parameters! ***")
            return run_reflection(weight, parameters, gate_error)
        elif choice == 62:
            weight = 3
            scheme = "carving"
            parameters = "cur"
            medium = "cav"
            print(f"*** Chosen Scheme: Wt.{weight}, {medium} {scheme}  with {parameters} parameters, SGL! ***")
            return run_carving(weight, parameters, medium, gate_error)
        elif choice == 64:
            weight = 3
            scheme = "carving"
            parameters = "cur"
            medium = "wg"
            print(f"*** Chosen Scheme: Wt.{weight}, {medium} {scheme}  with {parameters} parameters, SGL! ***")
            return run_carving(weight, parameters, medium, gate_error)
        elif choice == 70:
            weight = 4
            scheme = "reflection"
            parameters = "nf"
            print(f"*** Chosen Scheme: Wt.{weight}, {scheme} with {parameters} parameters! ***")
            return run_reflection(weight, parameters, gate_error)
        elif choice == 72:
            weight = 4
            scheme = "carving"
            parameters = "nf"
            medium = "cav"
            print(f"*** Chosen Scheme: Wt.{weight}, {medium} {scheme}  with {parameters} parameters, SGL! ***")
            return run_carving(weight, parameters, medium, gate_error)
        elif choice == 74:
            weight = 4
            scheme = "carving"
            parameters = "nf"
            medium = "wg"
            print(f"*** Chosen Scheme: Wt.{weight}, {medium} {scheme}  with {parameters} parameters, SGL! ***")
            return run_carving(weight, parameters, medium, gate_error)
        elif choice == 80:
            weight = 3
            scheme = "reflection"
            parameters = "nf"
            print(f"*** Chosen Scheme: Wt.{weight}, {scheme} with {parameters} parameters! ***")
            return run_reflection(weight, parameters, gate_error)
        elif choice == 82:
            weight = 3
            scheme = "carving"
            parameters = "nf"
            medium = "cav"
            print(f"*** Chosen Scheme: Wt.{weight}, {medium} {scheme}  with {parameters} parameters, SGL! ***")
            return run_carving(weight, parameters, medium, gate_error)
        elif choice == 84:
            weight = 3
            scheme = "carving"
            parameters = "nf"
            medium = "wg"
            print(f"*** Chosen Scheme: Wt.{weight}, {medium} {scheme}  with {parameters} parameters, SGL! ***")
            return run_carving(weight, parameters, medium, gate_error)
        else:
            sys.exit("No valid value for the network noise type was chosen with dynamic direct states!")

