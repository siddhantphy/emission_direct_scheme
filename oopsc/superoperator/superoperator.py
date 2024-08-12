import re
import os
import copy
import random
import inspect
import numpy as np
import pandas as pd
from circuit_simulation.node.node import nodes


class Superoperator:

    def __init__(self, file_name, GHZ_success=1.1, additional_superoperators=None, failed_ghz_superoperator=None,
                 supop_date_time=None):
        """
            Superoperator(file_name, graph, GHZ_success=1.1)

                Superoperator object that contains a list of SuperoperatorElements for both stabilizer types
                (plaquette and star) that specifies what errors occur on the stabilizer qubits and the
                corresponding probability of the that error occurring.

                Parameters:
                -----------
                file_name : str
                    File name of the csv file that specifies the errors on the stabilizer qubits. File must be
                    placed in the 'csv_files' folder.
                graph : graph object
                    The graph object that creates the Superoperator object. This is necessary to sort the existing
                    stabilizer per round
                GHZ_success : float [0-1], optional, default=1.1
                    The percentage of stabilizers that are successfully created by the protocol that the superoperator
                    is the result of.

                Attributes:
                -----------
                file_name : str
                    File name of the csv file that specifies the errors on the stabilizer qubits. File must be
                    placed in the 'csv_files' folder.
                GHZ_success : float [0-1], optional, default=1.1
                    The percentage of stabilizers that are successfully created by the protocol that the superoperator
                    is the result of.
                sup_op_elements_p : list
                    The list of SuperoperatorElement objects that specifies the errors and their probabilities
                    occurring on the plaquette stabilizers
                sup_op_elements_s : list
                    The list of SuperoperatorElement objects that specifies the errors and their probabilities
                    occurring on the star stabilizers
                stabs_p1 : dict
                    A dictionary with the z layer a key and the value a list of the stabilizers that are involved in
                    the first round of plaquette stabilizer measurements for that layer.
                stabs_p2 : dict
                    A dictionary with the z layer a key and the value a list of the stabilizers that are involved in
                    the second round of plaquette stabilizer measurements for that layer.
                stabs_s1 : dict
                    A dictionary with the z layer a key and the value a list of the stabilizers that are involved in
                    the first round of star stabilizer measurements for that layer.
                stabs_s2 : dict
                    A dictionary with the z layer a key and the value a list of the stabilizers that are involved in
                    the second round of star stabilizer measurements for that layer.
        """
        if isinstance(file_name, pd.DataFrame):
            self.file_name = file_name
            self._path_to_file = file_name
            self.file_name_failed_superoperator = None
            if not (failed_ghz_superoperator is None or isinstance(failed_ghz_superoperator, pd.DataFrame)):
                raise ValueError("Directly building a superoperator from a pandas DataFrame requires also the failed "
                                 "superoperator to be a pandas DataFrame.")
            self._path_to_file_failed = failed_ghz_superoperator
            if additional_superoperators is not None:
                raise ValueError("Directly building a superoperator from a pandas DataFrame is not yet compatible "
                                 "with also including additional superoperators.")
        else:
            self.file_name = file_name.replace('.csv', '')
            self._path_to_file = os.path.join(os.path.dirname(__file__), "csv_files", self.file_name + ".csv")
            self.file_name_failed_superoperator = (failed_ghz_superoperator.replace('.csv', '')
                                                   if failed_ghz_superoperator is not None else None)
            self._path_to_file_failed = (os.path.join(os.path.dirname(__file__), "csv_files",
                                                      self.file_name_failed_superoperator + ".csv")
                                         if failed_ghz_superoperator is not None else None)
        self._additional_superoperators_files = additional_superoperators
        self.GHZ_success = GHZ_success
        self.supop_date_time = supop_date_time

        self.protocol_name = None
        self.set_number = None
        self.bell_pair_type = None
        self.network_noise_type = None
        self.F_link = None
        self.p_link = None
        self.ent_prot = None
        self.F_prep = None
        self.p_DE = None
        self.mu = None
        self.labda = None
        self.eta = None
        self.alpha = None
        self.t_link = None
        self.t_meas = None
        self.T1n_idle = None
        self.T1n_link = None
        self.T1e_idle = None
        self.T2n_idle = None
        self.T2n_link = None
        self.T2e_idle = None
        self.t_pulse = None
        self.n_DD = None
        self.te_X = None
        self.te_Y = None
        self.tn_X = None
        self.tn_Y = None
        self.te_Z = None
        self.te_H = None
        self.tn_Z = None
        self.tn_H = None
        self.t_CZ = None
        self.t_CX = None
        self.t_CiY = None
        self.t_SWAP = None
        self.p_g = None
        self.p_m = None
        self.p_m_1 = None
        self.noiseless_swap = None
        self.basis_transformation_noise = None
        self.cut_off_time = None
        self.probabilistic = None
        self.decoherence = None
        self.combine = None
        # self.node = nodes[re.search(re.compile("_node(\w*)_"), file_name)[1]]
        self.node = None
        self.probability_sums = {}
        self.date_and_time = None

        # Convert additional superoperators if present
        self.additional_superoperators = self._handle_additional_superoperators(additional_superoperators)

        self.sup_op_elements_p, self.sup_op_elements_s = self._csv_to_superoperator(path_to_file=self._path_to_file,
                                                                                    check_sum=True, set_attributes=True)

        # print('\n\n\n\nsub_elements_s:')
        # for element in self.sup_op_elements_s:
        #     if element.p != 0.0:
        #         print(element.error_array, element.p, element.lie)
        # print('\n\n\n\n')
        # print('sub_elements_p:')
        # for element in self.sup_op_elements_p:
        #     if element.p != 0.0:
        #         print(element.error_array,  element.p, element.lie)
        # print('\n\n\n\n')

        self.failed_ghz_elements = self._set_failed_superoperator(self._path_to_file_failed) if GHZ_success < 1.0 else None
        self.sup_op_elements_p_before_meas, self.sup_op_elements_s_before_meas = \
            self._convert_elements_to_before_projection()

        # print('\n\n\n\nsub_elements_s_before_meas:')
        # for element in self.sup_op_elements_s_before_meas:
        #     if element.p != 0.0:
        #         print(element.error_array, element.p, element.lie)
        # print('\n\n\n\n')
        # print('sub_elements_p_before_meas:')
        # for element in self.sup_op_elements_p_before_meas:
        #     if element.p != 0.0:
        #         print(element.error_array,  element.p, element.lie)
        # print('\n\n\n\n')

        # For speed-up purposes, the superoperator has the stabilizers split into rounds as attributes
        self.stabs_p1, self.stabs_p2, self.stabs_s1, self.stabs_s2 = {}, {}, {}, {}

    def __repr__(self):
        return "Superoperator ({}: {})".format(self.protocol_name, self.file_name)

    def __str__(self):
        return self.__repr__()

    def _handle_additional_superoperators(self, additional_superoperators):
        if additional_superoperators is None:
            return {}

        superoperators = {}
        for id, superoperator in enumerate(additional_superoperators):
            superoperator.replace('.csv', '')
            path_to_file = os.path.join(os.path.dirname(__file__), "csv_files", superoperator + ".csv")
            if 'failed' in superoperator.lower():
                failed = self._set_failed_superoperator(path_to_file)
                superoperators[id] = {'failed': failed}
            else:
                p, s = self._csv_to_superoperator(path_to_file)
                p_before_meas, s_before_meas = self._convert_elements_to_before_projection(p, s)
                superoperators[id] = {'p': p, 's': s, 'p_before_meas': p_before_meas, 's_before_meas': s_before_meas}
        return superoperators

    def _set_failed_superoperator(self, path_to_file=None):
        # If not given use ideal case
        if path_to_file is None:
            return [SuperoperatorElement(1, False, ["I", "I", "I", "I"])]

        return self._csv_to_superoperator(path_to_file, check_sum=True)[0]

    def _csv_to_superoperator(self, path_to_file=None, set_attributes=False, check_sum=False):
        if path_to_file is None:
            return None, None

        if isinstance(path_to_file, pd.DataFrame):
            reader = path_to_file
        else:
            reader = pd.read_csv(path_to_file, sep=";", float_precision='round_trip')

        if 's' in reader:
            sup_op_elements_p, sup_op_elements_s = self._get_elements_superoperator(
                path_to_file=path_to_file, set_attributes=set_attributes)
        else:
            sup_op_elements_p, sup_op_elements_s = self._get_elements_superoperator_old(
                path_to_file=path_to_file, set_attributes=set_attributes)

        if check_sum:
            self._check_sum_probabilities(sup_op_elements_p, sup_op_elements_s, path_to_file)

        sup_op_elements_p = sorted(sup_op_elements_p, reverse=True)
        sup_op_elements_s = sorted(sup_op_elements_s, reverse=True)

        return sup_op_elements_p, sup_op_elements_s

    def _get_elements_superoperator(self, path_to_file=None, set_attributes=True):
        sup_op_elements_p = []
        sup_op_elements_s = []
        if isinstance(path_to_file, pd.DataFrame):
            data_frame = path_to_file
        else:
            file = open(path_to_file)
            data_frame = pd.read_csv(file, sep=';', float_precision='round_trip')
            index = ['error_config', 'lie'] if 'error_idle' not in data_frame else ['error_stab', 'error_idle', 'lie']
            data_frame = data_frame.set_index(index)
        data_frame.fillna({'p': 0., 's': 0.}, inplace=True)

        # If GHZ_success is 1.1 it has obtained the default value and can be overwritten
        if 'GHZ_success' in data_frame and self.GHZ_success == 1.1 and set_attributes:
            self.GHZ_success = float(str(data_frame.GHZ_success[0]).replace(',', '.').replace(" ", ""))
        self._set_superoperator_attributes_if_present(data_frame) if set_attributes else None

        for index, row in data_frame.iterrows():
            error_config = list(index[0])
            error_config_idle = list(index[1]) if len(index) == 3 else None
            lie = index[1] if error_config_idle is None else index[2]
            p_prob = row['p']
            s_prob = row['s']

            sup_op_elements_p.append(SuperoperatorElement(p_prob, lie, error_config,
                                                          error_array_idle=error_config_idle))
            sup_op_elements_s.append(SuperoperatorElement(s_prob, lie, error_config,
                                                          error_array_idle=error_config_idle))

        return sup_op_elements_p, sup_op_elements_s

    def _get_elements_superoperator_old(self, path_to_file, set_attributes=False):
        """
            Retrieves the list of SuperoperatorElements for both the stabilizer types from the supplied csv file
            name.

            CSV file column format
            -----------------------
            p_prob : float
                Column containing the probabilities for the specific errors on the stabilizer qubits.
            p_lie : bool
                Contains the information of whether or not a measurement error occurred.
            p_error : str
                Specify the occurred error configuration
            GHZ_success : float, optional
                Percentage of stabilizers that were able to be created successfully. If not specified, the success
                rate will be set to 1.1
            pm : float, optional
                Measurement error rate
            pg : float, optional
                Gate error rate
            pn : float, optional
                Network error rate

            CSV file format example
            -----------------------
            p_prob;     p_lie;  p_error;    s_prob;    s_lie;   s_error;    GHZ_success;    pg;     pm;    pn;
            0.9509;     0    ;  IIII   ;    0.950 ;    0    ;   IIII   ;    0.99       ;  0.01;   0.01;  0.1;
            0.0384;     0    ;  IIIX   ;    0.038 ;    0    ;   IIIX   ;               ;      ;       ;      ;
        """
        sup_op_elements_p = []
        sup_op_elements_s = []
        if isinstance(path_to_file, pd.DataFrame):
            reader = path_to_file
        else:
            file = open(path_to_file)
            reader = pd.read_csv(file, sep=';', float_precision='round_trip')

        # If GHZ_success is 1.1 it has obtained the default value and can be overwritten
        if 'GHZ_success' in reader and self.GHZ_success == 1.1:
            self.GHZ_success = float(str(reader.GHZ_success[0]).replace(',', '.').replace(" ", ""))

        self._set_superoperator_attributes_if_present(reader) if set_attributes else None

        for i in range(len(list(reader.p_prob))):
            # Do some parsing operations on the entries to ensure proper form
            p_prob = float(str(reader.p_prob[i]).replace(',', '.').replace(" ", ""))
            s_prob = float(str(reader.s_prob[i]).replace(',', '.').replace(" ", ""))
            p_error = [ch for ch in reader.p_error[i].replace(" ", "")]
            s_error = [ch for ch in reader.s_error[i].replace(" ", "")]

            sup_op_elements_p.append(SuperoperatorElement(p_prob, bool(int(reader.p_lie[i])), p_error))
            sup_op_elements_s.append(SuperoperatorElement(s_prob, bool(int(reader.s_lie[i])), s_error))

        return sup_op_elements_p, sup_op_elements_s

    def _set_superoperator_attributes_if_present(self, data_frame):
        attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
        attributes = [a[0] for a in attributes if not (a[0].startswith('__') and a[0].endswith('__'))]
        attributes.remove('GHZ_success')

        for a in data_frame:
            a = 'labda' if a == 'lambda' else a
            if a in attributes:
                a = 'lambda' if a == 'labda' else a
                if re.match("^[0-9,.]*$", str(data_frame[a][0])):
                    value = round(float(str(data_frame[a][0]).replace(",", ".").replace(" ", "")), 9)
                else:
                    value = data_frame[a][0]
                a = 'labda' if a == 'lambda' else a
                setattr(self, a, value)
        setattr(self, "node", self.set_number)

    def _check_sum_probabilities(self, sup_op_elements_p, sup_op_elements_s, filepath):
        # Check if the probabilities add up to 1 to ensure a valid decomposition
        if round(sum(sup_op_elements_p), 4) != 1.0 or round(sum(sup_op_elements_s), 4) != 1.0:
            raise ValueError(
                "\nExpected joint probabilities of the superoperator to add up to one, instead it is:\n"
                "{} for the plaquette errors (difference = {}),\n"
                "{} for the star errors (difference = {})."
                "\nCheck your superoperator csv: \n{}."
                .format(sum(sup_op_elements_p), 1.0 - sum(sup_op_elements_p),
                        sum(sup_op_elements_s), 1.0 - sum(sup_op_elements_s),
                        filepath))

    def _convert_elements_to_before_projection(self, sup_op_elements_p=None, sup_op_elements_s=None):
        """
            This method creates the superoperator elements when superoperator is applied before measurement projection.
            As described in Naomi Nickerson's PhD Thesis, this is necessary when error is applied before measurement
            projection instead of after, which changes the original superoperator elements accordingly.
        """
        if sup_op_elements_s is sup_op_elements_p is None:
            sup_op_elements_p = self.sup_op_elements_p
            sup_op_elements_s = self.sup_op_elements_s

        sup_op_elements_p_before_meas = copy.deepcopy(sup_op_elements_p)
        sup_op_elements_s_before_meas = copy.deepcopy(sup_op_elements_s)

        for sup_op_el_p2, sup_op_el_s2 in zip(sup_op_elements_p_before_meas, sup_op_elements_s_before_meas):
            if (sup_op_el_p2.error_array.count("Y") + sup_op_el_p2.error_array.count("X")) % 2 == 1:
                sup_op_el_p2.lie = not sup_op_el_p2.lie
            if (sup_op_el_s2.error_array.count("Y") + sup_op_el_s2.error_array.count("Z")) % 2 == 1:
                sup_op_el_s2.lie = not sup_op_el_s2.lie

        return sup_op_elements_p_before_meas, sup_op_elements_s_before_meas

    def set_stabilizer_rounds(self, graph):
        """
            Obtain for both type of stabilizers the stabilizers that will be measured each round for every
            measurement layer z. These rounds are necessary when non local stabilizer measurements protocols
            are used.

            Parameters
            ----------
            graph : graph object
                The graph object that the Superoperator object is applied to
        """
        cycles = graph.cycles if graph.cycles else 1
        # Return if the stabilizer rounds have already been configured
        if self.stabs_s1:
            return

        # Initialise stabilizer round dictionaries with empty arrays for each layer
        for z in range(cycles):
            self.stabs_p1[z] = []
            self.stabs_p2[z] = []
            self.stabs_s1[z] = []
            self.stabs_s2[z] = []

        # Append the stabilizer to the stabilizer round list according to the layer they belong
        def append_stabilizers(stabilizer_list, sID):
            for z in range(cycles):
                stabilizer_list[z].append(graph.S[z][sID])

        # Determine for a stabilizer position (no matter the layer) in which list it belongs
        for stab in graph.S[0].values():
            even_odd = stab.sID[1] % 2
            if stab.sID[2] % 2 == even_odd:
                if stab.sID[0] == 0:
                    append_stabilizers(self.stabs_s1, stab.sID)
                else:
                    append_stabilizers(self.stabs_p1, stab.sID)
            else:
                if stab.sID[0] == 0:
                    append_stabilizers(self.stabs_s2, stab.sID)
                else:
                    append_stabilizers(self.stabs_p2, stab.sID)

    def reset_stabilizer_rounds(self):
        self.stabs_p1.clear(), self.stabs_p2.clear(), self.stabs_s1.clear(), self.stabs_s2.clear()

    def _get_probability_sum(self, elements) -> float:
        key = elements[0].p
        if key in self.probability_sums:
            sum_prob = self.probability_sums[key]
        else:
            sum_prob = sum(el.p for el in elements)
            self.probability_sums[key] = sum_prob

        return sum_prob

    def get_supop_el_by_prob(self, superoperator_elements):
        """
            Retrieve a SuperoperatorElement from a list of SuperoperatorElements based on the probabilities of
            these SuperoperatorElements. This means, that the method is more likely to return a SuperoperatorElement
            with a high probability than one with a low probability.

            Parameters
            ----------
            superoperator_elements : list
                List containing SuperoperatorElements of which a SuperoperatorElement should be picked

            Returns
            -------
            superoperator_element : SuperoperatorElement
                A SuperoperatorElement is returned from the superoperator_elements list based on the probability
        """
        prob_sum = self._get_probability_sum(superoperator_elements)
        r = random.random() * prob_sum
        index = 0
        while r >= 0 and index <= len(superoperator_elements):
            # If total probability does not count up to 1, then return first element if 'r' lies outside the probability
            if index == len(superoperator_elements):
                return superoperator_elements[0]
            r -= superoperator_elements[index].p
            index += 1
        return superoperator_elements[index - 1]


class SuperoperatorElement:

    def __init__(self, p, lie, error_array, error_density_matrix=None, fused_configs=None, error_array_idle=None):
        """
            SuperoperatorElement(p, lie, error_array)

                Used as building block for the Superoperator object. It contains the error configuration on the
                stabilizer qubits, the presents of a measurement error and the corresponding probability.

                Parameters
                ----------
                p : float
                    Probability of the specific error configurations occurring on the stabilizer qubits
                lie : bool
                    Whether the a measurement error is involved.
                error_array : list
                    List of four characters that represent Pauli errors occurring on a qubit. One can choose from
                    'X', 'Y', 'Z' or 'I'.
        """
        self.p = p
        self.lie = lie
        self.error_array = error_array
        self.error_array_idle = error_array_idle
        self.error_density_matrix = error_density_matrix
        self.fused_configs = fused_configs if fused_configs is not None else {"".join(error_array):
                                                                              error_density_matrix}
        self.id = str(p) + str(lie) + str(error_array)

    def __repr__(self):
        return "SuperoperatorElement(p:{}, lie:{}, errors:{})".format(self.p, self.lie, self.error_array)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if type(other) != SuperoperatorElement:
            return False
        self_error_sorted = sorted(self.error_array)
        other_error_sorted = sorted(other.error_array)
        return self.p == other.p and self.lie == other.lie and self_error_sorted == other_error_sorted

    def __hash__(self):
        return hash(str(self.p) + str(self.lie) + str(self.error_array))

    def __ge__(self, other):
        return self.p >= other.p

    def __gt__(self, other):
        return self.p > other.p

    def __le__(self, other):
        return self.p <= other.p

    def __lt__(self, other):
        return self.p < other.p

    def __add__(self, other):
        return self.p + other.p

    def __radd__(self, other):
        return self.p + other

    @staticmethod
    def file_path():
        return str(os.path.dirname(__file__))

    def full_equals(self, other, rnd=8, sort_array=True):
        if sort_array:
            self.error_array.sort()
            other.error_array.sort()

        return (round(self.p, rnd) == round(other.p, rnd)
                and self.lie == other.lie
                and self.error_array == other.error_array)

    def error_array_lie_equals(self, other, sort_array=True):
        if sort_array:
            self.error_array.sort()
            other.error_array.sort()

        return self.lie == other.lie and self.error_array == other.error_array

    def probability_lie_equals(self, other, rnd=8):
        return round(self.p, rnd) == round(other.p, rnd) and self.lie == other.lie

    def error_density_matrix_equals(self, other):
        if type(other) != SuperoperatorElement:
            raise ValueError("Compared value should be of type SuperoperatorElement")
        return self._csr_matrix_equal(self.error_density_matrix, other.error_density_matrix)

    def any_error_density_matrix_equals(self, other):
        if type(other) != SuperoperatorElement:
            raise ValueError("Compared value should be of type SuperoperatorElement")
        return any([self._csr_matrix_equal(self.error_density_matrix, other_dens) for other_dens in
                    other.fused_configs.values()])

    @staticmethod
    def _csr_matrix_equal(a1, a2):
        return (np.array_equal(a1.indptr, a2.indptr) and
                np.array_equal(a1.indices, a2.indices) and
                np.array_equal(a1.data, a2.data))

