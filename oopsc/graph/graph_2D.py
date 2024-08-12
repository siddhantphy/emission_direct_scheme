'''
2020 Mark Shui Hu, QuTech

www.github.com/watermarkhu/oop_surface_code
_____________________________________________

We define the unit cell, which contains two qubits, a star operator and plaquette operator.

    |       |
- Star  -  Q_0 -     also top (T) qubit
    |       |
-  Q_1  - Plaq  -    also down (D) qubit
    |       |

Each cell is indicated by its y and x coordiantes. As such every qubit and stabilizer can by identified by a unique ID number:

Qubits: qID (td, y, x)          Stabilizers: sID (ertype, y, x)
    Q_0:    (0, y, x)               Star:   (0, y, x)
    Q_1:    (1, y, x)               Plaq:   (1, y, x)

The 2D graph (toric/planar) is a square lattice with 1 layer of these unit cells.
'''
from ..plot import plot_graph_lattice as pgl
from ..plot import plot_unionfind as puf
import random
import numpy as np
from ..superoperator import superoperator as so
from copy import copy


class toric(object):
    """
    The graph in which the vertices, edges and clusters exist. Has the following parameters

    size            1D size of the graph
    range           range over 1D that is often used
    decoder         decoder object to use for this graph, also saves graph object to decoder object
    decode_layer    z layer on which the qubits is decoded, 0 for 2D graph graph
    C               dict of clusters with
                        Key:    cID number
                        Value:  Cluster object
    S               dict of stabilizers with
                        Key:    sID number
                        Value:  Stab object
    Q               dict of qubits with
                        Key:    qID number
                        Value:  Qubit object with two Edge objects
    matching_weight total length of edges in the matching
    """
    def __init__(self, size, decoder, *args, plot_config={}, dim=2, **kwargs):
        self.dim = dim
        self.size = size
        self.range = range(size)
        self.decoder = decoder
        decoder.graph = self
        self.decode_layer = 0
        self.cID = 0
        self.C, self.S, self.Q = {}, {}, {}
        self.matching_weight = []
        self.cycles = 1

        self.init_graph_layer()

        for key, value in kwargs.items():
            setattr(self, key, value)
        self.plot_config = plot_config
        self.gl_plot = pgl.plot_2D(self, **plot_config) if self.plot2D else None

        self.superoperator = None

    def __repr__(self):
        return f"2D {self.__class__.__name__} graph object with"

    def init_uf_plot(self):
        '''
        Initializes plot of unionfind decoder.
        '''
        self.uf_plot = puf.plot_2D(self, **self.plot_config)
        return self.uf_plot


    def count_matching_weight(self, z=0):
        '''
        Loops through all qubits on the layer and counts the number of matchings edges
        '''
        weight = 0
        for qubit in self.Q[z].values():
            if qubit.E[0].matching == 1:
                weight += 1
            if qubit.E[1].matching == 1:
                weight += 1
        self.matching_weight.append(weight)
    '''
    ########################################################################################

                                    Surface code functions

    ########################################################################################
    '''

    def init_graph_layer(self, z=0):
        '''
        param z     layer
        Initializes a layer of the graph structure of a toric lattice
        '''
        self.dirs = ["n", "s", "e", "w"]
        self.S[z], self.Q[z], = {}, {}

        # Add stab objects to graph
        for ertype in [0,1]:
            for y in self.range:
                for x in self.range:
                    self.add_stab(ertype, y, x, z)

        # Add edges to graph
        for y in self.range:
            for x in self.range:
                vW, vE = self.S[z][(0, y, x)], self.S[z][(0, y, (x + 1) % self.size)]
                vN, vS = self.S[z][(1, (y - 1) % self.size, x)], self.S[z][(1, y, x)]
                self.add_qubit(0, y, x, z, vW, vE, vN, vS)

                vN, vS = self.S[z][(0, y, x)], self.S[z][(0, (y + 1) % self.size, x)]
                vW, vE = self.S[z][(1, y, (x - 1) % self.size)], self.S[z][(1, y, x)]
                self.add_qubit(1, y, x, z, vW, vE, vN, vS)

    def apply_and_measure_errors(self, pX=0, pZ=0, pE=0, **kwargs):
        '''
        Initilizes errors on the qubits and measures the stabilizers on the graph
        '''

        self.init_erasure(pE=pE)
        self.init_pauli(pX=pX, pZ=pZ)               # initialize errors
        self.measure_stab()                         # Measure stabilizers

    def perform_stabilizer_measurement_cycles_with_superoperator(self, superoperator, networked_architecture=False):
        self.superoperator = superoperator
        self.superoperator.set_stabilizer_rounds(self)

        if not networked_architecture:
            self.stabilizer_cycle_monolithic_architecture()
        else:
            self.stabilizer_cycle_with_superoperator_naomi_order()

    def init_erasure(self, pE=0, **kwargs):
        """
        Initializes an erasure error with probability pE, which will take form as a uniformly chosen pauli X and/or Z error.
        """
        if pE == 0:
            return

        for qubit in self.Q[0].values():
            if random.random() < pE:
                qubit.erasure = True
                rand = random.random()
                if rand < 0.25:
                    qubit.E[0].state = 1
                elif 0.25 <= rand < 0.5:
                    qubit.E[1].state = 1
                elif 0.5 <= rand < 0.75:
                    qubit.E[0].state = 1
                    qubit.E[1].state = 1

        if self.gl_plot: self.gl_plot.plot_erasures()


    def init_pauli(self, pX=0, pZ=0, **kwargs):
        """
        initiates Pauli X and Z errors on the lattice based on the error rates
        """
        for qubit in self.Q[0].values():
            if pX != 0 and random.random() < pX:
                qubit.E[0].state = 1
            if pZ != 0 and random.random() < pZ:
                qubit.E[1].state = 1

        if self.gl_plot: self.gl_plot.plot_errors()

    def measure_stab(self, **kwargs):
        """
        The measurement outcomes of the stabilizers, which are the vertices on the self are saved to their corresponding vertex objects.
        """
        stabs = self.S[0].values()
        if "stabs" in kwargs.keys():
            stabs = kwargs["stabs"]

        for i, stab in enumerate(stabs):
            for dir in self.dirs:
                if dir in stab.neighbors:
                    vertex, edge = stab.neighbors[dir]
                    if edge.state:
                        stab.parity = 1 - stab.parity
                    stab.state = stab.parity

        if self.gl_plot: self.gl_plot.plot_syndrome()


    def logical_error(self, z=0):
        """
        Finds whether there are any logical errors on the lattice/self. The logical error is returned as [Xvertical, Xhorizontal, Zvertical, Zhorizontal], where each item represents a homological Loop
        """
        if self.gl_plot: self.gl_plot.plot_final()

        logical_error = [0, 0, 0, 0]

        for i in self.range:
            if self.Q[z][(0, 0, i)].E[0].state:
                logical_error[0] = 1 - logical_error[0]
            if self.Q[z][(1, i, 0)].E[0].state:
                logical_error[1] = 1 - logical_error[1]
            if self.Q[z][(1, 0, i)].E[1].state:
                logical_error[2] = 1 - logical_error[2]
            if self.Q[z][(0, i, 0)].E[1].state:
                logical_error[3] = 1 - logical_error[3]

        errorless = True if logical_error == [0, 0, 0, 0] else False
        return logical_error, errorless

    def stabilizer_cycle_monolithic_architecture(self, z=0):
        """
            Method is used to run a full stabilizer measurement cycle that is used to verify the superoperator
            implementation. When a superoperator is used that represents iid error on the stabilizer qubits, then
            this method ensures a similar error implementation as the usual iid error implementation (see the
            'apply_and_measure_errors' method). This way, threshold simulation results with the superoperator

            implementation can be compared with the usual implementation to verify the implementation.
            Parameters
            ----------
            z : int, optional, z=0
                Integer value to indicate the layer for which the stabilizer measurement cycle should run.
        """
        self.set_qubit_states_to_state_previous_layer(z)

        # Only apply error once, since for each run of 'superoperator_error' there is looped over all qubits.
        measurement_errors_p1, _ = self.superoperator_error(self.superoperator.stabs_p1[z],
                                                            self.superoperator.sup_op_elements_p)

        # Run 'superoperator_error' three more times, but don't apply the selected error. It is only used to acquire the
        # measurement errors that will later be applied to the stabilizer measurements.
        measurement_errors_p2, _ = self.superoperator_error(self.superoperator.stabs_p2[z],
                                                            self.superoperator.sup_op_elements_p,
                                                            apply_error=False)
        measurement_errors_s1, _ = self.superoperator_error(self.superoperator.stabs_s1[z],
                                                            self.superoperator.sup_op_elements_s,
                                                            apply_error=False)
        measurement_errors_s2, _ = self.superoperator_error(self.superoperator.stabs_s2[z],
                                                            self.superoperator.sup_op_elements_s,
                                                            apply_error=False)

        # Measure all stabilizers and apply the according measurement errors found above.
        self.measure_stab(stabs=self.superoperator.stabs_p1[z],
                          z=z,
                          measurement_errors=measurement_errors_p1)
        self.measure_stab(stabs=self.superoperator.stabs_p2[z],
                          z=z,
                          measurement_errors=measurement_errors_p2)

        # The apply error and measure star stabilizers in two rounds.
        self.measure_stab(stabs=self.superoperator.stabs_s1[z],
                          z=z,
                          measurement_errors=measurement_errors_s1)
        self.measure_stab(stabs=self.superoperator.stabs_s2[z],
                          z=z,
                          measurement_errors=measurement_errors_s2)

    def stabilizer_cycle_with_superoperator(self, z=0, supop_elements="before_meas"):
        """
            Performs a full stabilizer measurement cycle divided into two rounds per stabilizer. It is done in rounds
            per stabilizer to simulate the situation where GHZ states are used to create a networked version of the
            surface code, as described in Naomi Nickerson's PhD Thesis. These rounds are used, since each qubit can only
            allow for one entanglement link at the same time.

            Parameters
            ----------
            z : int, optional, z=0
                Integer value to indicate the layer for which the stabilizer measurement cycle should run.
            supop_elements : str, default='before_meas'
                String describing which superoperator elements should be used: the ones that are altered to describe
                measurement errors before data qubit errors are applied, or the original ones
        """
        self.set_qubit_states_to_state_previous_layer(z)

        supop_elements_p = self.superoperator.sup_op_elements_p_before_meas if supop_elements == "before_meas" \
            else self.superoperator.sup_op_elements_p
        supop_elements_s = self.superoperator.sup_op_elements_s_before_meas if supop_elements == "before_meas" \
            else self.superoperator.sup_op_elements_s

        # First apply error and measure plaquette stabilizers in two rounds
        measurement_errors_p1, _ = self.superoperator_error(self.superoperator.stabs_p1[z], supop_elements_p)
        self.measure_stab(stabs=self.superoperator.stabs_p1[z],
                          z=z,
                          measurement_errors=measurement_errors_p1)

        measurement_errors_p2, _ = self.superoperator_error(self.superoperator.stabs_p2[z], supop_elements_p)
        self.measure_stab(stabs=self.superoperator.stabs_p2[z],
                          z=z,
                          measurement_errors=measurement_errors_p2)

        # The apply error and measure star stabilizers in two rounds
        measurement_errors_s1, _ = self.superoperator_error(self.superoperator.stabs_s1[z], supop_elements_s)
        self.measure_stab(stabs=self.superoperator.stabs_s1[z],
                          z=z,
                          measurement_errors=measurement_errors_s1)

        measurement_errors_s2, _ = self.superoperator_error(self.superoperator.stabs_s2[z], supop_elements_s)
        self.measure_stab(stabs=self.superoperator.stabs_s2[z],
                          z=z,
                          measurement_errors=measurement_errors_s2)

    def stabilizer_cycle_weight_two_four_architecture(self, z=0):
        self.set_qubit_states_to_state_previous_layer(z)

        first_round_p_sup = self.superoperator.additional_superoperators[0]['p_before_meas']
        first_round_s_sup = self.superoperator.additional_superoperators[0]['s_before_meas']
        failed_first_round_sup = (self.superoperator.additional_superoperators[1]['failed']
                                  if len(self.superoperator.additional_superoperators) > 1 else None)

        second_round_p_stabs = [stab for i, stab in enumerate(self.superoperator.stabs_p2[z])
                                if (divmod(i, self.size/2)[0] % 2) == 0]
        third_round_p_stabs = [stab for i, stab in enumerate(self.superoperator.stabs_p2[z])
                               if (divmod(i, self.size/2)[0] % 2) == 1]
        p_second_and_third_round_stabs = [second_round_p_stabs, third_round_p_stabs]

        second_round_s_stabs = [stab for i, stab in enumerate(self.superoperator.stabs_s2[z])
                                if (divmod(i, self.size/2)[0] % 2) == 0]
        third_round_s_stabs = [stab for i, stab in enumerate(self.superoperator.stabs_s2[z])
                               if (divmod(i, self.size/2)[0] % 2) == 1]
        s_second_and_third_round_stabs = [second_round_s_stabs, third_round_s_stabs]

        # ------------------------------- Plaquette stabilizers --------------------------------

        # First round is weight two superoperator, can be applied without any special alterations
        measurement_errors_p1, _ = self.superoperator_error(self.superoperator.stabs_p1[z], first_round_p_sup,
                                                            failed_superoperator_elements=failed_first_round_sup)
        self.measure_stab(stabs=self.superoperator.stabs_p1[z],
                          z=z,
                          measurement_errors=measurement_errors_p1)

        # Second and third round are different. Half of qubits are involved in stabilizer measurement,
        # other half is idle. Idle error should be included in the superoperator
        for stabs in p_second_and_third_round_stabs:
            meas_errors, _ = self.superoperator_error(stabs, self.superoperator.sup_op_elements_p_before_meas,
                                                      architecture="weight_two_four")

            self.measure_stab(stabs=stabs,
                              z=z,
                              measurement_errors=meas_errors)

        # ------------------------------- Star stabilizers --------------------------------

        # (same as above, but now for star stabilizers)
        measurement_errors_s1, _ = self.superoperator_error(self.superoperator.stabs_s1[z], first_round_s_sup,
                                                            failed_superoperator_elements=failed_first_round_sup)
        self.measure_stab(stabs=self.superoperator.stabs_s1[z],
                          z=z,
                          measurement_errors=measurement_errors_s1)

        for stabs in s_second_and_third_round_stabs:
            meas_errors, _ = self.superoperator_error(stabs, self.superoperator.sup_op_elements_s_before_meas,
                                                      architecture="weight_two_four")
            self.measure_stab(stabs=stabs,
                              z=z,
                              measurement_errors=meas_errors)

    def stabilizer_cycle_weight_three_architecture(self, z=0):
        self.set_qubit_states_to_state_previous_layer(z)

        first_round_p_stabs = [stab for i, stab in enumerate(self.superoperator.stabs_p1[z])
                               if (divmod(i, self.size / 2)[0] % 2) == 0]
        second_round_p_stabs = [stab for i, stab in enumerate(self.superoperator.stabs_p1[z])
                                if (divmod(i, self.size / 2)[0] % 2) == 1]
        third_round_p_stabs = [stab for i, stab in enumerate(self.superoperator.stabs_p2[z])
                               if (divmod(i, self.size / 2)[0] % 2) == 0]
        fourth_round_p_stabs = [stab for i, stab in enumerate(self.superoperator.stabs_p2[z])
                                if (divmod(i, self.size / 2)[0] % 2) == 1]
        p_stabilizers = [first_round_p_stabs, second_round_p_stabs, third_round_p_stabs, fourth_round_p_stabs]

        first_round_s_stabs = [stab for i, stab in enumerate(self.superoperator.stabs_s1[z])
                               if (divmod(i, self.size / 2)[0] % 2) == 0]
        second_round_s_stabs = [stab for i, stab in enumerate(self.superoperator.stabs_s1[z])
                                if (divmod(i, self.size / 2)[0] % 2) == 1]
        third_round_s_stabs = [stab for i, stab in enumerate(self.superoperator.stabs_s2[z])
                               if (divmod(i, self.size / 2)[0] % 2) == 0]
        fourth_round_s_stabs = [stab for i, stab in enumerate(self.superoperator.stabs_s2[z])
                                if (divmod(i, self.size / 2)[0] % 2) == 1]
        s_stabilizers = [first_round_s_stabs, second_round_s_stabs, third_round_s_stabs, fourth_round_s_stabs]

        # ------------------------------- Plaquette stabilizers --------------------------------

        for stabs in p_stabilizers:
            measurement_errors, _ = self.superoperator_error(stabs, self.superoperator.sup_op_elements_p_before_meas,
                                                             architecture="weight_three")
            self.measure_stab(stabs=stabs,
                              z=z,
                              measurement_errors=measurement_errors)
        # ------------------------------- Star stabilizers --------------------------------

        for stabs in s_stabilizers:
            measurement_errors, _ = self.superoperator_error(stabs, self.superoperator.sup_op_elements_s_before_meas,
                                                             architecture="weight_three")
            self.measure_stab(stabs=stabs,
                              z=z,
                              measurement_errors=measurement_errors)

    def stabilizer_cycle_with_superoperator_naomi_order(self, z=0):
        """
            Method applies qubit and measurement errors to the qubits for the specified layer (z).
            It is done in rounds per stabilizer to simulate the situation where GHZ states are used to create a
            networked version of the surface code, as described in Naomi Nickerson's PhD Thesis. These rounds are used,
            since each qubit can only allow for one entanglement link at the same time. The order of applying error and
            stabilizer measurements is similar to the order described in Naomi Nickerson's PhD thesis

            Parameters
            ----------
            z : int, optional, default=0
                Integer that indicates the layer on which the error should be applied
        """
        self.set_qubit_states_to_state_previous_layer(z)

        # First apply error to first round of plaquette stabilizers qubits
        measurement_errors_p1, _ = self.superoperator_error(self.superoperator.stabs_p1[z],
                                                            self.superoperator.sup_op_elements_p_before_meas)

        # Get the measurement errors and qubit errors from the second round, but do not yet apply the error
        measurement_errors_p2, qubit_errors_p2 = self.superoperator_error(self.superoperator.stabs_p2[z],
                                                                          self.superoperator.sup_op_elements_p,
                                                                          apply_error=False)

        # Measure all plaquette stabilizers and apply the corresponding measurement errors
        self.measure_stab(stabs=self.superoperator.stabs_p1[z],
                          z=z,
                          measurement_errors=measurement_errors_p1)
        self.measure_stab(stabs=self.superoperator.stabs_p2[z],
                          z=z,
                          measurement_errors=measurement_errors_p2)

        # Now apply error on the second round of plaquette stabilizers qubits
        self.superoperator_error(self.superoperator.stabs_p2[z], qubit_errors=qubit_errors_p2)

        # ----------------------------------------------------------------------------------
        # -------------- Same as above only now for the star stabilizers -------------------
        # ----------------------------------------------------------------------------------
        measurement_errors_s1, _ = self.superoperator_error(self.superoperator.stabs_s1[z],
                                                            self.superoperator.sup_op_elements_s_before_meas)

        measurement_errors_s2, qubit_errors_s2 = self.superoperator_error(self.superoperator.stabs_s2[z],
                                                                          self.superoperator.sup_op_elements_s,
                                                                          apply_error=False)
        self.measure_stab(stabs=self.superoperator.stabs_s1[z],
                          z=z,
                          measurement_errors=measurement_errors_s1)
        self.measure_stab(stabs=self.superoperator.stabs_s2[z],
                          z=z,
                          measurement_errors=measurement_errors_s2)

        self.superoperator_error(self.superoperator.stabs_s2[z], qubit_errors=qubit_errors_s2)

    def stabilizer_cycle_with_superoperator_phenomenological(self, z=0):
        """
            Performs a full stabilizer measurement cycle divided into two rounds per stabilizer. It is done in rounds
            per stabilizer to simulate the situation where GHZ states are used to create a networked version of the
            surface code, as described in Naomi Nickerson's PhD Thesis. These rounds are used, since each qubit can only
            allow for one entanglement link at the same time.

            Parameters
            ----------
            z : int, optional, z=0
                Integer value to indicate the layer for which the stabilizer measurement cycle should run.
        """
        self.set_qubit_states_to_state_previous_layer(z)

        # First apply error and measure plaquette stabilizers in two rounds
        measurement_errors_p1, _ = self.superoperator_error(self.superoperator.stabs_p1[z],
                                                            self.superoperator.sup_op_elements_p)

        measurement_errors_p2, _ = self.superoperator_error(self.superoperator.stabs_p2[z],
                                                            self.superoperator.sup_op_elements_p)

        # The apply error and measure star stabilizers in two rounds
        measurement_errors_s1, _ = self.superoperator_error(self.superoperator.stabs_s1[z],
                                                            self.superoperator.sup_op_elements_s)

        measurement_errors_s2, _ = self.superoperator_error(self.superoperator.stabs_s2[z],
                                                            self.superoperator.sup_op_elements_s)

        self.measure_stab(stabs=self.superoperator.stabs_p1[z],
                          z=z,
                          measurement_errors=measurement_errors_p1)

        self.measure_stab(stabs=self.superoperator.stabs_p2[z],
                          z=z,
                          measurement_errors=measurement_errors_p2)

        self.measure_stab(stabs=self.superoperator.stabs_s1[z],
                          z=z,
                          measurement_errors=measurement_errors_s1)

        self.measure_stab(stabs=self.superoperator.stabs_s2[z],
                          z=z,
                          measurement_errors=measurement_errors_s2)

    def superoperator_error(self, stabs, superoperator_elements=None, qubit_errors=None, apply_error=True,
                            architecture=None, failed_superoperator_elements=None):
        """
            Based on the probability of the superoperator elements, this method applies error to the qubits of the
            specified stabilizers and saves the according measurement error value (True or False) to a list. The
            measurement error list together with the applied errors on the qubits are returned.

            Parameters
            ----------
            stabs : list
                List of stabilizers on which qubits the (probabilistic) error should be applied on.
            superoperator_elements : list, optional, default=None
                List containing the superoperator elements corresponding to the stabilizers (stabs parameter) that have
                been passed.
            qubit_errors : list, optional, default=None
                List containing error configurations for the stabilizer qubits. The index corresponds with the index of
                the stabilizer list.
            apply_error : bool, optional, default=True
                Used to only obtain the measurement error list and the qubit error list without actually applying the
                error on the qubits. This can be used when the measurements are done prior to the application of the
                error to the qubits.
            architecture : str
                If superoperator for a different architecture (now the weight 4-2 and weight 3 architectures are
                known) is assessed, then there are idle qubits. These will obtain error differently. This can be
                specified with this parameter
            failed_superoperator_elements : list
                List of superoperator elements that represent the error on the qubits when the GHZ state has failed
                to be formed.

            Returns
            -------
            measurement_errors : list
                List containing boolean values that indicate if a measurement error has happened. The indices of the
                list correspond with the indices of the passed 'stabs' parameter.
            qubit_errors : list
                List containing the error configuration on the 4 qubits of a stabilizer. The indices of the list
                correspond with the indices of the passed 'stabs' parameter.
        """
        measurement_errors = []
        measurement_errors_2 = {}
        if qubit_errors is None:
            # qubit_errors = []
            sample_new_errors = True
        else:
            sample_new_errors = False
        qubit_errors_2 = {}
        if failed_superoperator_elements is None:
            failed_superoperator_elements = self.superoperator.failed_ghz_elements

        for stab_index, stab in enumerate(stabs):
            # Get the error config on the stabilizer qubits by picking an element from the superoperator
            if superoperator_elements is not None and sample_new_errors:
                # If GHZ state failed to be formed, the failed superoperator elements should be used
                if random.random() > self.superoperator.GHZ_success:
                    random_super_op_element = self.superoperator.get_supop_el_by_prob(failed_superoperator_elements)
                    measurement_errors.append("failed")
                    measurement_errors_2[stab.sID] = "failed"
                else:
                    random_super_op_element = self.superoperator.get_supop_el_by_prob(superoperator_elements)
                    measurement_errors.append(random_super_op_element.lie)
                    measurement_errors_2[stab.sID] = random_super_op_element.lie
                # qubit_errors.append(random_super_op_element.error_array)
                qubit_errors_2[stab.sID] = random_super_op_element

            if not apply_error:
                continue

            if sample_new_errors:
                super_op_element = copy(qubit_errors_2[stab.sID])
            else:
                super_op_element = copy(qubit_errors[stab.sID])

            random_error_array = copy(super_op_element.error_array)
            random_error_array_idle = copy(super_op_element.error_array_idle)

            # Skip error apply loop if error-array equals the noiseless case
            if (random_error_array == ["I", "I", "I", "I"] and
               (random_error_array_idle is None or random_error_array_idle == ["I", "I", "I", "I"])):
                continue

            # # Apply 'Twirling' by shuffling the error array for the 4 qubits (do not shuffle in case of idle qubits)
            # if random_error_array_idle is None and len(superoperator_elements) < 80:
            #     np.random.shuffle(random_error_array)

            for i, dir in enumerate(self.dirs):
                self._apply_error_stabilizer(dir, i, random_error_array, stab)
                self._apply_error_idle(dir, i, random_error_array_idle, stab, architecture)

        if self.gl_plot: self.gl_plot.plot_errors()

        return measurement_errors_2, qubit_errors_2

    def _apply_error_stabilizer(self, dir, i, random_error_array, stab):
        if random_error_array[i] == "I":
            return

        if dir in stab.neighbors:
            _, edge = stab.neighbors[dir]

            if random_error_array[i] == "X":
                # XOR (^) with current state of the qubit
                edge.qubit.E[0].state = 1 ^ edge.qubit.E[0].state

            elif random_error_array[i] == "Y":
                edge.qubit.E[0].state = 1 ^ edge.qubit.E[0].state
                edge.qubit.E[1].state = 1 ^ edge.qubit.E[1].state

            elif random_error_array[i] == "Z":
                edge.qubit.E[1].state = 1 ^ edge.qubit.E[1].state

    def _apply_error_idle(self, dir, i, random_error_array_idle, stab, architecture=None):
        stab_type = "S" if stab.sID[0] == 0 else "P"
        if random_error_array_idle in [None, ["I", "I", "I", "I"]]:
            return

        if architecture == "weight_two_four":
            translate = {"n": "e", "e": "n", "s": "w", "w": "s"}

        elif architecture == "weight_three":
            # No idle qubits at North of stabilizer (When star stabilizer is evaluated, the dir should be translated)
            if dir == ('n' if stab_type == "P" else "s"):
                return

            # Idle node (containing two qubits) at South-West of stabilizer
            elif dir == ('w' if stab_type == "P" else "e"):
                stab, _ = stab.neighbors[dir][0].neighbors[('s' if stab_type == "P" else "w")]
                self._apply_error_stabilizer("n", self.dirs.index("n"), random_error_array_idle, stab)
                self._apply_error_stabilizer("w", self.dirs.index("w"), random_error_array_idle, stab)
                return

            # Idle qubits at East and South of the stab. At East stab the North qubit, at South stab the West qubit.
            else:
                translate = {"e": "n", "s": "w"} if stab_type == "P" else {"w": "e", "n": "s"}

        if dir in stab.neighbors:
            neighbor_stab, _ = stab.neighbors[dir]
            new_dir = dir.translate(str.maketrans(translate))
            self._apply_error_stabilizer(new_dir, i, random_error_array_idle, neighbor_stab)

    '''
    ########################################################################################

                                    Constructor functions

    ########################################################################################
    '''

    def add_cluster(self, cID, vertex):
        """Adds a cluster with cluster ID number cID"""
        cluster = self.C[cID] = Cluster(cID, vertex)
        return cluster

    def get_cluster(self, cID, vertex):
        return Cluster(cID, vertex)

    def add_stab(self, ertype, y, x, z):
        """Adds a stabilizer with stab ID number sID"""
        stab = self.S[z][(ertype, y, x)] = Stab(sID=(ertype, y, x), z=z)
        return stab

    def add_boundary(self, ertype, y, x, z):
        """Adds a open bounday (stab like) with bounday ID number sID"""
        bound = self.B[z][(ertype, y, x)] = Bound(sID=(ertype, y, x), z=z)
        return bound

    def add_qubit(self, td, y, x, z, vW, vE, vN, vS):
        """Adds an edge with edge ID number qID with pointers to vertices. Also adds pointers to this edge on the vertices. """

        qubit = self.Q[z][(td, y, x)] = Qubit(qID=(td, y, x), z=z)
        E1, E2 = (qubit.E[0], qubit.E[1]) if td == 0 else (qubit.E[1], qubit.E[0])

        vW.neighbors["e"] = (vE, E2)
        vE.neighbors["w"] = (vW, E2)
        vN.neighbors["s"] = (vS, E1)
        vS.neighbors["n"] = (vN, E1)

    def reset(self):
        """
        Resets the graph by deleting all clusters and resetting the edges and vertices
        """
        self.C, self.cID = {}, 0
        for qlayer in self.Q.values():
            for qubit in qlayer.values():
                qubit.reset()
        for slayer in self.S.values():
            for stab in slayer.values():
                stab.reset()

    def set_qubit_states_to_state_previous_layer(self, z):
        return


'''
########################################################################################

                                        Planar class

########################################################################################
'''

class planar(toric):
    '''
    Inherits all the class variables and methods of the graph_2D.toric object.
    Additions:
        params:
            B   dict of boundary objects with
                    Key:    sID number
                    Value:  Stab object

    Replaces:
        methods:
            init_graph_layer()
            logical_error()
            reset()
    '''
    def __init__(self, *args, **kwargs):
        self.B = {}
        super().__init__(*args, **kwargs)

    def init_graph_layer(self, z=0):
        '''
        param z     layer
        Initializes a layer of the graph structure of a planar lattice
        '''
        self.dirs = ["n", "s", "e", "w"]
        self.S[z], self.Q[z], self.B[z]= {}, {}, {}

        # Add vertices and boundaries to graph
        for yx in self.range:
            for xy in range(self.size - 1):
                self.add_stab(0, yx, xy + 1, z)
                self.add_stab(1, xy, yx, z)

            self.add_boundary(0, yx, 0, z)
            self.add_boundary(0, yx, self.size, z)
            self.add_boundary(1, -1, yx, z)
            self.add_boundary(1, self.size - 1, yx, z)

        # Add edges to graph
        for y in self.range:
            for x in self.range:
                if x == 0:
                    vW, vE = self.B[z][(0, y, x)], self.S[z][(0, y, x + 1)]
                elif x == self.size - 1:
                    vW, vE = self.S[z][(0, y, x)], self.B[z][(0, y, x + 1)]
                else:
                    vW, vE = self.S[z][(0, y, x)], self.S[z][(0, y, x + 1)]
                if y == 0:
                    vN, vS = self.B[z][(1, y - 1, x)], self.S[z][(1, y, x)]
                elif y == self.size - 1:
                    vN, vS = self.S[z][(1, y - 1, x)], self.B[z][(1, y, x)]
                else:
                    vN, vS = self.S[z][(1, y - 1, x)], self.S[z][(1, y, x)]

                self.add_qubit(0, y, x, z, vW, vE, vN, vS)

                if y != self.size - 1 and x != self.size - 1:
                    vN, vS = self.S[z][(0, y, x + 1)], self.S[z][(0, y + 1, x + 1)]
                    vW, vE = self.S[z][(1, y, x)], self.S[z][(1, y, x + 1)]
                    self.add_qubit(1, y, x + 1, z, vW, vE, vN, vS)


    def logical_error(self, z=0):
        """
        Finds whether there are any logical errors on the lattice/self. The logical error is returned as
        [Xvertical, Zhorizontal], where each item represents a homological Loop
        """

        if self.gl_plot: self.gl_plot.plot_final()

        logical_error = [0, 0]

        for i in self.range:
            if self.Q[z][(0, 0, i)].E[0].state:
                logical_error[0] = 1 - logical_error[0]
            if self.Q[z][(0, i, 0)].E[1].state:
                logical_error[1] = 1 - logical_error[1]

        errorless = True if logical_error == [0, 0] else False
        return logical_error, errorless


    def reset(self):
        """
        Resets the graph by resetting all boudaries and interited objects
        """
        super().reset()
        for layer in self.B.values():
            for bound in layer.values():
                bound.reset()

'''
########################################################################################

                            Subclasses: Graph objects

########################################################################################
'''

class Cluster(object):
    """
    Cluster obejct with parameters:
    cID         ID number of cluster
    size        size of this cluster based on the number contained vertices
    parity      parity of this cluster based on the number of contained anyons
    parent      the parent cluster of this cluster
    childs      the children clusters of this cluster
    boundary    len(2) list containing 1) current boundary, 2) next boundary
    bucket      the appropiate bucket number of this cluster
    support     growth state of the cluster: 1 if False, 2 if True

    [planar]
    on_bound    whether this clusters is connected to the boundary

    [evengrow]
    root_node   the root node of the anyontree representing this cluster
    calc_delay  list of nodes in this anyontree for which it and its children has undefined delays
    self.mindl  the minimal delay value of anyonnodes in this anyontree/cluster, which can be <1
    """
    def __init__(self, cID, vertex):
        # self.inf = {"cID": cID, "size": 0, "parity": 0}
        self.cID        = cID
        self.size       = 0
        self.parity     = 0
        self.parent     = self
        self.childs     = [[], []]
        self.boundary   = [[], []]
        self.bucket     = 0
        self.support    = 0
        self.on_bound   = 0
        self.root_node  = vertex.node
        self.calc_delay = []
        self.mindl      = 0
        self.add_vertex(vertex)

    def __repr__(self):
        return "C" + str(self.cID) + "(" + str(self.size) + ":" + str(self.parity) + ")"

    def __hash__(self):
        return self.cID

    def add_vertex(self, vertex):
        """Adds a stabilizer to a cluster. Also update cluster value of this stabilizer."""
        self.size += 1
        if vertex.state:
            self.parity += 1
        vertex.cluster = self


class Stab(object):
    """
    Object that are both:
        - the stabilizers on the toric/planar lattice
        - the vertices on the uf-lattice

    [fixed parameters]
    type        0 for stab object, 1 for boundary (inherited)
    sID         location of stabilizer (ertype, y, x)
    z           layer of graph to which this stab belongs
    neighbors   dict of the neighobrs (in the graph) of this stabilizer with
                    Key:    direction
                    Value   (Stab object, Edge object)

    [iteration parameters]
    parity      boolean indicating the outcome of the parity measurement on this stab
    state       boolean indicating anyon state of stabilizer
    mstate      boolean indicating measurement error on this stab (for plotting)
    cluster     Cluster object of which this stabilizer is apart of
    tree        boolean indicating whether this stabilizer has been traversed

    [iteration parameters: evengrow]
    node        the anyonnode in which this stab/uf-lattice-vertex is rooted
    new_bound   temporary storage list for new boundary of a cluster

    """
    def __init__(self, sID, type=0, z=0):
        self.type       = type
        self.sID        = sID
        self.z          = z
        self.neighbors  = {}
        self.parity     = 0
        self.state      = 0
        self.mstate     = 0
        self.cluster    = None
        self.forest     = 0
        self.tree       = 0
        self.node       = None
        self.new_bound  = []

    def __repr__(self):
        type = "X" if self.sID[0] == 0 else "Z"
        return "v{}({},{}|{})".format(type, *self.sID[1:], self.z)

    def picker(self):
        cluster = self.cluster.parent if self.cluster else None
        if self.node:
            return "{}-{}-{}".format(self.__repr__(), self.node.tree_rep(), cluster)
        else:
            return "{}-{}".format(self.__repr__(), cluster)


    def reset(self):
        """
        Changes all iteration paramters to their initial value
        """
        self.state      = 0
        self.mstate     = 0
        self.parity     = 0
        self.cluster    = None
        self.forest     = 0
        self.tree       = 0
        self.node       = None


class Bound(Stab):
    '''
    Object that are both:
        - the boundaries on the toric/planar lattice
        - the vertices on the uf-lattice

    Iherits all class variables and methods of Stab object
    '''
    def __init__(self, sID, z=0):
        super().__init__(sID, type=1, z=z)

    def __repr__(self):
        type = "X" if self.sID[0] == 0 else "Z"
        return "b{}({},{}|{})".format(type, *self.sID[1:], self.z)


class Qubit(object):
    '''
    Qubit object representing the physical qubits on the lattice.

    [fixed parameters]
    qID         (td, y, x)
    z           layer of graph to which this stab belongs
    E           list countaining the two edges of the primal and secundary lattice

    [iteration parameters]
    erasure     boolean of erased qubit
    '''
    def __init__(self, qID, z=0):
        self.qID        = qID
        self.z          = z
        self.E          = [Edge(self, ertype=0, z=z), Edge(self, ertype=1, z=z)]
        self.erasure    = 0

    def __repr__(self):
        return "q({},{}:{}|{})".format(*self.qID[1:], self.qID[0], self.z)

    def picker(self):
        return self.__repr__()

    def reset(self):
        """
        Changes all iteration parameters to their default value
        """
        self.E[0].reset()
        self.E[1].reset()
        self.erasure = 0



class Edge(object):
    """
    Edges on the uf-lattice, of which each qubit on the surface lattice has two.

    [fixed parameters]
    edge_type       0 for horizontal edge (within layer, 2D), 1 for vertical edge (between layers, 3D)
    qubit           qubit object this edge belongs to
    z               layer of graph to which this stab belongs
    ertype          0 for primal lattice connecting X-type vertices,
                    1 for secundary lattice connecting Z-type vertices

    [iteration parameters]
    cluster         Cluster object of which this edge is apart of
    state           boolean indicating the state of the qubit
    support         0 for ungrown, 1 for half-edge, 2 for full-edge
    peeled          boolean indicating whether this edge has peeled
    matching        boolean indicating whether this edge is apart of the matching
    """
    def __init__(self, qubit, ertype, z=0, edge_type=0):
        # fixed parameters
        self.edge_type  = edge_type
        self.qubit      = qubit
        self.ertype     = ertype
        self.z          = z
        self.state      = 0
        self.support    = 0
        self.peeled     = 0
        self.matching   = 0
        self.forest     = 0


    def __repr__(self):
        if self.edge_type == 0:
            orientation = "-" if self.ertype == self.qubit.qID[0] else "|"
            errortype = "P" if self.ertype == 0 else "S"
        else:
            orientation = "~"
            errortype = "P" if self.ertype == 1 else "S"

        return "e{}{}({},{}|{})".format(errortype, orientation, *self.qubit.qID[1:], self.z)

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def __gt__(self, other):
        return self.__repr__() > other.__repr__()

    def __ge__(self, other):
        return self.__repr__() >= other.__repr__()

    def __lt__(self, other):
        return self.__repr__() < other.__repr__()

    def __le__(self, other):
        return self.__repr__() <= other.__repr__()

    def picker(self):
        return "{}-{}".format(self.__repr__(), self.qubit)

    def reset(self):
        """
        Changes all iteration paramters to their initial value
        """
        self.state      = 0
        self.support    = 0
        self.peeled     = 0
        self.matching   = 0
        self.forest     = 0
