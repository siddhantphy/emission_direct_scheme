

def N_decoherence(self, qubits=None, sub_circuit=None, sub_circuit_concurrent=False, decoherence=True):
    if not decoherence:
        return

    qubits = qubits if qubits is not None else [i for i in self.qubits.keys()]

    for qubit in qubits:
        current_qubit = self.qubits[qubit]
        waiting_time_link = current_qubit.waiting_time_link
        waiting_time_idle = current_qubit.waiting_time_idle
        if waiting_time_link == waiting_time_idle == 0:
            continue
        qubit_type = current_qubit.qubit_type
        T1_idle = current_qubit.T1_idle
        T2_idle = current_qubit.T2_idle
        T1_link = current_qubit.T1_link
        T2_link = current_qubit.T2_link
        density_matrix, qubits_dens, rel_qubit, rel_num_qubits = self._get_qubit_relative_objects(qubit)

        if waiting_time_idle > 0:
            alpha = 2 if qubit_type == 'e' else 1
            density_matrix = self._N_amplitude_damping_channel(rel_qubit, density_matrix, rel_num_qubits,
                                                               waiting_time_idle, T1_idle)
            density_matrix = self._N_phase_damping_channel(rel_qubit, density_matrix, rel_num_qubits, waiting_time_idle,
                                                           T2_idle, alpha)
            self._add_draw_operation("{:1.1e}xD[{}-{}]".format(waiting_time_idle, 'i', qubit_type), qubit, noise=True,
                                     sub_circuit=sub_circuit, sub_circuit_concurrent=sub_circuit_concurrent)
        if waiting_time_link > 0:
            # Nuclear qubits will only actually experience the T1_link and T2_link. This is namely the decoherence time
            # for the qubits in the nodes not participating in the link attempt
            if qubit_type == 'n':
                density_matrix = self._N_amplitude_damping_channel(rel_qubit, density_matrix, rel_num_qubits,
                                                                   waiting_time_link, T1_link)
                density_matrix = self._N_phase_damping_channel(rel_qubit, density_matrix, rel_num_qubits,
                                                               waiting_time_link, T2_link)
            self._add_draw_operation("{:1.1e}xD[{}-{}]".format(waiting_time_link, 'l', qubit_type), qubit, noise=True,
                                     sub_circuit=sub_circuit, sub_circuit_concurrent=sub_circuit_concurrent)

        self._set_density_matrix(qubit, density_matrix)
        # After everything, set qubit waiting time to 0 again
        current_qubit.reset_waiting_time()
