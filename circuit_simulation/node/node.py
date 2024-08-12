nodes = {
    "Natab": "Natural Abundance",
    "Pur": "Purified",
    None: "Ideal",
    "Pur10": "PurifiedImproved10",
    "Pur5": "PurifiedImproved5",
    "Pur20": "PurifiedImproved20",
    "Pur30": "PurifiedImproved30",
    "Pur50": "PurifiedImproved50",
    "Pur100": "PurifiedImproved100",
    "Pur1000": "PurifiedImproved1000",
    "Natabfast": "FastNaturalAbundance",
    "Natabimpr": "NaturalAbundanceImpr",
    "Natabimpr2": "NaturalAbundanceImpr2",
    "Natabimpr3": "NaturalAbundanceImpr3"
}


class Node(object):

    def __init__(self, name: str, qubits: list, qc, electron_qubits=None, data_qubits=None):

        self._name = name
        self._qubits = qubits
        self._qc = qc
        self._electron_qubits = electron_qubits
        self._data_qubits = data_qubits
        self._sub_circuit_time = 0
        self._total_time = 0

    @property
    def name(self):
        return self._name

    @property
    def qubits(self):
        return self._qubits

    @property
    def qc(self):
        return self._qubits

    @property
    def electron_qubits(self):
        return self._electron_qubits

    @property
    def data_qubits(self):
        return self._data_qubits

    @property
    def total_time(self):
        return self._total_time

    @property
    def sub_circuit_time(self):
        return self._sub_circuit_time

    def increase_total_time(self, amount):
        self._total_time += amount

    def increase_sub_circuit_time(self, amount):
        self._sub_circuit_time += amount
        self._total_time += amount

    def reset_total_time(self):
        self._total_time = 0

    def reset_sub_circuit_time(self):
        self._sub_circuit_time = 0

    def reset_all_times(self):
        self.reset_sub_circuit_time()
        self.reset_total_time()

    def qubit_in_node(self, qubit):
        return qubit in self.qubits
