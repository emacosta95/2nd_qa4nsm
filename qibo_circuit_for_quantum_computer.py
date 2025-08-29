import os
import qililab as ql
from qibo import Circuit, gates
import numpy as np
from src.qibo_utils import load_gate_list, build_qibo_circuit_from_gate_list
from qibo.gates import M


ql.logger.setLevel(40)  # Set qililab's logger to a higher level so it only shows error messages

PLATFORM_PATH = os.getenv("RUNCARD")
platform = ql.build_platform(runcard=PLATFORM_PATH)

# We load the main circuit
gate_list = load_gate_list("data/qa_circuit_2.json")

qibo_circuit = build_qibo_circuit_from_gate_list(gate_list, n_qubits=3)



set_backend("qibojit")
backend = get_backend()  # retrieves the backend object
# Copy your circuit to avoid modifying the original
meas_circuit = qibo_circuit.copy()

# Measure in Z, X, Y basis
basis_gates = {
    "Z": [],  # Z basis is just standard measurement
    "X": [gates.H(i) for i in range(3)],  # 3 qubits example
    "Y": [gates.S(i).dagger() for i in range(3)] + [gates.H(i) for i in range(3)]
}

shots = 5000
results = {}

for basis, rotations in basis_gates.items():
    circuit = meas_circuit.copy()
    for gate in rotations:
        circuit.add(gate)
    nqubits = circuit.nqubits
    circuit.add(M(*range(nqubits)))
    result = circuit(nshots=shots)
    results[basis] = result

# Circuit 1


# Circuit 2
circuit_2 = Circuit(2)
circuit_2.add(gates.M(0,1))

# Connect
platform.connect()
platform.turn_on_instruments()
platform.initial_setup()

# Execute
result = platform.execute(circuit, num_bins=100)
result_2 = platform.execute(circuit_2, num_bins=100)

## IMPORTANT: Disconnect
platform.disconnect()

print(result.counts())
print(result_2.counts())