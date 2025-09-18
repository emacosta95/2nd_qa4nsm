import os
import qililab as ql
from qibo import Circuit, gates
import numpy as np
from src.qibo_utils import load_gate_list, build_qibo_circuit_from_gate_list
from qibo.gates import M
import pickle

ql.logger.setLevel(40)  # Set qililab's logger to a higher level so it only shows error messages

PLATFORM_PATH = os.getenv("RUNCARD")
platform = ql.build_platform(runcard=PLATFORM_PATH)

# We load the main circuit
gate_list = load_gate_list("data/qa_circuit_2.json")

qibo_circuit = build_qibo_circuit_from_gate_list(gate_list, n_qubits=3)

# Measure in Z, X, Y basis
basis_gates = {
    "Z": [],  # Z basis is just standard measurement
    "X": [gates.H(i) for i in range(3)],  # 3 qubits example
    "Y": [gates.S(i).dagger() for i in range(3)] + [gates.H(i) for i in range(3)]
}

shots = 5000
results = {}
circuits=[]
for basis, rotations in basis_gates.items():
    circuit = qibo_circuit.copy()
    for gate in rotations:
        circuit.add(gate)
    nqubits = circuit.nqubits
    circuit.add(M(*range(nqubits)))
    circuits.append(circuit)

# Circuit 2
circuit_2 = Circuit(2)
circuit_2.add(gates.M(0,1))

# Connect
platform.connect()
platform.turn_on_instruments()
platform.initial_setup()

# Execute
# Execute & store
for basis, circuit in zip(basis_gates.keys(), circuits):
    result = platform.execute(circuit, shots=shots)
    results[basis] = result  # keep in dict
    print(basis, result.counts())

# Save all results in one file
with open("results.pkl", "wb") as f:
    pickle.dump(results, f)


## IMPORTANT: Disconnect
platform.disconnect()
