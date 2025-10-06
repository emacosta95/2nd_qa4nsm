import numpy as np
import json
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

# --------------------------
# Load one-body data
# --------------------------
data_onebody = np.load('data/matrix_elements_h_eff_2body/one_body_nn_p.npz')
keys = data_onebody['keys']
values = data_onebody['values']
n_qubits = 3

t_onebody = {}
for a, key in enumerate(keys):
    i, j = key
    t_onebody[(i, j)] = values[a]

# Build target Hamiltonian
from src.qiskit_utils import get_hamiltonian
hamiltonian_q = get_hamiltonian(t_onebody, n_qubits)

# Driver Hamiltonian
coupling_term = -8.4321
Z_tuples = [("Z", [0], -0.5*coupling_term)]
I_tuples = [('I', [0], 0.5*coupling_term)]
hamiltonian_driver = SparsePauliOp.from_sparse_list([*Z_tuples, *I_tuples], num_qubits=n_qubits)

# --------------------------
# Time evolution circuit
# --------------------------
time_steps = 10
tf = 1
time = np.linspace(0, tf, time_steps)
dt = tf / time_steps
driver = 1 - time/tf

circuit_time_evolution = QuantumCircuit(hamiltonian_q.num_qubits)
circuit_time_evolution.x([0])  # initial state

for n, t in enumerate(time):
    hamiltonian_t = (driver[n])*hamiltonian_driver + (1-driver[n])*hamiltonian_q
    exp_H_t = PauliEvolutionGate(hamiltonian_t, time=dt, synthesis=SuzukiTrotter(order=1))
    circuit_time_evolution.append(exp_H_t, range(hamiltonian_q.num_qubits))


# --------------------------
# Measurement circuits
# --------------------------
creg = ClassicalRegister(n_qubits)

# Z basis
circuit_z = circuit_time_evolution.copy()
circuit_z.add_register(creg)
circuit_z.measure(range(n_qubits), range(n_qubits))

# X basis
circuit_x = circuit_time_evolution.copy()
circuit_x.add_register(creg)
circuit_x.h(range(n_qubits))
circuit_x.measure(range(n_qubits), range(n_qubits))

# Y basis
circuit_y = circuit_time_evolution.copy()
circuit_y.add_register(creg)
circuit_y.sdg(range(n_qubits))
circuit_y.h(range(n_qubits))
circuit_y.measure(range(n_qubits), range(n_qubits))

# --------------------------
# Connect to IBM Quantum
# --------------------------
service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token="jfthsWLvgK-4k51QgbJQ7ZkfnUlEcC2PnHexhbYyklyi"
)

backend = service.least_busy(simulator=False, operational=True)
print("Running on:", backend.name)


# Transpile
circuit_x = transpile(
    circuit_x,
    optimization_level=3,
    backend=backend,
    #basis_gates=['cz','id','rz','x'] #cz, id, rx, rz, rzz, sx, x
)

circuit_y = transpile(
    circuit_y,
    optimization_level=3,
    backend=backend,
    #basis_gates=['cz','id','rz','x'] #cz, id, rx, rz, rzz, sx, x
)

circuit_z = transpile(
    circuit_z,
    optimization_level=3,
    backend=backend,
    #basis_gates=['cz','id','rz','x'] #cz, id, rx, rz, rzz, sx, x
)


# --------------------------
# Run Sampler
# --------------------------
sampler = Sampler(mode=backend)
job = sampler.run([circuit_z, circuit_x, circuit_y], shots=2048)
results = job.result()

# Extract histograms
hist_z = results[0].data.c0.get_counts()
hist_x = results[1].data.c0.get_counts()
hist_y = results[2].data.c0.get_counts()

# --------------------------
# Save histograms as JSON
# --------------------------
with open("histogram_z.json", "w") as f:
    json.dump(hist_z, f, indent=4)

with open("histogram_x.json", "w") as f:
    json.dump(hist_x, f, indent=4)

with open("histogram_y.json", "w") as f:
    json.dump(hist_y, f, indent=4)

print("Histograms saved to histogram_{z,x,y}.json")

