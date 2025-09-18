from qibo import Circuit, gates
import json

# --- 1. Load gate list ---
def load_gate_list(filename="gate_list.json"):
    with open(filename, "r") as f:
        return json.load(f)



def remap_qubits(qubit_indices, n_qubits):
    return [n_qubits - q - 1 for q in qubit_indices]

def gate_from_label(label, qubit_indices, param=None):
    if label == "h":
        return gates.H(qubit_indices[0])
    elif label == "x":
        return gates.X(qubit_indices[0])
    elif label == "rz":
        return gates.RZ(qubit_indices[0], theta=param)
    elif label == "rx":
        return gates.RX(qubit_indices[0], theta=param)
    elif label == "ry":
        return gates.RY(qubit_indices[0], theta=param)
    elif label in {"cx", "cnot"}:
        return gates.CNOT(*qubit_indices)
    elif label == "cz":
        return gates.CZ(*qubit_indices)
    elif label in {"ccx", "toffoli"}:
        return gates.Toffoli(*qubit_indices)
    elif label == "u3":
        return gates.U3(qubit_indices[0], *param)
    else:
        raise ValueError(f"Unsupported gate: {label}")

def build_qibo_circuit_from_gate_list(gate_list, n_qubits):
    circuit = Circuit(n_qubits,density_matrix=True)
    for label, qubit_indices, param in gate_list:
        remapped_indices = remap_qubits(qubit_indices, n_qubits)
        gate = gate_from_label(label, remapped_indices, param)
        circuit.add(gate)
    return circuit