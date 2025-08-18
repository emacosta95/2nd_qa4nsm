from src.hamiltonian_utils import FermiHubbardHamiltonian
from src.nuclear_physics_utils import get_twobody_nuclearshell_model,SingleParticleState,QuadrupoleOperator,J2operator
import numpy as np

from typing import Dict
import scipy
from src.qml_models import AdaptVQEFermiHubbard
from src.qml_utils.train import Fit
from src.qml_utils.utils import configuration
from scipy.sparse.linalg import eigsh,expm_multiply
from tqdm import trange
import matplotlib.pyplot as plt
from src.utils_quasiparticle_approximation import QuasiParticlesConverterOnlynnpp,HardcoreBosonsBasis
from scipy.sparse import lil_matrix
import pickle

file_name='data/GCN5082'

SPS=SingleParticleState(file_name=file_name)



nparts=[(2,0)]
titles=[r'$^{18}$O']

nparticles_a=nparts[0][0]
nparticles_b=nparts[0][1]


size_a=SPS.energies.shape[0]//2
size_b=SPS.energies.shape[0]//2

title=r'$^{18}$O'

energies=SPS.energies[:size_a+size_b]   
twobody_matrix=pickle.load(open('twobody_interaction_GCN5082.pkl','rb'))   

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper,BravyiKitaevMapper

# Start building terms
hamiltonian_terms = {}
n_spin_orbitals =  size_a  # Assuming spin-orbitals are double the single-particle states
# One-body terms: e_a * c†_a c_a
for a, e_a in enumerate(SPS.energies[:size_a]):
    label = f"+_{a} -_{a}"
    hamiltonian_terms[label] = hamiltonian_terms.get(label, 0) + e_a

# Two-body terms: (1/4) * v_abcd * c†_a c†_b c_d c_c
for (a, b, c, d), v_abcd in twobody_matrix.items():
    if a<size_a and b<size_a and c<size_a and d<size_a:
        label = f"+_{a} +_{b} -_{d} -_{c}"
        coef = 0.25 * v_abcd
        hamiltonian_terms[label] = hamiltonian_terms.get(label, 0) + coef

# Create FermionicOp
nsm_hamiltoninan_fermions = FermionicOp(hamiltonian_terms, num_spin_orbitals=n_spin_orbitals)

# (Optional) Print to inspect
print(nsm_hamiltoninan_fermions)

mapper = JordanWignerMapper()
nsm_hamiltonian_qubit = mapper.map(nsm_hamiltoninan_fermions)

from scipy.sparse import identity
#Quadrupole Operator
energy_errors=[]
fidelities=[]
for idx,npart in enumerate(nparts):
    nparticles_a,nparticles_b=npart
    title=titles[idx]
    
    # compute the NSM Hamiltonian
    NSMHamiltonian=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
    print('size=',size_a+size_b,size_b)
    NSMHamiltonian.get_external_potential(external_potential=SPS.energies[:size_a+size_b])
    if file_name=='data/cki':
        twobody_matrix,energies=get_twobody_nuclearshell_model(file_name=file_name)

        NSMHamiltonian.get_twobody_interaction(twobody_dict=twobody_matrix)
    else:
        NSMHamiltonian.twobody_operator=scipy.sparse.load_npz(f'data/nuclear_twobody_matrix/GCN5082_{nparticles_a}_{nparticles_b}.npz')
    NSMHamiltonian.get_hamiltonian()

    
    shifted_hamiltonian=NSMHamiltonian.hamiltonian


    egs,psi0=eigsh(shifted_hamiltonian,k=1,which='SA')

    print('ground state energy=',egs[0])
    
    
    #### Initial Hamiltonian
min_b=np.zeros(size_a+size_b)

if file_name=='data/gxpf1a' or file_name=='data/GCN5082':
    order_of_filling=np.asarray([0,7,1,6,2,5,3,4])
    order_of_filling_protons=order_of_filling+size_a
else:
    order_of_filling=np.asarray([0,5,1,4,2,3,6,7,8,11,9,10])
    order_of_filling_protons=order_of_filling+size_a


min_b[order_of_filling[:nparticles_a]]=1
min_b[order_of_filling_protons[:nparticles_b]]=1

print(min_b)

print('initial state=',min_b)
InitialHamiltonian=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
kinetic_term:Dict={}
adj_matrix=np.zeros((size_a+size_b,size_a+size_b))
idx=InitialHamiltonian._get_index(element=min_b)
print('idx=',idx)
psi_configuration=np.zeros(NSMHamiltonian.hamiltonian.shape[0])
psi_configuration[idx]=1
min=psi_configuration.transpose().dot(NSMHamiltonian.hamiltonian.dot(psi_configuration))      
external_field=np.zeros(size_a+size_b)
external_field=-1*(np.abs(min)/(nparticles_a+nparticles_b))*min_b
print('min energy=',min)
InitialHamiltonian.get_external_potential(external_field)
InitialHamiltonian.get_hamiltonian()

print('external field=',external_field)

e0,psis=InitialHamiltonian.get_spectrum(n_states=1)
psi_configuration=psis[:,0]

from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper,BravyiKitaevMapper

# Start building terms
hamiltonian_terms = {}
n_spin_orbitals =  size_a  # Assuming spin-orbitals are double the single-particle states
# One-body terms: e_a * c†_a c_a
for a, e_a in enumerate(external_field[:size_a]):
    label = f"+_{a} -_{a}"
    hamiltonian_terms[label] = hamiltonian_terms.get(label, 0) + e_a


# Create FermionicOp
driver_hamiltoninan_fermions = FermionicOp(hamiltonian_terms, num_spin_orbitals=n_spin_orbitals)

# (Optional) Print to inspect
print(driver_hamiltoninan_fermions)

mapper = JordanWignerMapper()
driver_hamiltoninan_qubit = mapper.map(driver_hamiltoninan_fermions)

from qiskit import QuantumCircuit
from tqdm import tqdm
from qiskit.quantum_info import Statevector
from qiskit.synthesis import SuzukiTrotter,QDrift
from qiskit.circuit.library import PauliEvolutionGate
from qiskit import transpile

n_qubits = nsm_hamiltonian_qubit.num_qubits
#final_state=Statevector(circuit_opt)
time_steps=1
tf=0.2
time=np.linspace(0.1,tf,time_steps)
dt=tf/time_steps
print(dt)
#b=0.8
#h=1-(1+b)*(time/tf)+b*(time/tf)**2
driver=1-time/tf
circuit_time_evolution=QuantumCircuit(nsm_hamiltonian_qubit.num_qubits)
circuit_time_evolution.x([0]) # initial state

energies_qiskit=np.zeros((time_steps))
tbar=tqdm(enumerate(time))

for n,t in tbar:
    hamiltonian_t=(driver[n])*nsm_hamiltonian_qubit+(1-driver[n])*driver_hamiltoninan_qubit
    exp_H_t=PauliEvolutionGate(hamiltonian_t,time=dt,synthesis=SuzukiTrotter(order=2))
    #exp_H_t=PauliEvolutionGate(hamiltonian_t,time=dt,synthesis=QDrift(reps=5))
    circuit_time_evolution.append(exp_H_t,range(nsm_hamiltonian_qubit.num_qubits))
    single_particle_vector=np.zeros(2**n_qubits,dtype=np.complex128)
    # psi_for_fidelity=np.zeros(n_qubits,dtype=np.complex128)
    # for a in range(n_qubits):
    #     a_mb=2**(a)
    #     final_state=Statevector(circuit_time_evolution).data
    #     single_particle_vector[a_mb]=final_state[a_mb]
    #     psi_for_fidelity[a]=final_state[a_mb]
    # energies_qiskit[n]=Statevector(single_particle_vector).expectation_value(hamiltonian_t).real
    tbar.refresh()
#final_state=Statevector(circuit_time_evolution)

transpiled_circuit_time_evolution=transpile(circuit_time_evolution, optimization_level=0,basis_gates=['cx','s','h','rz','x'])



print(
    f"""
Trotter step with Suzuki Trotter (1nd order)
--------------------------------------------

                  Depth: {transpiled_circuit_time_evolution.depth()}
             Gate count: {len(transpiled_circuit_time_evolution)}
    Nonlocal gate count: {transpiled_circuit_time_evolution.num_nonlocal_gates()}
         Gate breakdown: {", ".join([f"{k.upper()}: {v}" for k, v in transpiled_circuit_time_evolution.count_ops().items()])}

"""
)
