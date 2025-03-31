
from src.hamiltonian_utils import FermiHubbardHamiltonian
from src.nuclear_physics_utils import get_twobody_nuclearshell_model,SingleParticleState,QuadrupoleOperator,J2operator
import numpy as np
import torch
from typing import Dict
import scipy
from src.qml_models import AdaptVQEFermiHubbard
from src.qml_utils.train import Fit
from src.qml_utils.utils import configuration
from scipy.sparse.linalg import eigsh,expm_multiply
from tqdm import trange
import matplotlib.pyplot as plt
from src.utils_quasiparticle_approximation import QuasiParticlesConverter
file_name='data/usdb.nat'
qq_filename='data/qq.sd'
pp_filename='data/pair_inter.sd'
SPS=SingleParticleState(file_name=file_name)


nparticles_a=6
nparticles_b=6

size_a=SPS.energies.shape[0]//2
size_b=SPS.energies.shape[0]//2

title=r'$^{28}$Si'
filename='28Si'

# compute the NSM Hamiltonian
NSMHamiltonian=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
print('size=',size_a+size_b,size_b)
NSMHamiltonian.get_external_potential(external_potential=SPS.energies[:size_a+size_b])
if file_name=='data/cki':
    twobody_matrix,energies=get_twobody_nuclearshell_model(file_name=file_name)

    NSMHamiltonian.get_twobody_interaction(twobody_dict=twobody_matrix)
else:
    NSMHamiltonian.twobody_operator=scipy.sparse.load_npz(f'data/nuclear_twobody_matrix/usdb_{nparticles_a}_{nparticles_b}.npz')
NSMHamiltonian.get_hamiltonian()

egs,psi0=NSMHamiltonian.get_spectrum(n_states=1)

print(egs)

print('total_m=',SPS.compute_m_exp_value(psi=psi0,basis=NSMHamiltonian.basis))


QPC=QuasiParticlesConverter()

QPC.initialize_shell(state_encoding=SPS.state_encoding)


#just for the basis
QPC.get_the_basis_matrix_transformation(basis=NSMHamiltonian.basis)

print(QPC.particles2quasiparticles.shape)
print(QPC.quasiparticle_basis)

print(QPC.rest_basis.shape)
print(QPC.couples)



hamiltonian_qq=QPC.particles2quasiparticles @ NSMHamiltonian.hamiltonian @ QPC.particles2quasiparticles.T
hamiltonian_rr=QPC.particles2restofstates @ NSMHamiltonian.hamiltonian @ QPC.particles2restofstates.T
hamiltonian_qr=QPC.particles2quasiparticles @ NSMHamiltonian.hamiltonian @ QPC.particles2restofstates.T
hamiltonian_rq=QPC.particles2restofstates @ NSMHamiltonian.hamiltonian @ QPC.particles2quasiparticles.T

tot_hamiltonian=hamiltonian_qq
values,psi=eigsh(hamiltonian_qq,k=1)
e=values[0]
approximations=[]

single_term = hamiltonian_rq  # Start with initial term

delta_e_step=1000
e_old=1000
i=0
approximations=[]
history_errors_exact=[]
history_energy=[]
history_psi=[]
while(delta_e_step>10**-4):
    
    if i > 0:
        single_term = hamiltonian_rr @ single_term  # Efficient update
    approximations.append(hamiltonian_qr @ single_term)  # Store result
    
    tot_hamiltonian=hamiltonian_qq
    for j in range(i):
        tot_hamiltonian=tot_hamiltonian+approximations[j]/e**(j+1)
    values,psi=eigsh(tot_hamiltonian,k=1)
    e=values[0]
    print(e)
    print(np.abs((e-egs[0])/egs[0]),'index=',i)
    print('delta e=',delta_e_step)
    history_errors_exact.append(np.abs((e-egs[0])/egs[0]))
    history_energy.append(e)
    history_psi.append(psi[:,0])
    delta_e_step=np.abs(e_old-e)
    e_old=e
    i+=1
    print(i)
    if i>1000:
        np.savez('data/effective_hamiltonian_method_'+filename,energy=history_energy,psi=history_psi,errors=history_errors_exact,final_effective_hamiltonian=tot_hamiltonian,title=title)
        exit()


np.savez('data/effective_hamiltonian_method_'+filename,energy=history_energy,psi=history_psi,errors=history_errors_exact,final_effective_hamiltonian=tot_hamiltonian,title=title)