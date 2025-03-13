#%%
from collections import Counter
from src.hamiltonian_utils import FermiHubbardHamiltonian
from src.nuclear_physics_utils import get_twobody_nuclearshell_model,SingleParticleState,QuadrupoleOperator,J2operator,write_j_square_twobody_file
import numpy as np
import torch
from typing import Dict
from src.qml_models import AdaptVQEFermiHubbard
from src.qml_utils.train import Fit
from src.qml_utils.utils import configuration
from scipy.sparse.linalg import eigsh,expm_multiply
from tqdm import trange
import matplotlib.pyplot as plt
import scipy
from typing import List,Type
from scipy.optimize import minimize
from tqdm import trange
from src.qml_utils.utils import Schedule,SchedulerModel
from scipy.sparse import csr_matrix, diags
from numpy.linalg import eigh
import argparse

parser = argparse.ArgumentParser()


parser.add_argument(
    "--filename",
    type=str,
    help="name of the file with the nuclear potential entries.",

)

parser.add_argument(
    "--qq_filename",
    type=str,
    help="name of the file with quadrupole quadrupole interaction.",
)

parser.add_argument(
    "--pairing_filename",
    type=str,
    help="name of the file with pairing interaction.",
)


parser.add_argument(
    "--nlevels",
    type=int,
    help="number of energy levels to compute",
)

parser.add_argument(
    "--tf",
    type=float,
    help="final time of the quantum control protocol (tf=20)",
    default=20,
)

parser.add_argument(
    "--nsteps",
    type=int,
    help="number of time steps (nsteps=100)",
    default=100,
)

parser.add_argument(
    "--number_parameters",
    type=int,
    help="number of parameters (number_parameters=10) for each driving configuration",
    default=10,
)

parser.add_argument(
    "--nparticles_a",
    type=int,
    help="number of neutrons",
    default=2,
)


parser.add_argument(
    "--nparticles_b",
    type=int,
    help="number of protons",
    default=2,
)

parser.add_argument(
    "--pairing_coupling",
    type=float,
    help="strength of the pairing interaction",
    default=1.,
)

parser.add_argument(
    "--quadrupole_coupling",
    type=float,
    help="strength of the quadrupole quadrupole interaction",
    default=2.,
)

parser.add_argument(
    "--title",
    type=str,
    help="title in the r format",
    default=r"$^{8}$ Be",
)

parser.add_argument(
    "--save_filename",
    type=str,
    help="title in the r format",
    default='data/optimal_control/8be_results',
)


args=parser.parse_args()

#### Hyperparameters
nlevels=args.nlevels

tf=args.tf
nsteps=args.nsteps
number_of_parameters=args.number_parameters


file_name:str=args.filename
qq_filename:str=args.qq_filename
pairing_filename:str=args.pairing_filename

j_square_filename:str=file_name+'_j2'
SPS=SingleParticleState(file_name=file_name)
energies:List=SPS.energies


nparticles_a:int=args.nparticles_a
nparticles_b:int=args.nparticles_b

size_a:int=SPS.energies.shape[0]//2
size_b:int=SPS.energies.shape[0]//2
title=args.title

pairing_coupling=args.pairing_coupling
quadrupole_coupling=args.quadrupole_coupling


### file output
file=open('OUTPUTS/'+args.save_filename+"_simulation_output.txt", "w")
file.write("Simulation Results:\n")


#### Target Hamiltonian
TargetHamiltonian=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
if file_name=='data/cki':
    twobody_matrix,energies=get_twobody_nuclearshell_model(file_name=file_name)
    TargetHamiltonian.get_twobody_interaction(twobody_dict=twobody_matrix)
else:
    TargetHamiltonian.twobody_operator=scipy.sparse.load_npz(f'data/nuclear_twobody_matrix/usdb_{nparticles_a}_{nparticles_b}.npz')
file.write(f'size={size_a+size_b},{size_b} \n')
TargetHamiltonian.get_external_potential(external_potential=energies[:size_a+size_b])

TargetHamiltonian.get_hamiltonian()

egs,psis=TargetHamiltonian.get_spectrum(n_states=nlevels)
file.write(f'Hamiltonian shape={TargetHamiltonian.hamiltonian.shape}')
egs=egs[0]
file.write(f'energy ground state={egs}')
psi0=psis[:,:1]


#### Simpler Hamiltonian
matrix_qq,_=get_twobody_nuclearshell_model(file_name=qq_filename)
QQoperator=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
QQoperator.get_twobody_interaction(twobody_dict=matrix_qq)
QQoperator.get_hamiltonian()


matrix_pairing,_=get_twobody_nuclearshell_model(file_name=pairing_filename)
PPoperator=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
PPoperator.get_twobody_interaction(twobody_dict=matrix_pairing)
PPoperator.get_hamiltonian()
mean_field_hamiltonian=diags(TargetHamiltonian.hamiltonian.diagonal(), format='csr')


simpler_target_hamiltonian=(mean_field_hamiltonian+pairing_coupling*PPoperator.hamiltonian+quadrupole_coupling*QQoperator.hamiltonian)

eigenvalues,eigenstates=eigh(simpler_target_hamiltonian.todense())



#### Initial Hamiltonian
min_b=np.zeros(size_a+size_b)

if file_name=='data/cki':
    order_of_filling=np.asarray([0,3,1,2,4,5])
    order_of_filling_protons=order_of_filling+size_a
else:
    order_of_filling=np.asarray([0,5,1,4,2,3,6,7,8,11,9,10])
    order_of_filling_protons=order_of_filling+size_a


min_b[order_of_filling[:nparticles_a]]=1
min_b[order_of_filling_protons[:nparticles_b]]=1

file.write(f'initial state={min_b} \n')
InitialHamiltonian=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
kinetic_term:Dict={}
adj_matrix=np.zeros((size_a+size_b,size_a+size_b))
idx=InitialHamiltonian._get_index(element=min_b)

psi_configuration=np.zeros(TargetHamiltonian.hamiltonian.shape[0])
psi_configuration[idx]=1
min=psi_configuration.transpose().dot(TargetHamiltonian.hamiltonian.dot(psi_configuration))      
external_field=np.zeros(size_a+size_b)
external_field=-1*(np.abs(min)/(nparticles_a+nparticles_b))*min_b
InitialHamiltonian.get_external_potential(external_field)
InitialHamiltonian.get_hamiltonian()


#### Optimization

model=SchedulerModel(initial_state=psi_configuration,target_hamiltonian=mean_field_hamiltonian+PPoperator.hamiltonian,initial_hamiltonian=InitialHamiltonian.hamiltonian,second_target_hamiltonian=QQoperator.hamiltonian,tf=tf,nsteps=nsteps,number_of_parameters=number_of_parameters,type='fourier',seed=42,reference_hamiltonian=TargetHamiltonian.hamiltonian,mode='annealing ansatz')


res = minimize(
                    model.forward,
                    model.parameters,
                    method='L-BFGS-B',
                    tol=10**-6,
                    callback=model.callback,
                    options=None,
                    
                )

model.parameters = res.x
energy = model.forward(model.parameters)
file.write(f'energy,egs={energy},{egs}')
file.write(f'parameters={model.parameters}')

np.savez('data/optimal_control_results/'+args.save_filename,history=model.history,history_drivings=model.history_drivings,history_parameters=model.history_parameters,history_psi=model.history_psi,history_run=model.history_run,info=args)