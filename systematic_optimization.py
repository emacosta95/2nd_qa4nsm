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


tf=100
nsteps=10
range_of_number_of_parameters=np.arange(1,20)

file_name:str='data/cki'
qq_filename:str='data/qq.p'


j_square_filename:str=file_name+'_j2'
SPS=SingleParticleState(file_name=file_name)
energies:List=SPS.energies


nparticles_a:int=2
nparticles_b:int=2

size_a:int=SPS.energies.shape[0]//2
size_b:int=SPS.energies.shape[0]//2
title=r'$^{8}$Be'


# Target Hamiltonian
TargetHamiltonian=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
if file_name=='data/cki':
    twobody_matrix,energies=get_twobody_nuclearshell_model(file_name=file_name)
    TargetHamiltonian.get_twobody_interaction(twobody_dict=twobody_matrix)
else:
    TargetHamiltonian.twobody_operator=scipy.sparse.load_npz(f'data/nuclear_twobody_matrix/usdb_{nparticles_a}_{nparticles_b}.npz')
print('size=',size_a+size_b,size_b)
TargetHamiltonian.get_external_potential(external_potential=energies[:size_a+size_b])

TargetHamiltonian.get_hamiltonian()

matrix_qq,_=get_twobody_nuclearshell_model(file_name=qq_filename)
QQoperator=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
QQoperator.get_twobody_interaction(twobody_dict=matrix_qq)
QQoperator.get_hamiltonian()


# check the eigenstates
nlevels=5
egs,psis=TargetHamiltonian.get_spectrum(n_states=nlevels)
print('Hamiltonian shape=',TargetHamiltonian.hamiltonian.shape)
egs=egs[0]
print(egs)
psi0=psis[:,:1]

print('deformation=',psi0.transpose().conjugate().dot(QQoperator.hamiltonian.dot(psi0)),'\n')

min_b=np.zeros(size_a+size_b)
min_b[0]=1
min_b[3]=1

min_b[0+size_a]=1

min_b[3+size_a]=1




print('initial state=',min_b)
InitialHamiltonian=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
kinetic_term:Dict={}
adj_matrix=np.zeros((size_a+size_b,size_a+size_b))
idx=InitialHamiltonian._get_index(element=min_b)
print('idx=',idx)
psi_configuration=np.zeros(TargetHamiltonian.hamiltonian.shape[0])
psi_configuration[idx]=1
min=psi_configuration.transpose().dot(TargetHamiltonian.hamiltonian.dot(psi_configuration))      
external_field=np.zeros(size_a+size_b)
external_field=-1*(np.abs(min)/(nparticles_a+nparticles_b))*min_b
print('min energy=',min)
InitialHamiltonian.get_external_potential(external_field)
InitialHamiltonian.get_hamiltonian()

nlevels=3
es=np.zeros(range_of_number_of_parameters.shape[0])
hs_driver=np.zeros((range_of_number_of_parameters.shape[0],nsteps))
hs_target=np.zeros_like(hs_driver)
psis=[]

for a,number_of_parameters in enumerate(range_of_number_of_parameters):
    
    model=SchedulerModel(initial_state=psi_configuration,target_hamiltonian=TargetHamiltonian.hamiltonian,initial_hamiltonian=InitialHamiltonian.hamiltonian,tf=tf,nsteps=nsteps,number_of_parameters=number_of_parameters,type='power law',seed=42)
    print(model.parameters.shape)

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
    es[a]=energy
    print(energy,egs)
    print(model.parameters)
    psis.append(model.psi)
    hs_driver[a],hs_target[a]=model.get_driving()
    
    
np.savez('data/optimization_vs_number_of_parameters_trotter_erros',psi=psis,hs_driver=hs_driver,hs_target=hs_target,energy=es)
    