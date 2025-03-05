from src.fermi_hubbard_library import FemionicBasis
import numpy as np
from typing import List, Dict, Callable,Optional
from scipy.linalg import expm
import scipy
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize



def configuration(res,energy,grad_energy):
    
    print('Optimization Success=',res.success)
    print(f'energy={energy:.5f}')
    print(f'average gradient={np.average(grad_energy):.5f} \n')
    


class Schedule: 
    def __init__(self,tf:float,type:str,number_of_parameters:int,nsteps:int,seed:Optional[int]=None):
        
        self.tf=tf
        self.type=type
        self.time=np.linspace(0,self.tf,nsteps)
        self.parameters=np.zeros(2*number_of_parameters)
        if type=='F-CRAB':
            np.random.seed(seed)
            self.parameters=np.random.uniform(-1,1,size=2*number_of_parameters)
        self.number_parameters=number_of_parameters
        self.seed=seed
    def get_driving(self)-> np.ndarray:
        
        if self.type=='power law':
            matrix_driver=(self.parameters[:self.number_parameters,None]*(self.time[None,:]/self.tf)**np.arange(1,self.parameters.shape[0]//2+1)[:,None])
            matrix_target=(self.parameters[self.number_parameters:,None]*(self.time[None,:]/self.tf)**np.arange(1,self.parameters.shape[0]//2+1)[:,None])
            h_driver=(1-self.time/self.tf)*(1+np.sum(matrix_driver,axis=0))
            h_target=(self.time/self.tf)*(1+np.sum(matrix_target,axis=0))
        
        if self.type=='F-CRAB':
            #np.random.seed(self.seed)
            dim=self.parameters.shape[0]
            omegas=(1+np.random.uniform(-0.5,0.5,size=self.number_parameters))*np.pi*np.arange(self.number_parameters)/self.tf
            matrix_driver=(self.parameters[:dim//2,None]*np.sin(self.time[None,:]*omegas[:,None]))
            matrix_target=(self.parameters[dim//2:,None]*np.sin(self.time[None,:]*omegas[:,None]))
            
            h_driver=(1-self.time/self.tf)*(1+np.average(matrix_driver,axis=0))
            h_target=(self.time/self.tf)*(1+np.average(matrix_target,axis=0))
        return h_driver,h_target

  
class SchedulerModel(Schedule):    
    def __init__(self,initial_state:np.ndarray,target_hamiltonian: scipy.sparse.spmatrix,initial_hamiltonian: scipy.sparse.spmatrix,tf:float,number_of_parameters:int,nsteps:int,type:str,seed:int):
        
        super().__init__(tf=tf,type=type, number_of_parameters=number_of_parameters,nsteps=nsteps,seed=seed)
        self.target_hamiltonian=target_hamiltonian
        self.initial_hamiltonian=initial_hamiltonian
        self.initial_state=initial_state
        
        
        self.energy=1000
        self.psi=None
    def forward(self,parameters):
        psi=self.initial_state
        dt=self.time[1]-self.time[0]
        self.parameters=parameters
        
        h_driver,h_target=self.get_driving()
        
        for i,t in enumerate(self.time):
            time_hamiltonian=h_driver[i]*self.initial_hamiltonian+h_target[i]*self.target_hamiltonian
            psi=expm_multiply(-1j*dt*time_hamiltonian,psi)

        self.energy=psi.conjugate().transpose().dot(self.target_hamiltonian.dot(psi))
        self.psi=psi
        return self.energy
    def callback(self,args):
        print(self.energy)