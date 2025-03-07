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
    def __init__(self,tf:float,type:str,number_of_parameters:int,nsteps:int,seed:Optional[int]=None,mode:Optional[str]='annealing ansatz'):
        
        self.tf=tf
        self.nsteps=nsteps
        self.type=type
        self.mode=mode
        self.time=np.linspace(0,self.tf,nsteps)
        self.parameters=np.zeros(2*number_of_parameters)
        if self.type=='F-CRAB' or self.type=='fourier':
            self.parameters=np.zeros(4*number_of_parameters)
        self.number_parameters=number_of_parameters
        self.seed=seed
        
        
        
    def get_driving(self)-> np.ndarray:
        
        if self.type=='power law':
            matrix_driver=(self.parameters[:self.number_parameters,None]*(self.time[None,:]/self.tf)**np.arange(1,self.parameters.shape[0]//2+1)[:,None])
            matrix_target=(self.parameters[self.number_parameters:,None]*(self.time[None,:]/self.tf)**np.arange(1,self.parameters.shape[0]//2+1)[:,None])
            
        if self.type=='F-CRAB':
            #np.random.seed(self.seed)
            dim=self.parameters.shape[0]
            omegas=np.linspace(0,0.5*self.nsteps/self.tf,dim//4)*(1+np.random.uniform(-0.5,0.5,dim//4))
            print('omegas=',omegas)
            matrix_driver=(self.parameters[:dim//4,None]*np.sin(self.time[None,:]*omegas[:,None])+self.parameters[dim//4:dim//2,None]*np.cos(self.time[None,:]*omegas[:,None]))
            matrix_target=(self.parameters[dim//2:dim//2+dim//4,None]*np.sin(self.time[None,:]*omegas[:,None])+self.parameters[dim//2+dim//4:,None]*np.cos(self.time[None,:]*omegas[:,None]))
            
        if self.type=='fourier':
            #np.random.seed(self.seed)
            dim=self.parameters.shape[0]
            omegas=np.linspace(0,0.5*self.nsteps/self.tf,dim//4)

            
            matrix_driver=(self.parameters[:dim//4,None]*np.sin(self.time[None,:]*omegas[:,None])+self.parameters[dim//4:dim//2,None]*np.cos(self.time[None,:]*omegas[:,None]))
            matrix_target=(self.parameters[dim//2:dim//2+dim//4,None]*np.sin(self.time[None,:]*omegas[:,None])+self.parameters[dim//2+dim//4:,None]*np.cos(self.time[None,:]*omegas[:,None]))
            
            
        
        if self.mode=='annealing ansatz':
            h_driver=(1-self.time/self.tf)*(1+np.average(matrix_driver,axis=0))
            h_target=(self.time/self.tf)*(1+np.average(matrix_target,axis=0))
        
        else:
            h_driver=(self.time/self.tf)*(1+np.average(matrix_driver,axis=0))
            h_target=(self.time/self.tf)*(1+np.average(matrix_target,axis=0))
        
        return h_driver,h_target

  
class SchedulerModel(Schedule):    
    def __init__(self,initial_state:np.ndarray,target_hamiltonian: scipy.sparse.spmatrix,initial_hamiltonian: scipy.sparse.spmatrix,reference_hamiltonian:scipy.sparse.spmatrix,tf:float,number_of_parameters:int,nsteps:int,type:str,seed:int,mode:Optional[str]='annealing ansatz'):
        
        super().__init__(tf=tf,type=type, number_of_parameters=number_of_parameters,nsteps=nsteps,seed=seed,mode=mode)
        self.target_hamiltonian=target_hamiltonian
        self.initial_hamiltonian=initial_hamiltonian
        self.reference_hamiltonian=reference_hamiltonian
        self.initial_state=initial_state
        
        
        self.energy=1000
        self.psi=None
        
        #### memory
        self.history=[]
        self.history_psi=[]
        self.history_drivings=[]
        self.history_parameters=[]
        self.history_run=[]
        
        self.run_number=0
        
    def forward(self,parameters):
        psi=self.initial_state
        dt=self.time[1]-self.time[0]
        self.parameters=parameters
        
        h_driver,h_target=self.get_driving()
        
        for i,t in enumerate(self.time):
            time_hamiltonian=h_driver[i]*self.initial_hamiltonian+h_target[i]*self.target_hamiltonian
            psi=expm_multiply(-1j*dt*time_hamiltonian,psi)

        self.energy=psi.conjugate().transpose().dot(self.reference_hamiltonian.dot(psi))
        self.psi=psi
        #every time we run a simulation
        self.run_number+=1
        
        return self.energy
    def callback(self,args):
        self.history.append(self.energy)
        self.history_parameters.append(self.parameters)
        self.history_drivings.append(self.get_driving())
        self.history_psi.append(self.psi)
        self.history_run.append(self.run_number)
        print(self.energy)