import numpy as np
from typing import List,Optional
import scipy
from src.fermi_hubbard_library import FemionicBasis
import numpy as np
from typing import List, Dict, Callable,Optional
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from scipy.optimize import minimize
import scipy.sparse as sp

class EvolutionSolver():
    
    def __init__(self,time:np.ndarray,time_dependent_hamiltonian:List[scipy.sparse.spmatrix],initial_state:Optional[np.ndarray],):
        self.time=time
        self.time_dependent_hamiltonian=time_dependent_hamiltonian
        self.initial_state=initial_state
    
    
    def depolarization_option(self,noise_option:bool,noise_coupling:float):
        
        self.depolarization_coupling=noise_coupling
        self.noise_option=noise_option
        
        
    def pure_time_evolution(self,):
        
        dt=self.time[1]-self.time[0]
        psi=self.initial_state
        self.evolving_psi=[]
        for i,t in enumerate(self.time):
            time_hamiltonian=self.time_dependent_hamiltonian[i]
            psi=expm_multiply(-1j*dt*time_hamiltonian,psi)
            self.evolving_psi.append(psi)
        
        print('Time evolution successfully ended! \n')

    def noisy_time_evolution(self,):
        dt=self.time[1]-self.time[0]
        rho=self.initial_state
        
        for i,t in enumerate(self.time):
            time_hamiltonian=self.time_dependent_hamiltonian[i]
            rho=self.__evolve_density_matrix(hamiltonian=time_hamiltonian,rho=rho,t=dt)
            #depolarization term
            rho=(1-self.depolarization_coupling*dt)*rho+self.depolarization_coupling*dt*sp.identity(rho.shape[0])/rho.shape[0]
            
    
    def __evolve_density_matrix(self,hamiltonian, rho, t):
        # Compute Uρ using expm_multiply
        U_rho = expm_multiply(-1j * hamiltonian * t, rho)
        # Compute U†(Uρ)
        return expm_multiply(-1j * hamiltonian * t, U_rho.T.conjugate()).T.conjugate()  # Applying U†
    
    
def evolve_density_matrix(hamiltonian, rho, t):
        # Compute Uρ using expm_multiply
        U_rho = expm_multiply(-1j * hamiltonian * t, rho)
        # Compute U†(Uρ)
        return expm_multiply(-1j * hamiltonian * t, U_rho.T.conjugate()).T.conjugate()  # Applying U†
    