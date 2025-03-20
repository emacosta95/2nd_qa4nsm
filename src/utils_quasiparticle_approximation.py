from itertools import combinations

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
from typing import Optional,List
from scipy.sparse import lil_matrix

class QuasiParticlesConverter():
    
    def __init__(self,):
        pass
    
    def initialize_shell(self,state_encoding:List):
        
        #### nn and pp
        couples=[]
        for a,state_a in enumerate(state_encoding):
            for b,state_b in enumerate(state_encoding):
                
                if b>a:
                    _,_,ja,ma,_,tza=state_a
                    _,_,jb,mb,_,tzb=state_b
                    if ja==jb and ma==-mb and tza==tzb:
                        couples.append([a,b])
                        
        for a,state_a in enumerate(state_encoding):
            for b,state_b in enumerate(state_encoding):
                
                if b>a:
                    _,_,ja,ma,_,tza=state_a
                    _,_,jb,mb,_,tzb=state_b
                    if ja==jb and ma==-mb and tza==-tzb:
                        couples.append([a,b])
            
        self.couples=couples



    def new_base_computation(self,base:np.ndarray):
        
        indices=np.nonzero(base)[0]
        new_base=np.zeros(len(self.couples))
        value=np.sum(base)
        print('base=',base,'\n')     
                
        list_of_token_indices=[]
        
        for i in range(new_base.shape[0]):
            
            if base[self.couples[i][0]]+base[self.couples[i][1]]!=2 :
                continue
            else:
                new_base[i]+=1
                base[self.couples[i][0]]=0
                base[self.couples[i][1]]=0
        
        if np.sum(new_base)==value//2:
            return new_base
        

    def get_the_basis_matrix_transformation(self,basis:np.ndarray):
        
        self.quasiparticle_basis=[]
        self.particles2quasiparticles=lil_matrix((basis.shape[0],basis.shape[0]))
        qp_idx=0
        for i,b in enumerate(basis):
            qp_base=self.new_base_computation(base=b)
            
            if qp_base is not(None):
                print('qp base=',qp_base)
                print(qp_idx,i)    
                self.quasiparticle_basis.append(qp_base)
                self.particles2quasiparticles[qp_idx,i]=1.
                qp_idx+=1
        self.quasiparticle_basis=np.asarray(self.quasiparticle_basis)