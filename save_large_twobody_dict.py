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

twobody_matrix,energies=get_twobody_nuclearshell_model(file_name=file_name)

# Save to file
with open("twobody_interaction_GCN5082.pkl", "wb") as f:
    pickle.dump(twobody_matrix, f)