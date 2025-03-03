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
from typing import List

def plot_spectrum(eigenvalues):
    """
    Plot the vertical spectrum of a Hamiltonian, showing the eigenvalues as horizontal lines 
    and indicating their degeneracy.

    Parameters:
    eigenvalues (array-like): Array of eigenvalues of the Hamiltonian.
    """
    # Count the degeneracy of each eigenvalue
    degeneracy = Counter(eigenvalues)

    # Prepare data for plotting
    unique_eigenvalues = list(degeneracy.keys())
    degeneracies = list(degeneracy.values())

    # Plot the spectrum
    plt.figure(figsize=(6, 10))
    for i, (eig, deg) in enumerate(zip(unique_eigenvalues, degeneracies)):
        plt.hlines(eig, i - 0.2 * deg, i + 0.2 * deg, colors='b', linewidth=5)
        plt.text(i, eig, f'{deg}', horizontalalignment='center', verticalalignment='bottom', fontsize=24, color='r')

    # Make the plot fancy
    plt.title('Spectrum of the Hamiltonian', fontsize=16)
    plt.ylabel('Eigenvalue', fontsize=14)
    plt.xlabel('Index (degeneracy indicated by text)', fontsize=14)
    plt.xticks(range(len(unique_eigenvalues)), ['']*len(unique_eigenvalues))  # Remove x-axis ticks
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Show the plot
    plt.show()

file_name:str='data/cki'

write_j_square_twobody_file(filename=file_name)
j_square_filename:str=file_name+'_j2'
qq_filename='data/qq.p'
SPS=SingleParticleState(file_name=file_name)
energies:List=SPS.energies

matrix_qq,_=get_twobody_nuclearshell_model(file_name=qq_filename)
twobody_matrix,_=get_twobody_nuclearshell_model(file_name=file_name)


size_a:int=SPS.energies.shape[0]//2
size_b:int=SPS.energies.shape[0]//2
titles=[r'$^{8}$Li',r'$^{8}$Be',r'$^{10}$Be',r'$^{12}$Be',r'$^{10}$B',r'$^{12}$C']
nparts=[(1,3),(2,2),(2,4),(2,6),(3,3),(4,4)]
qq_list=[]
ipr_list=[]
eng_list=[]
entropy_proton_neutron_list=[]
entropy_initial_rest_list=[]
spectrum_list=[]

for i in range(len(titles)):
    nparticles_a,nparticles_b=nparts[i]
    title=titles[i]
    J2cki=J2operator(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,single_particle_states=SPS.state_encoding,j_square_filename=j_square_filename,symmetries=[SPS.total_M_zero])


    Moperator = FermiHubbardHamiltonian(
        size_a=size_a,
        size_b=size_b,
        nparticles_a=nparticles_a,
        nparticles_b=nparticles_b,
        symmetries=[SPS.total_M_zero]
    )
    diag_m:List=[]
    for state in SPS.state_encoding:
        n,l,j,m,t,tz=state
        diag_m.append(m)

    Moperator.get_external_potential(diag_m)
    Moperator.get_hamiltonian()


    QQoperator=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
    QQoperator.get_twobody_interaction(twobody_dict=matrix_qq)
    QQoperator.get_hamiltonian()

    TargetHamiltonian=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
    print('size=',size_a+size_b,size_b)
    TargetHamiltonian.get_external_potential(external_potential=energies[:size_a+size_b])
    TargetHamiltonian.get_twobody_interaction(twobody_dict=twobody_matrix)
    TargetHamiltonian.get_hamiltonian()


    nlevels=2

    egs,psis=TargetHamiltonian.get_spectrum(n_states=nlevels)

    egs=egs[0]
    psi0=psis[:,:1]

    print('M value ground state=',SPS.compute_m_exp_value(psi0,basis=TargetHamiltonian.basis),'\n')
    print('total_m=',psi0.transpose().conjugate().dot(Moperator.hamiltonian.dot(psi0)))
    print('total J value=',J2cki.j_value(psi=psi0),'\n')
    print('deformation=',psi0.transpose().conjugate().dot(QQoperator.hamiltonian.dot(psi0)))




    min_b=np.zeros(size_a+size_b)

    if nparticles_a==2:    
        min_b[0]=1
        min_b[3]=1    
    if nparticles_a==4:
        min_b[0:4]=1
    if nparticles_a==6:
        min_b[0:6]=1
        
    if nparticles_b==2:    
        min_b[0+size_a]=1
        min_b[3+size_a]=1    
    if nparticles_b==4:
        min_b[0+size_a:4+size_a]=1
    if nparticles_b==6:
        min_b[0+size_a:6+size_a]=1
    
    if nparticles_a==1 and nparticles_b==3:
        min_b[0]=1
        min_b[3+size_a]=1
        min_b[1+size_a]=1
        min_b[2+size_a]=1
        
    if nparticles_a==3 and nparticles_b==3:
        min_b[0]=1
        min_b[1]=1
        min_b[2]=1
        
        min_b[3+size_a]=1
        min_b[1+size_a]=1
        min_b[2+size_a]=1
        
    print('initial state=',min_b)

    InitialHamiltonian=FermiHubbardHamiltonian(size_a=size_a,size_b=size_b,nparticles_a=nparticles_a,nparticles_b=nparticles_b,symmetries=[SPS.total_M_zero])
   


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

    nlevels=2

    es,psis=InitialHamiltonian.get_spectrum(n_states=nlevels)
    einitial=es[0]
    psi_initial=psis[:,0]
    print(psi_initial)
    print('psi initial vs psi configuration',psi_initial-psi_configuration,'\n')
    print('total_m=',SPS.compute_m_exp_value(psi=psi_initial,basis=InitialHamiltonian.basis))
    #print('total J^2=',psi_initial.transpose().conjugate() @ Joperator.hamiltonian @ psi_initial)
    print('GS ENERGY=',es[0])
    from fractions import Fraction
    print('M value initial state=',psi_initial.transpose().conjugate().dot(Moperator.hamiltonian.dot(psi_initial)),'\n')

    ### final time of the simulation
    tf = 20
    ### we fix the nsteps such that dt=0.1
    nstep=int(tf*10)
    nlevels=3
    time = np.linspace(0.0, tf, nstep)

    ### psi is initialized to be the gs of the target hamiltonian
    psi = psi_initial

    ### we stored the spectrum
    spectrum = np.zeros((nlevels, nstep))
    probabilities=np.zeros((nlevels, nstep))

    ### we need the dt for the simulation
    dt=time[1]-time[0]

    ### stored energy (variance is deprecated actually)
    eng_t=[]
    variance_t=[]

    ### stored fidelity
    fidelity_t=[]
    fidelity_psi0_t=[]

    ### stored important quantities
    ipr_t=[]
    qq_t=[]
    entropy_proton_neutron=[]
    entropy_initial_rest=[]
    lambd=1-time/tf
    
    for i in trange(nstep):

        time_hamiltonian = (
            InitialHamiltonian.hamiltonian * ( lambd[i])
            + TargetHamiltonian.hamiltonian * (1-lambd[i])
        ) #+lambd[i]*(1-lambd[i]) * IntermediateHamiltonian.hamiltonian
        values, psis = eigsh(time_hamiltonian, k=nlevels, which="SA")
        psi=expm_multiply(-1j*dt*time_hamiltonian,psi)

        ### compute the IPR
        ipr_t.append(np.sum((psi.conjugate()*psi)**2))
        
        ### compute the qq
        qq_t.append(psi.conjugate().transpose().dot(QQoperator.hamiltonian.dot(psi)))
        
        e_ave=psi.conjugate().transpose()@ time_hamiltonian @ psi
        e_square_ave = (
            psi.conjugate().transpose() @ time_hamiltonian @ time_hamiltonian @ psi
        )
        eng_t.append(e_ave)
        variance_t.append(e_square_ave-e_ave**2)
        spectrum[:, i] = values
        
        fidelity_psi0_t.append((psi0.conjugate().transpose() @ psi[:]
                    ) * np.conj(psi0.conjugate().transpose() @ psi[:]))

        degenerate_fidelity=0.
        count=0
        for j in range(values.shape[0]):
            if np.isclose(values[j],values[0]):
                degenerate_fidelity += (
                    psis[:, j].conjugate().transpose() @ psi[:]
                ) * np.conj(psis[:, j].conjugate().transpose() @ psi[:])
                count=count+1
            
            probabilities[j,i]=(
                    psis[:, j].conjugate().transpose() @ psi[:]
                ) * np.conj(psis[:, j].conjugate().transpose() @ psi[:])

        fidelity=degenerate_fidelity        
        fidelity_t.append(fidelity)
        print([i for i in range(size_a)])
        entropy_proton_neutron.append(TargetHamiltonian.entanglement_entropy(indices=[i for i in range(size_a)],psi=psi))
        entropy_initial_rest.append(TargetHamiltonian.entanglement_entropy(indices=np.nonzero(min_b)[0],psi=psi))
    eng_t=np.asarray(eng_t)
    fidelity_t=np.asarray(fidelity_t)
    fidelity_psi0_t=np.asarray(fidelity_psi0_t)
    variance_t=np.asarray(variance_t)
    print(np.abs((egs-eng_t[-1])/egs))
    print(fidelity)
    print(eng_t.shape)
    ipr_t=np.asarray(ipr_t)-1/psi.shape[0] #we remove the fully mixed condition
    qq_t=np.asarray(qq_t)
    entropy_proton_neutron=np.asarray(entropy_proton_neutron)
    entropy_initial_rest=np.asarray(entropy_initial_rest)
    
    ipr_list.append(ipr_t)
    qq_list.append(qq_t)
    eng_list.append(eng_t)
    spectrum_list.append(spectrum)
    entropy_proton_neutron_list.append(entropy_proton_neutron)
    entropy_initial_rest_list.append(entropy_initial_rest)
np.savez('data/ipr_computation_pshell',ipr=np.asarray(ipr_list),qq=np.asarray(qq_list),energy=np.asarray(eng_list),spectrum=np.asarray(spectrum_list),labels=titles,time=time,entropy_proton_neutron=np.asarray(entropy_proton_neutron_list),entropy_initial_rest=np.asarray(entropy_initial_rest_list))
    