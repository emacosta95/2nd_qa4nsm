#!/bin/bash
#SBATCH --job-name=quantum_job
#SBATCH --account=ub221021
#SBATCH --qos=bl_short
#SBATCH --time=00:10:00
#SBATCH --output=quantum_%j.out
#SBATCH --error=quantum_%j.err

module load python
python qibo_circuit_for_quantum_computer.py