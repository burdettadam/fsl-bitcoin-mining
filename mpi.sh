#!/bin/bash

#SBATCH --time=00:10:00   # walltime
#SBATCH --nodes=32   # number of nodes
#SBATCH --mem-per-cpu=2024M   # memory per CPU core
#SBATCH -J "latancy test"   # job name
#SBATCH --mail-user=burdettadam@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Compatibility variables for PBS. Delete if not needed.
export PBS_NODEFILE=`/fslapps/fslutils/generate_pbs_nodefile`
export PBS_JOBID=$SLURM_JOB_ID
export PBS_O_WORKDIR="$SLURM_SUBMIT_DIR"
export PBS_QUEUE=batch


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
echo "bitcoin"
mpirun -np 32 bitcoin
