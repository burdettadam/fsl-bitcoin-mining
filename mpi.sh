#!/bin/bash

#SBATCH --time=00:10:00   # walltime
#SBATCH --nodes=256   # number of nodes
#SBATCH --mem-per-cpu=2024M   # memory per CPU core
#SBATCH -J "sha256 hashtest"   # job name
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
mpicc -o bitcoin bitCoinMpi.c -O3 -lssl -lcrypto
echo "bitcoin 2"
mpirun -np 2 bitcoin
echo "bitcoin 4"
mpirun -np 4 bitcoin
echo "bitcoin 8"
mpirun -np 8 bitcoin
echo "bitcoin 16"
mpirun -np 16 bitcoin
echo "bitcoin 32"
mpirun -np 32 bitcoin
echo "bitcoin 64 "
mpirun -np 64 bitcoin
echo "bitcoin 128"
mpirun -np 128 bitcoin
echo "bitcoin 256"
mpirun -np 256 bitcoin

