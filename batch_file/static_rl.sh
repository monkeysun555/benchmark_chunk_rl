#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=30:00:00
#SBATCH --mem=4GB
#SBATCH --job-name=chunk_rl_2s
#SBATCH --mail-type=END
#SBATCH --mail-user=ls3817@nyu.edu
#SBATCH --output=slurm_%j.out
  
module purge

VIRTUALENV=$SCRATCH
RUNDIR=$SCRATCH/low_latency_streaming/rate_version/benchmark_chunk_rl_2s

cd $VIRTUALENV

source ./tf_learn/bin/activate
  
cd $RUNDIR
python static_sim_chunk.py
