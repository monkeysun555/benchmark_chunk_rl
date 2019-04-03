#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=2:30:00
#SBATCH --mem=4GB
#SBATCH --job-name=chunk_rl_js
#SBATCH --mail-type=END
#SBATCH --mail-user=ls3817@nyu.edu
#SBATCH --output=slurm_%j.out
  
module purge

VIRTUALENV=$SCRATCH
RUNDIR=$SCRATCH/CMAF/JS_model

cd $VIRTUALENV

source ./tflearn/bin/activate
  
cd $RUNDIR
python static_sim_chunk.py
