#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20 
#SBATCH --mem-per-cpu=10GB 
#SBATCH --job-name=parallel_affine
#SBATCH --time=35:00:00 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<email address goes here>
#SBATCH --output=./out/%j.out
#SBATCH --error=./err/%j.err

source /home/path/to/python/environment/bin/activate

srun python /home/path/to/parallel_affine/parallel_affine_example.py  
