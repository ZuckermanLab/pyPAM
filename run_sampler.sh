#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20 ## 10
#SBATCH --mem-per-cpu=10GB ## 10 GB 
#SBATCH --job-name=affine
#SBATCH --time=35:00:00 ## 9-23:00:00 ask for 9 days + 23 hours hr on the node
##SBATCH --qos=long_jobs
#SBATCH --partition=exacloud
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=georgeau@ohsu.edu
#SBATCH --output=./out/%j.out
#SBATCH --error=./err/%j.err

source /home/groups/ZuckermanLab/georgeau/sampling_tests/py36_emcee/bin/activate

srun python /home/groups/ZuckermanLab/georgeau/sampling_tests/parallel_affine/parallel_affine_example.py 
