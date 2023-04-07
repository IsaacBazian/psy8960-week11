#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --mem=16gb
#SBATCH -t 00:30:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bazia001@umn.edu
#SBATCH -p amdsmall
cd ~/week11-cluster
module load R/4.2.2-openblas
Rscript week11-cluster.R
