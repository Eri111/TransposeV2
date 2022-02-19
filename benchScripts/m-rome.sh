#!/usr/bin/env bash
#SBATCH -o ../results/rome/COMP_all_Places8.txt
#SBATCH -p rome
#SBATCH --exclusive

export OMP_PLACES=cores
export OMP_PROC_BIND=true

./build/computational-benchmark-rome8 --benchmark_out_format=json --benchmark_out=results/rome/COMP_all_Places8.json