#!/usr/bin/env bash
#SBATCH -o ../results/media/test.txt
#SBATCH -p media
#SBATCH --exclusive


./memory-benchmark-media --benchmark_out_format=json --benchmark_out=../results/media/test.json