#!/usr/bin/env bash
#SBATCH -o ../results/media/test.txt
#SBATCH -p media
#SBATCH -w mp-media3
#SBATCH --exclusive


../build/memory-benchmark-media --benchmark_out_format=json --benchmark_out=../results/media/test.json