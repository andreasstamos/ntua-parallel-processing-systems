#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_omp_gameoflife

## Output and error files
#PBS -o run.out
#PBS -e run.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=8

##How long should the job run for?
#PBS -l walltime=01:10:00

## Start 

module load openmp
cd /home/parallel/parlab19/exer01
for n in 64 1024 4096; do
	for cpus in 1 2 3 4 5 6 7 8; do
		echo $n
		echo $cpus
		OMP_NUM_THREADS=$cpus ./game $n 1000
	done
done

