#!/bin/bash

## Give the Job a descriptive name
#PBS -N make_game_of_life

## Output and error files
#PBS -o make.out
#PBS -e make.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=1

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start 

module load openmp
cd /home/parallel/parlab19/exer01
rm -f game
gcc -std=c99 -march=native -fopenmp -O2 -o game Game_Of_Life.c

