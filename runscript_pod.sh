#!/bin/bash
#########################################################################
## Name of my job
#PBS -N lmg_periodic

## Name of the job queue
#PBS -q S30

## Walltime
#PBS -l walltime=04:00:00

##Number of nodes and procs per node.
#PBS -l nodes=1:ppn=10

##Send me email when my job aborts, begins, or ends
#PBS -m ea
#PBS -M daneel@utexas.edu
#########################################################################

## Name of python script to be executed
SCRIPT="./lipkin.py"
#########################################################################

##Export all PBS environment variables
#PBS -V
##Output file. Combine stdout and stderr into one
#PBS -j oe
cd $PBS_O_WORKDIR
## Number of OpenMP threads to be used by the blas library. Keep this small
export OMP_NUM_THREADS=2
##Load these modules before running
module load openblas anaconda/2019.03/python3.7
BEGINTIME=$(date +"%s")
python -W ignore $SCRIPT
ENDTIME=$(date +"%s")
ELAPSED_TIME=$(($ENDTIME-$BEGINTIME))
echo "#Runtime: $(($ELAPSED_TIME / 60)) minutes and $(($ELAPSED_TIME % 60)) seconds."
