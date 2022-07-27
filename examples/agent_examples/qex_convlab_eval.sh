##################### begin execute.sh ###############################
#! /bin/sh
#
# execute.sh
# Copyright (C) 2019 lubis <lubis@hpc-login7>
#
# Distributed under terms of the MIT license.
#

#PBS -l select=1:ncpus=1:mem=48gb:ngpus=2
#PBS -l walltime=12:00:00
#PBS -A "DialSys"
#PBS -q "DSML"
#PBS -r n
#PBS -e qlog/
#PBS -o qlog/

set -e
#module load Python/3.6.5
#module load CUDA/9.1.85
#module load cuDNN/7.1
i3
export PYTHONPATH="${PYTHONPATH}:/gpfs/project/lubis/convlab-2/tests"


HEAD_FOLDER=/gpfs/project/${USER}/convlab-2/tests
SCRATCH_FOLDER=/gpfs/scratch/${USER}/${PBS_JOBID}
PBS_O_WORKDIR=$HEAD_FOLDER

cd "$PBS_O_WORKDIR"


python ${HEAD_FOLDER}/test_LAVA.py

qstat -f $PBS_JOBID

####################### end execute.sh ##########################
