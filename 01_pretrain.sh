#!/bin/bash

set -x

#module load cuda/9.0
#module unload cuda/8.0
#module load cuda/9.0 cudnn/7.1.3_cuda-9.0
nvcc --version

processor=cpu
models=all
split_ratio=0.33
epochs=20
data='SADRA-Dataset'
path='models'


name_job="LGR_PreTrain_CPU"
#name_job="FCL_MANNERSDB_withaug_FedLGR"

#output="/home/nc528/rds/hpc-work/nc528/Models/FCLSocRob/Results/FCL/withaug"
echo $name_job
sbatch -J $name_job -o out_logs_${name_job}.out.log -e err_${name_job}.err.log 02_pretrain.script ${models} ${split_ratio} ${epochs} ${data} ${path} ${processor}



