#!/bin/bash

set -x

#module load cuda/9.0
#module unload cuda/8.0
#module load cuda/9.0 cudnn/7.1.3_cuda-9.0
nvcc --version

processor=gpu
strategy=all
strategy_cl=all
base=all
aug=False
model=MobileNet
#model=DeepLabMobileNet
rounds=10
epochs=10
icl=2
fcl=2
#tmp_dir='/home/nc528/rds/hpc-work/nc528/Code/FCLSocRob/tmp'
tmp_dir=''
reg_coef='all'

name_job="FCL_MANNERSDB_withoutaug"
#name_job="FCL_MANNERSDB_withaug"
data="/home/nc528/rds/hpc-work/nc528/Datasets/MANNERSDB/SADRA-Dataset"
output="/home/nc528/rds/hpc-work/nc528/Models/FCLSocRob/Results/FCL/withoutaug"
#output="/home/nc528/rds/hpc-work/nc528/Models/FCLSocRob/Results/FCL/withaug"
echo $name_job
sbatch -J $name_job -o out_logs_${name_job}.out.log -e err_${name_job}.err.log 02_slurm_script_CL.script ${strategy} ${strategy_cl} ${data} ${output} ${base} ${aug} ${processor} ${model} ${rounds} ${epochs} ${icl} ${fcl} ${reg_coef} ${tmp_dir}


