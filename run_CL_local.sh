#!/bin/bash

processor=gpu
strategy=all
strategy_cl=all
base=all
model=MobileNet
#model=DeepLabMobileNet
aug=True
rounds=1
epochs=10
icl=2
fcl=2
coef='all'

name_job="FL_MANNERSDB"
data="/local/scratch/nc528/Datasets/MannersDB/SADRA-Dataset"
output="/local/scratch/nc528/FCLSocRob_Internship/Results/outputs/FCL/withaug"
#output="/local/scratch/nc528/FCLSocRob_Internship/Results/outputs/FCL/withoutaug"
temp_dir='/local/scratch/nc528/raytmp2'
#temp_dir=''
python main_fcl.py --strategy_fl ${strategy} --strategy_cl ${strategy_cl} --model ${model} --rounds ${rounds} --epochs ${epochs} --icl ${icl} --fcl ${fcl} --path ${data} --output ${output} --aug ${aug} --processor_type ${processor} --base ${base} --temp_dir ${temp_dir} --reg_coef ${coef}
