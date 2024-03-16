#!/bin/bash

processor=gpu
strategy=all
base=all
model=MobileNet
#model=DeepLabMobileNet
aug=True
rounds=10
epochs=10
icl=2
fcl=2
name_job="FL_MANNERSDB"
data="/local/scratch/nc528/Datasets/MannersDB/SADRA-Dataset"
output="/local/scratch/nc528/FCLSocRob_Internship/Results/outputs/FL/withaug"

temp_dir='/local/scratch/nc528/raytmp'
#temp_dir=''
#python main.py -s ${strategy} -m MobileNet -n 4 -e 4 -c 2 -f 3 -p ${data} -o ${output} -b ${base} -a ${aug -t ${processor}
python main.py --strategy ${strategy} --model ${model} --rounds ${rounds} --epochs ${epochs} --icl ${icl} --fcl ${fcl} --path ${data} --output ${output} --aug ${aug} --processor_type ${processor} --base ${base} --temp_dir ${temp_dir}
