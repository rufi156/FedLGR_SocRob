#!/bin/bash

processor=gpu
strategy=all
strategy_cl=all
base=all
model=MobileNet
aug=True
rounds=10
epochs=10
icl=2
fcl=2
coef='all'

data="Data" # Path to dataset
output="" #Provide path to store output files

python main_fcl.py --strategy_fl ${strategy} --strategy_cl ${strategy_cl} --model ${model} --rounds ${rounds} --epochs ${epochs} --icl ${icl} --fcl ${fcl} --path ${data} --output ${output} --aug ${aug} --processor_type ${processor} --base ${base} --temp_dir ${temp_dir} --reg_coef ${coef}
