#!/bin/bash

processor=gpu
strategy=all
base=all
model=MobileNet
aug=True
rounds=10
epochs=10
icl=2
fcl=2
data="Data" # Path to dataset
output="" #Provide path to store output files

python main.py --strategy ${strategy} --model ${model} --rounds ${rounds} --epochs ${epochs} --icl ${icl} --fcl ${fcl} --path ${data} --output ${output} --aug ${aug} --processor_type ${processor} --base ${base}
