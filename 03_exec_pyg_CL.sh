#!/bin/bash

#set -x

echo "You provided the arguments:" "$@"
echo "You provided $# arguments"

#/usr/local/cuda/bin/nvcc --version
nvcc --version

echo "Sourcing Virtual Environent"
source /home/nc528/rds/hpc-work/nc528/Code/FCLSocRob/FCL_venv/bin/activate
#python3 --version
# python3 -u main.py --batch_size "$1" --epochs 100
# python main.py -s "$1" -m MobileNet -n 10 -e 10 -c 2 -f 10 -p SADRA-Dataset -o output -b "$2" -a "$3" -t "$4"
#python main_fcl.py -sfl "$1" -scl "$2" -m DeepLabMobileNet -n 10 -e 10 -c 2 -f 2 -p "$3" -o "$7" -a "$5" -pro "$4" -b "$6" -r "all"

python main_fcl.py --strategy_fl "$1" --strategy_cl "$2" --model "$8" --rounds "$9" --epochs "${10}" --icl "${11}" --fcl "${12}" --path "$3" --output "$4" --base "$5" --aug "$6" --processor_type "$7" --reg_coef "${13}" --temp_dir "${14}"

echo "Deactivating Virtual Environment"
deactivate
