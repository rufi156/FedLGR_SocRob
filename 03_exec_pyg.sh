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
#python main.py -s "$1" -m MobileNet -n 10 -e 10 -c 2 -f 10 -p "$2" -o "$3" -b "$4" -a "$5" -t "$6"

python main.py --strategy "$1" --model "$7" --rounds "$8" --epochs "$9" --icl "${10}" --fcl "${11}" --path "$2" --output "$3" --base "$4" --aug "$5" --processor_type "$6" --temp_dir "${12}"

echo "Deactivating Virtual Environment"
deactivate
