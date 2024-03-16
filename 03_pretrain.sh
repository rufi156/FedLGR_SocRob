#!/bin/bash

#set -x

echo "You provided the arguments:" "$@"
echo "You provided $# arguments"

#/usr/local/cuda/bin/nvcc --version
nvcc --version

echo "Sourcing Virtual Environent"
source ~/jupyter-env/bin/activate
#python3 --version
# python3 -u main.py --batch_size "$1" --epochs 100
# python main.py -s "$1" -m MobileNet -n 10 -e 10 -c 2 -f 10 -p SADRA-Dataset -o output -b "$2" -a "$3" -t "$4"
#python main_fcl.py -sfl "$1" -scl "$2" -m DeepLabMobileNet -n 10 -e 10 -c 2 -f 2 -p "$3" -o "$7" -a "$5" -pro "$4" -b "$6" -r "all"

python pretrain.py --models "$1" --split_ratio "$2" --epochs "$3" --data "$4" --path "$5" -t "$6"

echo "Deactivating Virtual Environment"
deactivate
