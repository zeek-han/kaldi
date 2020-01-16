#!/usr/bin/env bash

sudo nvidia-smi -c 3



## stage1까지만 수행
#./run_zeek_voxceleb1.sh 

mv ./mfcc     ./mfcc.original
mv ./mfcc.aws ./mfcc
from_mfcc_dir="./mfcc"
mfcc_npy_root_dir="/media/sangjik/hdd2/dataset/speech/English/VoxCeleb1/npy_mfcc"
mfcc_output_dir="./mfcc.output"
mkdir -p $mfcc_output_dir
./rewrite_mfcc_from_scp_ark.py $from_mfcc_dir $mfcc_npy_root_dir $mfcc_output_dir ./mfcc.original
mv ./mfcc ./mfcc.aws
mv $mfcc_output_dir ./mfcc
