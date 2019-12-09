#!/usr/bin/env bash

djs_dir=$1
gpu_id=$2
input_wav_dir=$3

../../../../deejaytransform-sound_length_no_limit/djt freq -t 4 \
   -d $djs_dir -sd -spt 0 -gpu $gpu_id 
./djs2mfcc.py $djs_dir $input_wav_dir
rm -rf $djs_dir
