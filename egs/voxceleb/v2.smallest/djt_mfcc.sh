#!/usr/bin/env bash

input_prefix=$1
input_wav_dir=$2

#echo $input_prefix

djs_dir=./djs_dir/$(basename $input_wav_dir)

rm -rf $djs_dir 
mkdir -p $djs_dir ./log
./resample_and_cut.py $input_prefix $input_wav_dir $djs_dir
echo "resample done!! date="`date`
# wav2djs
for gpu_dir in $djs_dir/gpu_*; do
    gpu_id=${gpu_dir##*/gpu_};
    ../../../../deejaytransform-sound_length_no_limit/djt freq  \
        -t 4 -d $gpu_dir -sd -spt 0 -gpu $gpu_id >./log/djt_out_$gpu_id 2>./log/djt_err_$gpu_id &
done 
wait
echo "djt-multiGPU done!! date="`date`
../../../../deejaytransform-sound_length_no_limit/djt freq -t 4 -d $djs_dir/long_wav \
    -sd -spt 0  >./log/djt_out_long 2>./log/djt_err_long 
echo "djt-singleGPU for long wav done!! djs2mfcc start! date="`date`
./djs2mfcc.py $djs_dir $input_wav_dir
echo " djs2mfcc done, date="`date`
concat_long_wav_mfcc $djs_dir/long_wav $input_wav_dir
rm -rf $djs_dir 
