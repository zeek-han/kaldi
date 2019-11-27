#!/usr/bin/env bash

input_prefix=$1
input_wav_dir=$2

#echo $input_prefix

djs_dir=./djs_dir/$(basename $input_wav_dir)

rm -rf $djs_dir 
mkdir -p $djs_dir ./log
./resample_and_cut.py $input_prefix $input_wav_dir $djs_dir 10
echo "resample done!! date="`date`

# wav2djs2mfcc for short wavs by multi-GPU
for chunk_dir in $djs_dir/chunk_*; do
    chunk_id=${chunk_dir##*/chunk_};
    for gpu_dir in $chunk_dir/gpu_*; do
        gpu_id=${gpu_dir##*/gpu_};
        ./wav2djs2mfcc.sh $gpu_dir $gpu_id $input_wav_dir   >./log/djt_out_${chunk_id}_$gpu_id 2>./log/djt_err_${chunk_id}_$gpu_id &
    done 
    wait
done
echo "djt-multiGPU done!! date="`date`

# wav2djs2mfcc for long wavs by single-GPU
../../../../deejaytransform-sound_length_no_limit/djt freq -t 4 -d $djs_dir/long_wav \
    -sd -spt 0  >./log/djt_out_long 2>./log/djt_err_long 
echo "$djs_dir/long_wav djt-singleGPU done!! djs2mfcc start! date="`date`
./djs2mfcc.py $djs_dir/long_wav $input_wav_dir
echo "$djs_dir/long_wav : djs2mfcc done, date="`date`
concat_long_wav_mfcc $djs_dir/long_wav $input_wav_dir

rm -rf $djs_dir 
