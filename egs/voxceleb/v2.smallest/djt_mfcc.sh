#!/usr/bin/env bash

input_prefix=$1
input_wav_dir=$2

num_wav_chunk=20
cutted_long_wavs="/home/sangjik/cutted_long_wavs"
dir_prefix="/media/sangjik/storage_for_gpu"
dir_suffix=$(basename $input_wav_dir)

./resample_and_cut.py $input_prefix $input_wav_dir $dir_prefix $dir_suffix $cutted_long_wavs $num_wav_chunk 
echo "resample done!! date="`date`

# wav2djs2mfcc for short wavs by multi-GPU
num_gpu=`nvidia-smi -L | wc -l`
for chunk_id in `seq $num_wav_chunk | awk '{print $1 - 1}'`; do
    for gpu_id in `seq $num_gpu | awk '{print $1 - 1}'`; do
        djs_dir=${dir_prefix}_$gpu_id/$dir_suffix/chunk_${chunk_id}
        ./wav2djs2mfcc.sh $djs_dir $gpu_id $input_wav_dir   >./log/djt_out_${gpu_id}_$chunk_id 2>./log/djt_err_${gpu_id}_$chunk_id &
        #echo $djs_dir - `ls $djs_dir | wc -l`
        #echo ----
    done
    wait
done
echo "djt-multiGPU done!! date="`date`

# wav2djs2mfcc for long wavs by single-GPU
../../../../deejaytransform-sound_length_no_limit/djt freq -t 4 -d $cutted_long_wavs/long_wav \
    -sd -spt 0  >./log/djt_out_long 2>./log/djt_err_long 
echo "$cutted_long_wavs/long_wav djt-singleGPU done!! djs2mfcc start! date="`date`
./djs2mfcc.py $cutted_long_wavs/long_wav $input_wav_dir
echo "$cutted_long_wavs/long_wav : djs2mfcc done, date="`date`
concat_long_wav_mfcc $cutted_long_wavs/long_wav $input_wav_dir

rm -rf $cutted_long_wavs
