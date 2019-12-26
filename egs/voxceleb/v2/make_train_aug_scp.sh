#!/usr/bin/env bash

total_num=744
num_job=12
# 그래서 한개의 scp파일에 62개씩 있으면 됨

find /media/sangjik/hdd2/dataset/speech/English/VoxCeleb2/npy_mfcc/train_aug_1m.* -name "*.npy" |
sort            |
awk -v FS="/" '{
    nn = int(NR/62) + 1;
    if(nn > 12){ nn = 12};
    output_path = "/media/sangjik/hdd1/zeek_github/ASR/kaldi/egs/voxceleb/v2/mfcc_scp/raw_mfcc_train_aug_1m."nn".scp";
    #print($0" "$11" "output_path)
    system("echo "substr($11, 1, length($11)-4)" >>"output_path)
}'

