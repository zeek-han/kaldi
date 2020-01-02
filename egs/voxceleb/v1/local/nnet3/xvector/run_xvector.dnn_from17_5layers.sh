#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
#
# Copied from egs/sre16/v1/local/nnet3/xvector/tuning/run_xvector_1a.sh (commit e082c17d4a8f8a791428ae4d9f7ceb776aef3f0b).
#
# Apache 2.0.

# This script trains a DNN similar to the recipe described in
# http://www.danielpovey.com/files/2018_icassp_xvectors.pdf

. ./cmd.sh
set -e

stage=1
train_stage=0
use_gpu=true
remove_egs=false

data=data/train
nnet_dir="/media/sangjik/hdd2/speaker_verification/kaldi/xvector_nnet_1a"
egs_dir="${nnet_dir}/egs"


. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

num_pdfs=$(awk '{print $2}' $data/utt2spk | sort | uniq -c | wc -l)

# Now we create the nnet examples using sid/nnet3/xvector/get_egs.sh.
# The argument --num-repeats is related to the number of times a speaker
# repeats per archive.  If it seems like you're getting too many archives
# (e.g., more than 200) try increasing the --frames-per-iter option.  The
# arguments --min-frames-per-chunk and --max-frames-per-chunk specify the
# minimum and maximum length (in terms of number of frames) of the features
# in the examples.
#
# To make sense of the egs script, it may be necessary to put an "exit 1"
# command immediately after stage 3.  Then, inspect
# exp/<your-dir>/egs/temp/ranges.* . The ranges files specify the examples that
# will be created, and which archives they will be stored in.  Each line of
# ranges.* has the following form:
#    <utt-id> <local-ark-indx> <global-ark-indx> <start-frame> <end-frame> <spk-id>
# For example:
#    100304-f-sre2006-kacg-A 1 2 4079 881 23

# If you're satisfied with the number of archives (e.g., 50-150 archives is
# reasonable) and with the number of examples per speaker (e.g., 1000-5000
# is reasonable) then you can let the script continue to the later stages.
# Otherwise, try increasing or decreasing the --num-repeats option.  You might
# need to fiddle with --frames-per-iter.  Increasing this value decreases the
# the number of archives and increases the number of examples per archive.
# Decreasing this value increases the number of archives, while decreasing the
## number of examples per archive.
#if [ $stage -le 6 ]; then
#  echo "stage=6"
#  echo "$0: Getting neural network training egs";
#  # dump egs.
#  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $egs_dir/storage ]; then
#    utils/create_split_dir.pl \
#     /media/sangjik/hdd2/kaldi_voxceleb/b{03,04,05,06}/$USER/kaldi-data/egs/voxceleb2/v2/xvector-$(date +'%m_%d_%H_%M')/$egs_dir/storage $egs_dir/storage
#  fi
#  sid/nnet3/xvector/get_egs.sh --cmd "$train_cmd" \
#    --nj 2 \
#    --stage 0 \
#    --frames-per-iter 500000000 \
#    --frames-per-iter-diagnostic 100000 \
#    --min-frames-per-chunk 2000 \
#    --max-frames-per-chunk 4000 \
#    --num-diagnostic-archives 3 \
#    --num-repeats 50 \
#    "$data" $egs_dir
#fi
#
#if [ $stage -le 7 ]; then
#  echo "stage=7"
#  echo "$0: creating neural net configs using the xconfig parser";
#  num_targets=$(wc -w $egs_dir/pdf2num | awk '{print $1}')
#  feat_dim=$(cat $egs_dir/info/feat_dim)
#
#  # This chunk-size corresponds to the maximum number of frames the
#  # stats layer is able to pool over.  In this script, it corresponds
#  # to 100 seconds.  If the input recording is greater than 100 seconds,
#  # we will compute multiple xvectors from the same recording and average
#  # to produce the final xvector.
#  max_chunk_size=100000
#
#  # The smallest number of frames we're comfortable computing an xvector from.
#  # Note that the hard minimum is given by the left and right context of the
#  # frame-level layers.
#  min_chunk_size=250
#  mkdir -p $nnet_dir/configs
#  cat <<EOF > $nnet_dir/configs/network.xconfig
#  # please note that it is important to have input layer with the name=input
#
#  # The frame-level layers
#  input dim=${feat_dim} name=input
#  relu-batchnorm-layer name=tdnn0 input=Append(-17, -16, -15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17) dim=${feat_dim}
#  relu-batchnorm-layer name=tdnn1 input=Append(-14,-12,-10,-8,-6,-4,-2,-1,0,1,2,4,6,8,10,12,14) dim=512
#  relu-batchnorm-layer name=tdnn2 input=Append(-12,-9,-6,-3,0,3,6,9,12) dim=512
#  relu-batchnorm-layer name=tdnn3 input=Append(-12,-8,-4,0,4,8,12) dim=512
#  relu-batchnorm-layer name=tdnn4 input=Append(-10,-5,0,5,10) dim=512
#  relu-batchnorm-layer name=tdnn5 input=Append(-6,0,6) dim=512
#  relu-batchnorm-layer name=tdnn6 dim=512
#  relu-batchnorm-layer name=tdnn7 dim=1500
#
#  # The stats pooling layer. Layers after this are segment-level.
#  # In the config below, the first and last argument (0, and ${max_chunk_size})
#  # means that we pool over an input segment starting at frame 0
#  # and ending at frame ${max_chunk_size} or earlier.  The other arguments (1:1)
#  # mean that no subsampling is performed.
#  stats-layer name=stats config=mean+stddev(0:1:1:${max_chunk_size})
#
#  # This is where we usually extract the embedding (aka xvector) from.
#  relu-batchnorm-layer name=tdnn8 dim=512 input=stats
#
#  # This is where another layer the embedding could be extracted
#  # from, but usually the previous one works better.
#  relu-batchnorm-layer name=tdnn9 dim=512
#  output-layer name=output include-log-softmax=true dim=${num_targets}
#EOF
#
#  steps/nnet3/xconfig_to_configs.py \
#      --xconfig-file $nnet_dir/configs/network.xconfig \
#      --config-dir $nnet_dir/configs/
#  cp $nnet_dir/configs/final.config $nnet_dir/nnet.config
#
#  # These three files will be used by sid/nnet3/xvector/extract_xvectors.sh
#  echo "output-node name=output input=tdnn8.affine" > $nnet_dir/extract.config
#  echo "$max_chunk_size" > $nnet_dir/max_chunk_size
#  echo "$min_chunk_size" > $nnet_dir/min_chunk_size
#fi
#
dropout_schedule='0,0@0.20,0.1@0.50,0'
srand=123
if [ $stage -le 8 ]; then
  echo "stage=8"
  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --trainer.optimization.proportional-shrink 10 \
    --trainer.optimization.momentum=0.5 \
    --trainer.optimization.num-jobs-initial=3 \
    --trainer.optimization.num-jobs-final=8 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.minibatch-size=16 \
    --trainer.srand=$srand \
    --trainer.max-param-change=2 \
    --trainer.num-epochs=3 \
    --trainer.dropout-schedule="$dropout_schedule" \
    --trainer.shuffle-buffer-size=1000 \
    --egs.frames-per-eg=1 \
    --egs.dir="$egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=wait \
    --dir=$nnet_dir  || exit 1;
    #--trainer.optimization.minibatch-size=64 \
fi

exit 0;
