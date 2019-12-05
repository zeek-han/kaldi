#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh

featdir="./exp.tmp/train_combined_no_sil"
rm -rf $featdir
mkdir -p $featdir

write_num_frames_opt="--write-num-frames=ark,t:$featdir/log/utt2num_frames.1"

apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:./data/train_combined.tmp/split12/1/feats.scp ark:-    |
select-voiced-frames ark:- scp,s,cs:./data/train_combined.tmp/split12/1/vad.scp ark:-   |
copy-feats --compress=$compress $write_num_frames_opt ark:- \
  ark,scp:$featdir/xvector_feats_train_combined.1.ark,$featdir/xvector_feats_train_combined.1.scp
