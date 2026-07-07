#!/usr/bin/env bash 
set -eu  

epochs=200
batch_size=8
gpuid=0
num_workers=24
cpt_dir=data_cpt_unet_tse_steplr_clip


./train_tinypse.py \
  --gpu $gpuid \
  --epochs $epochs \
  --batch-size $batch_size \
  --num-workers $num_workers \
  --checkpoint $cpt_dir \
> train.log 2>&1