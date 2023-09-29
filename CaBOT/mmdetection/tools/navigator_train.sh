#!/usr/bin/env bash

# train navigator

# history-aware navigator: 
#    ../configs/single_navigator/navigator_region2layer_time1layer_waction.py
# history-aware navigator w/o action history: 
#   ../configs/single_navigator/navigator_region2layer_time1layer.py
# history-aware navigator w/o action history, w/o historical vision encoder:
#   ../configs/single_navigator/navigator_region2layer.py

#e.g. train history-aware navigator
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port 1010 embodied_nav_train.py \
../configs/single_navigator/navigator_region2layer_time1layer_waction.py \
--work-dir ./work_dirs/navigator_waction_region2layer_time1layer_de2layer_lr1e4_epoch10 \
--launcher pytorch