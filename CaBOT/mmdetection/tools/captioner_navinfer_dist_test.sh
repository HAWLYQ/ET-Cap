#!/usr/bin/env bash
# eval_set:
## navigation_v1_trajcapinfer_test_common.json
## navigation_v1_trajcapinfer_test_novel_instance.json
## navigation_v1_trajcapinfer_test_novel_category.json
## navigation_v1_trajcapinfer_test.json


# when use_val_best_checkpoint is True, the checkpoint is not the latest.pth
config=captioner_timeglobal_meanview_detrinit_navinfer.py
model_dir=captioner_detrinit-tune-backbone_timeglobal_meanview_de2layer_lr3e5-liner-anneal_epoch20

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 1000 embodied_trajcap_test.py \
../configs/single_captioner/$config \
./work_dirs/$model_dir/latest.pth \
--use_val_best_checkpoint True \
--eval_set navigation_v1_trajcapinfer_val.json \
--eval CIDEr \
--calculate_metrics_with_saved_result False \
--path_len_weight_eval True \
--launcher pytorch
