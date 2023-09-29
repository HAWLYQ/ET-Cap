#!/usr/bin/env bash
# eval_set:
## navicaption_v1_test_common.json
## navicaption_v1_test_novel_instance.json
## navicaption_v1_test_novel_category.json
## navicaption_v1_test.json


# test trajectory-aware captioner
## when use_val_best_checkpoint is True, the checkpoint is not the latest.pth, reading training log to find best checkpoint
model_dir=captioner_detrinit-tune-backbone_timeglobal_meanview_de2layer_lr3e5-liner-anneal_epoch20
config=captioner_timeglobal_meanview_detrinit.py

CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.run --nproc_per_node=3 --master_port 1111 embodied_trajcap_test.py \
../configs/single_captioner/$config \
./work_dirs/$model_dir/latest.pth \
--use_val_best_checkpoint True \
--eval CIDEr \
--eval_set anno/navicaption_v1_val_common.json \
--calculate_metrics_with_saved_result False \
--launcher pytorch
