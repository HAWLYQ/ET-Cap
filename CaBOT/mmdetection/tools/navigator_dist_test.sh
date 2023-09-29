#!/usr/bin/env bash
# eval_mode in ['test', 'pseudo_test', 'rule_test']
# eval_render_type ['render', 'simulate_render']
# eval_set:
## embodiedcap_navigation_v1_test_common.json
## embodiedcap_navigation_v1_test_novel_instance.json
## embodiedcap_navigation_v1_test_novel_category.json
## embodiedcap_navigation_v1_test.json


## test history-aware navigator
model_dir=navigator_waction_region2layer_time1layer_de2layer_lr1e4_epoch10
config=navigator_region2layer_time1layer_waction.py
model_name=latest.pth
render_dir=simu_rel_inference6_debug

# test is separated to multiple processes and rendered with cpus
## when test a model first time, set calculate_metrics_with_saved_result=False, the image rendered will be saved to $render_dir 
## when re-evaluate a model's test result, set calculate_metrics_with_saved_result=True to avoid rendering again

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 1210 embodied_nav_test.py \
../configs/single_navigator/$config \
./work_dirs/$model_dir/$model_name \
--work-dir ./work_dirs/$model_dir/online_metrics/ \
--eval_mode test \
--eval_render_type simulate_render \
--render_save_dir $render_dir \
--eval_set anno/navigation_v1_test_novel_instance.json \
--calculate_metrics_with_saved_result False \
--path_len_weight_eval True \
--launcher pytorch
