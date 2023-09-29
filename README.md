# Explore and Tell: Embodied Visual Captioning in 3D Environments (ICCV 2023)
Anwen Hu, Shizhe Chen, Liang Zhang, Qin Jin

# Environments
- refer to [kubric](https://github.com/google-research/kubric) for preparing basic experimental environment by 'docker pull kubricdockerhub/kubruntu'. (Note this docker image may not support GPU, refer to https://github.com/google-research/kubric/issues/224, you can try this docker image 'docker pull ganow/kubric:latest' for GPU supporting.)

- python 3.9.5, torch 1.12, mmcv 1.6.0

# ET-CAP Download
Download ETCAP from baidu cloud driver (), including:
- 3D assets (~64G): 3D assets used in ET-Cap come from ShapeNet and GSO, before donwloading 3D assets, please make sure you have accepted the license from [shapenet.org](https://shapenet.org/). 
- 3D scenes: ~165G
- ET-Cap annotations: ~600MB

# Create Your Own Scenes or Training Trajectories
## 3D Scenes Simulation
```
python scene_construct.py
```

## Trajectory Generation
```
python path_action_construction.py
```

# CaBOT
The CaBOT code is organized based on [mmdetection](https://github.com/open-mmlab/mmdetection). The checkpoint of CaBOT (including the Navigator and the Captioner can be downloaded from ...)

```
cd ./mmdetection/tools
```
## History-aware Navigator
### Train (refer to navigator_train.sh)
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port 1010 embodied_nav_train.py \
../configs/single_navigator/navigator_region2layer_time1layer_waction.py \
--work-dir {navigator_save_dir} \
--launcher pytorch
```

### Evaluate on validation/test set (refer to navigator_dist_test.sh)
```
## test history-aware navigator
model_dir={navigator_save_dir}
config=navigator_region2layer_time1layer_waction.py
model_name=latest.pth
render_dir={render_save_dir_name}

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 1210 embodied_nav_test.py \
../configs/single_navigator/$config \
./work_dirs/$model_dir/$model_name \
--work-dir ./work_dirs/$model_dir/online_metrics/ \
--eval_mode test \
--eval_render_type simulate_render \
--render_save_dir $render_dir \
--eval_set anno/navigation_v1_val.json \
--calculate_metrics_with_saved_result False \
--path_len_weight_eval True \
--launcher pytorch
```

## Trajectory-aware Captioner
### Train (refer to captioner_train.sh)
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port 1400 embodied_trajcap_train.py \
../configs/single_captioner/captioner_timeglobal_meanview_detrinit.py \
--work-dir {captioner_save_dir} \
--launcher pytorch
```

### Captioning Evaluate (with oracle trajectories) on validation/test set (refer to navigator_dist_test.sh)
```
model_dir={captioner_save_dir}
config=captioner_timeglobal_meanview_detrinit.py

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 1111 embodied_trajcap_test.py \
../configs/single_captioner/$config \
./work_dirs/$model_dir/latest.pth \
--use_val_best_checkpoint True \
--eval_set anno/navicaption_v1_val.json \
--eval CIDEr \
--calculate_metrics_with_saved_result False \
--launcher pytorch
```

### Captioning Evaluate (with predicted trajectories) on validation/test set (refer to navigator_dist_test.sh)
- After test the Navigator on the val/tes set, run the following script to tranfer navigation results to the input format of the Captioner
```
# revise the navi_model_dir first
python navi_result_format_transfer.py
```
- Test the Captioner with trajectories given by the Navigator
```
# before running the following script, 
# revise navi_model_dir in the corresponding config file (suffixed with '_navinfer.py'). 

config=captioner_timeglobal_meanview_detrinit_navinfer.py
model_dir={captioner_model_dir}
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port 1000 embodied_trajcap_test.py \
../configs/single_captioner/$config \
./work_dirs/$model_dir/latest.pth \
--use_val_best_checkpoint True \
--eval_set navigation_v1_trajcapinfer_val.json \
--eval CIDEr \
--calculate_metrics_with_saved_result False \
--path_len_weight_eval True \
--launcher pytorch
```

### Spice Calculation
The raw Spice in pycocoevalcap is not suitable for paragraph evaluation. To calcutae Spice for this task, we build a ParagraphSpice project (~2.25G). Download it from baidu cloud driver ()
```
# revise caption_model, pred_file and gt_file, then
python embodied_caption_spice_eval.py
```


## Citation
if you find this code useful for your research, please consider citing:
```
@article{Hu2023ExploreAT,
  title={Explore and Tell: Embodied Visual Captioning in 3D Environments},
  author={Anwen Hu and Shizhe Chen and Liang Zhang and Qin Jin},
  journal={ArXiv},
  year={2023},
  volume={abs/2308.10447},
  url={https://api.semanticscholar.org/CorpusID:261048852}
}
```



