# Explore and Tell: Embodied Visual Captioning in 3D Environments (ICCV 2023)
Anwen Hu, Shizhe Chen, Liang Zhang, Qin Jin

# Environments
- refer to [kubric](https://github.com/google-research/kubric) for preparing basic experimental environment by 'docker pull kubricdockerhub/kubruntu'. (Note this docker image may not support GPU, refer to https://github.com/google-research/kubric/issues/224, you can try this docker image 'docker pull ganow/kubric:latest' for GPU supporting.)

- python 3.9.5, torch 1.12, mmcv 1.6.0

# ET-CAP Download
Download ETCAP from 
- [Baidu Cloud Driver](https://pan.baidu.com/s/19JFRyqiq9TjRGSE89QUcfA) (pwd: rsuh): asssets+scenes+annotation
- [OneDriver](https://1drv.ms/f/s!AocXJ7uKxt6XdtEnO8p0Fr-5fl4?e=waE1Y8): annotation (assets and scenes are comming soon...)

including:
- 3D assets (zip ~64G, unzip ~142G): 3D assets used in ET-Cap come from ShapeNet and GSO, before donwloading 3D assets, please make sure you have accepted the license from [shapenet.org](https://shapenet.org/). 
    ```
    cd ETCAP
    unzip kubric_asstets.zip
    ```
- 3D scenes: zip ~165G, unzip ~300G
    ```
    cat scenes_split* > scenes.zip
    unzip scenes.zip
    unzip scenes_redo.zip
    ```
- ET-Cap annotations: ~558MB, including:
    * only navigation data: navigation_v1_{split}_{subset}.json

        format:
        ```
        list of {
            'scene_id': string, 
            'naivgation_data': list of {
                'pathid': string, 
                'render_dir': string, 
                'images': list of rendered image names, 'actions': list, 
                'path': list of position ids (int),
                'positions': list of grid-level position coordinates, 
                # after skip some points in the path
                'shorter_images': list of rendered image names,
                'shorter_actions': list,
                'shorter_path': list of grid-level position coordinates,
                'shorter_positions': list of grid-level position coordinates,
            }
        }
        ``` 
        
    * both navigation and caption data: navicaption_v1_{split}_{subset}.json

        format:
        ```
        list of {
            'scene_id': string, 
            'naivgation_data': list of {
                # compared 'naivgation_data' above, add  'final_view_captions'
                'final_view_captions': list of captions of the trajectory
            },
            'scene_captions': list of captions of the scene
        }
        ``` 

revise the {DATASET_DIR} to you own dataset location in following files:
- ./ET-Cap/CaBOT/mmdetection/mmdet/datasets/embodiedcap.py
- ./ET-Cap/CaBOT/mmdetection/mmdet/models/navigators/kubric_render.py
- ./ET-Cap/CaBOT/mmdetection/mmdet/models/navigators/single_navigator.py
- ./ET-Cap/dataset_construct/path_action_construct.py
- ./ET-Cap/dataset_construct/scene_construct.py

revise the {data_root} to you own dataset location in config files in 
- ./ET-Cap/CaBOT/mmdetection/configs/single_captioner
- ./ET-Cap/CaBOT/mmdetection/configs/single_navigator

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
The CaBOT code is organized based on [mmdetection](https://github.com/open-mmlab/mmdetection). The checkpoint of CaBOT (including the Navigator and the Captioner, ~7G) can be downloaded from:
- [Baidu Cloud Driver](https://pan.baidu.com/s/1ejtF5SheOQ4APquXkGoM2g) (pwd:i5xi) 
- [OneDriver](https://1drv.ms/f/s!AocXJ7uKxt6Xg0nwDfYfxax16iOh?e=GtKJC0)

Put models under ./ET-Cap/CaBOT/mmdetection/tools/work_dirs

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

### Captioning Evaluate (with oracle trajectories) on validation/test set (refer to captioner_dist_test.sh)
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

### Captioning Evaluate (with predicted trajectories) on validation/test set (refer to captioner_navinfer_dist_test.sh)
- After test the Navigator on the val/test set, run the following script to transfer navigation results to the input format of the Captioner
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
The raw Spice in pycocoevalcap is not suitable for paragraph evaluation. To calculate Spice for this task, we build a ParagraphSpice project (~2.25G). Download it from:
- [Baidu Cloud Driver](https://pan.baidu.com/s/1R5PUNGdg5IA6GgnpKOI5KA) (pwd：hted)
- [OneDriver](https://1drv.ms/f/s!AocXJ7uKxt6XgQXSacV-Ha6s860A?e=SzxKDJ) 
```
# revise caption_model, pred_file and gt_file, then
python embodied_caption_spice_eval.py
```


## Citation
if you find this code useful for your research, please consider citing:
```
@inproceedings{hu2023explore,
  title={Explore and Tell: Embodied Visual Captioning in 3D Environments},
  author={Hu, Anwen and Chen, Shizhe and Zhang, Liang and Jin, Qin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2482--2491},
  year={2023}
}
```



