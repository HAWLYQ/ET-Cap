_base_ = [
    'captioner_meanview.py'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

model = dict(
    cap_head=dict(
        med_config='/root/code/ET-Cap/CaBOT/mmdetection/configs/single_captioner/med_config.json',
        ),
    )

max_seq_len = 13 # 13

test_pipeline = [
    dict(type='LoadNavigationImages'),
    dict(type='NormalizeImages', **img_norm_cfg),
    dict(type='ImageSeqPad', max_seq_len=max_seq_len),
    dict(type='NavigationDefaultFormatBundle'), # format imgs and img_seq_mask
    dict(type='CaptionDefaultFormatBundle'), # format img
    dict(type='NaviCollect',
        meta_keys=('scene_id', 'pathid'),
        keys=['imgs', 'img_seq_mask','img']),
]

# eval_metrics = ['BLEU','METEOR','ROUGE_L','CIDEr','SPICE']
test_metrics = ['BLEU','METEOR','ROUGE_L','CIDEr']
# dataset settings
dataset_type = 'EmbodiedCapTrajCapInferenceDataset'

navi_model_dir = 'navigator_waction_region2layer_time1layer_de2layer_lr1e4_epoch10'
data_root='/root/code/ET-Cap/CaBOT/mmdetection/tools/work_dirs/'+navi_model_dir

data = dict(
    samples_per_gpu=16, 
    workers_per_gpu=4,
    val=dict(
        type=dataset_type,
        ann_file='navigation_v1_trajcapinfer_val_common.json',
        data_root=data_root,
        pred_result_save_dir = None,
        pipeline=test_pipeline,
        eval_metrics=test_metrics),
    test=dict(
        type=dataset_type,
        ann_file='navigation_v1_trajcapinfer_test_common.json', 
        data_root=data_root,
        pred_result_save_dir = None,
        pipeline=test_pipeline,
        eval_metrics=test_metrics),
    test_dataloader = dict(
        samples_per_gpu=32
    )
    )

