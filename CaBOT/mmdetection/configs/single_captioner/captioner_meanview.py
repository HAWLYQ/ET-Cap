_base_ = [
    '../_base_/default_runtime.py'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
max_seq_len = 13
max_cap_len = 77 # word num: avg: 50.2;  <77 95% 
use_shorter_path = True

model = dict(
    type='SingeTrajectoryCaptioner',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck = dict(
        type='NaviImageTransformerNeck',
        in_channels = 2048,
        output_channels = 256,
        encoder=dict(
            type='NaviTransformerEncoder',
            num_layers=2, # 2, 6
            transformerlayers=dict(
                type='NaviBaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='NaviMultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1)
                ],
                feedforward_channels=2048,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        encode_positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True)),
    cap_head=dict(
        type='SingleCaptionerHead',
        encoder_width=256,
        min_dec_len=5,
        max_dec_len=max_cap_len,
        dec_beam_size=3,
        repetition_penalty=1.0,
        med_config='/root/code/ET-Cap/CaBOT/mmdetection/configs/single_captioner/med_config.json',
        ),
    cap_input_feat_type='MeanViewLocal',
    )

train_pipeline = [
    dict(type='LoadNavigationImages'),
    dict(type='LoadCapAnnotation'),
    dict(type='NormalizeImages', **img_norm_cfg),
    dict(type='ImageSeqPad', max_seq_len=max_seq_len), # pad or truncate img seq and gt labels
    dict(type='NavigationDefaultFormatBundle'),
    dict(type='CaptionDefaultFormatBundle'),
    dict(type='NaviCollect',
        meta_keys=('scene_id',),
        keys=['imgs', 'img_seq_mask', 'img', 'text_ids', 'text_mask']),
]
test_pipeline = [
    dict(type='LoadNavigationImages'),
    dict(type='NormalizeImages', **img_norm_cfg),
    dict(type='ImageSeqPad', max_seq_len=max_seq_len),
    dict(type='NavigationDefaultFormatBundle'),
    dict(type='CaptionDefaultFormatBundle'),
    dict(type='NaviCollect',
        meta_keys=('scene_id', 'pathid'),
        keys=['imgs', 'img_seq_mask','img']),
]

# eval_metrics = ['BLEU','METEOR','ROUGE_L','CIDEr','SPICE']
val_metrics = ['BLEU','CIDEr']
test_metrics = ['BLEU','METEOR','ROUGE_L','CIDEr']
# dataset settings
dataset_type = 'EmbodiedCapTrajCapDataset'
data_root = '/data5/haw/ETCAP/'
data = dict(
    samples_per_gpu=16, # 8 x 4 GPU(11G); 16 x 2 GPU (19G)
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file='anno/navicaption_v1_train.json',
        data_root=data_root,
        use_shorter_path = use_shorter_path,
        pred_result_save_dir = None,
        max_cap_len = max_cap_len,
        pipeline=train_pipeline,
        eval_metrics=val_metrics),
    val=dict(
        type=dataset_type,
        ann_file='anno/navicaption_v1_val.json',
        data_root=data_root,
        use_shorter_path = use_shorter_path,
        pred_result_save_dir = None,
        max_cap_len = max_cap_len,
        pipeline=test_pipeline,
        eval_metrics=val_metrics),
    test=dict(
        type=dataset_type,
        ann_file='anno/navicaption_v1_test_common.json',
        data_root=data_root,
        use_shorter_path = use_shorter_path,
        pred_result_save_dir = None,
        max_cap_len = max_cap_len,
        pipeline=test_pipeline,
        eval_metrics=test_metrics),
    test_dataloader = dict(
        samples_per_gpu=16
    ),
    val_dataloader = dict(
        samples_per_gpu=64
    )
    )

# optimizer
# set paramwise_cfg cause that the lr for decoder embedding and decoder cls is not consistent 
optimizer = dict(
    type='AdamW',
    lr=3e-5, # 1e-5
    weight_decay=0.05, #0.05
    # eps=1e-8,
    # betas=(0.9, 0.999)
)
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))

# learning policy
lr_config = dict(
    policy='LinearAnnealing', # refer to mmcv LinearAnnealingLrUpdaterHook
    by_epoch=True,
    min_lr = 0, # min lr
    warmup='linear',
    warmup_by_epoch=False, # warmup_by_epoch=True means the number of warmup_iters is epoch count
    warmup_ratio=0.1,  # set warmup inital lr=lr*warmup_ratio
    warmup_iters=10)

runner = dict(type='EpochBasedRunner', max_epochs=20) # 10

checkpoint_config = dict(interval=1, save_optimizer=False)

log_config = dict(
    interval=200, 
    hooks=[
        
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])