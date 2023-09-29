_base_ = [
    '../_base_/default_runtime.py'
]

CLASSES = {
            'fb_move': {0:'none', 1:'forward', 2:'backward'},
            'rl_move': {0:'none', 1:'right', 2:'left'},
            'ud_move': {0:'none', 1:'up', 2:'down'},
            'yaw':{0:'none', 1:'left', 2:'right'},
            'pitch':{0:'none', 1:'up', 2:'down'}
        }

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# max_seq_len: normal 26; shorter 13
max_seq_len = 13
use_shorter_path = True
use_camera_info = False
navi_head_in_channels=2048
label_keys=['gt_fbmove_labels', 'gt_fbmove_steps',
            'gt_rlmove_labels', 'gt_rlmove_steps',
            'gt_udmove_labels', 'gt_udmove_steps', 
            'gt_yaw_labels', 'gt_yaw_angles', 
            'gt_pitch_labels', 'gt_pitch_angles']

model = dict(
    type='SingleNavigator',
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
            num_layers = 2, # 2, 6
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
    navi_head=dict(
        type='SingleRelNavigatorHead',
        fb_move_classes=len(CLASSES['fb_move'].keys()),
        rl_move_classes=len(CLASSES['rl_move'].keys()),
        ud_move_classes=len(CLASSES['ud_move'].keys()),
        pitch_classes=len(CLASSES['pitch'].keys()),
        yaw_classes=len(CLASSES['yaw'].keys()),
        in_channels=navi_head_in_channels, 
        decoder=dict(
            type='NaviTransformerDecoder',
            return_intermediate=True,
            num_layers=2, # 6
            transformerlayers=dict(
                type='NaviTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention', # refer to torch.nn.MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1),
                feedforward_channels=2048,
                ffn_dropout=0.1,
                # no cross attention
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
        decode_positional_encoding=dict(
            type='SeqLearnedPositionalEncoding', num_feats=256),
        loss_cls=dict(
            type='SeqMaskedCrossEntropyLoss',
            loss_weight=1.0),
        loss_reg=dict(type='SeqMaskedL1Loss', loss_weight=2.0)
        ),
    classes=CLASSES,
    max_seq_len = max_seq_len, 
    inference_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Normalize', **img_norm_cfg)],
    test_cfg=dict(eval_mode='pseudo_test', # ['test', 'pseudo_test', 'pseudo_test_stepbystep']
                eval_render_type='render', # ['render', 'simulate_render'], only used when eval_mode is test
                render_save_dir='simu_inference'),  
    )

train_pipeline = [
    dict(type='LoadNavigationImages'),
    dict(type='LoadNavigationCameraInfos'),
    dict(type='LoadRelNavigationAnnotations'),
    dict(type='NormalizeImages', **img_norm_cfg),
    dict(type='ImageSeqPad', max_seq_len=max_seq_len, label_keys=label_keys), # pad or truncate img seq and gt labels
    dict(type='NavigationDefaultFormatBundle', label_keys=label_keys), # lable type from list to tensor
    dict(type='NaviCollect',
        meta_keys=('scene_id', 'positions'),
        keys=['imgs', 'camera_infos', 'img_seq_mask', 
            'gt_fbmove_labels','gt_fbmove_steps', 
            'gt_rlmove_labels','gt_rlmove_steps', 
            'gt_udmove_labels','gt_udmove_steps', 
            'gt_yaw_labels', 'gt_yaw_angles',
            'gt_pitch_labels', 'gt_pitch_angles']),
]
test_pipeline = [
    dict(type='LoadNavigationImages'),
    dict(type='LoadNavigationCameraInfos'),
    dict(type='NormalizeImages', **img_norm_cfg),
    dict(type='ImageSeqPad', max_seq_len=max_seq_len, label_keys=label_keys),
    dict(type='NavigationDefaultFormatBundle', label_keys=label_keys),
    dict(type='NaviCollect',
        meta_keys=('scene_id', 'positions', 'pathid', 'path_len', 'path', 'actions', 'asset_infos'),
        keys=['imgs', 'camera_infos', 'img_seq_mask']),
]


val_metrics = ['manhattan_distance']
test_metrics = ['manhattan_distance','euclidean_distance','clip_score','seg_score']
# dataset settings
dataset_type = 'EmbodiedCapNaviDataset'
data_root = '/data5/haw/ETCAP/'
data = dict(
    # bs=16, 6 region encoder cost 20G in 1 GPU
    # bs=32, 2 region encoder cost 36G in 1 GPU
    samples_per_gpu=32, # 8 x 4 GPU (11 GB); 16 x 2 GPU
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file='anno/navigation_v1_train.json',
        data_root=data_root,
        classes = CLASSES,
        use_shorter_path = use_shorter_path,
        pred_result_save_dir = None,
        pipeline=train_pipeline,
        eval_metrics=val_metrics),
    val=dict(
        type=dataset_type,
        ann_file='anno/navigation_v1_val.json',
        data_root=data_root,
        classes = CLASSES,
        use_shorter_path = use_shorter_path,
        pred_result_save_dir = None,
        pipeline=test_pipeline,
        eval_metrics=val_metrics),
    test=dict(
        type=dataset_type,
        ann_file='anno/navigation_v1_test_common.json',
        data_root=data_root,
        classes = CLASSES,
        use_shorter_path = use_shorter_path,
        pred_result_save_dir = None,
        pipeline=test_pipeline,
        eval_metrics=test_metrics))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0)
        },
        norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=True,
    step=[50],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10)


runner = dict(type='EpochBasedRunner', max_epochs=10) # 10

checkpoint_config = dict(interval=1, save_optimizer=False)

log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])