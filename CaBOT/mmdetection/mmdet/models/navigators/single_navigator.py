# Copyright (c) aim3lab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
import torch

from mmcv.utils import build_from_cfg
from mmcv.runner import get_dist_info
from mmcv.cnn import Linear, initialize
from mmcv.cnn.utils.weight_init import update_init_info
from mmcv.utils.logging import get_logger, logger_initialized, print_log
from mmcv.runner import BaseModule, auto_fp16

from mmdet.datasets.pipelines import Compose
from mmdet.core import multi_apply
from ..builder import NAVIGATORS, build_backbone, build_head, build_neck

from collections import OrderedDict, defaultdict
import torch.distributed as dist
from .utils import new_pos_and_lookat, vector_angle
from .kubric_render import NavigationKubricRenderer, NavigationKubricSimulateRenderer
import numpy as np
import os
import sys
# 
sys.path.append('/root/code/kubric/')
import kubric_haw as kb_haw

# revise to your own dataset directory
DATASET_DIR='/data5/haw/ETCAP/' 
import gc
import random


def backbone_init_from_pretrained(model, checkpoint_info):
    checkpoint = torch.load(checkpoint_info['checkpoint'], map_location='cpu') 
    # print(checkpoint.keys()) ['meta', 'state_dict', 'optimizer']
    all_state_dict = checkpoint['state_dict']
    # print('model parameters:', model.state_dict().keys())
    # print('pretrained parameters:', all_state_dict.keys())

    # retrain part-of parameters and rename
    state_dict = {}
    for k, v in all_state_dict.items():
        if 'backbone' in k:
            # remove prefx 'text_decoder' to keep consistent with model.state_dict()
            state_dict[k.replace('backbone.', '')] = v
            """"if k == 'backbone.layer1.0.bn1.weight':
                print('pretrained weight:', k, v)"""
    
    # remove same name but diff shape parameters
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                # print(key, state_dict[key].shape, model.state_dict()[key].shape)
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('initialize backbone from %s, missing %d parameters ' % (checkpoint_info['checkpoint'], len(msg.missing_keys)))  
    # print('missing parameters:', msg.missing_keys)
    return model

def neck_init_from_pretrained(model, checkpoint_info, name='neck'):
    if model.__class__.__name__ == 'NaviImageTransformerNeck':
        region_encoder_name = 'encoder.'
    if model.__class__.__name__ == 'NaviImageTransformerPreNeck':
        region_encoder_name = 'encoder.'
    elif model.__class__.__name__ == 'NaviImageRegionTimeTransformerNeck':
        region_encoder_name = 'region_encoder.'

    checkpoint = torch.load(checkpoint_info['checkpoint'], map_location='cpu') 
    all_state_dict = checkpoint['state_dict']
    # retrain part-of parameters and rename 
    state_dict = {}
    if checkpoint_info['type']=='DETR':
        for k, v in all_state_dict.items():
            if 'bbox_head.transformer.encoder.' in k:
                # remove prefx 'text_decoder' to keep consistent with model.state_dict()
                if  model.__class__.__name__ == 'NaviImageRegionTimeBlockTransformerNeck':
                    # e.g. bbox_head.transformer.encoder.layers.1.ffns.0.layers.0.0.weight
                    # change to block_region_layers.1.layers.0.ffns.0.layers.0.0.weight
                    k = k.replace('bbox_head.transformer.encoder.layers.', '')
                    layer_num = k.split('.')[0]
                    k = '.'.join(k.split('.')[1:])
                    new_k = 'block_region_layers.'+layer_num+'.layers.0.'+k
                    state_dict[new_k] = v
                else:
                    state_dict[k.replace('bbox_head.transformer.encoder.', region_encoder_name)] = v


            elif 'bbox_head.input_proj.' in k:
                state_dict[k.replace('bbox_head.input_proj.', 'input_proj.')] = v
            
            """if k == 'bbox_head.transformer.encoder.layers.0.ffns.0.layers.0.0.weight':
                print('pretrained weight:', k, v)"""

    # remove same name but diff shape parameters
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                # print(key, state_dict[key].shape, model.state_dict()[key].shape)
                del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    print('initialize %s from %s, missing %d parameters ' % (name, checkpoint_info['checkpoint'], len(msg.missing_keys)))  
    # print('missing parameters:', msg.missing_keys)
    return model


@NAVIGATORS.register_module()
class SingleNavigator(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 backbone,
                 neck=None,
                 navi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 classes=None,
                 frozen_backbone=False,
                 backbone_init=None,
                 frozen_neck=False,
                 neck_init=None,
                 inference_pipeline=None,
                 max_seq_len=None,
                 use_camera_action=False,
                 init_cfg=None):
        # __init__ only set self.init_cfg = init_cfg
        # weights are initialzed by call model.init_weights() or write by yourself
        super(SingleNavigator, self).__init__(init_cfg)

        self.backbone = build_backbone(backbone)
        self.backbone_init = backbone_init

        # anwen hu 2022/10/18: customized  pretrianed weights loading outside .init_weights()
        if self.backbone_init is not None:
            self.backbone = backbone_init_from_pretrained(self.backbone, self.backbone_init)
 
        if neck is not None:
            self.neck = build_neck(neck)
            self.neck_init = neck_init
            if self.neck_init is not None:
                self.neck = neck_init_from_pretrained(self.neck, self.neck_init)
        
        if frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        

        if neck is not None and frozen_neck:
            for m in self.neck.children():
                if m.__class__.__name__ == 'NaviTransformerEncoder':
                    print('no freeze ', m.__class__.__name__)
                    continue
                if m.__class__.__name__ == 'ModuleList' and m[0].__class__.__name__ == 'NaviTransformerEncoder':
                    print('no freeze ', m.__class__.__name__, 'of (' + m[0].__class__.__name__, ')')
                    continue
                for param in m.parameters():
                    param.requires_grad = False

        navi_head.update(train_cfg=train_cfg)
        navi_head.update(test_cfg=test_cfg)
        self.navi_head = build_head(navi_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # for processing new images during testing
        self.CLASSES = classes # label to action name
        # processing pipeline
        self.infer_pipeline = Compose(inference_pipeline)
        self.max_seq_len = max_seq_len
        self.use_camera_action = use_camera_action
        if self.use_camera_action:
            self.camerainfo_proj = Linear(10, 128)
            neck_output_dim = neck['output_channels'] # 256 default
            navi_head_in = navi_head['decoder']['transformerlayers']['attn_cfgs']['embed_dims']
            self.img_camerainfo_fusion = Linear(neck_output_dim+128, navi_head_in)
        
        self.eval_mode = self.test_cfg['eval_mode'] 
        assert self.eval_mode in ['test', 'pseudo_test', 'rule_test']
        self.eval_render_type = self.test_cfg['eval_render_type']
        assert self.eval_render_type in ['render', 'simulate_render']
        self.render_save_dir = self.test_cfg['render_save_dir']
        if self.eval_render_type == 'simulate_render':
            asset_dir = DATASET_DIR+'kubric_assets/'
            assert os.path.exists(asset_dir)
            self.shapenet = kb_haw.AssetSource.from_manifest(DATASET_DIR+'ShapeNetCore.v2.json', scratch_dir=asset_dir, random_name=False)
            self.gso = kb_haw.AssetSource.from_manifest(DATASET_DIR+"GSO.json",scratch_dir=asset_dir, random_name=False)


    def init_weights(self):
        """Initialize the weights."""

        is_top_level_module = False
        # check if it is top-level module
        if not hasattr(self, '_params_init_info'):
            # The `_params_init_info` is used to record the initialization
            # information of the parameters
            # the key should be the obj:`nn.Parameter` of model and the value
            # should be a dict containing
            # - init_info (str): The string that describes the initialization.
            # - tmp_mean_value (FloatTensor): The mean of the parameter,
            #       which indicates whether the parameter has been modified.
            # this attribute would be deleted after all parameters
            # is initialized.
            self._params_init_info: defaultdict = defaultdict(dict)
            is_top_level_module = True

            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param][
                    'init_info'] = f'The value is the same before and ' \
                                   f'after calling `init_weights` ' \
                                   f'of {self.__class__.__name__} '
                self._params_init_info[param][
                    'tmp_mean_value'] = param.data.mean()

            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        # Get the initialized logger, if not exist,
        # create a logger named `mmcv`
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else 'mmcv'

        # from ..cnn import initialize
        # from ..cnn.utils.weight_init import update_init_info
        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f'initialize {module_name} with init_cfg {self.init_cfg}',
                    logger=logger_name)
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    # prevent the parameters of
                    # the pre-trained model
                    # from being overwritten by
                    # the `init_weights`
                    if self.init_cfg['type'] == 'Pretrained':
                        return

            for m in self.children():
                # anwenhu 2022/10/17: revise here to avoid overwritten pretrained parameters
                """if hasattr(m, 'init_weights'):
                    m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m,
                        init_info=f'Initialized by '
                        f'user-defined `init_weights`'
                        f' in {m.__class__.__name__} ')"""
                if hasattr(m, 'init_weights'):
                    # anwenhu 2022/10/17: these two parts have been initialized before call init_weights()
                    if m.__class__.__name__ == 'ResNet' and self.backbone_init:
                        print_log(f'skip backbone {m.__class__.__name__}.init_weights()',logger=logger_name)
                        continue
                    elif m.__class__.__name__ == 'NaviImageTransformerNeck' and self.neck_init:
                        print_log(f'skip neck {m.__class__.__name__}.init_weights()',logger=logger_name)
                        continue
                    elif m.__class__.__name__ == 'NaviImageTransformerNeck' and self.preneck_init:
                        print_log(f'skip pre_neck {m.__class__.__name__}.init_weights()',logger=logger_name)
                        continue
                    elif m.__class__.__name__ == 'NaviImageRegionTimeTransformerNeck' and self.neck_init:
                        m.init_weights(skip_region_encoder=True)
                    elif m.__class__.__name__ == 'NaviImageRegionTimeBlockTransformerNeck' and self.neck_init:
                        m.init_weights(skip_region_encoder=True)
                    else:
                        m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m,
                        init_info=f'Initialized by '
                        f'user-defined `init_weights`'
                        f' in {m.__class__.__name__} ')

            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has '
                          f'been called more than once.')

        if is_top_level_module:
            self._dump_init_info(logger_name)

            for sub_module in self.modules():
                del sub_module._params_init_info

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None
    
    def extract_feat(self, imgs, camera_infos):
        """Directly extract features from the backbone+neck."""
        # print('SingleNavigator imgs:', imgs.size())
        batch_size, seq_len, C, H, W = imgs.size()
        imgs = imgs.reshape(batch_size*seq_len, C, H, W)
        # print('SingleNavigator input imgs:', imgs)
        x = self.backbone(imgs)[-1] # [batch_size*seq, 2048, 16, 16]
        # print('SingleNavigator after backbone x:', x)

        if self.with_neck:
            _, c, h, w = x.size()
            x = x.reshape(batch_size, seq_len, c, h, w) # [batch, img_seq, c, h, w]
            x = self.neck(x) # [batch, img_seq, dim]
            # print('SingleNavigator after neck x:', x.size())
        if  self.use_camera_action:
            # print('SingleNavigator camera_infos:', camera_infos.size())
            camera_info_emb = self.camerainfo_proj(camera_infos.reshape(batch_size*seq_len, -1))
            # print('SingleNavigator camera_info_emb:', camera_info_emb.size())
            x = torch.cat([x.reshape(batch_size*seq_len, -1), camera_info_emb], dim=-1) # [batch*seq, dim+128]
            x = self.img_camerainfo_fusion(x) # [batch*seq, dim]
            x = x.reshape(batch_size, seq_len, -1)
        return x # [batch, img_seq, dim]

    def forward_train(self,
                      imgs,
                      camera_infos, 
                      # img_metas,
                      img_seq_mask, 
                      gt_fbmove_labels,
                      gt_fbmove_steps,
                      gt_rlmove_labels,
                      gt_rlmove_steps,
                      gt_udmove_labels,
                      gt_udmove_steps,
                      gt_yaw_labels,
                      gt_yaw_angles,
                      gt_pitch_labels,
                      gt_pitch_angles):
        """
        Args:
            imgs (Tensor): Input images of shape (B, Seq, C, H, W).
                Typically these should be mean centered and std scaled.
            
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(imgs, camera_infos) # [batch, seq, dim]
        # print('SingleNavigator x.size:', x.size())
        # print('SingleNavigator x:', x)
        # print('SingleNavigator img_seq_mask:', img_seq_mask)
        losses = self.navi_head.forward_train(x, img_seq_mask, 
                                            gt_fbmove_labels, gt_fbmove_steps,
                                            gt_rlmove_labels, gt_rlmove_steps,
                                            gt_udmove_labels, gt_udmove_steps,
                                            gt_yaw_labels, gt_yaw_angles, 
                                            gt_pitch_labels, gt_pitch_angles)
        return losses
    
    def forward_pseudo_test(self, imgs, path_info, camera_infos, img_seq_mask, **kwargs):
        x = self.extract_feat(imgs, camera_infos) # [batch, seq, dim]
        bs, seq, _ = x.size()
        fbmove_cls, rlmove_cls, udmove_cls, yaw_cls, pitch_cls, fbmove_steps, rlmove_steps, udmove_steps, yaw_angles, pitch_angles = self.navi_head.forward_sudo_test(x, img_seq_mask)
        # move_cls/yaw_cls/pitch_cl : [ndec, bs, seq, cls_channels]
        # move_steps/yaw_angles/pitch_angles: [ndec, bs, seq]
        fbmove_cls = fbmove_cls[-1] # last layer preds of  all step; [bs, seq, cls_channels]
        rlmove_cls = rlmove_cls[-1] # last layer preds of  all step; [bs, seq, cls_channels]
        udmove_cls = udmove_cls[-1] # last layer preds of  all step; [bs, seq, cls_channels]
        yaw_cls = yaw_cls[-1] # last layer preds of  all step; [bs, seq, cls_channels]
        pitch_cls = pitch_cls[-1] # last layer preds of  all step; [bs, seq, cls_channels]

        fbmove_steps = fbmove_steps[-1] # last layer preds of all step; [bs, seq]
        rlmove_steps = rlmove_steps[-1] # last layer preds of all step; [bs, seq]
        udmove_steps = udmove_steps[-1] # last layer preds of all step; [bs, seq]
        yaw_angles = yaw_angles[-1] # last layer preds of all step; [bs, seq]
        pitch_angles = pitch_angles[-1] # last layer preds of all step; [bs, seq]
        # print('SingleNavigator img_seq_mask:', img_seq_mask.size())
        # print('SingleNavigator img_seq_mask:', img_seq_mask)
        # print('SingleNavigator move_cls:', move_cls.size())
        results = []
        for i in range(bs):
            single_result = {'scene_id':path_info[i]['scene_id'], 'pathid':path_info[i]['pathid'],
                            'pred_actions': []}
            # single_img_seq_mask = img_seq_mask[i]
            # seq_len = torch.nonzero(single_img_seq_mask).size(0)
            path_len = path_info[i]['path_len']
            for j in range(path_len):
                fbmove_action = self.CLASSES['fb_move'][torch.argmax(fbmove_cls[i][j]).item()]
                rlmove_action = self.CLASSES['rl_move'][torch.argmax(rlmove_cls[i][j]).item()]
                udmove_action = self.CLASSES['ud_move'][torch.argmax(udmove_cls[i][j]).item()]
                yaw_action = self.CLASSES['yaw'][torch.argmax(yaw_cls[i][j]).item()]
                pitch_action = self.CLASSES['pitch'][torch.argmax(pitch_cls[i][j]).item()]

                fbmove_step = fbmove_steps[i][j].item() * 4 # last layer and last step preds
                rlmove_step = rlmove_steps[i][j].item() * 4 # last layer and last step preds
                udmove_step = udmove_steps[i][j].item() * 4 # last layer and last step preds
                yaw_angle = yaw_angles[i][j].item() * 360 # last layer and last step preds
                pitch_angle = pitch_angles[i][j].item() * 180 # last layer and last step preds
                single_result['pred_actions'].append({'move':
                                                        {
                                                            'fb':[fbmove_action, int(fbmove_step)],
                                                            'rl':[rlmove_action, int(rlmove_step)],
                                                            'ud':[udmove_action, int(udmove_step)],
                                                        },
                                                      'yaw':[yaw_action, yaw_angle],
                                                      'pitch':[pitch_action, pitch_angle]
                                                    })
            results.append(single_result)
        return results
    
    def forward_test(self, imgs, path_info, camera_infos, **kwargs):
        bs, seq, c, h, w = imgs.size()
        start_imgs = imgs[:, 0, :,:,:] # [bs, c, h, w]
        start_img_list = [start_imgs[i] for i in range(bs)]
        path_info_list = [path_info[i] for i in range(bs)]
        start_camera_infos = camera_infos[:,0,:]
        camera_info_list = [start_camera_infos[i] for i in range(bs)]
        # return multi_apply(self.forward_single_test, start_img_list, path_info_list)
        results = []
        for i in range(bs):
            results.append(self.forward_single_test(start_img_list[i], path_info_list[i], camera_info_list[i]))
        # print('single_navigator.py results:', results)
        return results
    
    def forward_rule_test(self, imgs, path_info, camera_infos, **kwargs):
        bs, seq, c, h, w = imgs.size()
        start_imgs = imgs[:, 0, :,:,:] # [bs, c, h, w]
        start_img_list = [start_imgs[i] for i in range(bs)]
        path_info_list = [path_info[i] for i in range(bs)]
        start_camera_infos = camera_infos[:,0,:]
        camera_info_list = [start_camera_infos[i] for i in range(bs)]
        # return multi_apply(self.forward_single_test, start_img_list, path_info_list)
        results = []
        for i in range(bs):
            results.append(self.forward_rule_single_test(start_img_list[i], path_info_list[i], camera_info_list[i]))
        # print('single_navigator.py results:', results)
        return results

    def forward_single_test(self, start_img, path_info, start_camera_info):
        # for one start image
        """
        start_img: [c, h, w]
        """
        rank, world_size = get_dist_info()
        start_position = path_info['positions'][0]
        start_position_id = path_info['path'][0]
        if self.eval_render_type == 'render':
            nav_kbrender = NavigationKubricRenderer(scene_id=path_info['scene_id'], 
                                                    cache_dir='tmp/rank'+str(rank)+'_navigation_inference/', 
                                                    render_frame=41, 
                                                    save_dir=self.render_save_dir+'/scene'+str(path_info['scene_id'])+'_start'+str(start_position_id)+'/',
                                                    keep_rendered_imgs=True)
        elif self.eval_render_type == 'simulate_render':
            # NavigationKubricSimulateRenderer will re-simulate the scene and save the segmentation of the last step
            nav_kbrender = NavigationKubricSimulateRenderer(scene_id=path_info['scene_id'], 
                                                    cache_dir='tmp/rank'+str(rank)+'_navigation_inference/', 
                                                    asset_infos=path_info['asset_infos'],
                                                    simulator_frames=40, 
                                                    save_dir=self.render_save_dir+'/scene'+str(path_info['scene_id'])+'_start'+str(start_position_id)+'/',
                                                    keep_rendered_imgs_and_segs=True,
                                                    shapenet=self.shapenet,
                                                    gso=self.gso
                                                    )
        imgs = start_img.unsqueeze(0).unsqueeze(1) # [bs(1), seq(1), dim]
        camera_infos = start_camera_info.unsqueeze(0).unsqueeze(1) # [bs(1), seq(1), 10]
        old_position = start_position
        old_lookat = [0,0,0]
        old_view = np.array(old_lookat) - np.array(old_position)
        old_view_vec_xy = np.array(list(old_view[0:2])+[0])
        old_view_x_angle = vector_angle(old_view_vec_xy, [1,0,0])
        if old_view_x_angle == 'nan':
            old_view_x_angle = 0
        # anwen hu 2023/3/5: fix old_view_x_angle bug
        if old_view[1] < 0: # view angle with (1,0,0), always counterclockwise direction
            old_view_x_angle = 360-old_view_x_angle
        single_result = {'scene_id':path_info['scene_id'], 'start_pos_id':start_position_id, 'preds': [{'step':0, 'position':old_position, 'look_at':old_lookat}]}
        while True:
            if self.use_camera_action:
                x = self.extract_feat(imgs, camera_infos) # [bs(1), seq, dim]
            else:
                x = self.extract_feat(imgs, None) # [bs(1), seq, dim]
            bs, seq, _ = x.size()
            # print('SingleNavigator x.size:', x.size())
            seq_mask = torch.ones([1, seq], device=x.device)
            fbmove_cls, rlmove_cls, udmove_cls, yaw_cls, pitch_cls, fbmove_steps, rlmove_steps, udmove_steps, yaw_angles, pitch_angles = self.navi_head.forward_single(x, seq_mask)
            # move_cls/yaw_cls/pitch_cl : [ndec, bs, seq, cls_channels]
            # yaw_angles/pitch_angles: [ndec, bs, seq]
            fbmove_cls = fbmove_cls[-1].squeeze(0)[-1] # last layer and last step preds
            rlmove_cls = rlmove_cls[-1].squeeze(0)[-1] # last layer and last step preds
            udmove_cls = udmove_cls[-1].squeeze(0)[-1] # last layer and last step preds
            yaw_cls = yaw_cls[-1].squeeze(0)[-1] # last layer and last step preds
            pitch_cls = pitch_cls[-1].squeeze(0)[-1] # last layer and last step preds

            fbmove_step = fbmove_steps[-1].squeeze(0)[-1].item() * 4 # last layer and last step preds
            rlmove_step = rlmove_steps[-1].squeeze(0)[-1].item() * 4 # last layer and last step preds
            udmove_step = udmove_steps[-1].squeeze(0)[-1].item() * 4 # last layer and last step preds
            yaw_angle = yaw_angles[-1].squeeze(0)[-1].item() * 360 # last layer and last step preds
            pitch_angle = pitch_angles[-1].squeeze(0)[-1].item() * 180 # last layer and last step preds

            fbmove_action = self.CLASSES['fb_move'][torch.argmax(fbmove_cls).item()]
            rlmove_action = self.CLASSES['rl_move'][torch.argmax(rlmove_cls).item()]
            udmove_action = self.CLASSES['ud_move'][torch.argmax(udmove_cls).item()]
            yaw_action = self.CLASSES['yaw'][torch.argmax(yaw_cls).item()]
            pitch_action = self.CLASSES['pitch'][torch.argmax(pitch_cls).item()]

            if fbmove_action == 'none' and rlmove_action == 'none' and udmove_action == 'none' and yaw_action == 'none' and pitch_action == 'none':
                # print('move action:', move_action, 'yaw action:', yaw_action, 'pitch action:', pitch_action)
                break
            

            new_position, new_lookat, new_view_x_angle, new_view  = new_pos_and_lookat(np.array(old_position), np.array(old_lookat), old_view_x_angle, 
                                                        [fbmove_action, rlmove_action, udmove_action], [fbmove_step, rlmove_step, udmove_step], 
                                                        yaw_action, yaw_angle,
                                                        pitch_action, pitch_angle)
            # assert np.sum(np.absolute(np.array(new_lookat)-np.array(new_position))) > 0
            if self.use_camera_action:
                new_camera_info = np.array([torch.argmax(fbmove_cls).item(), fbmove_step/4,
                                            torch.argmax(rlmove_cls).item(), rlmove_step/4,
                                            torch.argmax(udmove_cls).item(), udmove_step/4,
                                            torch.argmax(yaw_cls).item(), yaw_angle/360,
                                            torch.argmax(pitch_cls).item(), pitch_angle/360,
                                            ]).astype(np.float32)
                

            image_name = 'step'+str(seq)+'_move'+'-fb-'+fbmove_action+str(int(fbmove_step))+ \
                    '-rl-'+rlmove_action+str(int(rlmove_step))+ '-ud-'+udmove_action+str(int(udmove_step))+ \
                    '_yaw-'+yaw_action+str(round(yaw_angle, 2))+'_pitch-'+pitch_action+str(round(pitch_angle, 2))
            
            new_image_path = nav_kbrender.render(new_position.tolist(), new_lookat.tolist(), image_name)

            single_result['preds'].append({'step':seq, 'position':new_position.tolist(), 
                                            'look_at':new_lookat.tolist(), 'img_path':new_image_path})
            
            print('NavigationKubricRenderer: rank:%d, image:%s' % (rank, new_image_path))
            
            # debug
            # if seq+1 == 10 and rank == 0:
            #     break
            if seq+1 == self.max_seq_len:
                break
            new_results = self.infer_pipeline({'img_info':{'filename':new_image_path}, 'img_prefix':''})
            img = torch.tensor(new_results['img'].transpose(2,0,1), device=start_img.device) # C * H * W
            img = img.unsqueeze(0).unsqueeze(1) # bs (1) * seq(1) * c * h * w
            imgs = torch.cat([imgs, img], dim=1) # bs (1) * seq+1 * c * h * w

            if self.use_camera_action:
                camera_infos = torch.cat([camera_infos, torch.tensor(new_camera_info, device=start_img.device).unsqueeze(0).unsqueeze(1)], dim=1) # bs (1) * seq+1 * 6
            old_position = new_position
            old_lookat = new_lookat
            # assert np.sum(np.absolute(np.array(old_lookat)-np.array(old_position))) > 0
            old_view_x_angle = new_view_x_angle
        
        nav_kbrender.close()
        # gc.collect()
        # print('SingleNavigator rank %d pred seq len:%d' % (rank, len(single_result['preds'])))
        return single_result
    

    def forward_rule_single_test(self, start_img, path_info, start_camera_info):
        # for one start image
        """
        start_img: [c, h, w]
        """
        def obj_area(seg):
            area=0
            h, w = seg.shape
            values = np.resize(seg, [h*w]).tolist()
            for value in values:
                if value != 0:
                    area+=1
            return area
        
        def obj_num(seg):
            h, w = seg.shape
            values = np.resize(seg, [h*w]).tolist()
            num = len(set(values))-1 # remove background
            return num
        
        # directions here is the direction in camera coordinate system 
        def determine_move_dierctions(seg):
            """
            seg: 256*256
            """
            updown_direction = ''
            height, width = seg.shape
            # divide seg to up and down, calculate the object area gap
            up_seg = seg[:int(height/2)]
            down_seg = seg[int(height/2):]
            up_area = obj_area(up_seg)
            down_area = obj_area(down_seg)
            if up_area > down_area:
                updown_direction="up"
            elif up_area < down_area:
                updown_direction="down"
            else:
                updown_direction=random.choice(["up", "down"])

            forward_direction=''
            leftright_direction=''

            # divide seg to left, middle and right, calculate the object area gap
            left_seg = seg[:][:int(width/3)]
            mid_seg = seg[:][int(width/3):int(2*width/3)]
            right_seg = seg[:][int(2*width/3):]
            left_area = obj_area(left_seg)
            mid_seg = obj_area(mid_seg)
            right_area = obj_area(right_seg)
            
            # forward_direction has a higher priority than leftright_direction
            if mid_seg >= left_area and mid_seg >= right_area:
                forward_direction = 'forward'

            # decide a left/right direction for candidate when forward action results in fewer visible objects
            if left_area > right_area:
                leftright_direction = 'left'
            elif left_area < right_area:
                leftright_direction = 'right'
            else:
                leftright_direction = random.choice(["left", "right"])

            return updown_direction, forward_direction, leftright_direction


        # transfer directions in "camera coordinate system" to ones in "world coordinate system"
        def transfer_move_directions(old_view, move_direction):
            new_directions = []
            old_view_x, old_view_y = old_view[:2]
            # find two directions of vertical vectors for the current view
            left_vertical_view = np.array([-old_view_y, old_view_x])
            right_vertical_view = np.array([old_view_y, -old_view_x])

            if move_direction == 'forward':
                if old_view[0] < 0: # means x should be smaller
                    new_directions.append('forward')
                elif old_view[0] > 0: # means x should be bigger
                    new_directions.append('backward')

                if old_view[1] < 0: # means y should be smaller
                    new_directions.append('left')
                elif old_view[1] > 0: # means y should be bigger
                    new_directions.append('right')

            
            elif move_direction == 'left':
                # determine move directions in "world coordinate system" accorind  vertical vector
                if left_vertical_view[0] < 0: # means x should be smaller
                    new_directions.append('forward')
                elif left_vertical_view[0] > 0: # means x should be bigger
                    new_directions.append('backward')
                
                if left_vertical_view[1] < 0: # means y should be smaller
                    new_directions.append('left')
                elif left_vertical_view[1] > 0: # means y should be bigger
                        new_directions.append('right')

            elif move_direction == 'right':
                if right_vertical_view[0] < 0: # means x should be smaller
                    new_directions.append('forward')
                elif right_vertical_view[0] > 0: # means x should be bigger
                    new_directions.append('backward')
                
                if right_vertical_view[1] < 0: # means y should be smaller
                    new_directions.append('left')
                elif right_vertical_view[1] > 0: # means y should be bigger
                    new_directions.append('right')

            if len(new_directions) == 0:
                new_directions.append(random.choice(["forward", "backward", 'left', 'right']))
            # new_directions contain at most 2 directions 
            return new_directions

        def opposite_move_directions(directions):
            new_directions = []
            for direction in directions:
                assert direction in ['forward', 'backward', 'left', 'right']
                if direction == 'forward':
                    new_directions.append('backward')
                elif direction == 'backward':
                    new_directions.append('forward')
                elif direction == 'left':
                    new_directions.append('right')
                elif direction == 'right':
                    new_directions.append('left')
            return new_directions

        rank, world_size = get_dist_info()
        start_position = path_info['positions'][0]
        start_position_id = path_info['path'][0]
        if self.eval_render_type == 'render':
            nav_kbrender = NavigationKubricRenderer(scene_id=path_info['scene_id'], 
                                                    cache_dir='tmp/rank'+str(rank)+'_navigation_inference/', 
                                                    render_frame=41, 
                                                    save_dir=self.render_save_dir+'/scene'+str(path_info['scene_id'])+'_start'+str(start_position_id)+'/',
                                                    keep_rendered_imgs=True)
        elif self.eval_render_type == 'simulate_render':
            # NavigationKubricSimulateRenderer will re-simulate the scene and save the segmentation of the last step
            nav_kbrender = NavigationKubricSimulateRenderer(scene_id=path_info['scene_id'], 
                                                    cache_dir='tmp_debug/rank'+str(rank)+'_navigation_inference/', 
                                                    asset_infos=path_info['asset_infos'],
                                                    simulator_frames=40, 
                                                    save_dir=self.render_save_dir+'/scene'+str(path_info['scene_id'])+'_start'+str(start_position_id)+'/',
                                                    keep_rendered_imgs_and_segs=True,
                                                    shapenet=self.shapenet,
                                                    gso=self.gso
                                                    )
        imgs = start_img.unsqueeze(0).unsqueeze(1) # [bs(1), seq(1), dim]
        camera_infos = start_camera_info.unsqueeze(0).unsqueeze(1) # [bs(1), seq(1), 6]
        old_position = start_position
        old_lookat = [0,0,0]
        old_view = np.array(old_lookat) - np.array(old_position)
        old_view_vec_xy = np.array(list(old_view[0:2])+[0])
        old_view_x_angle = vector_angle(old_view_vec_xy, [1,0,0])
        if old_view_x_angle == 'nan':
            old_view_x_angle = 0
        if old_view[1] < 0: # view angle with (1,0,0), always counterclockwise direction
            old_view_x_angle = 360-old_view_x_angle

        seq = 0
        image_name = 'step'+str(seq)
        new_image_path = nav_kbrender.render(old_position, old_lookat, image_name)
        single_result = {'scene_id':path_info['scene_id'], 'start_pos_id':start_position_id, 'rule':'True', 
                        'preds': [{'step':seq, 'position':old_position, 'look_at':old_lookat, 'img_path':new_image_path}]}
        seq+=1

        while True:
            seg_path = new_image_path.replace('png', 'npz')
            seg_data = np.load(seg_path, allow_pickle=True)
            seg = seg_data['segmentation'] # 256*256
            visible_obj_num = obj_num(seg)
            camera_updown_dire, camera_forward_dire, camera_leftright_dire = determine_move_dierctions(seg)


            move_forward_done=False
            # if forward, try move forward first
            if camera_forward_dire != '':
                worldsystem_move_directions = transfer_move_directions(old_view, camera_forward_dire)
                for direction in worldsystem_move_directions:
                    move_action = direction
                    yaw_action = 'none'
                    pitch_action='none'
                    move_step= 1
                    yaw_angle = 0
                    pitch_angle = 0
                    new_position, new_lookat, new_view_x_angle, new_view  = new_pos_and_lookat(np.array(old_position), np.array(old_lookat), old_view_x_angle, 
                                                                move_action, move_step, yaw_action, yaw_angle, pitch_action, pitch_angle)
                    old_position = new_position
                    old_lookat = new_lookat
                    old_view = new_view
                    old_view_x_angle = new_view_x_angle

                joint_move_action='-'.join(worldsystem_move_directions)
                joint_move_step=len(worldsystem_move_directions)
                image_name = 'step'+str(seq)+'_move-'+joint_move_action+str(joint_move_step)+ \
                        '_yaw-'+yaw_action+str(round(yaw_angle, 2))+'_pitch-'+pitch_action+str(round(pitch_angle, 2))

                new_image_path = nav_kbrender.render(new_position.tolist(), new_lookat.tolist(), image_name)

                single_result['preds'].append({'step':seq, 'position':new_position.tolist(), 
                                                'look_at':new_lookat.tolist(), 'img_path':new_image_path})
                
                if seq+1 == self.max_seq_len:
                    break
                else:
                    seq+=1
                
                new_seg_path = new_image_path.replace('png', 'npz')
                new_seg_data = np.load(new_seg_path, allow_pickle=True)
                new_seg = new_seg_data['segmentation'] # 256*256
                new_visible_obj_num = obj_num(new_seg)

                # if move forward results in less visible objects, return to the original location 
                if new_visible_obj_num < visible_obj_num:
                    worldsystem_move_directions = opposite_move_directions(worldsystem_move_directions)
                    for direction in worldsystem_move_directions:
                        move_action = direction
                        yaw_action = 'none'
                        pitch_action='none'
                        move_step= 1
                        yaw_angle = 0
                        pitch_angle = 0
                        new_position, new_lookat, new_view_x_angle, new_view  = new_pos_and_lookat(np.array(old_position), np.array(old_lookat), old_view_x_angle, 
                                                                    move_action, move_step, yaw_action, yaw_angle, pitch_action, pitch_angle)
                        old_position = new_position
                        old_lookat = new_lookat
                        old_view = new_view
                        old_view_x_angle = new_view_x_angle
                else:
                    move_forward_done=True

            worldsystem_move_directions = [camera_updown_dire]
            if not move_forward_done: # if not forward or forward is rollbacked, excute left/rght actions 
                worldsystem_move_directions += transfer_move_directions(old_view, camera_leftright_dire)

            # print('camera-system directions:', camerasystem_move_directions)
            
            # print('word-system directions:', worldsystem_move_directions)

            # arrive a new position and doesn't render, so doesn't add seq
            for direction in worldsystem_move_directions:
                move_action = direction
                yaw_action = 'none'
                pitch_action='none'
                move_step= 1
                yaw_angle = 0
                pitch_angle = 0
                new_position, new_lookat, new_view_x_angle, new_view  = new_pos_and_lookat(np.array(old_position), np.array(old_lookat), old_view_x_angle, 
                                                            move_action, move_step, yaw_action, yaw_angle, pitch_action, pitch_angle)
                old_position = new_position
                old_lookat = new_lookat
                old_view = new_view
                old_view_x_angle = new_view_x_angle
        

            joint_move_action='-'.join(worldsystem_move_directions)
            joint_move_step=len(worldsystem_move_directions)

            # turn around the camera (only yaw) and render, each render, seq+=1, store seg area of each orientation
            seg_areas = []
            end_nav = False
            for i in range(4):
                move_action = 'none'
                yaw_action = 'left'
                pitch_action='none'
                move_step= 0
                yaw_angle = 90.0
                pitch_angle = 0
                new_position, new_lookat, new_view_x_angle, new_view  = new_pos_and_lookat(np.array(old_position), np.array(old_lookat), old_view_x_angle, 
                                                            move_action, move_step, yaw_action, yaw_angle, pitch_action, pitch_angle)
                old_position = new_position
                old_lookat = new_lookat
                old_view = new_view
                old_view_x_angle = new_view_x_angle
                if i == 0:
                    image_name = 'step'+str(seq)+'_move-'+joint_move_action+str(joint_move_step)+ \
                            '_yaw-'+yaw_action+str(round(yaw_angle, 2))+'_pitch-'+pitch_action+str(round(pitch_angle, 2))
                else:
                    image_name = 'step'+str(seq)+'_move-'+move_action+str(move_step)+ \
                            '_yaw-'+yaw_action+str(round(yaw_angle, 2))+'_pitch-'+pitch_action+str(round(pitch_angle, 2))
                
                new_image_path = nav_kbrender.render(new_position.tolist(), new_lookat.tolist(), image_name)

                single_result['preds'].append({'step':seq, 'position':new_position.tolist(), 
                                                'look_at':new_lookat.tolist(), 'img_path':new_image_path})
                print('NavigationKubricRenderer: rank:%d, image:%s' % (rank, new_image_path))

                if seq+1 == self.max_seq_len:
                    end_nav = True
                    break
                else:
                    new_seg_path = new_image_path.replace('png', 'npz')
                    new_seg_data = np.load(new_seg_path, allow_pickle=True)
                    new_seg = new_seg_data['segmentation'] # 256*256
                    new_seg_area = obj_area(new_seg)
                    seg_areas.append({'area':new_seg_area, 'angle':yaw_angle*(i+1)})
                    seq+=1

            if end_nav:
                break
        
            # chose a best camera orientation and rotate to this orientation
            best_area = seg_areas[0]['area']
            best_angle = seg_areas[0]['angle']
            for seg_area in seg_areas:
                if seg_area['area'] > best_area:
                    best_area = seg_area['area']
                    best_angle = seg_area['angle']
            
            if best_angle < 360.0:
                move_action = 'none'
                yaw_action = 'left'
                pitch_action='none'
                move_step= 0
                yaw_angle = best_angle
                pitch_angle = 0
                new_position, new_lookat, new_view_x_angle, new_view  = new_pos_and_lookat(np.array(old_position), np.array(old_lookat), old_view_x_angle, 
                                                            move_action, move_step, yaw_action, yaw_angle, pitch_action, pitch_angle)
                old_position = new_position
                old_lookat = new_lookat
                old_view = new_view
                old_view_x_angle = new_view_x_angle

                image_name = 'step'+str(seq)+'_move-'+move_action+str(move_step)+ \
                        '_yaw-'+yaw_action+str(round(yaw_angle, 2))+'_pitch-'+pitch_action+str(round(pitch_angle, 2))
                
                new_image_path = nav_kbrender.render(new_position.tolist(), new_lookat.tolist(), image_name)

                single_result['preds'].append({'step':seq, 'position':new_position.tolist(), 
                                                'look_at':new_lookat.tolist(), 'img_path':new_image_path})
                print('NavigationKubricRenderer: rank:%d, image:%s' % (rank, new_image_path))

                if seq+1 == self.max_seq_len:
                    break
                else:
                    seq+=1 

        nav_kbrender.close()
        # gc.collect()
        # print('SingleNavigator rank %d pred seq len:%d' % (rank, len(single_result['preds'])))
        return single_result


    @auto_fp16(apply_to=('imgs', ))
    def forward(self, imgs, path_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.
        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            # print('SingleNavigator imgs:', imgs)
            return self.forward_train(imgs, **kwargs)
        else:
            if self.eval_mode == 'pseudo_test':
                # print('mmdet/models/navigators/single_navigator.py: pesudo test')
                return self.forward_pseudo_test(imgs, path_metas, **kwargs)
            elif self.eval_mode == 'test':
                # print('mmdet/models/navigators/single_navigator.py: test')
                return self.forward_test(imgs, path_metas, **kwargs)
            elif self.eval_mode == 'rule_test':
                return self.forward_rule_test(imgs, path_metas, **kwargs)
            else:
                print('unexpected eval mode:', self.eval_mode)
                exit(0)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        # print(data.keys())
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=data['imgs'].size(0))

        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=data['imgs'].size(0))

        return outputs

    
    
    

    