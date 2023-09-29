# Copyright (c) aim3lab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
import torch

from mmcv.utils import build_from_cfg
from mmcv.runner import get_dist_info, BaseModule, auto_fp16
from mmcv.cnn import Linear, initialize
from mmcv.cnn.utils.weight_init import update_init_info
from mmcv.utils.logging import get_logger, logger_initialized, print_log

from mmdet.datasets.pipelines import Compose
from mmdet.core import multi_apply

from ..builder import CAPTIONERS, build_backbone, build_head, build_neck 
from collections import OrderedDict, defaultdict
import torch.distributed as dist
from .utils import new_pos_and_lookat, vector_angle
from .kubric_render import NavigationKubricRenderer
import numpy as np
from .single_navigator import backbone_init_from_pretrained
from .single_navigator import neck_init_from_pretrained


# SingeTrajectoryCaptioner process photos on trajectory to generate caption
# MeanViewLocal: only use region-level features of mean view
# EndViewLocal: only use region-level features of end view
# TimeGlobal: only use time-level features
# TimeGlobal_EndViewLocal: parallel time-level cross att and region-level cross att (only end view)
# TimeGlobal_MeanViewLocal: parallel time-level cross att and region-level cross att (mean view)
@CAPTIONERS.register_module()
class SingeTrajectoryCaptioner(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 backbone,
                 pre_neck=None,
                 neck=None,
                 cap_head=None,
                 frozen_backbone=False,
                 backbone_init=None,
                 frozen_preneck=False,
                 preneck_init=None,
                 frozen_neck=False,
                 neck_init=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 cap_input_feat_type=None):
        super(SingeTrajectoryCaptioner, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)    

        self.backbone_init = backbone_init
        if self.backbone_init is not None:
            self.backbone = backbone_init_from_pretrained(self.backbone, self.backbone_init)

        if pre_neck is not None:
            self.pre_neck = build_neck(pre_neck)
            self.preneck_init = preneck_init
            if self.preneck_init is not None:
                self.pre_neck = neck_init_from_pretrained(self.pre_neck, self.preneck_init, 'pre_neck')

        if neck is not None:
            self.neck = build_neck(neck)
            self.neck_init = neck_init
            if self.neck_init is not None:
                self.neck = neck_init_from_pretrained(self.neck, self.neck_init)
        
        if frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        if pre_neck is not None and frozen_preneck:
            for m in self.pre_neck.children():
                if m.__class__.__name__ == 'NaviTransformerEncoder':
                    print('no freeze ', m.__class__.__name__)
                    continue
                for param in m.parameters():
                    param.requires_grad = False

        if neck is not None and frozen_neck:
            for m in self.neck.children():
                if m.__class__.__name__ == 'NaviTransformerEncoder':
                    print('no freeze ', m.__class__.__name__)
                    continue
                for param in m.parameters():
                    param.requires_grad = False

        cap_head.update(train_cfg=train_cfg)
        cap_head.update(test_cfg=test_cfg)
        self.cap_head = build_head(cap_head)
        self.caphead_init = cap_head.get('initialize_from_pretrain', False)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # ['MeanViewLocal'，'EndViewLocal', 'TimeGlobal', 'TimeGlobal_EndViewLocal', 'TimeGlobal_MeanViewLocal']
        self.cap_input_feat_type = cap_input_feat_type 

        # anwenhu 2022/10/17: revise init_weights to avoid overwritten pretrained parameters
    
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
                if hasattr(m, 'init_weights'):
                    # anwenhu 2022/10/17: these two parts have been initialized before call init_weights()
                    if m.__class__.__name__ == 'SingleCaptionerHead' and self.caphead_init:
                        print_log(f'skip caphead {m.__class__.__name__}.init_weights()',logger=logger_name)
                        continue
                    elif m.__class__.__name__ == 'ResNet' and self.backbone_init:
                        print_log(f'skip backbone {m.__class__.__name__}.init_weights()',logger=logger_name)
                        continue
                    elif m.__class__.__name__ == 'NaviImageTransformerNeck' and self.neck_init:
                        print_log(f'skip neck {m.__class__.__name__}.init_weights()',logger=logger_name)
                        continue
                    elif m.__class__.__name__ == 'NaviImageTransformerPreNeck' and self.preneck_init:
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
    
    @property
    def with_preneck(self):
        """bool: whether the detector has a pre_neck"""
        return hasattr(self, 'pre_neck') and self.pre_neck is not None

    def extract_global_feat(self, imgs, img_seq_mask):
        """Directly extract features from the backbone+neck."""
        # print('SingleNavigator imgs:', imgs.size())
        batch_size, seq_len, C, H, W = imgs.size()
        imgs = imgs.reshape(batch_size*seq_len, C, H, W)
        # print('SingleNavigator input imgs:', imgs)
        x = self.backbone(imgs)[-1] # [batch_size*seq, 2048, 8, 8]
        # print('SingeTrajectoryCaptioner after backbone x:', x.size())

        if self.with_preneck:
            _, c, h, w = x.size()
            x = x.reshape(batch_size, seq_len, c, h, w) # [batch, img_seq, c, h, w]
            x = self.pre_neck(x, patch_mean_pool=False, img_seq_mask=img_seq_mask) # [h*w, batch, img_seq, dim]
            x = x.permute(1,2,3,0).reshape(batch_size*seq_len, -1, h, w) # [batch*img_seq, dim, h, w,]

        if self.with_neck:
            _, c, h, w = x.size()
            x = x.reshape(batch_size, seq_len, c, h, w) # [batch, img_seq, c, h, w]
            x = self.neck(x, patch_mean_pool=True, img_seq_mask=img_seq_mask) # [batch, img_seq, dim]
            # print('SingeTrajectoryCaptioner after neck x:', x.size()) # [bs, 13, 256]
        return x # [batch, img_seq, dim]
        
    def extract_finegrained_feat(self, imgs):
        """Directly extract features from the backbone+neck."""
        # print('SingleNavigator imgs:', imgs.size())
        batch_size, C, H, W = imgs.size()
        # print('SingleNavigator input imgs:', imgs)
        x = self.backbone(imgs)[-1] # [batch_size, 2048, 8, 8]
        # print('SingeTrajectoryCaptioner after backbone x:', x.size())

        if self.with_preneck:
            _, c, h, w = x.size()
            x = x.unsqueeze(1) # [batch_size, 1, 2048, 16, 16]
            x = self.pre_neck(x, patch_mean_pool=False) # [h*w, batch, 1, dim]
            x = x.squeeze(2).permute(1,2,0).reshape(batch_size, -1, h, w) # [batch*img_seq, dim, h, w,]

        if self.with_neck:
            # to satisfiy embodied_neck input format, add img_seq==1
            x = x.unsqueeze(1) # [batch_size, 1, 2048, 8, 8]
            x = self.neck(x, patch_mean_pool=False) # [h*w, batch_size, 1, dim]
            x = x.squeeze(2).permute(1,0,2) # [batch_size, h*w, dim]
            # print('SingeTrajectoryCaptioner after neck x:', x.size()) # [bs, 64, 256]
        return x
    
    def extract_global_and_mean_finegrained_feat(self, imgs, img_seq_mask):
        """Directly extract features from the backbone+neck."""
        # print('SingleNavigator imgs:', imgs.size())
        batch_size, seq_len, C, H, W = imgs.size()
        imgs = imgs.reshape(batch_size*seq_len, C, H, W)
        # print('SingleNavigator input imgs:', imgs)
        x = self.backbone(imgs)[-1] # [batch_size*seq, 2048, 8, 8]
        # print('SingeTrajectoryCaptioner after backbone x:', x.size())

        if self.with_preneck:
            _, c, h, w = x.size()
            x = x.reshape(batch_size, seq_len, c, h, w) # [batch, img_seq, c, h, w]
            x = self.pre_neck(x, patch_mean_pool=False, img_seq_mask=img_seq_mask) # [h*w, batch, img_seq, dim]
            x = x.permute(1,2,3,0).reshape(batch_size*seq_len, -1, h, w) # [batch*img_seq, dim, h, w,]

        assert self.with_neck
        _, c, h, w = x.size()
        x = x.reshape(batch_size, seq_len, c, h, w) # [batch, img_seq, c, h, w]
        x = self.neck(x, patch_mean_pool=False, img_seq_mask=img_seq_mask) # [h*w, batch, img_seq, dim]
        global_x = torch.mean(x, dim=0) # [batch, img_seq, dim]
        ## calculate time-level mean region features
        img_num = torch.sum(img_seq_mask, dim=1).unsqueeze(0).unsqueeze(2).repeat(h*w, 1, 1) # [h*w, batch, 1]
        # img_seq_mask: [batch, img_seq] > [1, batch, img_seq, 1] > [h*w, batch, img_seq, 1]
        img_seq_mask = img_seq_mask.unsqueeze(0).unsqueeze(3)
        img_seq_mask = img_seq_mask.repeat(h*w, 1, 1, 1)
        mean_finegrained_x = torch.sum(x * img_seq_mask, dim=2) /  img_num # [h*w, batch, dim]
        mean_finegrained_x = mean_finegrained_x.permute(1, 0, 2) # [batch, h*w, dim]
        # print('SingeTrajectoryCaptioner after neck x:', x.size()) # [bs, 13, 256]
        return global_x, mean_finegrained_x # [batch, img_seq, dim]

    def forward_train(self,
                      imgs,
                      img_seq_mask,
                      img,
                      text_ids,
                      text_mask,):
        """
        Args:
            imgs (Tensor): Input images of shape (B, Seq, C, H, W).
            img (Tensor): Input good view images of shape (B, C, H, W).
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        if self.cap_input_feat_type == 'TimeGlobal_EndViewLocal':
            time_x = self.extract_global_feat(imgs, img_seq_mask) # [batch, nav_seq, dim]
            region_x = self.extract_finegrained_feat(img) # [batch, h*w, dim]
            losses = self.cap_head.forward_train(region_x, time_x, img_seq_mask, text_ids, text_mask)
        elif self.cap_input_feat_type == 'TimeGlobal_MeanViewLocal':
            # [batch, nav_seq, dim], [batch, h*w, dim]
            time_x, region_x = self.extract_global_and_mean_finegrained_feat(imgs, img_seq_mask)
            losses = self.cap_head.forward_train(region_x, time_x, img_seq_mask, text_ids, text_mask)
        elif self.cap_input_feat_type == 'MeanViewLocal':
            _, region_x = self.extract_global_and_mean_finegrained_feat(imgs, img_seq_mask)
            losses = self.cap_head.forward_train(region_x, text_ids, text_mask)
        elif self.cap_input_feat_type == 'EndViewLocal':
            region_x = self.extract_finegrained_feat(img) # [batch, h*w, dim]
            losses = self.cap_head.forward_train(region_x, text_ids, text_mask)
        elif self.cap_input_feat_type == 'TimeGlobal':
            time_x = self.extract_global_feat(imgs, img_seq_mask) # [batch, nav_seq, dim]
            losses = self.cap_head.forward_train(time_x, img_seq_mask, text_ids, text_mask)
        return losses

    def forward_test(self, imgs, path_info, img_seq_mask, img, **kwargs):
        if self.cap_input_feat_type == 'TimeGlobal_EndViewLocal':
            time_x = self.extract_global_feat(imgs, img_seq_mask) # [batch, nav_seq, dim]
            region_x = self.extract_finegrained_feat(img) # [batch, h*w, dim]
            bs, _, _ = time_x.size()
            captions = self.cap_head.forward_test(region_x, time_x, img_seq_mask)
        elif self.cap_input_feat_type == 'TimeGlobal_MeanViewLocal':
            # [batch, nav_seq, dim], [batch, h*w, dim]
            time_x, region_x = self.extract_global_and_mean_finegrained_feat(imgs, img_seq_mask)
            bs, _, _ = time_x.size()
            captions = self.cap_head.forward_test(region_x, time_x, img_seq_mask)
        elif self.cap_input_feat_type == 'MeanViewLocal':
            _, region_x = self.extract_global_and_mean_finegrained_feat(imgs, img_seq_mask)
            bs, _, _ = region_x.size()
            captions = self.cap_head.forward_test(region_x)
        elif self.cap_input_feat_type == 'EndViewLocal':
            region_x = self.extract_finegrained_feat(img) # [batch, h*w, dim]
            bs, _, _ = region_x.size()
            captions = self.cap_head.forward_test(region_x)
        elif self.cap_input_feat_type == 'TimeGlobal':
            time_x = self.extract_global_feat(imgs, img_seq_mask) # [batch, nav_seq, dim]
            bs, _, _ = time_x.size()
            captions = self.cap_head.forward_test(time_x, img_seq_mask)
        # print('single_captioner.py img_info:', img_info)
        # print('single_captioner.py captions:', captions)
        results = []
        for i in range(bs):
            single_result = {'scene_id':path_info[i]['scene_id'], 
                             'pathid':path_info[i]['pathid'],
                             'pred_caption': captions[i]}
            results.append(single_result)
        return results
    
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

            return self.forward_test(imgs, path_metas, **kwargs)

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

