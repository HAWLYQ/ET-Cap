# Copyright (c) AIM3 Lab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import (FFN, build_positional_encoding, build_transformer_layer_sequence)
from mmcv.runner import BaseModule,force_fp32

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from ..builder import HEADS, build_loss
from abc import ABCMeta, abstractmethod


def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask, 
    # True means masked item; as torch.nn.MultiheadAttention
    # mask = torch.zeros(seq_length, seq_length, device=device)
    mask = torch.ones(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i+1):
            mask[i, j] = 0
    return mask.to(torch.bool)

@HEADS.register_module()
class SingleRelNavigatorHead(BaseModule, metaclass=ABCMeta):
    """Implements the navigation transformer head.

    Args:
        num_classes (int): Number of action categrories.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    # _version = 2

    def __init__(self,
                 fb_move_classes,
                 rl_move_classes,
                 ud_move_classes,
                 pitch_classes,
                 yaw_classes,
                 in_channels,
                 num_reg_fcs=2,
                 decoder=None,
                 decode_positional_encoding=dict(
                     type='SeqLearnedPositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 loss_cls=dict(
                    type='SeqMaskedCrossEntropyLoss',
                    loss_weight=1.0),
                 loss_reg=dict(
                    type='SeqMaskedL1Loss', 
                    loss_weight=2.0),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        super(SingleRelNavigatorHead, self).__init__(init_cfg)

        self.fb_move_classes = fb_move_classes # 3 means forward/backward/none
        self.rl_move_classes = rl_move_classes # 3 means left/right/none
        self.ud_move_classes = ud_move_classes # 3 means up/down/none
        self.yaw_classes = yaw_classes
        self.pitch_classes = pitch_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        # self.train_cfg = train_cfg
        # self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)

        self.fb_move_cls_out_channels = fb_move_classes
        self.rl_move_cls_out_channels = rl_move_classes
        self.ud_move_cls_out_channels = ud_move_classes
        self.yaw_cls_out_channels = yaw_classes
        self.pitch_cls_out_channels = pitch_classes

        self.act_cfg = decoder.get('act_cfg', dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)


        self.decode_positional_encoding = build_positional_encoding(
            decode_positional_encoding)

        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        """assert 'num_feats' in decode_positional_encoding
        num_feats = decode_positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'"""
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)
        self.fb_move_fc_cls = Linear(self.embed_dims, self.fb_move_cls_out_channels)
        self.rl_move_fc_cls = Linear(self.embed_dims, self.rl_move_cls_out_channels)
        self.ud_move_fc_cls = Linear(self.embed_dims, self.ud_move_cls_out_channels)
        self.yaw_fc_cls = Linear(self.embed_dims, self.yaw_cls_out_channels)
        self.pitch_fc_cls = Linear(self.embed_dims, self.pitch_cls_out_channels)
        self.reg_ffn = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_reg_fcs,
            self.act_cfg,
            dropout=0.0,
            add_residual=False)
        # predict forward/backward move step, left/right move step. up/down move step, yaw and pitch angles
        self.fc_reg = Linear(self.embed_dims, 5)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.decoder.init_weights()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        """version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is DETRHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]"""

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, feats, seq_masks):
        """Forward function.
        """
        num_levels = len(feats)
        # img_metas_list = [img_metas for _ in range(num_levels)]
        seq_masks_list = [seq_masks for _ in range(num_levels)]
        return multi_apply(self.forward_single, feats, seq_masks_list)


    def forward_single(self, x, seq_masks):
        """"Forward function for a single feature level.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, seq, dim].
            mask:[bs, seq]

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        decode_pos_embed = self.decode_positional_encoding(seq_masks)  # [bs, embed_dim, seq]
        decode_pos_embed = decode_pos_embed.permute(2, 0, 1) # [seq, bs, c]
        x = x.permute(1, 0, 2)  # [seq, bs, c]
        # print(x.type(), decode_pos_embed.type(), seq_masks.type())
        _, seq = seq_masks.size()
        attn_mask = _get_causal_mask(seq, seq_masks.device) # [seq, seq] 
        # print('mmdet/models/decoding_heads/navi_head.py attn_mask:', attn_mask)
        # print('SingleNavigatorHead attn_mask:', attn_mask)
        # outs_dec = self.decoder(query=x, query_pos=decode_pos_embed, query_key_padding_mask=seq_masks)  # outs_dec: [nb_dec, num_query, bs, embed_dim]
        outs_dec = self.decoder(query=x, query_pos=decode_pos_embed, attn_masks=attn_mask)  # outs_dec: [nb_dec, num_query, bs, embed_dim]
        outs_dec = outs_dec.transpose(1, 2) # [nb_dec, bs, seq, embed_dim]

        all_fbmove_cls_scores = self.fb_move_fc_cls(outs_dec) # [nb_dec, bs, seq, 3]
        all_rlmove_cls_scores = self.rl_move_fc_cls(outs_dec) # [nb_dec, bs, seq, 3]
        all_udmove_cls_scores = self.ud_move_fc_cls(outs_dec) # [nb_dec, bs, seq, 3]
        all_yaw_cls_scores = self.yaw_fc_cls(outs_dec) # [nb_dec, bs, seq, 3]
        all_pitch_cls_scores = self.pitch_fc_cls(outs_dec) # [nb_dec, bs, seq, 3]
        
        all_step_angles_preds = self.fc_reg(self.activate(self.reg_ffn(outs_dec))).sigmoid() # [nb_dec, bs, seq, 5]
        all_fbmove_steps_preds = all_step_angles_preds[:,:,:,0]
        all_rlmove_steps_preds = all_step_angles_preds[:,:,:,1]
        all_udmove_steps_preds = all_step_angles_preds[:,:,:,2]
        all_yaw_angles_preds = all_step_angles_preds[:,:,:,3]
        all_pitch_angles_preds = all_step_angles_preds[:,:,:,4]
        
        return all_fbmove_cls_scores, all_rlmove_cls_scores, all_udmove_cls_scores, all_yaw_cls_scores, all_pitch_cls_scores, all_fbmove_steps_preds, all_rlmove_steps_preds, all_udmove_steps_preds, all_yaw_angles_preds, all_pitch_angles_preds

    
    @force_fp32(apply_to=('all_move_cls_scores_list', 'all_yaw_cls_scores_list','all_pitch_cls_scores_list'))
    def loss(self,
             all_fbmove_cls_scores_list,
             all_rlmove_cls_scores_list,
             all_udmove_cls_scores_list,
             all_yaw_cls_scores_list,
             all_pitch_cls_scores_list,
             all_fbmove_step_preds_list,
             all_rlmove_step_preds_list,
             all_udmove_step_preds_list,
             all_yaw_angle_preds_list,
             all_pitch_angle_preds_list,
             gt_fbmove_labels_list,
             gt_rlmove_labels_list,
             gt_udmove_labels_list,
             gt_yaw_labels_list,
             gt_pitch_labels_list,
             gt_fbmove_steps_list,
             gt_rlmove_steps_list,
             gt_udmove_steps_list,
             gt_yaw_angles_list,
             gt_pitch_angles_list,
             seq_mask
             ):
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_move_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, seq, move_cls_out_channels].
            all_yaw_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, seq, mvoe_cls_out_channels].
            all_pitch_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, seq, mvoe_cls_out_channels].

            gt_move_labels_list (list[Tensor]): Ground truth class indices for each
                step with shape (num_gts, ).
            gt_yaw_labels_list (list[Tensor]): Ground truth class indices for each
                step with shape (num_gts, ).
            gt_pitch_labels_list (list[Tensor]): Ground truth class indices for each
                step with shape (num_gts, ).

            img_metas (list[dict]): List of image meta information.


        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # NOTE defaultly only the outputs from the last feature scale is used.
        all_fbmove_cls_scores = all_fbmove_cls_scores_list[-1]
        all_rlmove_cls_scores = all_rlmove_cls_scores_list[-1]
        all_udmove_cls_scores = all_udmove_cls_scores_list[-1]
        all_yaw_cls_scores = all_yaw_cls_scores_list[-1]
        all_pitch_cls_scores = all_pitch_cls_scores_list[-1]

        all_fbmove_step_preds = all_fbmove_step_preds_list[-1]
        all_rlmove_step_preds = all_rlmove_step_preds_list[-1]
        all_udmove_step_preds = all_udmove_step_preds_list[-1]
        all_yaw_angle_preds = all_yaw_angle_preds_list[-1]
        all_pitch_angle_preds = all_pitch_angle_preds_list[-1]

        # print('SingleNavigatorHead all_move_cls_scores:', all_move_cls_scores.size())
        num_dec_layers = len(all_fbmove_cls_scores)

        all_fbmove_gt_labels_list = [gt_fbmove_labels_list for _ in range(num_dec_layers)]
        all_rlmove_gt_labels_list = [gt_rlmove_labels_list for _ in range(num_dec_layers)]
        all_udmove_gt_labels_list = [gt_udmove_labels_list for _ in range(num_dec_layers)]
        all_yaw_gt_labels_list = [gt_yaw_labels_list for _ in range(num_dec_layers)]
        all_pitch_gt_labels_list = [gt_pitch_labels_list for _ in range(num_dec_layers)]

        all_gt_fbmove_steps_list = [gt_fbmove_steps_list for _ in range(num_dec_layers)]
        all_gt_rlmove_steps_list = [gt_rlmove_steps_list for _ in range(num_dec_layers)]
        all_gt_udmove_steps_list = [gt_udmove_steps_list for _ in range(num_dec_layers)]
        all_gt_yaw_angles_list = [gt_yaw_angles_list for _ in range(num_dec_layers)]
        all_gt_pitch_angles_list = [gt_pitch_angles_list for _ in range(num_dec_layers)]

        all_seq_mask = [seq_mask for _ in range(num_dec_layers)]
        
        # multiple-layer loss
        losses_fbmove_cls, losses_rlmove_cls, losses_udmove_cls, losses_yaw_cls, losses_pitch_cls, losses_fbmove_step, losses_rlmove_step, losses_udmove_step, losses_yaw_angle, losses_pitch_angle = multi_apply(self.loss_single,
                                                                                                                 all_fbmove_cls_scores, all_fbmove_gt_labels_list,
                                                                                                                 all_rlmove_cls_scores, all_rlmove_gt_labels_list,
                                                                                                                 all_udmove_cls_scores, all_udmove_gt_labels_list,
                                                                                                                 all_yaw_cls_scores, all_yaw_gt_labels_list,
                                                                                                                 all_pitch_cls_scores, all_pitch_gt_labels_list,
                                                                                                                 all_fbmove_step_preds, all_gt_fbmove_steps_list,
                                                                                                                 all_rlmove_step_preds, all_gt_rlmove_steps_list,
                                                                                                                 all_udmove_step_preds, all_gt_udmove_steps_list,
                                                                                                                 all_yaw_angle_preds, all_gt_yaw_angles_list,
                                                                                                                 all_pitch_angle_preds, all_gt_pitch_angles_list,
                                                                                                                 all_seq_mask,
                                                                                                                 )

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_fbmove_cls'] = losses_fbmove_cls[-1]
        loss_dict['loss_rlmove_cls'] = losses_rlmove_cls[-1]
        loss_dict['loss_udmove_cls'] = losses_udmove_cls[-1]
        loss_dict['loss_yaw_cls'] = losses_yaw_cls[-1]
        loss_dict['loss_pitch_cls'] = losses_pitch_cls[-1]
        loss_dict['loss_fbmove_step'] = losses_fbmove_step[-1]
        loss_dict['loss_rlmove_step'] = losses_rlmove_step[-1]
        loss_dict['loss_udmove_step'] = losses_udmove_step[-1]
        loss_dict['loss_yaw_angle'] = losses_yaw_angle[-1]
        loss_dict['loss_pitch_angle'] = losses_pitch_angle[-1]
        # loss from other decoder layers
        return loss_dict

    def loss_single(self,
                    fbmove_cls_scores, fbmove_gt_labels,
                    rlmove_cls_scores, rlmove_gt_labels,
                    udmove_cls_scores, udmove_gt_labels,
                    yaw_cls_scores, yaw_gt_labels,
                    pitch_cls_scores, pitch_gt_labels,
                    fbmove_step_preds, gt_fbmove_steps,
                    rlmove_step_preds, gt_rlmove_steps,
                    udmove_step_preds, gt_udmove_steps,
                    yaw_angle_preds, gt_yaw_angles,
                    pitch_angle_preds, gt_pitch_angles,
                    seq_mask, 
                    ):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            move_cls_scores: [bs, seq, 3]
            yaw_cls_scores: [bs, seq, 2]
            pitch_cls_scores: [bs, seq, 2]
            yaw_angle_preds: [bs, seq]
            pitch_angle_preds: [bs, seq]

            move_gt_labels:[bs, seq]
            yaw_gt_labels: [bs, seq]
            pitch_gt_labels: [bs, seq]
            gt_yaw_angles [bs, seq]
            gt_pitch_angles [bs, seq]

            seq_mask: [bs, seq]


        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # classification losses 
        losses_fbmove_cls = self.loss_cls(fbmove_cls_scores, fbmove_gt_labels, seq_mask)
        losses_rlmove_cls = self.loss_cls(rlmove_cls_scores, rlmove_gt_labels, seq_mask)
        losses_udmove_cls = self.loss_cls(udmove_cls_scores, udmove_gt_labels, seq_mask)
        losses_yaw_cls = self.loss_cls(yaw_cls_scores, yaw_gt_labels, seq_mask)
        losses_pitch_cls = self.loss_cls(pitch_cls_scores, pitch_gt_labels, seq_mask)

        losses_fbmove_step = self.loss_reg(fbmove_step_preds, gt_fbmove_steps, seq_mask)
        losses_rlmove_step = self.loss_reg(rlmove_step_preds, gt_rlmove_steps, seq_mask)
        losses_udmove_step = self.loss_reg(udmove_step_preds, gt_udmove_steps, seq_mask)
        losses_yaw_angle = self.loss_reg(yaw_angle_preds, gt_yaw_angles, seq_mask)
        losses_pitch_angle = self.loss_reg(pitch_angle_preds, gt_pitch_angles, seq_mask)

        return losses_fbmove_cls, losses_rlmove_cls, losses_udmove_cls, losses_yaw_cls, losses_pitch_cls, losses_fbmove_step, losses_rlmove_step, losses_udmove_step, losses_yaw_angle, losses_pitch_angle

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,x, x_mask, gt_fbmove_labels, gt_fbmove_steps, gt_rlmove_labels, gt_rlmove_steps, gt_udmove_labels, gt_udmove_steps, gt_yaw_labels, gt_yaw_angles, gt_pitch_labels, gt_pitch_angles):
        """Forward function for training mode.

        Args:
            x : [bs, seq, dim]
            x_mask: [bs, seq]

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # outs = self(x, x_mask)
        # print('SingleNavigatorHead x:', x)
        outs = self([x], x_mask)
        # print('SingleNavigatorHead outs:', outs)
        # outs = self.forward_single(x, x_mask)
        loss_inputs = outs + (gt_fbmove_labels, gt_rlmove_labels, gt_udmove_labels, gt_yaw_labels, gt_pitch_labels, gt_fbmove_steps, gt_rlmove_steps, gt_udmove_steps, gt_yaw_angles, gt_pitch_angles)
        losses = self.loss(*loss_inputs, x_mask)
        return losses
    
    def forward_sudo_test(self, x, x_mask):
        return self.forward_single(x, x_mask)




    


