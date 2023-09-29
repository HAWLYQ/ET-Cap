# Copyright (c) AIM3 Lab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
from mmcv.runner import BaseModule


@POSITIONAL_ENCODING.register_module()
class SeqSinePositionalEncoding(BaseModule):
    """Position encoding with sine and cosine functions.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(SeqSinePositionalEncoding, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SeqSinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, seq].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, seq].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, SEQ = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
            dim=3).view(B, SEQ, -1)
        pos = pos.permute(0, 2, 1)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'temperature={self.temperature}, '
        repr_str += f'normalize={self.normalize}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'eps={self.eps})'
        return repr_str


@POSITIONAL_ENCODING.register_module()
class SeqLearnedPositionalEncoding(BaseModule):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        seq_num_embed (int, optional): The dictionary size of seq embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 seq_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(SeqLearnedPositionalEncoding, self).__init__(init_cfg)
        self.seq_embed = nn.Embedding(seq_num_embed, num_feats)
        self.num_feats = num_feats
        self.seq_num_embed = seq_num_embed
    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, seq].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, seq].
        """
        seq = mask.shape[1]
        x = torch.arange(seq, device=mask.device)
        x_embed = self.seq_embed(x)
        pos = x_embed.permute(1, 0).unsqueeze(0).repeat(mask.shape[0], 1, 1)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'seq_num_embed={self.seq_num_embed})'
        return repr_str
