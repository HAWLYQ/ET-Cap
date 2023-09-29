# Copyright (c) AIM3. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

@LOSSES.register_module()
class SeqMaskedCrossEntropyLoss(nn.Module):

    def __init__(self,
                 loss_weight=1.0):
        """CrossEntropyLoss with seq mask

        Args:
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(SeqMaskedCrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        
    def forward(self,
                preds,
                labels,
                seq_mask):
        """Forward function.

        Args:
            preds:[bs, seq, cls_num]
            labels:[bs, seq]
            seq_mask: [bs, seq]
        Returns:
            torch.Tensor: The calculated loss.
        """
        bs, seq, cls_num = preds.size()
        preds = preds.view(-1, cls_num)
        labels = labels.view(bs*seq)
        loss = F.cross_entropy(preds, labels, reduction='none') # [bs*seq]
        loss = loss.view(bs, seq) * seq_mask 
        non_mask_num = len(torch.nonzero(seq_mask))
        loss = self.loss_weight * torch.sum(loss) / non_mask_num
        # print('preds:', preds)
        # print('labels:', labels)
        # print('cross entroy loss:', loss)
        return loss


@LOSSES.register_module()
class SeqMaskedL1Loss(nn.Module):

    def __init__(self,
                 loss_weight=1.0):
        """L1 loss with seq mask

        Args:
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(SeqMaskedL1Loss, self).__init__()
        self.loss_weight = loss_weight
        
    def forward(self,
                preds,
                targets,
                seq_mask):
        """Forward function.

        Args:
            preds:[bs, seq]
            targets:[bs, seq]
            seq_mask: [bs, seq]
        Returns:
            torch.Tensor: The calculated loss.
        """
        
        loss = torch.abs(preds - targets) * seq_mask 
        non_mask_num = len(torch.nonzero(seq_mask))
        return self.loss_weight * torch.sum(loss) / non_mask_num