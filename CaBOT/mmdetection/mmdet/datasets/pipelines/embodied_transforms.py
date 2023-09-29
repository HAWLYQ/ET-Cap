import copy
import inspect
import math
import warnings

import cv2
import mmcv
import numpy as np
from numpy import random

from mmdet.core import BitmapMasks, PolygonMasks, find_inside_bboxes
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.utils import log_img_scale
from ..builder import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


@PIPELINES.register_module()
class NormalizeImages:
    """Normalize multiple images.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        normalized_imgs = []
        for img in results['imgs']:
            normalized_img = mmcv.imnormalize(img, self.mean, self.std,self.to_rgb) # ndarry
            normalized_imgs.append(normalized_img)
        results['imgs'] = normalized_imgs
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class ImageSeqPad:
    def __init__(self, max_seq_len, label_keys=['gt_move_labels', 'gt_move_steps', 'gt_yaw_labels', 'gt_yaw_angles', 'gt_pitch_labels', 'gt_pitch_angles']):
        self.max_seq_len = max_seq_len
        self.label_keys = label_keys

    def __call__(self, results):
        seq_len = len(results['imgs'])
        mask = np.ones(self.max_seq_len)
        if seq_len < self.max_seq_len:
            # 'img' refer to the final view  in the path 
            results['img'] = results['imgs'][-1]
            for i in range(self.max_seq_len-seq_len):
                results['imgs'].append(np.zeros(results['imgs_shape'][0]))
                if 'camera_infos' in results:
                    results['camera_infos'].append(np.zeros(results['camera_infos'][0].shape))
                mask[seq_len+i] = 0
            results['img_seq_mask'] = mask
            for key in self.label_keys:
                if key not in results:
                    continue
                results[key] += [0]*(self.max_seq_len-seq_len)
        else:
            # remove previous imgs, camera_infos and annos
            results['imgs'] = results['imgs'][-self.max_seq_len:]
            # 'img' refer to the final view  in the path 
            results['img'] = results['imgs'][-1]
            if 'camera_infos' in results:
                results['camera_infos'] = results['camera_infos'][-self.max_seq_len:] 
            results['img_seq_mask'] = mask
            for key in self.label_keys:
                if key not in results:
                    continue
                results[key] = results[key][-self.max_seq_len:]
        # print('ImageSeqPad len(imgs):', len(results['imgs']))
        # print('ImageSeqPad imgs[0].shape:', results['imgs'][0].shape)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_seq_len={self.max_seq_len}, label_keys={self.label_keys})'
        return repr_str