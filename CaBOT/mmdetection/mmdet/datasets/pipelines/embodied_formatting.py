# Copyright (c) AIM3 Lab. All rights reserved.
from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PIPELINES.register_module()
class NaviCollect:
    def __init__(self,
                 keys, 
                 meta_keys=('scene_id', 'positions')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys

                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        for key in self.keys:
            data[key] = results[key]
        path_meta = {}
        for key in self.meta_keys:
            path_meta[key] = results['path_info'][key]
        data['path_metas'] = DC(path_meta, cpu_only=True)
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'


@PIPELINES.register_module()
class CapCollect:
    def __init__(self,
                 keys, 
                 meta_keys=('scene_id', 'imgid')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys

                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        for key in self.keys:
            data[key] = results[key]
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results['img_info'][key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'




@PIPELINES.register_module()
class ImagesToTensor:
    """Convert images to :obj:`torch.Tensor` by given keys.

    transpose each img and concatenate them to one `torch.Tensor` 
    
    The dimension order of each image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys=['imgs']):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert images in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """
        for key in self.keys:
            img_tensors = []
            for img in results[key]:
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
            img_tensors.append((to_tensor(img.transpose(2, 0, 1))).contiguous())
            results[key] = torch.cat(img_tensors, dim=0)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class NavigationDefaultFormatBundle:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields

    - imgs: (1)transpose, (2)to tensor, (3) concatenate, (4)to DataContainer (stack=True)
    - camera_infos: (1)to tensor,(2) concatenate (3) to DataContainer
    - img_seq_mask: (1)to tensor, (2)to DataContainer
    - gt_move_labels: (1)to tensor, (2)to DataContainer
    - gt_yaw_labels: (1)to tensor, (2)to DataContainer
    - gt_yaw_angles: (1)to tensor, (2)to DataContainer
    - gt_pitch_labels: (1)to tensor, (2)to DataContainer
    - gt_pitch_angles: (1)to tensor, (2)to DataContainer
    

    Args:
        img_to_float (bool): Whether to force the image to be converted to
            float type. Default: True.
        pad_val (dict): A dict for padding value in batch collating,
            the default value is `dict(img=0, masks=0, seg=255)`.
            Without this argument, the padding value of "gt_semantic_seg"
            will be set to 0 by default, which should be 255.
    """

    def __init__(self,
                 label_keys=['gt_move_labels', 'gt_move_steps', 'gt_yaw_labels', 'gt_yaw_angles', 'gt_pitch_labels', 'gt_pitch_angles'],
                 img_to_float=True,
                 pad_val=dict(imgs=0, masks=0)):
        self.label_keys=label_keys
        self.img_to_float = img_to_float
        self.pad_val = pad_val

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'imgs' in results:
            imgs = results['imgs']
            img_tensors = []
            for img in imgs:
                # if self.img_to_float is True and img.dtype == np.uint8:
                    # Normally, image is of uint8 type without normalization.
                    # At this time, it needs to be forced to be converted to
                    # flot32, otherwise the model training and inference
                    # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                img_tensors.append((to_tensor(np.expand_dims(img.transpose(2, 0, 1), 0))).contiguous())
            imgs = torch.cat(img_tensors, dim=0) # max_seq_len * C * H * W
            results['imgs'] = DC(imgs, padding_value=self.pad_val['imgs'], stack=True, pad_dims=None)

        if 'camera_infos' in results:
            infos = results['camera_infos']
            camera_infos_tensors = []
            for info in infos:
                camera_infos_tensors.append(to_tensor(np.expand_dims(info, 0).astype(np.float32)).contiguous())
            camera_infos = torch.cat(camera_infos_tensors, dim=0) # max_seq_len * 6
            results['camera_infos'] = DC(camera_infos, padding_value=0, stack=True, pad_dims=None)

        for key in self.label_keys:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack = True, pad_dims=None)
        
        if 'img_seq_mask' in results:
            results['img_seq_mask'] = DC(
                to_tensor(results['img_seq_mask'].astype(np.float32)),
                padding_value=self.pad_val['masks'],
                stack = True,
                pad_dims=None)
                #cpu_only=True)

        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(img_to_float={self.img_to_float}, label_keys={self.label_keys})'


@PIPELINES.register_module()
class CaptionDefaultFormatBundle:
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields
    - img: (1)transpose, (2)to tensor (3) to DataContainer (stack=True)
    - text_ids: (1)to tensor (3) to DataContainer
    - text_mask: (1)to tensor, (2)to DataContainer

    Args:
        img_to_float (bool): Whether to force the image to be converted to
            float type. Default: True.
        pad_val (dict): A dict for padding value in batch collating,
            the default value is `dict(img=0, masks=0, seg=255)`.
            Without this argument, the padding value of "gt_semantic_seg"
            will be set to 0 by default, which should be 255.
    """

    def __init__(self,
                 img_to_float=True,
                 pad_val=dict(img=0, mask=0)):
        self.img_to_float = img_to_float
        self.pad_val = pad_val

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(
                to_tensor(img), padding_value=self.pad_val['img'], stack=True)

        for key in ['text_ids', 'text_mask']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=True, pad_dims=None)

        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(img_to_float={self.img_to_float})'