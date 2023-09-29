# Copyright (c) AIM3 lab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None


@PIPELINES.register_module()
class LoadNavigationImages:
    """Load images.
    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).
    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load images and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.EmbodiedCapDataset`.

        Returns:
            dict: The dict contains loaded images and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filenames = [osp.join(results['img_prefix'], filename) for filename in results['path_info']['images']]
        else:
            filenames = results['path_info']['images']

        imgs = []
        imgs_shape = []
        for filename in filenames:
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, channel_order=self.channel_order)
            if self.to_float32:
                img = img.astype(np.float32)
            imgs.append(img)
            imgs_shape.append(img.shape)

        results['filenames'] = filenames
        results['ori_filenames'] = results['path_info']['images']
        results['imgs'] = imgs
        # print('LoadNavigationImages len(imgs):', len(results['imgs']))
        # print('LoadNavigationImages imgs[0].shape:', results['imgs'][0].shape)
        results['imgs_shape'] = imgs_shape
        results['oris_shape'] = imgs_shape
        results['img_fields'] = ['imgs']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"channel_order='{self.channel_order}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadNavigationCameraInfos:
    """Load camera infos.

    Args:
        lookat (list): (0,0,-0.25)  lookat*0.4+(0,0,0.1)=(0,0,0)
        info_type:
            'position&view': current position + view
            'action': previous actions
    """

    def __init__(self,
                 look_at=(0,0,-0.25),
                 info_type='position_view',
                 action_classes=None):
        self.look_at = np.array(look_at)
        self.info_type=info_type
        if info_type == 'action':
            self.load_class2label(action_classes)

    def normalize_view(self, view):
        norm = np.sqrt((view*view).sum())
        normalized_view = view / norm
        return normalized_view 
    
    def position_view_infos(self, results):
        positions = results['path_info']['positions']
        camera_infos = []
        for position in positions:
            position = np.array(position)
            view = self.look_at-position
            normalized_view = self.normalize_view(view)
            camera_info = np.concatenate((position*0.4 + np.array([0,0,0.1]), normalized_view), axis=-1)
            camera_infos.append(camera_info)
        return camera_infos
    
    def load_class2label(self, classes):
        self.max_yaw_angle = 360
        self.max_pitch_angle = 180
        self.max_move_step = 4
        self.fbmove_class2label = {}
        self.rlmove_class2label = {}
        self.udmove_class2label = {}
        self.yaw_class2label = {}
        self.pitch_class2label = {}
        for label, name in classes['fb_move'].items():
            self.fbmove_class2label[name] = label
        for label, name in classes['rl_move'].items():
            self.rlmove_class2label[name] = label
        for label, name in classes['ud_move'].items():
            self.udmove_class2label[name] = label
        for label, name in classes['yaw'].items():
            self.yaw_class2label[name] = label
        for label, name in classes['pitch'].items():
            self.pitch_class2label[name] = label
    
    def action_infos(self, results):
        
        """ a sample of action:
                {
                    'move':
                    {
                        'fb':['forward', 1], 
                        'rl':['none', 0], 
                        'ud':['down', 2]}, 
                    },
                    'yaw': ['left', xx],
                    'pitch': ['up', xx]
                }
        """
        actions = results['path_info']['actions']
        camera_actions = []
        for i in range(len(actions)):
            if i ==0:
                fb_move_label = self.fbmove_class2label['none']
                rl_move_label = self.rlmove_class2label['none']
                ud_move_label = self.udmove_class2label['none']
                yaw_label = self.yaw_class2label['none']
                pitch_label = self.pitch_class2label['none']
                fb_move_step = 0.0
                rl_move_step = 0.0
                ud_move_step = 0.0
                yaw_angle = 0.0
                pitch_angle = 0.0
            else:
                fb_move_label = self.fbmove_class2label[actions[i-1]['move']['fb'][0]]
                rl_move_label = self.rlmove_class2label[actions[i-1]['move']['rl'][0]]
                ud_move_label = self.udmove_class2label[actions[i-1]['move']['ud'][0]]
                yaw_label = self.yaw_class2label[actions[i-1]['yaw'][0]]
                pitch_label = self.pitch_class2label[actions[i-1]['pitch'][0]]
                fb_move_step = actions[i-1]['move']['fb'][1]/self.max_move_step
                rl_move_step = actions[i-1]['move']['rl'][1]/self.max_move_step
                ud_move_step = actions[i-1]['move']['ud'][1]/self.max_move_step
                yaw_angle = actions[i-1]['yaw'][1]/self.max_yaw_angle
                pitch_angle = actions[i-1]['pitch'][1]/self.max_pitch_angle
            camera_action = np.array([fb_move_label, fb_move_step, 
                                    rl_move_label, rl_move_step,
                                    ud_move_label, ud_move_step,
                                    yaw_label, yaw_angle,
                                    pitch_label, pitch_angle]).astype(np.float32)
            
            camera_actions.append(camera_action)
        return camera_actions


        # TODO: 

    def __call__(self, results):
        """Call functions to load camera positions and calculate normalized view.

        Args:
            results (dict): Result dict from :obj:`mmdet.EmbodiedCapDataset`.

        Returns:
            dict: The dict contains loaded camera infos and meta information.
        """
        if self.info_type == 'action':
            results['camera_infos'] = self.action_infos(results)
        elif self.info_type == 'position_view':
            results['camera_infos'] = self.position_view_infos(results)
        else:
            print('unsopported camera info type:', self,info_type)
            exit(0)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'look_at={self.look_at}, info_type={self.info_type}'
                    )
        return repr_str


@PIPELINES.register_module()
class LoadNavigationAnnotations:
    """Load multiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
    """

    def __init__(self, with_move=True, with_yaw=True, with_pitch=True):
        self.with_move = with_move
        self.with_yaw = with_yaw
        self.with_pitch = with_pitch

    def _load_move(self, results):
        """Private function to move action annotations."""
        ann_info = results['ann_info']
        results['gt_move_labels'] = ann_info['gt_move_labels'].copy()
        results['gt_move_steps'] = ann_info['gt_move_steps'].copy()
        return results
    
    def _load_yaw(self, results):
        """Private function to yaw action annotations."""
        ann_info = results['ann_info']
        results['gt_yaw_labels'] = ann_info['gt_yaw_labels'].copy()
        results['gt_yaw_angles'] = ann_info['gt_yaw_angles'].copy()
        return results

    def _load_pitch(self, results):
        """Private function to pitch action annotations."""
        ann_info = results['ann_info']
        results['gt_pitch_labels'] = ann_info['gt_pitch_labels'].copy()
        results['gt_pitch_angles'] = ann_info['gt_pitch_angles'].copy()
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.EmbodiedCapDataset`.

        Returns:
            dict: The dict contains loaded move, yaw, pitch actions.
        """

        if self.with_move:
            results = self._load_move(results)
        if self.with_yaw:
            results = self._load_yaw(results)
        if self.with_pitch:
            results = self._load_pitch(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_move={self.with_move}, '
        repr_str += f'with_yaw={self.with_yaw}, '
        repr_str += f'with_pitch={self.with_pitch})'
        return repr_str


@PIPELINES.register_module()
class LoadRelNavigationAnnotations(LoadNavigationAnnotations):
    def _load_move(self, results):
        """Private function to move action annotations."""
        ann_info = results['ann_info']
        results['gt_fbmove_labels'] = ann_info['gt_fbmove_labels'].copy()
        results['gt_fbmove_steps'] = ann_info['gt_fbmove_steps'].copy()
        results['gt_rlmove_labels'] = ann_info['gt_rlmove_labels'].copy()
        results['gt_rlmove_steps'] = ann_info['gt_rlmove_steps'].copy()
        results['gt_udmove_labels'] = ann_info['gt_udmove_labels'].copy()
        results['gt_udmove_steps'] = ann_info['gt_udmove_steps'].copy()
        return results
    

@PIPELINES.register_module()
class LoadCapAnnotation:
    """Load multiple types of annotations.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
    """

    def __init__(self, with_text_ids=True, with_text_mask=True):
        self.with_text_ids = with_text_ids
        self.with_text_mask = with_text_mask

    def _load_text_ids(self, results):
        ann_info = results['ann_info']
        results['text_ids'] = ann_info['text_ids'].copy()
        return results
    
    def _load_text_mask(self, results):
        """Private function to yaw action annotations."""
        ann_info = results['ann_info']
        results['text_mask'] = ann_info['text_mask'].copy()
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.EmbodiedCapDataset`.

        Returns:
            dict: The dict contains loaded move, yaw, pitch actions.
        """

        if self.with_text_ids:
            results = self._load_text_ids(results)
        if self.with_text_mask:
            results = self._load_text_mask(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_text_ids={self.with_text_ids}, '
        repr_str += f'with_text_mask={self.with_text_mask}, '
        return repr_str