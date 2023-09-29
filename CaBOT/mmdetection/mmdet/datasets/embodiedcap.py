# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from torch.utils.data import Dataset
from .pipelines import Compose
import json
import cv2
from skimage.metrics import structural_similarity
import os
from tqdm import tqdm
from transformers import BertTokenizer
# import coco_caption_eval
from mmdet.datasets import coco_caption_eval
from .coco import CocoDataset
import sys
sys.path.append('/root/code/CLIP/')
import clip
import torch
from PIL import Image
import random
from icecream import ic

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

DATASET_DIR='/data5/haw/ETCAP/'
SCENES_DIR = DATASET_DIR + 'scenes/'
SCENES_SEG_DIR= DATASET_DIR + 'scenes_redo/'
# INFERENCE_DIR = DATASET_DIR + 'inference/'
MAX_NAVI_LEN=13


def mean_scores(scores):
    return sum(scores)/len(scores)


def cal_clip_similarity(samples, threshold=0.7, path_len_weight=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print('calculating clip score...')
    mean_nor_all_scores = []
    # samples = samples[:2]
    with torch.no_grad():
        for sample in tqdm(samples):
            pred_image_paths = sample['pred_images'][:MAX_NAVI_LEN]
            gt_end_images_path = sample['gt_end_images']
            pred_num = len(pred_image_paths)
            gt_num = len(gt_end_images_path)
            pred_path_len = len(pred_image_paths)
            gt_path_len = sample['gt_path_len']
            pred_images = []
            for pred_image_path in pred_image_paths:
                pred_images.append(preprocess(Image.open(pred_image_path)).unsqueeze(0).to(device))
            pred_images = torch.cat(pred_images, dim=0)
            pred_images = model.encode_image(pred_images)
            pred_images = pred_images.unsqueeze(1).repeat(1, gt_num, 1)
            pred_images = pred_images.reshape(pred_num*gt_num, -1)


            gt_images = []
            for gt_end_image_path in gt_end_images_path:
                gt_images.append(preprocess(Image.open(gt_end_image_path)).unsqueeze(0).to(device))
            gt_images = torch.cat(gt_images, dim=0)
            gt_images = model.encode_image(gt_images).repeat(pred_num, 1)
            scores = torch.cosine_similarity(pred_images, gt_images) # [pred_num*gt_num]
            scores = scores.reshape(pred_num, gt_num).cpu().numpy()
                    
            
            mean_score = np.mean(np.max(scores, axis=1))
            # if score > 0.8:
            # print('pred:', pred_end_image_path, ' gt:', gt_end_image_path, score) 

            if not path_len_weight:
                mean_nor_all_scores.append((mean_score-threshold)/(1.0-threshold))
            else:
                mean_nor_all_scores.append((mean_score-threshold)/(1.0-threshold)*(gt_path_len/max(gt_path_len, pred_path_len)))
    
    return mean_nor_all_scores

def cal_manhattan_distance(samples, path_len_weight=False):
    print('calculating manhattan distance...')
    mean_all_manhattan_dis = []
    for sample in samples:
        pred_positions = sample['pred_poss'][:MAX_NAVI_LEN]
        gt_end_positions = sample['gt_end_poss']
        pred_path_len = len(pred_positions)
        gt_path_len = sample['gt_path_len']
        pred_dis_list = []
        for pred_position in pred_positions:
            tmp_dis = []
            for gt_end_position in gt_end_positions:
                tmp_dis.append(manhattan_distance(gt_end_position, pred_position))
            pred_dis_list.append(min(tmp_dis))
        
        if not path_len_weight:
            mean_all_manhattan_dis.append(sum(pred_dis_list)/len(pred_positions))
        else:
            mean_all_manhattan_dis.append(sum(pred_dis_list)/len(pred_positions)*(max(gt_path_len, pred_path_len)/gt_path_len))

    return mean_all_manhattan_dis


def cal_euclidean_distance(samples, path_len_weight=False):
    print('calculating Euclidean distance...')
    mean_all_dis = []
    for sample in samples:
        pred_positions = sample['pred_poss'][:MAX_NAVI_LEN]
        gt_end_positions = sample['gt_end_poss']
        pred_path_len = len(pred_positions)
        gt_path_len = sample['gt_path_len']
        pred_dis_list = []
        for pred_position in pred_positions:
            tmp_dis = []
            for gt_end_position in gt_end_positions:
                tmp_dis.append(euclidean_distance(gt_end_position, pred_position))
            pred_dis_list.append(min(tmp_dis))

        # ic(gt_path_len, pred_path_len)
        if not path_len_weight:
            mean_all_dis.append(sum(pred_dis_list)/len(pred_positions))
        else:
            mean_all_dis.append(sum(pred_dis_list)/len(pred_positions)*(max(gt_path_len, pred_path_len)/gt_path_len))

    return mean_all_dis

def cal_seg_similarity(samples, path_len_weight=False):
    # 0~1
    print('calculating segmenation similarity...')
    mean_all_scores = []
    for sample in tqdm(samples):
        pred_segs = sample['pred_segs'][:MAX_NAVI_LEN]
        gt_end_segs = sample['gt_end_segs']
        pred_path_len = len(pred_segs)
        gt_path_len = sample['gt_path_len']

        pred_score_list = []
        for pred_seg in pred_segs:
            pred_category2count = read_seg(pred_seg)
            tmp_list = []
            for gt_end_seg in gt_end_segs:
                gt_category2count = read_seg(gt_end_seg)
                score = 0
                for category, count in gt_category2count.items():
                    score += min(pred_category2count.get(category, 0) / count, 1.0)
                try:
                    assert len(gt_category2count.keys()) > 0
                except AssertionError as e:
                    print('gt category num is 0 in seg:', gt_end_seg)
                    exit(0)
                score = score / len(gt_category2count.keys())
                tmp_list.append(score)
            pred_score_list.append(max(tmp_list))

        if not path_len_weight:
            mean_all_scores.append(sum(pred_score_list)/len(pred_segs))
        else:
            mean_all_scores.append(sum(pred_score_list)/len(pred_segs)*(gt_path_len/max(gt_path_len, pred_path_len)))
    
    return mean_all_scores


def euclidean_distance(position_a, position_b):
    return np.sqrt(np.sum(np.square(np.array(position_a)-np.array(position_b))))

def manhattan_distance(position_a, position_b):
    return np.sum(abs(np.array(position_a)-np.array(position_b)))

def read_seg(seg_path):
    seg_data = np.load(seg_path, allow_pickle=True)
    seg = seg_data['segmentation'] # 256*256
    v2category = seg_data['value2category'][()]
    category2count = {}
    values = np.resize(seg, [256*256]).tolist()
    for value in values:
        if value != 0:
            category2count[v2category[value]] = category2count.get(v2category[value], 0)+1
    return category2count

def get_img_path(scene_id, start_pos_id, step):
    img_dir = DATASET_DIR + 'simu_inference/scene'+str(scene_id)+'_start'+str(start_pos_id)+'/'
    for filename in os.listdir(img_dir):
        if 'png' in filename and 'step'+str(step) in filename:
            return img_dir+filename


# load navigation for navigation
class EmbodiedCapNavi():
    def __init__(self, annotation_file, classes, use_shorter_path=True):
        data = json.load(open(annotation_file, 'r', encoding='utf-8'))
        self.use_shorter_path=use_shorter_path
        self.pathid2anno = {}
        self.pathids = []
        self.data_infos = [] 
        self.max_yaw_angle = 360
        self.max_pitch_angle = 180
        self.max_move_step = 4
        self.load_class2label(classes)
        self.load_scene2assetinfo()
        self.scene2targets = {}

        for scene in data:
            scene_id = scene['scene_id'] # str
            # asset_infos are used for re-simulate a scene during inference to get segmentation
            asset_infos = self.scene2assetinfo[scene_id]
            for navigation in scene['navigation_data']:
                self.pathids.append(navigation['pathid'])
                if scene_id not in self.scene2targets:
                    self.scene2targets[scene_id] = {'positions':[], 'images':[], 'segmentations':[]}
                    if 'train' not in annotation_file:
                        seg_dir = SCENES_SEG_DIR + str(scene_id) +'/chosen_views_seg/'
                        seg_files = [seg_dir+filename for filename in os.listdir(seg_dir) if 'segmentation.npz' in filename]
                        self.scene2targets[scene_id]['segmentations'] = seg_files
                if self.use_shorter_path:
                    self.data_infos.append({'scene_id':scene_id,
                                            'asset_infos':asset_infos,
                                            'pathid':navigation['pathid'],
                                            'path':navigation['shorter_path'],
                                            'path_len':len(navigation['shorter_path']), 
                                            'images':navigation['shorter_images'], 
                                            'positions':navigation['shorter_positions'],
                                            'render_dir': scene_id+navigation['render_dir'],
                                            'actions':navigation['shorter_actions']})
                    self.pathid2anno[navigation['pathid']] = {'actions':navigation['shorter_actions']}
                    target_image_path = SCENES_DIR + str(scene_id)+navigation['render_dir']+navigation['shorter_images'][-1]
                    self.scene2targets[scene_id]['positions'].append(navigation['shorter_positions'][-1])
                    self.scene2targets[scene_id]['images'].append(target_image_path)
                else:
                    self.data_infos.append({'scene_id':scene_id,
                                            'asset_infos':asset_infos,
                                            'pathid':navigation['pathid'],
                                            'path':navigation['path'],
                                            'path_len':len(navigation['path']), 
                                            'images':navigation['images'], 
                                            'positions':navigation['positions'],
                                            'render_dir': scene_id+navigation['render_dir'],
                                            'actions':navigation['actions']})
                    self.pathid2anno[navigation['pathid']] = {'actions':navigation['actions']}
                    target_image_path = SCENES_DIR + str(scene_id)+navigation['render_dir']+navigation['images'][-1]
                    self.scene2targets[scene_id]['positions'].append(navigation['positions'][-1])
                    self.scene2targets[scene_id]['images'].append(target_image_path)

    def load_scene2assetinfo(self):
        scene_info_path= DATASET_DIR+'activecap_scenes_v0.json'
        scene_infos = json.load(open(scene_info_path, 'r', encoding='utf-8'))
        self.scene2assetinfo = {}
        for scene_info in scene_infos:
            scene_id = str(scene_info['scene_id'])
            self.scene2assetinfo[scene_id] = {'base':scene_info['base'],
                                            'other':scene_info['other']}

    def load_class2label(self, classes):
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

    def load_anns(self, pathid):
        actions = self.pathid2anno[pathid]['actions']
        """action format:
             {
                'move':{
                    'fb':['forward'/'backward', x], 
                    'rl':['left/right', x],
                    'ud':['up/down', x], 
                    },
                'pitch':['up'/'down', x], 
                'yaw': ['left'/'left', y]
            }
        """
        gt_fbmove_labels = []
        gt_rlmove_labels = []
        gt_udmove_labels = []
        gt_pitch_labels = []
        gt_yaw_labels = []

        gt_fbmove_steps = []
        gt_rlmove_steps = []
        gt_udmove_steps = []
        gt_pitch_angles = []
        gt_yaw_angles = []
        for action in actions:
            # print(action, type(action))
            gt_fbmove_labels.append(self.fbmove_class2label[action['move']['fb'][0]])
            gt_rlmove_labels.append(self.rlmove_class2label[action['move']['rl'][0]])
            gt_udmove_labels.append(self.udmove_class2label[action['move']['ud'][0]])
            gt_pitch_labels.append(self.pitch_class2label[action['pitch'][0]])
            gt_yaw_labels.append(self.yaw_class2label[action['yaw'][0]])

            gt_fbmove_steps.append(action['move']['fb'][1]/self.max_move_step) # normalize
            gt_rlmove_steps.append(action['move']['rl'][1]/self.max_move_step) # normalize
            gt_udmove_steps.append(action['move']['ud'][1]/self.max_move_step) # normalize
            gt_pitch_angles.append(action['pitch'][1]/self.max_pitch_angle) # normalize
            gt_yaw_angles.append(action['yaw'][1]/self.max_yaw_angle) # normalize

        anno = {
            'gt_fbmove_labels':gt_fbmove_labels,
            'gt_rlmove_labels':gt_rlmove_labels,
            'gt_udmove_labels':gt_udmove_labels,
            'gt_pitch_labels':gt_pitch_labels,
            'gt_yaw_labels':gt_yaw_labels,

            'gt_fbmove_steps':gt_fbmove_steps,
            'gt_rlmove_steps':gt_rlmove_steps,
            'gt_udmove_steps':gt_udmove_steps,
            'gt_pitch_angles':gt_pitch_angles,
            'gt_yaw_angles':gt_yaw_angles
        }
        return anno
    
    def get_pathids(self):
        return self.pathids

# load images of whole trajectory for captioning
class EmbodiedCapTrajCap():
    def __init__(self, annotation_file, use_shorter_path=True, max_cap_len=77):
        data = json.load(open(annotation_file, 'r', encoding='utf-8'))
        self.use_shorter_path=use_shorter_path
        self.pathid2anno = {}
        self.pathids = []
        self.data_infos = [] 
        self.scene2targets = {} # for online evaluation
        self.gtview2targets = {} # for pseudo (offline) caption evaluaton
        self.tokenizer = init_tokenizer()
        self.max_cap_len=max_cap_len

        for scene in data:
            scene_id = scene['scene_id'] # str
            for navigation in scene['navigation_data']:
                self.pathids.append(navigation['pathid'])
        
                gtview_id = scene_id + '_' + navigation['pathid'].split('_')[-1] # pathid: $scene_$start_$end
                if gtview_id not in self.gtview2targets:
                    self.gtview2targets[gtview_id] = navigation['final_view_captions']
                if self.use_shorter_path:
                    self.data_infos.append({'scene_id':scene_id,
                                            'pathid':navigation['pathid'],
                                            'path':navigation['shorter_path'],
                                            'path_len':len(navigation['shorter_path']), 
                                            'images':navigation['shorter_images'],
                                            'render_dir': scene_id+navigation['render_dir']})
                else:
                    self.data_infos.append({'scene_id':scene_id,
                                            'pathid':navigation['pathid'],
                                            'path':navigation['path'],
                                            'path_len':len(navigation['path']), 
                                            'images':navigation['images'], 
                                            'render_dir': scene_id+navigation['render_dir']})
                # randomly choose a gt caption of the view as label 
                caption = random.choice(navigation['final_view_captions'])
                self.pathid2anno[navigation['pathid']] = {'caption':caption}
                """if scene_id not in self.scene2targets:
                    self.scene2targets[scene_id] = {'captions':set()}
                self.scene2targets[scene_id]['captions']=self.scene2targets[scene_id]['captions'].union(set(navigation['final_view_captions']))"""
            self.scene2targets[scene_id] = {'captions':[]}
            self.scene2targets[scene_id]['captions'] = scene['scene_captions']

    def load_anns(self, pathid):
        ## caption annotation ##
        caption = self.pathid2anno[pathid]['caption']
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_cap_len, 
                              return_tensors="pt")
        text_mask = text.attention_mask[0].numpy()
        text_ids = text.input_ids.clone()[0].numpy()  
        text_ids[0] = self.tokenizer.bos_token_id
        # print('datasets/embodiedcap.py text_ids:', text_ids.shape)
        # print('datasets/embodiedcap.py text_mask:', text_mask.shape)
        # print('datasets/embodiedcap.py text_mask:', text_mask)

        anno = {
            'text_ids':text_ids,
            'text_mask':text_mask,
        }
        return anno
    
    def get_pathids(self):
        return self.pathids

# used for loading image pred by navigation model for trajcaption model
class EmbodiedCapTrajCapInference():
    def __init__(self, annotation_file):
        data = json.load(open(annotation_file, 'r', encoding='utf-8'))
        self.data_infos = [] 
        self.scene2targets = {}
        for scene in data:
            scene_id = scene['scene_id']
            for nav in scene['navigation_data']:
                if 'gt_path_len' not in nav:
                    self.data_infos.append({'scene_id':scene_id,
                                            'pathid':nav['pathid'],
                                            'images':nav['images'], # img paths
                                            'path_len':len(nav['images']),
                                            'gt_path_len': len(nav['images']),
                                            'render_dir': nav['render_dir'] # ''
                                            })
                else:
                    self.data_infos.append({'scene_id':scene_id,
                                            'pathid':nav['pathid'],
                                            'images':nav['images'], # img paths
                                            'path_len':len(nav['images']),
                                            'gt_path_len': nav['gt_path_len'],
                                            'render_dir': nav['render_dir'] # ''
                                            })
            assert len(scene['gt_captions']) == 3
            self.scene2targets[scene_id]=scene['gt_captions']

# for navigation only model
@DATASETS.register_module()
class EmbodiedCapNaviDataset(Dataset):
    def __init__(self,
                 ann_file,
                 classes,
                 pipeline,
                 data_root=None,
                 use_shorter_path=True,
                 pred_result_save_dir=None,
                 test_mode=False,
                 eval_metrics=None, 
                 path_len_weight_eval=False,
                 stage=1):
        self.ann_file = ann_file
        self.data_root = data_root
        self.test_mode = test_mode
        self.CLASSES = classes
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
        # load annotations
        self.load_annotations(self.ann_file, use_shorter_path)
        # processing pipeline
        self.pipeline = Compose(pipeline)
        self.pred_result_save_dir=pred_result_save_dir
        # filter images too small and containing no annotations
        if not test_mode:
            # set group flag for the sampler
            self._set_group_flag()
        self.eval_metrics = eval_metrics
        self.path_len_weight = path_len_weight_eval
        self.stage = stage
    
    def _set_group_flag(self):
        """Set flag according to path length
        path length > 10 will be set as group 1, otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            path_info = self.data_infos[i]
            if path_info['path_len'] > 10:
                self.flag[i] = 1
    
    def __len__(self):
        return len(self.data_infos)
    
    def load_annotations(self, ann_file, use_shorter_path):
        self.embodiedcap = EmbodiedCapNavi(ann_file, self.CLASSES, use_shorter_path)
        self.pathids = self.embodiedcap.pathids
        self.data_infos = self.embodiedcap.data_infos
        self.scene2targets = self.embodiedcap.scene2targets

    def get_ann_info(self, idx):
        """Get embodiedcap annotation by index.
        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        pathid = self.pathids[idx]
        ann = self.embodiedcap.load_anns(pathid)
        return ann
    
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_path(idx)
            """while True:
                data = self.prepare_train_path(idx)
                if data is None:
                    idx = self._rand_another(idx)
                    continue
                return data"""
        else:
            return self.prepare_train_path(idx)

    def prepare_train_path(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        path_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(path_info=path_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        # at stage 1, all images from dataset
        if self.stage == 1:
            results['img_prefix'] = self.data_root + 'scenes/' + results['path_info']['render_dir']
        # at stage 2, start images from dataset, left images come from inference dir
        elif self.stage == 2: 
            results['img_prefix'] = ''

    def prepare_test_path(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        path_info = self.data_infos[idx]
        results = dict(path_info=path_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    

    def pesudo_evaluate(self, results):
        eval_results = {}
        fbmove_cls_acc=0
        rlmove_cls_acc=0
        udmove_cls_acc=0
        pitch_cls_acc=0
        yaw_cls_acc=0

        fbmove_bias = 0
        rlmove_bias = 0
        udmove_bias = 0
        pitch_bias = 0
        yaw_bias = 0
        step=0
        # print(len(results))
        # print(len(self.data_infos))
        assert len(results) == len(self.data_infos)
        for i in range(len(results)):
            scene_id = self.data_infos[i]['scene_id']
            pathid = self.data_infos[i]['pathid']
            path_len = self.data_infos[i]['path_len']
            assert scene_id == results[i]['scene_id']
            assert pathid == results[i]['pathid']
            gt_actions = self.embodiedcap.pathid2anno[pathid]['actions']
            pred_actions = results[i]['pred_actions']
            for j in range(path_len):
                gt_fbmove = gt_actions[j]['move']['fb']
                gt_rlmove = gt_actions[j]['move']['rl']
                gt_udmove = gt_actions[j]['move']['ud']
                gt_pitch = gt_actions[j]['pitch']
                gt_yaw = gt_actions[j]['yaw']

                pred_fbmove = pred_actions[j]['move']['fb']
                pred_rlmove = pred_actions[j]['move']['rl']
                pred_udmove = pred_actions[j]['move']['ud']
                pred_pitch = pred_actions[j]['pitch']
                pred_yaw = pred_actions[j]['yaw']
                # print('gt:',gt_move,'pred:',pred_move)
                # print('gt:',gt_pitch,'pred:', pred_pitch)
                # print('gt:',gt_yaw, 'pred:', pred_yaw)

                if gt_fbmove[0]==pred_fbmove[0]:
                    fbmove_cls_acc+=1
                if gt_rlmove[0]==pred_rlmove[0]:
                    rlmove_cls_acc+=1
                if gt_udmove[0]==pred_udmove[0]:
                    udmove_cls_acc+=1
                if gt_pitch[0]==pred_pitch[0]:
                    pitch_cls_acc+=1
                if gt_yaw[0]==pred_yaw[0]:
                    yaw_cls_acc+=1
                fbmove_bias+=abs(gt_fbmove[1]-pred_fbmove[1])
                rlmove_bias+=abs(gt_rlmove[1]-pred_rlmove[1])
                udmove_bias+=abs(gt_udmove[1]-pred_udmove[1])
                pitch_bias+=abs(gt_pitch[1]-pred_pitch[1])
                yaw_bias+=abs(gt_yaw[1]-pred_yaw[1])
                step+=1
        eval_results['fbmove_cls_acc'] = fbmove_cls_acc / step
        eval_results['rlmove_cls_acc'] = rlmove_cls_acc / step
        eval_results['udmove_cls_acc'] = udmove_cls_acc / step
        eval_results['pitch_cls_acc'] = pitch_cls_acc / step
        eval_results['yaw_cls_acc'] = yaw_cls_acc / step

        eval_results['fbmove_bias'] = fbmove_bias*0.4 / step
        eval_results['rlmove_bias'] = rlmove_bias*0.4 / step
        eval_results['udmove_bias'] = udmove_bias*0.4 / step
        eval_results['pitch_bias'] = pitch_bias / step
        eval_results['yaw_bias'] = yaw_bias / step
        print('eval_results:', eval_results)
        return eval_results



    def evaluate(self, results, save_results=True, **kwargs):

        """
        results:
        [{"scene_id": "2141", 
        "start_pos_id": 4458, 
        "preds": [{"step": 0, "position": [8, -8, 10], "look_at": [0, 0, 0]},
                 {"step": 1, "position": [8, -8, 8], "look_at": [7.416310194661958, -7.411957336789462, 7.4400785435539225]},
                 ...]}, 
        """
        # print('EmbodiedCapDataset:', len(results))
        # print('EmbodiedCapDataset:', results)

        if 'pred_actions' in results[0]:
            # if self.pred_result_save_dir != None:
            if self.pred_result_save_dir != None and save_results:
                if results[0].get('random', 'False') == 'True':
                    pred_result_save_name = self.ann_file.split('/')[-1].replace('.json', '_offline-random-pred.json')
                else:
                    pred_result_save_name = self.ann_file.split('/')[-1].replace('.json', '_offline-pred.json')
                pred_result_save_path = self.pred_result_save_dir + '/'+ pred_result_save_name
                json.dump(results, open(pred_result_save_path, 'w', encoding='utf-8'))
                print('save %d offline preds to %s' % (len(results), pred_result_save_path))
            return self.pesudo_evaluate(results)
        else:
            if self.pred_result_save_dir != None and save_results:
                if results[0].get('random', 'False') == 'True':
                    pred_result_save_name = self.ann_file.split('/')[-1].replace('.json', '_online-random-pred.json')
                elif results[0].get('surround', 'False') == 'True':
                    pred_result_save_name = self.ann_file.split('/')[-1].replace('.json', '_online-surround-pred.json')
                elif results[0].get('rule', 'False') == 'True':
                    pred_result_save_name = self.ann_file.split('/')[-1].replace('.json', '_online-rule-pred.json')
                else:
                    pred_result_save_name = self.ann_file.split('/')[-1].replace('.json', '_online-pred.json')
                pred_result_save_path = self.pred_result_save_dir + '/'+ pred_result_save_name
                json.dump(results, open(pred_result_save_path, 'w', encoding='utf-8'))
                print('save %d online preds to %s' % (len(results), pred_result_save_path))
            eval_results = {}

            print(len(results), len(self.data_infos))
            
            assert len(results) == len(self.data_infos)
            postion_samples = []
            image_samples = []
            seg_samples = []
            for i in range(len(results)):
                scene_id = self.data_infos[i]['scene_id']
                positions = self.data_infos[i]['positions']
                path_len = self.data_infos[i]['path_len']
                start_position = positions[0]
                gt_end_positions = self.scene2targets[scene_id]['positions']
                gt_end_images = self.scene2targets[scene_id]['images']
                gt_end_segs = self.scene2targets[scene_id]['segmentations']
                scene_id = self.data_infos[i]['scene_id']

                assert scene_id == results[i]['scene_id']
                assert start_position ==  results[i]['preds'][0]['position']
            
                
                pred = results[i]
                pred_positions = []
                pred_imgs = []
                pred_segs = []
                if len(pred['preds']) == 1:
                    print(pred['preds'])
                for step_pred in pred['preds'][1:]:
                    pred_positions.append(step_pred['position'])
                    pred_imgs.append(step_pred['img_path'])
                    pred_segs.append(step_pred['img_path'].replace('png', 'npz'))
                
                postion_samples.append({'pred_poss':pred_positions,'gt_end_poss':gt_end_positions, 'gt_path_len':path_len-1})
                image_samples.append({'pred_images':pred_imgs,'gt_end_images':gt_end_images, 'gt_path_len':path_len-1})
                seg_samples.append({'pred_segs':pred_segs,'gt_end_segs':gt_end_segs, 'gt_path_len':path_len-1})


            if 'manhattan_distance' in self.eval_metrics:
                mean_manhattan_distances = cal_manhattan_distance(postion_samples, path_len_weight=self.path_len_weight)
                mean_manhattan_distance = mean_scores(mean_manhattan_distances)
                eval_results['mean_navigation_error'] = mean_manhattan_distance * 0.4

            if 'clip_score' in self.eval_metrics:
                mean_nor_clip_scores = cal_clip_similarity(image_samples, path_len_weight=self.path_len_weight)
                eval_results['mean_nor_clip_score'] = mean_scores(mean_nor_clip_scores)

            if 'seg_score' in self.eval_metrics:
                mean_seg_scores = cal_seg_similarity(seg_samples, path_len_weight=self.path_len_weight)
                eval_results['mean_seg_score'] = mean_scores(mean_seg_scores)
                
            for i in range(len(results)):
                results[i]['metrics'] = {}
                if 'manhattan_distance' in self.eval_metrics:
                    results[i]['metrics']['mean_navigation_error'] = mean_manhattan_distances[i]*0.4
                    
                if 'clip_score' in self.eval_metrics:
                    results[i]['metrics']['mean_nor_clip_score'] = mean_nor_clip_scores[i]
                    
                if 'seg_score' in self.eval_metrics:
                    results[i]['metrics']['mean_seg_score'] = mean_seg_scores[i]
                    
            result_metrics_save_name = self.ann_file.split('/')[-1].replace('.json', '_online-pred-metrics.json')
            result_metrics_save_path = self.pred_result_save_dir + '/'+ result_metrics_save_name
            json.dump(results, open(result_metrics_save_path, 'w', encoding='utf-8'))
            print('save each metrics of %d online preds to %s' % (len(results), result_metrics_save_path))

            return eval_results

# for trajectory caption model
@DATASETS.register_module()
class EmbodiedCapTrajCapDataset(Dataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 use_shorter_path=True,
                 pred_result_save_dir=None,
                 max_cap_len = 77,
                 test_mode=False,
                 eval_metrics=None):
                 
        self.CLASSES = None
        self.ann_file = ann_file
        self.data_root = data_root
        self.test_mode = test_mode
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
        # load annotations
        self.load_annotations(self.ann_file, use_shorter_path, max_cap_len)
        # processing pipeline
        self.pipeline = Compose(pipeline)
        self.pred_result_save_dir=pred_result_save_dir
        # filter images too small and containing no annotations
        if not test_mode:
            # set group flag for the sampler
            self._set_group_flag()
        self.eval_metrics = eval_metrics
    
    def _set_group_flag(self):
        """Set flag according to path length
        path length > 10 will be set as group 1, otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            path_info = self.data_infos[i]
            if path_info['path_len'] > 10:
                self.flag[i] = 1
    
    def __len__(self):
        return len(self.data_infos)
    
    def load_annotations(self, ann_file, use_shorter_path, max_cap_len):
        self.embodiedcap = EmbodiedCapTrajCap(ann_file, use_shorter_path, max_cap_len)
        self.pathids = self.embodiedcap.pathids
        self.data_infos = self.embodiedcap.data_infos
        self.scene2targets = self.embodiedcap.scene2targets
        self.gtview2targets = self.embodiedcap.gtview2targets

    def get_ann_info(self, idx):
        """Get embodiedcap annotation by index.
        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        pathid = self.pathids[idx]
        ann = self.embodiedcap.load_anns(pathid)
        return ann
    
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_path(idx)
            """while True:
                data = self.prepare_train_path(idx)
                if data is None:
                    idx = self._rand_another(idx)
                    continue
                return data"""
        else:
            return self.prepare_train_path(idx)

    def prepare_train_path(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        path_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(path_info=path_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.data_root + 'scenes/' + results['path_info']['render_dir']

    def prepare_test_path(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        path_info = self.data_infos[idx]
        results = dict(path_info=path_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def evaluate(self, results, **kwargs):

        """
        results:
        [{"scene_id": "xxx", 
        "pathid": xxxx,
        "pred_caption": xxxx}] 
        """
        
        if self.pred_result_save_dir != None:
            pred_result_save_name = self.ann_file.split('/')[-1].replace('.json', '_pred.json')
            pred_result_save_path = self.pred_result_save_dir + '/'+ pred_result_save_name
            json.dump(results, open(pred_result_save_path, 'w', encoding='utf-8'))
            print('save %d caption preds to %s' % (len(results), pred_result_save_path))

        # for caption evaluation
        eval_results = {}
        gts = []
        preds = []
        pathids = set()
        assert len(results) == len(self.data_infos)
        for i in range(len(results)):
            scene_id = self.data_infos[i]['scene_id']
            pathid = self.data_infos[i]['pathid']
            path_len = self.data_infos[i]['path_len']
            assert scene_id == results[i]['scene_id']
            assert pathid == results[i]['pathid']
            
            gt_caps = list(self.scene2targets[scene_id]['captions'])
            # print('embodiedcap gt_caps:', gt_caps)
            pred_cap = results[i]['pred_caption']
            for gt_cap in gt_caps:
                gts.append({'image_id':pathid, 'caption':gt_cap})
            if pathid not in pathids:
                pathids.add(pathid)
                preds.append({'image_id':pathid, 'caption':pred_cap})

        # caption evaluation
        metrics, img_metrics = coco_caption_eval.calculate_metrics(list(pathids), 
                                                                {'annotations': gts}, 
                                                                {'annotations': preds},
                                                                self.eval_metrics)
        eval_results.update(metrics)
        print('eval_results:', eval_results)
        return eval_results

# for inference navgation result with tarjectory caption only model
@DATASETS.register_module()
class EmbodiedCapTrajCapInferenceDataset(Dataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 pred_result_save_dir=None,
                 test_mode=True,
                 eval_metrics=None,
                 # following parameters are useless,
                 path_len_weight_eval=False,
                 max_cap_len=77,
                 use_shorter_path=True,
                 ):
        self.CLASSES = None
        self.ann_file = ann_file
        self.data_root = data_root
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
        # load annotations
        self.load_annotations(self.ann_file)
        # processing pipeline
        self.pipeline = Compose(pipeline)
        self.pred_result_save_dir=pred_result_save_dir
        # filter images too small and containing no annotations
        self.test_mode = test_mode
        assert test_mode
        self.eval_metrics = eval_metrics
        self.path_len_weight = path_len_weight_eval
        if 'earlystop' in self.ann_file:
            self.early_stop_eval=True
        else:
            self.early_stop_eval=False
    

    def __len__(self):
        return len(self.data_infos)
    
    def load_annotations(self, ann_file):
        self.embodiedcap = EmbodiedCapTrajCapInference(ann_file)
        self.data_infos = self.embodiedcap.data_infos
        self.scene2targets = self.embodiedcap.scene2targets

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        assert self.test_mode
        return self.prepare_test_path(idx)

    
    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = '' # ''

    def prepare_test_path(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        path_info = self.data_infos[idx]
        results = dict(path_info=path_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    def early_stop_metric_cal(self, path_metrics, path_len_weights):
        # path_metrics: metrics of each image, dict (key:image_id, value:dict(metric_name, metric_value))
        # choose a best one according CIDEr
        pathid2best_early_stop_pathid = {}
        for early_stop_pathid in path_metrics.keys(): # e.g. 2153_3045_step0
            CIDEr_value = path_metrics[early_stop_pathid]['CIDEr']
            pathid = '_'.join(early_stop_pathid.split('_')[:2])
            if pathid not in pathid2best_early_stop_pathid:
                pathid2best_early_stop_pathid[pathid] = (early_stop_pathid, CIDEr_value)
            elif pathid2best_early_stop_pathid[pathid][1] < CIDEr_value:
                pathid2best_early_stop_pathid[pathid] = (early_stop_pathid, CIDEr_value)


        best_metric = {}
        best_path_metrics = {}
        best_path_len_weights = {}
        chosen_early_stop_pathids = []
        for pathid in pathid2best_early_stop_pathid.keys():
            best_early_stop_pathid = pathid2best_early_stop_pathid[pathid][0]
            chosen_early_stop_pathids.append(best_early_stop_pathid)
            metrics = path_metrics[best_early_stop_pathid]
            best_path_metrics[pathid] = metrics
            best_path_len_weights[pathid] = path_len_weights[best_early_stop_pathid]
            for metric_name, metric_value in metrics.items():
                if metric_name != 'image_id':
                    if metric_name not in best_metric:
                        best_metric[metric_name] = metric_value
                    else:
                        best_metric[metric_name] += metric_value
        
        path_num = len(pathid2best_early_stop_pathid.keys())
        for metric_name in best_metric.keys():
            best_metric[metric_name] = best_metric[metric_name]/path_num
        return best_metric, best_path_metrics, best_path_len_weights, chosen_early_stop_pathids

    
    def evaluate(self, results, return_each_path_metrics=False, **kwargs):

        """
        results:
        [{"scene_id": "2141", 
        "imgid": ..., 
        "image":...,
        "pred_caption":xxx}, 
        """
        # print('EmbodiedCapDataset:', len(results))
        # print('EmbodiedCapDataset:', results)

        # record path_len and gt_path_len to save in file
        assert len(results) == len(self.data_infos)

        id2result = {}
        for i in range(len(results)):
            path_len = self.data_infos[i]['path_len']
            gt_path_len = self.data_infos[i]['gt_path_len']
            results[i]['pred_path_len'] = path_len
            results[i]['gt_path_len'] = gt_path_len
            id2result[results[i]['pathid']]=results[i]

        if self.pred_result_save_dir != None:
            pred_result_save_name = self.ann_file.split('/')[-1].replace('.json', '_pred.json')
            pred_result_save_path = self.pred_result_save_dir + '/'+ pred_result_save_name
            json.dump(results, open(pred_result_save_path, 'w', encoding='utf-8'), indent=2)
            print('save %d caption preds to %s' % (len(results), pred_result_save_path))
        # eval_results = {}
        
        # for caption evaluation
        # eval_results = {}
        gts = []
        preds = []
        pathids = set()
        assert len(results) == len(self.data_infos)
        path_len_weights = {}
        
        for i in range(len(results)):
            scene_id = self.data_infos[i]['scene_id']
            pathid = self.data_infos[i]['pathid']
            path_len = self.data_infos[i]['path_len']
            gt_path_len = self.data_infos[i]['gt_path_len']
            path_len_weights[pathid] = gt_path_len/(max(gt_path_len, path_len))
            assert scene_id == results[i]['scene_id']
            assert pathid == results[i]['pathid']
            
            gt_caps = list(self.scene2targets[scene_id])
            # print('embodiedcap gt_caps:', gt_caps)
            pred_cap = results[i]['pred_caption']
            for gt_cap in gt_caps:
                gts.append({'image_id':pathid, 'caption':gt_cap})
            if pathid not in pathids:
                pathids.add(pathid)
                preds.append({'image_id':pathid, 'caption':pred_cap})

        # print('eval img num:', len(imgids))
        # metrics: dict
        # "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4","METEOR", "ROUGE_L","CIDEr", "SPICE"
        # path_metrics: metrics of each image, dict (key:image_id, value:dict(metric_name, metric_value)) 
        # 

        metrics, path_metrics = coco_caption_eval.calculate_metrics(list(pathids), 
                                                                    {'annotations': gts}, 
                                                                    {'annotations': preds},
                                                                    self.eval_metrics)
        
        
        if self.early_stop_eval:
            path_metrics_save_name = self.ann_file.split('/')[-1].replace('.json', '_pred_metrics.json')
            path_metrics_save_path = self.pred_result_save_dir + '/'+ path_metrics_save_name
            json.dump(path_metrics, open(path_metrics_save_path, 'w', encoding='utf-8'), indent=2)
            print('save path metrics to %s' % (path_metrics_save_path))

            metrics, path_metrics, path_len_weights, chosen_early_stop_pathids = self.early_stop_metric_cal(path_metrics, path_len_weights)
            # save chosen result for spice evaluation
            chosen_results = []
            for chosen_id in chosen_early_stop_pathids:
                chosen_result = id2result[chosen_id]
                chosen_result['pathid'] = '_'.join(chosen_result['pathid'].split('_')[:2])
                chosen_results.append(chosen_result)

            chosen_result_save_name = self.ann_file.split('/')[-1].replace('.json', '_pred_ciderchosen.json')
            chosen_result_save_path = self.pred_result_save_dir + '/'+ chosen_result_save_name
            json.dump(chosen_results, open(chosen_result_save_path, 'w', encoding='utf-8'), indent=2)
            print('save %d caption preds to %s' % (len(chosen_results), chosen_result_save_path))


                
        if self.path_len_weight:
            print(len(path_len_weights.keys()), len(path_metrics.keys()))
            assert len(path_len_weights.keys()) == len(path_metrics.keys())
            for metric_name in metrics:
                metrics[metric_name] = 0.0

            for path_id in path_metrics.keys():
                weight = path_len_weights[path_id]
                for key in path_metrics[path_id]:
                    # key in ['image_id', "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4","METEOR", "ROUGE_L","CIDEr", "SPICE"]
                    if key != 'image_id':
                        weighted_value = path_metrics[path_id][key]*weight
                        path_metrics[path_id][key] = weighted_value
                        metrics[key]+=weighted_value
            
            for metric_name in metrics.keys():
                metrics[metric_name] = metrics[metric_name]/len(path_metrics.keys())
        
    
        # print(metrics)
        if return_each_path_metrics:
            return metrics, path_metrics
        else:
            return metrics


