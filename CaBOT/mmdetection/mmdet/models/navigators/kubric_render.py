import sys
sys.path.append('/root/code/kubric/')
import logging
import random
import kubric as kb
import kubric_haw as kb_haw
import numpy as np
# from kubric.renderer.blender import Blender as KubricRenderer
from kubric_haw.renderer.blender import Blender as KubricRenderer
from kubric_haw.simulator.pybullet_siegelordex import PyBullet as KubricSimulator
import os
os.environ['KUBRIC_USE_GPU'] = '0'
import time
import copy
import json
import gc
import math

from tqdm import tqdm

DATASET_DIR='/data5/haw/ETCAP/'
SCENES_DIR = DATASET_DIR + 'scenes/'

CATEGORY2SIZE = {
                'base':{
                    # ShapeNet objects
                    'bed': 2.0, # 233
                    'bench':1.2, # 1811
                    # 'birdhouse': 0.4, # 73
                    'bookshelf':1, # 451
                    'bathtub': 1.2, # 856
                    'cabinet':1, # 1571
                    'file':1, # 298
                    'sofa':1.5, # 3172
                    'piano':1.5, # 239
                    'table':1.2, # 8436
                    'washer':1.0, # 169
                    },
                'other':{
                    # ShapeNet objects
                    ## toy objects
                    'airplane': 0.3, # 4044
                    'bus': 0.3, # 937
                    'car':0.2, # 3486
                    'motorcycle':0.2, # 337
                    'train':0.3, # 389
                    'vessel':0.2, # 1935
                    'rocket':0.3, # 85
                    'tower':0.2, # 133
                    'pistol':0.2, # 307
                    'rifle':0.4,# 2373
                    ## container objects
                    'ashcan': 0.4, # 343
                    'bag':0.4, # 83
                    'basket':0.4, # 113
                    'bottle': 0.1, # 498
                    'bowl': 0.2, # 186
                    'can':0.1, # 108
                    'mug':0.1, # 214
                    'pot':0.2, # 601
                    'jar':0.2, # 596
                    ## electric objects
                    'remote control':0.2, # 66
                    'camera':0.2, # 113
                    'cellular telephone':0.16, # 831
                    'telephone':0.16, # 1088
                    'earphone':0.2, # 73
                    'computer keyboard':0.4, # 65
                    'display':0.4, # 1093
                    'laptop':0.4, # 460
                    'loudspeaker':0.3, # 1597
                    'microphone':0.3, # 67
                    'microwave':0.5, # 152
                    'printer':0.4, # 166
                    'dishwasher':0.5, # 92
                    ## others
                    # 'faucet':0.2, # 744 
                    'guitar':1.0, # 797
                    'helmet':0.3, # 162
                    'cap':0.2, # 56
                    'clock':0.3, # 650
                    'chair':0.4, # 6718
                    'knife':0.3, # 424
                    # 'lamp':0.2, # 2318
                    # 'mailbox':0, # 94
                    'pillow':0.4, # 96
                    'skateboard':0.6, # 152
                    # 'stove':0.7, # 218
                    # GSO objects:
                    'Shoe':0.3, # 254
                    'Consumer Goods':0.2, # box with words # 238
                    # 'Toys':0.2, # hard to recognize # 150
                    # 'Bottles and Cans and Cups':0.1, # 53
                    'Legos':0.3, # 10
                    'Media Cases':0.2, # box with words # 21
                    'Action Figures':0.1, # 17
                    'Board Games':0.2, # box with words # 17
                    'Bag':0.4, # 28
                    # 'Hat':0.2, # 2
                    # 'Headphones':0.2, # box with words # 4
                    # 'Keyboard':0.3, # 4
                    'Stuffed Toys':0.2, # 3
                    # 'Camera':0.1, # lens # 1
                    # 'Car Seat':0.4, # 1
                    # 'Mouse':0.1, # 4
                    },
                }

def rescaled_object(asset_id, object, asset, base_object=False):
    raw_obj = asset.create(asset_id=asset_id, scale=(1, 1, 1))
    if base_object:
        role = 'base'
        raw_obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
    else:
        role = 'other'
    # print('raw aabbox:', raw_obj.aabbox)
    if object in ['bookshelf', 'cabinet', 'file']:
        x_length = raw_obj.aabbox[1][0] - raw_obj.aabbox[0][0] # x_max - x_min
        y_length = raw_obj.aabbox[1][1] - raw_obj.aabbox[0][1] # y_max - y_min
        z_length = raw_obj.aabbox[1][2] - raw_obj.aabbox[0][2] # y_max - y_min
        xy_scale = CATEGORY2SIZE[role][object] / max(x_length, y_length)
        z_scale = 2.0 / z_length
        scale = min(xy_scale, z_scale) # avoid high > 2.0
    elif object in ['bed', 'bench', 'bathtub','sofa', 'piano']:
        x_length = raw_obj.aabbox[1][0] - raw_obj.aabbox[0][0] # x_max - x_min
        y_length = raw_obj.aabbox[1][1] - raw_obj.aabbox[0][1] # y_max - y_min
        scale = CATEGORY2SIZE[role][object] / max(x_length, y_length)
    elif object in ['Shoe', 'knife', 'microphone', 'remote control', 'rifle', 'pistol', 'tower', 
                    'cellular telephone', 'telephone',
                    'bus', 'car', 'train', 'vessel', 'airplane', 'motorcycle', 'rocket', 
                    'skateboard', 'guitar', 'washer', 'table']:
        x_length = raw_obj.aabbox[1][0] - raw_obj.aabbox[0][0] # x_max - x_min
        y_length = raw_obj.aabbox[1][1] - raw_obj.aabbox[0][1] # y_max - y_min
        z_length = raw_obj.aabbox[1][2] - raw_obj.aabbox[0][2] # z_max - z_min
        scale = CATEGORY2SIZE[role][object] /  max(x_length, y_length, z_length)
    else:
        x_length = raw_obj.aabbox[1][0] - raw_obj.aabbox[0][0] # x_max - x_min
        scale = CATEGORY2SIZE[role][object] / x_length

    # print(scale)
    rescaled_obj = asset.create(asset_id=asset_id, scale=(scale, scale, scale))
    if base_object:
        rescaled_obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
    # print('rescaled aabbox:', rescaled_obj.aabbox)
    return rescaled_obj


class NavigationKubricRenderer():
    def __init__(self, scene_id, cache_dir='tmp/navigation_inference/', render_frame=41, 
                save_dir='tmp/evaluate/', keep_rendered_imgs=False):
        self.scene_dir = SCENES_DIR + str(scene_id) + '/'
        self.scene = kb.Scene(resolution=(256, 256), frame_start=1, frame_end=render_frame)
        self.blend_path = self.scene_dir +'keyframing.blend'
        self.scratch_dir = DATASET_DIR+cache_dir
        self.renderer = KubricRenderer(self.scene, custom_scene=self.blend_path, scratch_dir=self.scratch_dir) # next render_frames are used to take photos from multiple views
        # reset a camera
        self.scene += kb.PerspectiveCamera(name="camera", position=(2, 0, 0), look_at=(0, 0, 1))
        self.render_frame = render_frame
        # self.save_dir = self.scene_dir + save_dir
        self.save_dir = DATASET_DIR + save_dir
        self.load_grid()
        self.keep_rendered_imgs = keep_rendered_imgs
    
    def load_grid(self):
        grid_info_path = self.scene_dir +'navigation_space_info.json'
        grid_info = json.load(open(grid_info_path, 'r', encoding='utf-8'))
        id2coor = grid_info['id2coordinates']
        self.unit2coor = {}
        for pos_id, info in id2coor.items():
            self.unit2coor[str(info['unit'])] = info['coor']

    def render(self, position, look_at, image_name):
        """if str(position) in self.unit2coor:
            self.scene.camera.position = self.unit2coor[str(position)]
        else:"""
        self.scene.camera.position = np.array(position)*0.4 + np.array([0,0,0.1])
        self.scene.camera.look_at(np.array(look_at)*0.4+np.array([0,0,0.1]))
        self.scene.camera.keyframe_insert("position", self.render_frame)
        self.scene.camera.keyframe_insert("quaternion", self.render_frame)
         # --- render the data
        # print('rendering...')
        # start_time = time.time()
        # data_stack = self.renderer.render(range(self.render_frame, self.render_frame+1))
        img_path = self.renderer.render_single_rgba(frame=self.render_frame, target_dir=self.save_dir, image_name=image_name)
        # end_time = time.time()
        # print('render cost: %.4f seconds each frame' % ((end_time-start_time)/render_frames))
        # --- save output files
        # kb.file_io.write_rgba_batch(data_stack["rgba"], kb.as_path(self.save_dir))
        # img_path = self.save_dir+'rgba_00000.png'
        # print('NavigationKubricRenderer img_path:', img_path)
        return img_path
    
    def close(self):
        if os.path.exists(self.scratch_dir):
            os.system('rm -r '+self.scratch_dir)
        if not self.keep_rendered_imgs:
            if os.path.exists(self.save_dir):
                os.system('rm -r '+self.save_dir)
        kb.done()
        gc.collect()


class NavigationKubricSimulateRenderer():
    def __init__(self, scene_id, cache_dir='tmp/navigation_inference/', asset_infos=None,
                simulator_frames=40, 
                save_dir='tmp/evaluate/', keep_rendered_imgs_and_segs=False,
                shapenet=None, gso=None):

        self.save_dir = DATASET_DIR + save_dir
        # self.load_grid()
        self.keep_rendered_imgs_and_segs = keep_rendered_imgs_and_segs
        self.simulator_frames = simulator_frames
        self.render_frame = simulator_frames+1
        self.scene_dir = SCENES_DIR + str(scene_id) + '/'
        # add 1 frame used for rendering each predicted step, simulate only $simulator_frames frames
        self.scene = kb.Scene(resolution=(256, 256), frame_start=1, frame_end=simulator_frames+1)
        self.scratch_dir = DATASET_DIR+cache_dir

        self.simulator = KubricSimulator(self.scene, scratch_dir=self.scratch_dir) 
        self.renderer = KubricRenderer(self.scene, scratch_dir=self.scratch_dir)


        """asset_dir = DATASET_DIR+'kubric_assets/'
        assert os.path.exists(asset_dir)
        shapenet = kb_haw.AssetSource.from_manifest(DATASET_DIR+'ShapeNetCore.v2.json', scratch_dir=asset_dir, random_name=False)
        gso = kb_haw.AssetSource.from_manifest(DATASET_DIR+"GSO.json",scratch_dir=asset_dir, random_name=False)"""
        """shapenet = kb.AssetSource.from_manifest(DATASET_DIR+'ShapeNetCore.v2.json', scratch_dir=self.scratch_dir)
        gso = kb.AssetSource.from_manifest(DATASET_DIR+"GSO.json", scratch_dir=self.scratch_dir)"""
        """shapenet = kb.AssetSource.from_manifest(DATASET_DIR+'ShapeNetCore.v2.json')
        gso = kb.AssetSource.from_manifest(DATASET_DIR+"GSO.json")"""

        # simulate
        # --- populate the scene with objects, lights, cameras
        # scene += kb.Sphere(name="floor", scale=1000, position=(0, 0, +1000), background=True)
        floor = kb.Cube(name="floor", scale=(100, 100, 0.1), position=(0, 0, -0.05), static=True)
        self.scene += floor
        # scene += kb.Cube(name="floor", scale=(.5,.7,1.0), position=(0, 0, 1.1))
        self.scene += kb.PerspectiveCamera(name="camera", position=(2, 0, 3), look_at=(0, 0, 0))
        # --- Add Klevr-like lights to the scene
        self.scene += kb.assets.utils.get_clevr_lights()
        self.scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

        self.objects = [] # for saving meta info of all objetcs in the scene, such as position
        # adding objects
        # print('# scene id:', scene_id)
        # print('## base object:', base_obj['category'], 'asset_id:', base_obj['asset_id'])
        base_obj = rescaled_object(asset_infos['base']['asset_id'], asset_infos['base']['category'], shapenet, base_object=True)
        base_obj.position = base_obj.position - (0, 0, base_obj.aabbox[0][2]) # (0,0,0.056291)
        self.scene += base_obj
        self.objects.append(base_obj)

        for other_obj in asset_infos['other']:
            # print('## other object:', other_obj['category'], 'asset_id:', other_obj['asset_id'])
            if other_obj['source']=='shapenet':
                source = shapenet
            elif other_obj['source'] == 'gso':
                source = gso
            else:
                print('invalid asset source:', other_obj['source'])
                exit(0)
            obj = rescaled_object(other_obj['asset_id'], other_obj['category'], source, base_object=False)
            obj.position = obj.position - (0, 0, obj.aabbox[0][2]) + (0, 0, base_obj.aabbox[1][2]) + (0, 0, 0.5)
            self.scene += obj
            self.objects.append(obj)

        # --- executes the simulation
        self.simulator.run(frame_start=1, frame_end=simulator_frames)

        self.object_3d_info = kb.get_instance_info(self.scene, self.objects)
        self.value2category = {}
        for i, object_info in enumerate(self.object_3d_info):
            self.value2category[i+1] = object_info['category']
    

    def get_surrounding_path(self):
        objects_3dbboxes = []
        for object_3d_info in self.object_3d_info:
            all_frames_3dbboxes = object_3d_info['bboxes_3d']
            # print(len(all_frames_3dbboxes))
            objects_3dbboxes.append(all_frames_3dbboxes[self.simulator_frames-1])

        points = abs(np.array(objects_3dbboxes).reshape(-1, 3))
        x_length, y_length, z_length = points.max(axis=0)
        start_z_coor = math.ceil((z_length-0.1+0.5)/0.4) # +0.5 to ensure a high-quality view
        start_x_coor = math.ceil((x_length+0.1)/0.4) # +0.1 to avoid camera position is overlaped with objects
        start_y_coor = math.ceil((y_length+0.1)/0.4) # +0.1 to avoid camera position is overlaped with objects
        start_position = np.array([start_x_coor, start_y_coor, start_z_coor])
        # do a horizontal and rectangular surrounding from start position to start position
        path_list = []
        path_list.append(start_position)
        tmp_x = start_x_coor
        tmp_y = start_y_coor
        while tmp_y > -start_y_coor:
            tmp_y -=1
            path_list.append(np.array([tmp_x, tmp_y, start_z_coor]))
        while tmp_x > -start_x_coor:
            tmp_x-=1
            path_list.append(np.array([tmp_x, tmp_y, start_z_coor]))
        while tmp_y < start_y_coor:
            tmp_y+=1
            path_list.append(np.array([tmp_x, tmp_y, start_z_coor]))
        while tmp_x < start_x_coor-1:
            tmp_x+=1
            path_list.append(np.array([tmp_x, tmp_y, start_z_coor]))
        
        path_list_shorter = [path_list[i] for i in range(len(path_list)) if i%2==0]
        # print('max(x,y,z):',x_length, y_length, z_length, 'start position:', start_position, ' surround path len:', len(path_list))

        return path_list_shorter


    def load_grid(self):
        grid_info_path = self.scene_dir +'navigation_space_info.json'
        grid_info = json.load(open(grid_info_path, 'r', encoding='utf-8'))
        id2coor = grid_info['id2coordinates']
        self.unit2coor = {}
        for pos_id, info in id2coor.items():
            self.unit2coor[str(info['unit'])] = info['coor']

    def render(self, position, look_at, image_name):
        """if str(position) in self.unit2coor:
            self.scene.camera.position = self.unit2coor[str(position)]
        else:"""
        self.scene.camera.position = np.array(position)*0.4 + np.array([0,0,0.1])
        self.scene.camera.look_at(np.array(look_at)*0.4+np.array([0,0,0.1]))
        self.scene.camera.keyframe_insert("position", self.render_frame)
        self.scene.camera.keyframe_insert("quaternion", self.render_frame)
         # --- render the data
        # print('rendering...')
        # start_time = time.time()
        # data_stack = self.renderer.render(range(self.render_frame, self.render_frame+1))
        # img_path = self.renderer.render_single_rgba(frame=self.render_frame, target_dir=self.save_dir, image_name=image_name)
        img_path, segmentation = self.renderer.render_single_rgba(frame=self.render_frame,
                                                             target_dir=self.save_dir, 
                                                             image_name=image_name,
                                                             return_segmentation=True)
        segmentation = kb.adjust_segmentation_idxs(
            segmentation,
            self.scene.assets,
            self.objects, # there are actually n+1 objects in foreground_assets (including the floor)
            ).astype(np.uint8)
        segmentation = np.squeeze(segmentation) # 256*256
        seg_npz_path = self.save_dir + image_name+'.npz'
        np.savez(seg_npz_path, segmentation=segmentation, value2category=self.value2category)
        
        # end_time = time.time()
        # print('render cost: %.4f seconds each frame' % ((end_time-start_time)/render_frames))
        # --- save output files
        # kb.file_io.write_rgba_batch(data_stack["rgba"], kb.as_path(self.save_dir))
        # img_path = self.save_dir+'rgba_00000.png'
        # print('NavigationKubricRenderer img_path:', img_path)
        return img_path
    
    def close(self):
        # when loading assets, run kb.done before rm dir
        if os.path.exists(self.scratch_dir):
            os.system('rm -r '+self.scratch_dir)
           # os.system('rm -r /tmp/tmp*')
        if not self.keep_rendered_imgs_and_segs:
            if os.path.exists(self.save_dir):
                os.system('rm -r '+self.save_dir)

        kb.done()
        gc.collect()


