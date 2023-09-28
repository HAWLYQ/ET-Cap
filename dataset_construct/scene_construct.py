# Copyright 2021 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
sys.path.append('../')

import logging
import random
import kubric as kb
import numpy as np
from kubric.renderer.blender import Blender as KubricRenderer
from kubric.simulator.pybullet import PyBullet as KubricSimulator
import os
import time
import copy
import json
import gc

from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

from tqdm import tqdm
from utils import vector_angle

DATASET_DIR='/data5/haw/ActiveCap/'

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

def generate_category_info():
    shapenet = kb.AssetSource.from_manifest('ShapeNetCore.v2.json')
    gso = kb.AssetSource.from_manifest("GSO.json")
    category2info = CATEGORY2SIZE
    
    category2num = {}
    for name, spec in gso._assets.items():
        category = spec["metadata"]["category"]
        category2num[category] = category2num.get(category, 0)+1
    for name, spec in shapenet._assets.items():
        category = spec["metadata"]["category"]
        category2num[category] = category2num.get(category, 0)+1

    for role, categories in category2info.items():
        for category, value in categories.items():
            category2info[role][category] = {'scale':value, 'num':category2num[category]}
    
    json.dump(category2info, open('category2info.json', 'w', encoding='utf-8'), indent=2)
    

def sample_object(start_id=0, scene_num=10000, save_path=''):
    save_path = DATASET_DIR+save_path
    # other object num distribution: 2,6:500, 3,5:2000, 4:5000
    object_nums = [2]*int(0.05*scene_num) + [3]*int(0.2*scene_num) + [4]*int(0.5*scene_num) +[5]*int(0.2*scene_num) + [6]*int(0.05*scene_num)
    print('len:', len(object_nums), 'mean:', np.mean(object_nums), 'std:', np.sqrt(np.var(object_nums)))
    random.shuffle(object_nums)
    category2info = json.load(open('category2info.json', 'r', encoding='utf-8'))

    base_category2ratio = {}
    
    for category, info in category2info['base'].items():
        base_category2ratio[category] = info['num']
    
    # calculate sample ratio for each base category according asset number
    base_category_sum = sum(base_category2ratio.values())
    for category, num in base_category2ratio.items():
        base_category2ratio[category] = num/base_category_sum

    print(base_category2ratio)
    base_categories = list(base_category2ratio.keys())
    base_categorieis_ratio = list(base_category2ratio.values())

    # calculate sample ratio for each placing category according asset number
    other_category2ratio = {}
    for category, info in category2info['other'].items():
        other_category2ratio[category] = min(info['num'], 1000) # set a max number 1000 to avoid ratio of a category too big

    other_category_sum = sum(other_category2ratio.values())
    for category, num in other_category2ratio.items():
        other_category2ratio[category] = num/other_category_sum
    print(other_category2ratio)
    other_categories = list(other_category2ratio.keys())
    other_categories_ratio = list(other_category2ratio.values())

    shapenet = kb.AssetSource.from_manifest('ShapeNetCore.v2.json')
    gso = kb.AssetSource.from_manifest("GSO.json")
    category2assets = {}
    for assetid, spec in gso._assets.items():
        category = spec["metadata"]["category"]
        if category not in category2assets:
            category2assets[category] = []
        category2assets[category].append(assetid)
    for assetid, spec in shapenet._assets.items():
        category = spec["metadata"]["category"]
        if category not in category2assets:
            category2assets[category] = []
        category2assets[category].append(assetid)
    
    scene_infos = []
    for i in tqdm(range(scene_num)):
        # choose a base object
        scene_info = {'scene_id':i+start_id}
        sampled_base_category = np.random.choice(base_categories, p = np.array(base_categorieis_ratio))
        sampled_other_categories = np.random.choice(other_categories, p = np.array(other_categories_ratio), size=object_nums[i])
        base_asset_id = np.random.choice(category2assets[sampled_base_category])
        scene_info['base'] = {'category':sampled_base_category, 'asset_id':base_asset_id, 'source':'shapenet'}
        scene_info['other'] = []
        for c in sampled_other_categories:
            asset_id = np.random.choice(category2assets[c])
            if c.istitle():
                source = 'gso'
            else:
                source = 'shapenet'
            scene_info['other'].append({'category':c, 'asset_id':asset_id, 'source':source})

        scene_infos.append(scene_info)
    
    json.dump(scene_infos, open(save_path,'w', encoding='utf-8'), indent=2)
    print('scene info save to ', save_path)




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

def loading_objects_3dbboxes(meta_path, target_frame):
    meta_info = json.load(open(meta_path, 'r', encoding='utf-8'))
    objects_info = meta_info['object']
    categories = []
    objects_3dbboxes = []
    for object_info in objects_info:
        categories.append(object_info['category'])
        all_frames_3dbboxes = object_info['bboxes_3d']
        # print(len(all_frames_3dbboxes))
        objects_3dbboxes.append(all_frames_3dbboxes[target_frame-1])
    return objects_3dbboxes, categories

def point_within_regions(point=None, delaunaies=None):
    # judge whether a point is in object regions

    for delaunay in delaunaies:
        if delaunay.find_simplex(point) >= 0:
            return True
    return False


    
def build_navigation_3d_grid(objects_region=None, max_pos_boundry=4.0, min_step=0.2, z_offset=0.1):

    points = abs(np.array(objects_region).reshape(-1, 3))
    max_x, max_y, max_z = points.max(axis=0) 

    pos_step_num = int(max_pos_boundry / min_step) # 20
    # print('radius %f, min_step %f, pos_step_num %d' % (radius, min_step, pos_step_num))
    objects_delaunaies = []
    for region in objects_region:
        objects_delaunaies.append(Delaunay(np.array(region)))
    print('object num:', len(objects_delaunaies))
    id2coor = {}
    id = 0
    # a circumscribed rectangle of the semicircle
    for z in range(0, pos_step_num+1):
        for y in range(-pos_step_num, pos_step_num+1):
            for x in range(-pos_step_num, pos_step_num+1):
                coor = [round(x*min_step, 4), round(y*min_step, 4), round(z*min_step+z_offset, 4)]
                if not point_within_regions(np.array(coor), objects_delaunaies):
                    id2coor[id] = {'coor':coor, 'unit':[x, y, z]}
                    id += 1
                """else:
                    print('occupied point:', coor)"""
    print('point num:', len(id2coor.keys()))
    return id2coor, max_x, max_y, max_z


def create_scene_and_simulate(scene_id, base_obj, other_objs, shapenet, gso, save_path, simulator_frames=40):
    # x+:forward y+: right; z+:up
    # --- create scene and attach a renderer to it
    # simulator_frames = 20
    scene = kb.Scene(resolution=(512, 512), frame_start=1, frame_end=simulator_frames)
    scene.frame_rate = 24  # < rendering framerate
    scene.step_rate = 240  # < simulation framerate
    simulator = KubricSimulator(scene) # first simulator_frames are used to place multiple objects
    renderer = KubricRenderer(scene) # next render_frames are used to take photos from multiple views

    # --- populate the scene with objects, lights, cameras
    # scene += kb.Sphere(name="floor", scale=1000, position=(0, 0, +1000), background=True)
    floor = kb.Cube(name="floor", scale=(100, 100, 0.1), position=(0, 0, -0.05), static=True)
    scene += floor
    # scene += kb.Cube(name="floor", scale=(.5,.7,1.0), position=(0, 0, 1.1))
    scene += kb.PerspectiveCamera(name="camera", position=(2, 0, 3), look_at=(0, 0, 0))
    # --- Add Klevr-like lights to the scene
    scene += kb.assets.utils.get_clevr_lights()
    scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

    # adding objects
    objects = [] # for saving meta info of all objetcs in the scene, such as position
    print('# scene id:', scene_id)
    print('## base object:', base_obj['category'], 'asset_id:', base_obj['asset_id'])
    base_obj = rescaled_object(base_obj['asset_id'], base_obj['category'], shapenet, base_object=True)
    base_obj.position = base_obj.position - (0, 0, base_obj.aabbox[0][2]) # (0,0,0.056291)
    scene += base_obj
    objects.append(base_obj)

    for other_obj in other_objs:
        print('## other object:', other_obj['category'], 'asset_id:', other_obj['asset_id'])
        if other_obj['source']=='shapenet':
            source = shapenet
        elif other_obj['source'] == 'gso':
            source = gso
        else:
            print('invalid asset source:', other_obj['source'])
            exit(0)
        obj = rescaled_object(other_obj['asset_id'], other_obj['category'], source, base_object=False)
        obj.position = obj.position - (0, 0, obj.aabbox[0][2]) + (0, 0, base_obj.aabbox[1][2]) + (0, 0, 0.5)
        scene += obj
        objects.append(obj)

    # --- executes the simulation
    simulator.run(frame_start=1, frame_end=simulator_frames)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # --- save scene for quick inspection
    renderer.save_state(save_path+"/keyframing.blend")
    
    # --- save simulate process (optional, not necessary)
    """data_stack = renderer.render()
    kb.file_io.write_rgba_batch(data_stack["rgba"][:simulator_frames], kb.as_path(save_path+"/simulate/"))
    os.system('convert -delay 8 -loop 0 '+save_path+'/simulate/rgba_*.png '+save_path+'/simulate/simulator.gif')"""

    # --- Collect metadata
    # Collecting and storing metadata for each object
    data = {
        "metadata": kb.get_scene_metadata(scene),
        "camera": kb.get_camera_info(scene.camera),
        "object": kb.get_instance_info(scene, objects)
    }
    kb.file_io.write_json(filename=kb.as_path(save_path) / "metadata.json", data=data)
    print('save scene to ', save_path)
    # kb.done() # done() will remove files in /tmp/ShapeNetCore.v2*/ or /tmp/GSO.v2*/
    # obj 3d files will be saved to /tmp/ShapeNetCore.v2*/ or /tmp/GSO.v2*/
    # other tmp dirs /tmp/tmp*/
    os.system('rm -r /tmp/*')
    gc.collect()


def create_scenes(scene_info_path, start=0, end=10000, simulator_frames=40, reset=False):
    dataset_path = DATASET_DIR+'scenes'
    scene_info_path = DATASET_DIR+scene_info_path
    # remove pre-constructed scenes
    if reset:
        os.system('rm -r '+dataset_path+'*')
    # --- Fetch shapenet and gso
    shapenet = kb.AssetSource.from_manifest('ShapeNetCore.v2.json')
    gso = kb.AssetSource.from_manifest("GSO.json")
    scene_infos = json.load(open(scene_info_path, 'r', encoding='utf-8'))
    missing_num = 0
    for scene_info in tqdm(scene_infos[start:end]):
        scene_id = scene_info['scene_id']
        base_obj = scene_info['base']
        other_objs = scene_info['other']
        save_path = dataset_path + '/' + str(scene_id)
        if os.path.exists(save_path+'/keyframing.blend'):
            continue
        try:
            create_scene_and_simulate(scene_id, base_obj, other_objs, shapenet, gso, save_path, simulator_frames)
        except Exception as e:
            print(str(e))
            missing_num+=1
            print('missing num:', missing_num)
            continue
        gc.collect()
    print('missing num:', missing_num)


def clean_scenes(scene_info_paths, new_scene_info_path):
    dataset_path = DATASET_DIR+'scenes'
    new_scene_info_path = DATASET_DIR+new_scene_info_path
    
    cleaned_scenes = []
    for scene_info_path in scene_info_paths:
        scene_info_path = DATASET_DIR+scene_info_path
        scene_infos = json.load(open(scene_info_path, 'r', encoding='utf-8'))
        for scene_info in tqdm(scene_infos):
            scene_id = scene_info['scene_id']
            scene_path =  dataset_path + '/' + str(scene_id) + '/keyframing.blend'
            meta_path = dataset_path + '/' + str(scene_id) + '/metadata.json'
            if os.path.exists(meta_path) and os.path.exists(scene_path):
                objects_3dbboxes, _ = loading_objects_3dbboxes(meta_path, target_frame=40)
                # radius of each axis
                points = abs(np.array(objects_3dbboxes).reshape(-1, 3))
                x_length, y_length, z_length = points.max(axis=0) 
                # scene can't be too wide and too high
                if x_length <= 3.0 and y_length <= 3.0 and z_length <= 3.0:
                    cleaned_scenes.append(scene_info)

    print('cleaned scene num:', len(cleaned_scenes))
    json.dump(cleaned_scenes, open(new_scene_info_path, 'w', encoding='utf-8'), indent=2)
    print('save to ', new_scene_info_path)


def random_render(scene_dir, pre_simulate_frames=40, render_frames=20, max_pos_boundry=4.0, min_step=0.2, z_offset=0.1, save_dir='render', cache_dir='tmp/render_cache/'):
    print('render scene:', scene_dir)
    # --- create scene and attach a renderer to it
    scene = kb.Scene(resolution=(512, 512), frame_start=1, frame_end=pre_simulate_frames+render_frames)
    blend_path = scene_dir+'/keyframing.blend'
    scratch_dir = DATASET_DIR+cache_dir
    if not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)
    renderer = KubricRenderer(scene, custom_scene=blend_path, scratch_dir=scratch_dir) # next render_frames are used to take photos from multiple views
    # reset a camera
    scene += kb.PerspectiveCamera(name="camera", position=(2, 0, 0), look_at=(0, 0, 1))
    objects_3dbboxes, categories = loading_objects_3dbboxes(scene_dir+'/metadata.json', target_frame=pre_simulate_frames)
    print('contained categories: ', categories)
    id2coor, max_x, max_y, max_z = build_navigation_3d_grid(objects_3dbboxes, max_pos_boundry, min_step, z_offset)

    print('max x:%.1f, max y:%.1f, max z:%.1f' % (max_x, max_y, max_z))
    radius = max(max_x, max_y, max_z)
    min_radius = radius
    max_radius = radius+1.0
    print('min radius:%.1f, max radius:%.1f' % (min_radius, max_radius))

    coor_candidates = list(id2coor.keys())
    render_points = []
    for frame in range(pre_simulate_frames+1, pre_simulate_frames+render_frames+1):
        coor_id = random.choice(coor_candidates)
        coor = id2coor[coor_id]['coor']
        # min_radius <= x <= max_radius
        # min_radius <= y <= max_radius
        # z >= max_z + 0.5
        # if the chosen point is too close or too far, re choose a point
        while abs(coor[0]) < max_x or abs(coor[0]) > max_x+1.0 or abs(coor[1]) < max_y or abs(coor[1]) > max_y+1.0 or abs(coor[2]) < max_z+0.5:
        # while abs(coor[0]) < min_radius or abs(coor[0]) > max_radius or abs(coor[1]) < min_radius or abs(coor[1]) > max_radius or abs(coor[2]) < max_z+0.5:
        # while np.linalg.norm(np.array(coor)) < min_radius or np.linalg.norm(np.array(coor)) > max_radius or abs(coor[2]) < max_z+1.0:
            coor_id = random.choice(coor_candidates)
            coor = id2coor[coor_id]['coor']
        render_points.append(coor_id)
        # coor = [2, 2, 2]
        # print('frame:', frame)
        print('chosen point:', (frame-pre_simulate_frames), coor_id, coor)
        scene.camera.position = coor
        scene.camera.look_at((0, 0, 0))
        scene.camera.keyframe_insert("position", frame)
        scene.camera.keyframe_insert("quaternion", frame)

    # --- render the data
    start_time = time.time()
    data_stack = renderer.render(range(pre_simulate_frames+1, pre_simulate_frames+render_frames+1))
    print('len(data_stack[rgba])', len(data_stack["rgba"]))
    end_time = time.time()
    print('render cost: %.4f seconds each frame' % ((end_time-start_time)/render_frames))
    # --- save output files
    kb.file_io.write_rgba_batch(data_stack["rgba"], kb.as_path(scene_dir+"/"+save_dir+"/"))
    os.system('rm -r '+scratch_dir+'*')
    json.dump({'id2coordinates':id2coor, 'chosen_ids':render_points}, open(scene_dir+"/navigation_space_info.json", 'w', encoding='utf-8'))
    kb.done()
    # gc.collect()

def render_scenes(scene_info_path, start=0, end=10000, pre_simulate_frames=40, render_frames=20, max_pos_boundry=4.0, min_step=0.2, z_offset=0.1, reset=False):
    
    dataset_path = DATASET_DIR+'scenes'
    scene_info_path = DATASET_DIR+scene_info_path
    scene_infos = json.load(open(scene_info_path, 'r', encoding='utf-8'))
    print('max_pos_boundry %f, min_step %f, pos_step_num %d, z_offset %f' % (max_pos_boundry, min_step, int(max_pos_boundry/min_step), z_offset))
    missing_num = 0
    cache_dir = 'tmp/'+str(start)+'-'+str(end)+'_render_cache/'
    for scene_info in tqdm(scene_infos[start:end]):
        scene_id = scene_info['scene_id']
        scene_dir = dataset_path + '/' + str(scene_id)
        if not os.path.exists(scene_dir):
            missing_num+=1
            continue
        if os.path.exists(scene_dir+"/navigation_space_info.json") and not reset:
            continue
        try:
            random_render(scene_dir, pre_simulate_frames, render_frames, max_pos_boundry, min_step, z_offset, save_dir='chosen_views', cache_dir=cache_dir)
        except Exception as e:
            print(str(e))
            missing_num+=1
            continue
        # gc.collect()
    print('missing num:', missing_num)



if __name__ == '__main__':
    ## steps for scene construction

    # 1.identify scale and number of each category
    generate_category_info() 

    # 2.sample objects for each scene, store asset id for each scene
    sample_object(start_id=0, scene_num=12000, save_path='scene_object_info_12k.json')
    
    # 3.constuct scene according pre-stored scene info
    # create_scenes(scene_info_path='scene_object_info_1w.json', start=0, end=10000, simulator_frames=40, reset=False)
    clean_scenes(scene_info_paths=['scene_object_info_12k.json'], new_scene_info_path='activecap_scenes_v0.json')

    # 4.random render 20 views for goodview annotation
    render_scenes(scene_info_path='activecap_scenes_v0.json', start=0, end=1, max_pos_boundry=4.0, min_step=0.4, reset=True)












