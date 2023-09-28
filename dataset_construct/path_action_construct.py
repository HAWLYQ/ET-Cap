import sys
sys.path.append('../')
import os
import json
import random
import numpy as np
from utils import vector_angle, rotate_action, move_action, relative_move_action
from tqdm import tqdm
import time
import gc
from scene_construct import loading_objects_3dbboxes, point_within_regions
from scipy.spatial import Delaunay

import kubric as kb
from kubric_haw.renderer.blender import Blender as KubricRenderer
import multiprocessing
import math
import argparse
import cv2

# revise to your own dataset directory
DATASET_DIR='/data5/haw/ETCAP/' 
os.environ['KUBRIC_USE_GPU'] = '0'

def filter_points(scene_id):
    scene_dir = DATASET_DIR + 'scenes/'+scene_id
    objects_3dbboxes, categories = loading_objects_3dbboxes(scene_dir+'/metadata.json', target_frame=40)
    obj_points = np.array(objects_3dbboxes).reshape(-1, 3)
    # radius = points.max()
    max_x, max_y, max_z = obj_points.max(axis=0) 
    min_x, min_y, min_z = obj_points.min(axis=0) 
    print('==========scene %s==========' % scene_id)
    print('max(x, y, z)', max_x, max_y, max_z)
    print('min(x, y, z)', min_x, min_y, min_z)
    objects_delaunaies = []
    for region in objects_3dbboxes:
        objects_delaunaies.append(Delaunay(np.array(region)))
    

    print('object num:', len(objects_delaunaies))
    navigation_info_path = scene_dir + "/navigation_space_info.json"
    assert os.path.exists(navigation_info_path)
    navigation_info = json.load(open(navigation_info_path, 'r', encoding='utf-8'))
    id2coor = navigation_info['id2coordinates']
    accesible_points = []
    for id, info in id2coor.items():
        coor = info['coor']
        if not point_within_regions(np.array(coor), objects_delaunaies):
            accesible_points.append(id)
    print('raw num:', len(id2coor.keys()), 'acce num:', len(accesible_points))
    return accesible_points


def dijkstra_shortest_path(src_id, id2coor=None):
    """
    get the shortest path from the soruce point to other points O(n^2)
    src_id: int
    """
    coor_num = len(id2coor.keys())
    max_dis = coor_num
    distance_path_list = []    
    # initialize the distance_list
    # print('initializing distance list...')
    visited = []
    unvisited = []

    for coor_id in id2coor.keys():
    # for coor_id in tqdm(id2coor.keys()):
        # each point is represented as a tuple where
        # tuple[0]: id (int)
        # tuple[1]: the min distance for src -> i
        # tuple[2]=k means the shortest path for src->i can be decomposed as src-> k -> i 
        init_tuple = [int(coor_id), max_dis, -1] 
        if int(coor_id)== int(src_id):
            init_tuple[1] = 0.0
            init_tuple[2] = src_id
            visited.append(init_tuple)
        else:
            distance = abs(np.array(id2coor[coor_id]['unit'])-np.array(id2coor[str(src_id)]['unit'])).sum()
            assert distance >= 1
            if distance == 1:
                init_tuple[1] = 1
                init_tuple[2] = src_id 
            distance_path_list.append(init_tuple)
            unvisited.append(init_tuple)


    # iterates all unvisited points as a middle point O(n^2)
    # print('updating distance list for ', src_id)
    for step in range(coor_num-1):
    # for step in tqdm(range(coor_num-1)):
        unvisited = sorted(unvisited, key=lambda t: t[1])
        link_point = unvisited.pop(0)
        visited.append(link_point)
        if len(unvisited) > 0:
            for target_point in unvisited:
                src2link_dis = link_point[1]

                link2tar_3d_dis = abs(np.array(id2coor[str(link_point[0])]['unit'])-np.array(id2coor[str(target_point[0])]['unit'])).sum()
                assert link2tar_3d_dis >= 1

                # link2tar_3d_dis == 1 means link point and target point are adjacent
                if link2tar_3d_dis == 1:
                    link2tar_dis = 1
                else:
                    # set a max distance for unadjacent points
                    link2tar_dis = max_dis
                
                if src2link_dis + link2tar_dis < target_point[1]:
                    target_point[1] = src2link_dis + link2tar_dis
                    target_point[2] = link_point[0]
        else:
            break

    assert len(visited) == coor_num

    id2dis_path = {}
    for point in visited:
        id2dis_path[point[0]] = point[1:]
    return id2dis_path

def generate_complete_path(id2dis_path, src_id, target_id):
    distance = id2dis_path[target_id][0]
    path = [target_id]
    while id2dis_path[target_id][1] != src_id:
        target_id = id2dis_path[target_id][1]
        path.append(target_id)
    path.append(src_id)
    assert distance == (len(path)-1)
    return distance, path


def construct_gt_pathes(id2coor, src_id, target_ids):
    # with goodview as the source, calculate the shortest path for each point
    # this could reduce computation cost when badview is changed
    id2dis_path = dijkstra_shortest_path(src_id=src_id, id2coor=id2coor)
    pathes = []
    for target_id in target_ids:
        distance, path = generate_complete_path(id2dis_path, src_id=src_id, target_id=target_id)
        pathes.append(path)
        # print('shortest distance from badview %d to goodview %d is %d' % (target_id, src_id, distance))
    return pathes

# generate a path for a targte point in a scene
def render_single_scene_path(scene_id, grid_info, navigation_path, simulator_frames=40,
                             cache_dir='tmp/path_render_cache/', save_dir='', pixel=256):
    
    scene_dir = DATASET_DIR+'scenes/'+scene_id
    id2coor = grid_info['id2coordinates']
    # simulator_frames = 20
    render_frames = len(navigation_path)
    # print(navigation_path, render_frames)
    # --- create scene and attach a renderer to it
    scene = kb.Scene(resolution=(pixel, pixel), frame_start=1, frame_end=simulator_frames+render_frames)
    blend_path = scene_dir+'/keyframing.blend'
    scratch_dir = DATASET_DIR+cache_dir
    renderer = KubricRenderer(scene, custom_scene=blend_path, scratch_dir=scratch_dir) # next render_frames are used to take photos from multiple views
    # reset a camera
    scene += kb.PerspectiveCamera(name="camera", position=(2, 0, 0), look_at=(0, 0, 1))
    for frame in range(simulator_frames+1, simulator_frames+render_frames+1):
        scene.camera.position = id2coor[str(navigation_path[frame-simulator_frames-1])]['coor']
        scene.camera.look_at((0, 0, 0))
        scene.camera.keyframe_insert("position", frame)
        scene.camera.keyframe_insert("quaternion", frame)

    # --- render the data
    print('pid %d rendering...' % os.getpid())
    start_time = time.time()
    # data_stack = renderer.render(range(simulator_frames+1, simulator_frames+render_frames+1))
    # remove some redundant steps to save time
    # renderer.render_rgbas_detail() prints render details
    render_save_dir = renderer.render_rgbas(frames=range(simulator_frames+1, simulator_frames+render_frames+1),
                                      target_dir=scene_dir+'/'+ save_dir+'/', prefix='rgba',
                                      frame_start=simulator_frames+1)
    end_time = time.time()
    print('pid %d render save dir: %s' % (os.getpid(), render_save_dir))
    print('pid %d, render cost: %.4f seconds (ave %.4fs each frame)' % (os.getpid(), end_time-start_time, (end_time-start_time)/render_frames))
    # --- save output files
    # image_dir = scene_dir+'/'+ save_dir
    # kb.file_io.write_rgba_batch(data_stack["rgba"], kb.as_path(image_dir+'/'))
    # os.system('convert -delay 8 -loop 0 '+image_dir+'/rgba_*.png '+image_dir+'/path_render.gif')
    os.system('rm -r '+scratch_dir+'*')
    kb.done()
    gc.collect()
    return navigation_path

def path_construction_multiprocess(description_path='/data5/haw/ActiveCap/embodiedcap_v1.json', 
                                    start=0, end=1000, path_num_for_each_view=3, process_num=10):
    scene_num = end-start
    process_scene_num = math.ceil(scene_num/process_num)
    args = []
    tmp_start = start
    for i in range(process_num):
        if i == process_num - 1:
            args.append((tmp_start, end))
        else:
            args.append((tmp_start, tmp_start+process_scene_num))
            tmp_start+=process_scene_num
    print('scenes split:', args)
    for arg in args:
        _process = multiprocessing.Process(target=path_construction, args=(description_path, arg[0], arg[1], path_num_for_each_view))
        _process.start()

    
def path_construction(description_path='/data5/haw/ActiveCap/embodiedcap_v1.json', start=0, end=1000, path_num_for_each_view=3):
    scene_dir = DATASET_DIR+'scenes'
    description_data = json.load(open(description_path, 'r', encoding='utf-8'))
    for scene_anno in tqdm(description_data[start:end]):
        scene_id = scene_anno['scene_id']
        # os.system('rm -r '+ scene_dir + '/'+scene_id+'/badview*-goodview*_render')
        # save navigation path
        navigation_save_path = scene_dir + '/'+scene_id+'/navigation_path.json'
        # if os.path.exists(navigation_save_path):
        #     continue
        descriptions = scene_anno['descriptions']
        # filter_points(scene_id)
        views = list(set([des['view'] for des in descriptions]))
        grid_info_path = scene_dir + '/'+scene_id+'/navigation_space_info.json'
        grid_info = json.load(open(grid_info_path, 'r', encoding='utf-8'))

        can_id2coor = {}
        for coor_id in grid_info['id2coordinates'].keys():
            # exclude [0,0,0] from path
            coor = grid_info['id2coordinates'][coor_id]['unit']
            if np.sum(abs(np.array(coor)))>0: 
                can_id2coor[coor_id] = grid_info['id2coordinates'][coor_id]
            # else:
            #     print('exclude ', coor_id, coor)
        candidate_ids = list(can_id2coor.keys())
        goodview_ids = []
        # collect good views
        for view in views:
            goodview_index = int(view.replace('rgba_000', '').replace('.png', ''))
            goodview_id =  grid_info['chosen_ids'][goodview_index]
            if goodview_id not in goodview_ids:
                goodview_ids.append(goodview_id)
                candidate_ids.remove(str(goodview_id))
        navigation_data = []
        for goodview_id in goodview_ids:
            # sample multiple badviews for each goodview
            badview_ids = [int(id) for id in random.sample(candidate_ids, path_num_for_each_view)]
            # print('generating pathes for scene %s , view %d' % (scene_id, goodview_id))
            navigation_pathes = construct_gt_pathes(can_id2coor, goodview_id, badview_ids)
            for i, navigation_path in enumerate(navigation_pathes):
                navigation_data.append({'goodview_id':goodview_id, 'badview_id':badview_ids[i], 'path':navigation_path})

        json.dump(navigation_data, open(navigation_save_path, 'w', encoding='utf-8'))
        # print('save %s navigation path to %s' % (scene_id, navigation_save_path))


def path_render_multiprocess(description_path='/data5/haw/ActiveCap/embodiedcap_v1.json', 
                                    start=0, end=1000, process_num=10):
    scene_num = end-start
    process_scene_num = math.ceil(scene_num/process_num)
    args = []
    tmp_start = start
    for i in range(process_num):
        if i == process_num - 1:
            args.append((tmp_start, end))
        else:
            args.append((tmp_start, tmp_start+process_scene_num))
            tmp_start+=process_scene_num
    print('scenes split:', args)
    for arg in args:
        _process = multiprocessing.Process(target=path_render, args=(description_path, arg[0], arg[1]))
        _process.start()

def path_render(description_path='/data5/haw/ActiveCap/embodiedcap_v1.json', start=0, end=1000):
    scene_dir = DATASET_DIR+'scenes'
    cache_dir = 'tmp/scene'+str(start)+'-'+str(end)+'_path_render_cache/'
    # print('start:', start, 'end:', end)
    description_data = json.load(open(description_path, 'r', encoding='utf-8'))
    rendered_path_num = 0
    for scene_anno in tqdm(description_data[start:end]):
        scene_id = scene_anno['scene_id']
        # os.system('rm -r '+ scene_dir + '/'+scene_id+'/badview*-goodview*_render')        
        grid_info_path = scene_dir + '/'+scene_id+'/navigation_space_info.json'
        grid_info = json.load(open(grid_info_path, 'r', encoding='utf-8'))
        navigation_save_path = scene_dir + '/'+scene_id+'/navigation_path.json'
        navigations = json.load(open(navigation_save_path, 'r', encoding='utf-8'))
        for navigation in navigations:
            # sample multiple badviews for each goodview
            navigation_path = navigation['path']
            goodview_id = navigation['goodview_id']
            badview_id = navigation['badview_id']
            save_dir = 'badview'+str(badview_id)+'-'+'goodview'+str(goodview_id)+'_render'
            save_path = scene_dir + '/'+scene_id+'/'+ save_dir+'/'
            if os.path.exists(save_path):
                image_num = len(os.listdir(save_path))
                if image_num == len(navigation_path):
                    # print('pid %d render dir existed: %s' % (os.getpid(), save_dir))
                    continue
                else:
                    os.system('rm -r ' + save_path)
            print('pid %d to render dir : %s' % (os.getpid(), save_path))
            rendered_path_num += 1
            render_single_scene_path(scene_id, grid_info, navigation_path, simulator_frames=40, cache_dir=cache_dir, 
                            save_dir=save_dir)
    # print(rendered_path_num)


def shorter_path(description_path='/data5/haw/ActiveCap/embodiedcap_v1.json', start=0, end=1000, max_repeat_step=3):
    scene_dir = DATASET_DIR+'scenes'
    description_data = json.load(open(description_path, 'r', encoding='utf-8'))
    for scene_anno in tqdm(description_data[start:end]):
        scene_id = scene_anno['scene_id']
        # save navigation path
        navigation_save_path = scene_dir + '/'+scene_id+'/navigation_path.json'
        assert os.path.exists(navigation_save_path)
        """if not os.path.exists(navigation_save_path):
            continue"""
        navigation_data = json.load(open(navigation_save_path, 'r', encoding='utf-8'))
        # list of {'goodview_id':x, 'badview_id':x, 'path':[x]}
        grid_info_path = scene_dir + '/'+scene_id+'/navigation_space_info.json'
        grid_info = json.load(open(grid_info_path, 'r', encoding='utf-8'))
        id2coor = grid_info['id2coordinates']
        new_navigation_data = []
        for navigation in navigation_data:
            path = navigation['path'] # list of position id
            shorter_path = []
            pre_action = 'None'
            repeat_move_step = 0
            for i in range(len(path)-1):
                start_coor = id2coor[str(path[i])]['unit']
                end_coor = id2coor[str(path[i+1])]['unit']
                if start_coor[0] == 0 and start_coor[1] == 0 and start_coor[2] == 0:
                    print('[0,0,0] in '+ str(scene_id) +'path')
                    # exit(0)
                if end_coor[0] == 0 and end_coor[1] == 0 and end_coor[2] == 0:
                    print('[0,0,0] in '+ str(scene_id) +'path')
                    # exit(0)
                
                # note move action here is absolute move action in the world coordinate system
                # move_act is just used to shorter path rather than traning
                move_act = move_action(start_coor, end_coor)
                # if this move action is the same as previous one, skip this point
                if move_act != pre_action: 
                    shorter_path.append(path[i])
                    pre_action = move_act
                    repeat_move_step = 1
                else:
                    repeat_move_step += 1
                    if repeat_move_step == max_repeat_step:
                        pre_action = 'None'
                        repeat_move_step = 0

            # add the target position
            shorter_path.append(path[-1])
            navigation['shorter_path'] = shorter_path
            new_navigation_data.append(navigation)

        shorter_navigation_save_path = scene_dir + '/'+scene_id+'/navigation_shorter_path.json'
        json.dump(navigation_data, open(shorter_navigation_save_path, 'w', encoding='utf-8'))
        # print('save %s shorter navigation path to %s' % (scene_id, shorter_navigation_save_path))

def action_generation(path, id2coor):
    actions = []
    positions = []
    for i in range(len(path)-1):
        start_coor = id2coor[str(path[i])]['unit']
        end_coor = id2coor[str(path[i+1])]['unit']
        positions.append(start_coor)
        if i == 0:
            start_coor = np.array(start_coor) 
            target_coor = np.array([0,0,-0.25]) # target_coor *0.4+[0,0,0.1] = [0,0,0]
            start_view_vec = target_coor-start_coor
            start_view_vec_xy = np.array(list(start_view_vec[0:2])+[0])
            start_view_x_angle = vector_angle(start_view_vec_xy, [1,0,0])
            if start_view_x_angle == 'nan':
                start_view_x_angle = 0
            else:
                if start_view_vec[1] < 0:
                    assert start_view_x_angle < 180
                    start_view_x_angle = 360-start_view_x_angle
        
        # move_act = move_action(start_coor, end_coor)
        move_act = relative_move_action(start_coor, end_coor, start_view_x_angle) # e.g. {'move':{'fb':['forward', 1], 'rl':['none', 0], 'ud':['down', 2]}}
        pitch_act, yaw_act, end_view_x_angle = rotate_action(start_coor, start_view_x_angle, end_coor)
        start_view_x_angle = end_view_x_angle
        # combine 3 actions (dict type)
        action = move_act | pitch_act | yaw_act
        # print('from ', start_coor, ' to ', end_coor, ' :', action)
        actions.append(action)
    positions.append(end_coor)
    # actions.append({'move':['none', 0], 'pitch':['none', 0], 'yaw': ['none', 0]})
    actions.append({'move':{'fb':['none', 0], 'rl':['none',0], 'ud':['none', 0]}, 'pitch':['none', 0], 'yaw': ['none', 0]})
    assert len(actions) == len(path)
    assert len(positions) == len(path)
    return actions, positions


def action_construction(description_path='/data5/haw/ActiveCap/embodiedcap_v1.json', start=0, end=100):
    scene_dir = DATASET_DIR+'scenes'
    description_data = json.load(open(description_path, 'r', encoding='utf-8'))
    for scene_anno in tqdm(description_data[start:end]):
        scene_id = scene_anno['scene_id']
        # save navigation path
        navigation_save_path = scene_dir + '/'+scene_id+'/navigation_shorter_path.json'
        assert os.path.exists(navigation_save_path)
        """if not os.path.exists(navigation_save_path):
            continue"""
        navigation_data = json.load(open(navigation_save_path, 'r', encoding='utf-8'))
        grid_info_path = scene_dir + '/'+scene_id+'/navigation_space_info.json'
        grid_info = json.load(open(grid_info_path, 'r', encoding='utf-8'))
        id2coor = grid_info['id2coordinates']
        for navigation in navigation_data:
            # normal path (only move one unit each step)
            path = navigation['path']
            actions, positions = action_generation(path, id2coor)
            navigation['actions'] = actions
            navigation['positions'] = positions
            # shorter path
            shorter_path = navigation['shorter_path']
            shorter_actions, shorter_positions = action_generation(shorter_path, id2coor)
            navigation['shorter_actions'] = shorter_actions
            navigation['shorter_positions'] = shorter_positions

        # 'rel' means relative move action
        navigation_action_save_path = scene_dir + '/'+scene_id+'/navigation_shorter_path_action_new_rel.json'
        json.dump(navigation_data, open(navigation_action_save_path, 'w', encoding='utf-8'))
        # print('save %s navigation path and actions to %s' % (scene_id, navigation_action_save_path))
        # exit(0)


if __name__ == '__main__':

    ## with annotated good views, construct navigation path and actions for imitation training

    # 1. randomly choose start point and generate path
    path_construction_multiprocess(description_path='/data5/haw/ActiveCap/embodiedcap_v1_train_val.json', start=0, end=9162, process_num=30)
    
    # 2. render according the path (pre-render images to save supervised training time)
    path_render_multiprocess(description_path='/data5/haw/ActiveCap/embodiedcap_v1_train_val.json', start=0, end=9162, process_num=5)
    
    # 3. shorter path by combining same-direction move
    shorter_path(description_path='/data5/haw/ActiveCap/embodiedcap_v1_train_val.json', start=0, end=9162, max_repeat_step=4)
    
    # 3. generate actions for each path
    action_construction(description_path='/data5/haw/ActiveCap/embodiedcap_v1_tain_val.json', start=0, end=9162)
    
    
