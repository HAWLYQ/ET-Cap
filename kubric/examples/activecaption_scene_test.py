import sys
sys.path.append('/root/code/kubric/')
import logging
import random
import kubric as kb
import numpy as np
# from kubric.renderer.blender import Blender as KubricRenderer
from kubric_haw.renderer.blender import Blender as KubricRenderer
from kubric.simulator.pybullet import PyBullet as KubricSimulator
import os
os.environ['KUBRIC_USE_GPU'] = '0'
import time
import copy
import json
import gc

from tqdm import tqdm
logging.basicConfig(level="INFO")


def lookat_transfer(position, look_at, x_set=0):
    x1 = position[0]
    y1 = position[1]
    z1 = position[2]
    x2 = look_at[0]
    y2 = look_at[1]
    z2 = look_at[2]
    # (x-x1)/(x1-x2) = (y-y1)/(y1-y2) = (z-z1)/(z1-z2)
    x = x_set
    y = (y1-y2)*(x-x1)/(x1-x2) + y1 
    z = (z1-z2)*(x-x1)/(x1-x2) + z1
    return np.array([x,y,z])

DATASET_DIR='/data5/haw/ActiveCap/'
SCENES_DIR = DATASET_DIR + 'scenes/'

scene_id=2140
scene_dir = SCENES_DIR + str(scene_id) + '/'
render_frame=41
cache_dir = 'tmp/navigation_inference/'
save_dir = 'output/'

scene = kb.Scene(resolution=(256, 256), frame_start=1, frame_end=render_frame)
blend_path = scene_dir +'keyframing.blend'
scratch_dir = DATASET_DIR+cache_dir
renderer = KubricRenderer(scene, custom_scene=blend_path, scratch_dir=scratch_dir) # next render_frames are used to take photos from multiple views
# reset a camera
scene += kb.PerspectiveCamera(name="camera", position=(2, 0, 0), look_at=(0, 0, 1))
render_frame = render_frame

# position = [10,-8 ,5] 
# look_at = [9.26352918,-7.43358873,4.63015003]
position = [5,5,5]
look_at = [0,0,0]
look_at = lookat_transfer(position, look_at, x_set=4.5)
print(position, look_at)
scene.camera.position = np.array(position)*0.4+np.array([0,0,0.1])
scene.camera.look_at(np.array(look_at)*0.4+np.array([0,0,0.1]))
# scene.camera.position = np.array(position)
# scene.camera.look_at(look_at)
scene.camera.keyframe_insert("position", render_frame)
scene.camera.keyframe_insert("quaternion", render_frame)
img_path = renderer.render_single_rgba(frame=render_frame, target_dir=save_dir, image_name='test')

os.system('rm -r '+scratch_dir)

