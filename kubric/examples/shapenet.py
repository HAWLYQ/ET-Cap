import os
import logging
import numpy as np

import kubric as kb
from kubric.renderer import Blender
import cv2
import copy

# --- CLI arguments
parser = kb.ArgumentParser()
parser.set_defaults(
    frame_end=1,
    resolution=(512, 512),
)
FLAGS = parser.parse_args()

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
renderer = Blender(scene, scratch_dir,
                   samples_per_pixel=64,
                   background_transparency=True)

# --- Fetch shapenet
# source_path = os.getenv("SHAPENET_GCP_BUCKET", "gs://kubric-public/assets/ShapeNetCore.v2.json")
source_path = 'ShapeNetCore.v2.json'
shapenet = kb.AssetSource.from_manifest(source_path)

# --- Add Klevr-like lights to the scene
scene += kb.assets.utils.get_clevr_lights(rng=rng)
scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

# --- Add shadow-catcher floor
floor = kb.Cube(name="floor", scale=(100, 100, 1), position=(0, 0, -1))
scene += floor
# Make the floor transparent except for catching shadows
# Together with background_transparency=True (above) this results in
# the background being transparent except for the object shadows.
floor.linked_objects[renderer].cycles.is_shadow_catcher = True

# --- Keyframe the camera
scene.camera = kb.PerspectiveCamera(position=(1.0,1.0,1.0), look_at=(0.0,0.0,0.0))
# print(scene.camera.matrix_world)
# print(scene.camera.intrinsics)



print(FLAGS.frame_start, FLAGS.frame_end)
for frame in range(FLAGS.frame_start, FLAGS.frame_end + 1):
  # scene.camera.position = kb.sample_point_in_half_sphere_shell(1.5, 1.7, 0.1)
  scene.camera.position = [1.0, 1.0, 1.0]
  scene.camera.look_at((0.0, 0.0, 0.0))
  scene.camera.keyframe_insert("position", frame)
  scene.camera.keyframe_insert("quaternion", frame)

# --- Fetch a random (airplane) asset
"""airplane_ids = [name for name, spec in shapenet._assets.items()
                if spec["metadata"]["category"] == "airplane"]

asset_id = rng.choice(airplane_ids) #< e.g. 02691156_10155655850468db78d106ce0a280f87"""
asset_id = '02691156/443be81bfa6e5b5ef6babb7d9ead7011'
obj = shapenet.create(asset_id=asset_id)
"""
obj.aabbox: Axis-aligned bounding box [(min_x, min_y, min_y), (max_x, max_y, max_z)].
obj.bbox: 3D bounding box as an array of 8 corners (shape = [8, 3])
obj.position (vec3d): the (x, y, z) position of the object.
obj.quaternion (vec4d): a (W, X, Y, Z) quaternion for describing the rotation.
"""
logging.info(f"selected '{asset_id}'")
output_dir = output_dir / asset_id
# --- make object flat on X/Y and not penetrate floor
obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
obj.position = obj.position - (0, 0, obj.aabbox[0][2]) # (0,0,0.056291)
scene.add(obj)

# add another obj (haw)
asset_id = '02691156/10155655850468db78d106ce0a280f87'
obj2 = shapenet.create(asset_id=asset_id)
logging.info(f"selected '{asset_id}'")
# --- make object flat on X/Y and not penetrate floor
obj2.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=180)
obj2.position = (0.4, 0.0, 0.3)
scene.add(obj2)


# --- Rendering
logging.info("Rendering the scene ...")
renderer.save_state(output_dir / "scene.blend")
data_stack = renderer.render()

# --- Postprocessing
print(data_stack["segmentation"].shape)
# print(data_stack["segmentation"])
kb.compute_visibility(data_stack["segmentation"], scene.assets)
kb.compute_bboxes(data_stack['segmentation'], scene.assets)
for t in range(data_stack["segmentation"].shape[0]):
  num2count = {}
  values = np.resize(data_stack["segmentation"], [512*512]).tolist()
  for value in values:
    num2count[value] = num2count.get(value, 0)+1
  print(num2count)

# there are actually 3 objects (including the floor) in the scene,
# after adjust, 0 means floor, 1 means obj, 2 means obj2 
data_stack["segmentation"] = kb.adjust_segmentation_idxs(
    data_stack["segmentation"],
    scene.assets,
    # [obj, obj2]
    scene.foreground_assets[1:],
    ).astype(np.uint8)
  
for i, instance in enumerate(scene.foreground_assets):
  info = copy.copy(instance.metadata)
  print(i, info)

for t in range(data_stack["segmentation"].shape[0]):
  num2count = {}
  values = np.resize(data_stack["segmentation"], [512*512]).tolist()
  for value in values:
    num2count[value] = num2count.get(value, 0)+1
  print(num2count)


kb.file_io.write_rgba_batch(data_stack["rgba"], output_dir)
kb.file_io.write_depth_batch(data_stack["depth"], output_dir)
kb.file_io.write_segmentation_batch(data_stack["segmentation"], output_dir)

# --- Collect metadata
logging.info("Collecting and storing metadata for each object.")
data = {
  "metadata": kb.get_scene_metadata(scene),
  # store the camera parameters(e.g. Rotation and Intrinsics) at the final frame 
  "camera": kb.get_camera_info(scene.camera), 
  "object": kb.get_instance_info(scene, [obj, obj2])
}
kb.file_io.write_json(filename=output_dir / "metadata5.json", data=data)
kb.done()


