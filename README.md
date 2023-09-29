# Explore and Tell: Embodied Visual Captioning in 3D Environments

# Environments
- refer to kubric(https://github.com/google-research/kubric) for preparing basic experimental environment by docker pull. (Note this docker image may not support GPU, refer to https://github.com/google-research/kubric/issues/224, you can try this docker image 'docker pull ganow/kubric:latest' for GPU supporting.)

- python 3.9.5, torch 1.12

# ET-CAP Download
- 3D assets (~64G): 3D assets used in ET-Cap come from ShapeNet and GSO, before donwloading 3D assets, please make sure you have accepted the license from shapenet.org (https://shapenet.org/). 
- 3D scenes (~160G): ...
- ET-Cap annotations: ...

# Create Your Own Scenes or Training Trajectories
## 3D Scenes Simulation
```
python scene_construct.py
```

## Trajectory Generation
```
python path_action_construction.py
```


