from .coco import CocoDataset
from .builder import DATASETS
# for object detection
@DATASETS.register_module()
class EmbodiedCocoDataset(CocoDataset):
    """Coco dataset for Embodied object detectiona and segmentation.
    """
    # 57 classes
    CLASSES = ['sofa', 'shoe', 'consumer goods', 'jar', 'guitar', 'vessel', 
        'loudspeaker', 'bench', 'display', 'bus', 'chair', 'table', 'car', 
        'microwave', 'rifle', 'dishwasher', 'piano', 'motorcycle', 'clock',
        'ashcan', 'microphone', 'knife', 'airplane', 'mug', 'tower', 'cabinet',
        'telephone', 'bottle', 'bowl', 'pot', 'bag', 'rocket', 'bookshelf', 
        'cellular telephone', 'bathtub', 'cap', 'camera', 'can', 'laptop', 
        'pistol', 'remote control', 'pillow', 'board games', 'helmet', 'train', 
        'earphone', 'skateboard', 'printer', 'washer', 'file', 'computer keyboard', 
        'basket', 'legos', 'media cases', 'action figures', 'bed', 'stuffed toys']
    
    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),
               (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
               (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
               (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
               (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1)]
