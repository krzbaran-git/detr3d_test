import os
from dataclasses import dataclass, field
import numpy as np

from tools.CameraVisualizer import CameraVisualizer
from tools.NuScenesVisualizer import NuscCameraVisualizer

@dataclass
class Config:
    # path: str = r"D:\Programiki do nauki i inne\Szkolne\Studia\Projekt inzynierski\Logi NuScenes"
    path: str = r"E:\scenes\val_scenes"
    output_dir: str = ''
    selected_scenes: tuple | None = None

    # (distance filter [m], heading filter)
    class_params: dict = field(default_factory=lambda: {
        'car':                  (3.0, True),
        'truck':                (3.0, True),
        'bus':                  (3.0, True),
        'trailer':              (4.0, True),
        'construction_vehicle': (3.0, True),
        'barrier':              (2.0, False),
        'bicycle':              (2.0, True),
        'motorcycle':           (2.0, True),
        'pedestrian':           (1.0, False),
        'traffic_cone':         (1.0, False),
    })
    default_params: tuple = (1.5, True)
    heading_limit: float = np.pi / 2

    detr_score_threshold: float = 0.5
    adatrack_score_threshold: float = 0.5

    iou_samples: int = 10_000
    iou_threshold: float  = 0.35

    n_jobs: int = -1
    visualize: bool = False
    visualizer_type: type = CameraVisualizer

    class_range: dict = field(default_factory=lambda: {
        'car': 50,
        'truck': 50,
        'bus': 50,
        'trailer': 50,
        'construction_vehicle': 50,
        'pedestrian': 40,
        'motorcycle': 40,
        'bicycle': 40,
        'traffic_cone': 30,
        'barrier': 30,
    })
    default_range: float = 50.0
    min_gt_pts: int = 1

    eval_classes: dict = field(default_factory=lambda: {
        'car': True,
        'truck': True,
        'bus': True,
        'trailer': True,
        'motorcycle': True,
        'bicycle': True,
        'pedestrian': True,
        'construction_vehicle': False, # Not returned by ADATrack
        'barrier': False,              # Not returned by ADATrack
        'traffic_cone': False,         # Not returned by ADATrack
    })

    sweep_thresholds: tuple = field(default_factory=lambda: tuple(round(x, 2) for x in np.arange(0.05, 1.0, 0.05)))

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = os.path.join(self.path, 'output')