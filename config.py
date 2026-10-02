import os
from dataclasses import dataclass, field
import numpy as np

from tools.CameraVisualizer import CameraVisualizer
from tools.NuScenesVisualizer import NuscCameraVisualizer

@dataclass
class Config:
    path: str = r"D:\Programiki do nauki i inne\Szkolne\Studia\Projekt inzynierski\Logi NuScenes"
    output_dir: str = ''

    class_params: dict = field(default_factory=lambda: {   # (distance filter [m], heading filter)
        'car':          (3.0, True),
        'truck':        (3.0, True),
        'bus':          (3.0, True),
        'bicycle':      (2.0, True),
        'motorcycle':   (2.0, True),
        'pedestrian':   (1.0, False),
        'traffic_cone': (1.0, False),
    })
    default_params: tuple = (1.5, True)
    heading_limit: float = np.pi / 2

    detr_score_threshold: float = 0.45
    adatrack_score_threshold: float = 0.3

    iou_samples: int = 10_000
    iou_threshold: float  = 0.5

    n_jobs: int = -1
    visualize: bool = True
    visualizer_type: type = CameraVisualizer

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = os.path.join(self.path, 'output')