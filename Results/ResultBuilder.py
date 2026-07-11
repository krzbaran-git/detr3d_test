import os
import pandas as pd
from tqdm import tqdm

from Scene import Scene

class ResultBuilder:
    def __init__(self, path, vis = False):
        self.path = path
        self.dataset = []
        self.df = None

        # Does get visualized
        self.visualize = vis

    def build(self):
        scene_dirs = [d for d in os.listdir(self.path)
                      if os.path.isdir(os.path.join(self.path, d))
                      and d.startswith('scene-')]

        for scene_dir in tqdm(scene_dirs, desc='Building scenes'):
            scene_path = os.path.join(self.path, scene_dir)

            scene = Scene(scene_path)
            scene.build_scene()
            scene.build_pairs()

            scene.paired_df['Scene'] = scene_dir

            if self.visualize:
                output_path = os.path.join(self.path, 'output', scene_dir)
                scene.visualize_scene(output_path)

            self.dataset.append(scene)

        self.df = pd.concat([scene.paired_df for scene in self.dataset], ignore_index=True)

    def calculate_metrics(self):
        pass