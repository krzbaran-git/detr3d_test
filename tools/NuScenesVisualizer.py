import os
import cv2
import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

class NuscCameraVisualizer:
    EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Front
        (4, 5), (5, 6), (6, 7), (7, 4), # Back
        (0, 4), (1, 5), (2, 6), (3, 7), # Sides
    ]

    def __init__(self, camera, image_path: str, vis_level=BoxVisibility.ANY):
        self.camera = camera
        self.vis_level = vis_level
        self.K = np.array(camera.camera_intrinsic)
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise FileNotFoundError(f"Nie można wczytać obrazu: {image_path}")
        self.imsize = (self.img.shape[1], self.img.shape[0])

    def render(self, paired: pd.DataFrame, unpaired_sys: pd.DataFrame,
               unpaired_ref: pd.DataFrame, ego: dict):
        for _, row in paired.iterrows():
            sys_pixels = self.project_box(self.row_to_box(row, 'Sys'), ego)
            ref_pixels = self.project_box(self.row_to_box(row, 'Ref'), ego)
            self.draw_box(sys_pixels, color=(0, 0, 255))
            self.draw_box(ref_pixels, color=(0, 255, 0))
            self.draw_match(sys_pixels, ref_pixels, iou=row.get('IoU'))

        for _, row in unpaired_sys.iterrows():
            self.draw_box(self.project_box(self.row_to_box(row, 'Sys'), ego), color=(255, 0, 0))

        for _, row in unpaired_ref.iterrows():
            self.draw_box(self.project_box(self.row_to_box(row, 'Ref'), ego), color=(0, 255, 255))

    def project_box(self, box: Box, ego: dict) -> np.ndarray | None:
        box = box.copy()

        box.translate(-np.array(ego['translation']))
        box.rotate(Quaternion(ego['rotation']).inverse)

        box.translate(-np.array(self.camera.translation))
        box.rotate(Quaternion(self.camera.rotation).inverse)

        if not box_in_image(box, self.K, self.imsize, vis_level=self.vis_level):
            return None

        pixels = view_points(box.corners(), self.K, normalize=True)[:2].T
        return pixels.astype(int)

    def draw_box(self, pixels, color=(0, 0, 255), thickness=2):
        if pixels is None:
            return
        for i, j in self.EDGES:
            cv2.line(self.img, tuple(pixels[i]), tuple(pixels[j]), color, thickness)

        bottom_fwd = pixels[[2, 3]].mean(axis=0).astype(int)
        bottom_ctr = pixels[[2, 3, 6, 7]].mean(axis=0).astype(int)
        cv2.line(self.img, tuple(bottom_ctr), tuple(bottom_fwd), color, thickness)

    def draw_match(self, sys_pixels, ref_pixels, iou=None):
        if sys_pixels is None or ref_pixels is None:
            return
        sys_center = tuple(sys_pixels.mean(axis=0).astype(int))
        ref_center = tuple(ref_pixels.mean(axis=0).astype(int))
        cv2.line(self.img, sys_center, ref_center, (255, 255, 255), 1)
        if iou is not None:
            mid = ((sys_center[0] + ref_center[0]) // 2,
                   (sys_center[1] + ref_center[1]) // 2)
            cv2.putText(self.img, f'{iou:.2f}', mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def save(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, self.img)

    def get_image(self) -> np.ndarray:
        return self.img

    @staticmethod
    def row_to_box(row, prefix: str) -> Box:
        qx, qy, qz, qw = Rotation.from_euler(
            'ZYX', [row[f'{prefix}_Yaw'], row[f'{prefix}_Pitch'], row[f'{prefix}_Roll']]
        ).as_quat()
        return Box(
            center=[row[f'{prefix}_PosX'], row[f'{prefix}_PosY'], row[f'{prefix}_PosZ']],
            size=[row[f'{prefix}_Width'], row[f'{prefix}_Length'], row[f'{prefix}_Height']],
            orientation=Quaternion(qw, qx, qy, qz),
            name=str(row[f'{prefix}_Class']),
        )