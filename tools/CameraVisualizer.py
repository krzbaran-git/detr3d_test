import cv2
import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from Sensors.Sensors import CameraSensor

class CameraVisualizer:
    def __init__(self, camera: CameraSensor, image_path: str):
        self.camera = camera
        self.img    = cv2.imread(image_path)

    def render(self, paired: pd.DataFrame, unpaired_sys: pd.DataFrame, unpaired_ref: pd.DataFrame, ego: dict):
        for _, row in paired.iterrows():
            sys_pixels = self.project_object(self.row_to_obj(row, 'Sys'), ego)
            ref_pixels = self.project_object(self.row_to_obj(row, 'Ref'), ego)
            self.draw_box(sys_pixels, color=(0, 0, 255))
            self.draw_box(ref_pixels, color=(0, 255, 0))
            self.draw_match(sys_pixels, ref_pixels, iou=row.get('PascalMeasure'))

        for _, row in unpaired_sys.iterrows():
            self.draw_box(self.project_object(self.row_to_obj(row, 'Sys'), ego), color=(255, 0, 0))

        for _, row in unpaired_ref.iterrows():
            self.draw_box(self.project_object(self.row_to_obj(row, 'Ref'), ego), color=(0, 255, 255))

    def save(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, self.img)

    def draw_box(self, pixels, color=(0, 0, 255), thickness=2):
        if pixels is None:
            return
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7),
        ]
        for i, j in edges:
            cv2.line(self.img, tuple(pixels[i]), tuple(pixels[j]), color, thickness)

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

    def project_object(self, obj: dict, ego: dict) -> np.ndarray | None:
        corners_global = self._get_box_corners(obj)

        R_ego = Rotation.from_quat([*ego['rotation'][1:], ego['rotation'][0]])
        corners_ego = R_ego.inv().apply(corners_global - np.array(ego['translation']))

        R_cam = Rotation.from_quat([*self.camera.rotation[1:], self.camera.rotation[0]])
        corners_cam = R_cam.inv().apply(corners_ego - np.array(self.camera.translation))

        if np.any(corners_cam[:, 2] <= 0):
            return None

        K = np.array(self.camera.camera_intrinsic)
        projected = (K @ corners_cam.T).T
        pixels = projected[:, :2] / projected[:, 2:3]
        return pixels.astype(int)

    def _get_box_corners(self, obj: dict) -> np.ndarray:
        l, w, h = obj['Length'] / 2, obj['Width'] / 2, obj['Height'] / 2
        corners = np.array([
            [ l,  w, -h],
            [ l, -w, -h],
            [-l, -w, -h],
            [-l,  w, -h],
            [ l,  w,  h],
            [ l, -w,  h],
            [-l, -w,  h],
            [-l,  w,  h],
        ])
        R = Rotation.from_euler('ZYX', [obj['Yaw'], obj['Pitch'], obj['Roll']])
        center = np.array([obj['PosX'], obj['PosY'], obj['PosZ']])
        return (R.as_matrix() @ corners.T).T + center

    def get_image(self) -> np.ndarray:
        return self.img

    @staticmethod
    def row_to_obj(row, prefix: str) -> dict:
        return {
            'PosX':   row[f'{prefix}_PosX'],   'PosY':   row[f'{prefix}_PosY'],   'PosZ':   row[f'{prefix}_PosZ'],
            'Yaw':    row[f'{prefix}_Yaw'],    'Pitch':  row[f'{prefix}_Pitch'],  'Roll':   row[f'{prefix}_Roll'],
            'Width':  row[f'{prefix}_Width'],  'Length': row[f'{prefix}_Length'], 'Height': row[f'{prefix}_Height'],
        }