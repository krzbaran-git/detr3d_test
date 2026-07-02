import cv2
import numpy as np

from Sensors.Sensors import CameraSensor

class CameraVisualizer:
    def __init__(self, camera: CameraSensor, image_path: str):
        self.camera = camera
        self.img = cv2.imread(image_path)

    def draw_box(self, img, pixels, color=(0, 0, 255), thickness=2):
        pass

    def project_object(self, obj: dict):
        pass

    def _get_box_corners(self, obj: dict):
        pass

    @staticmethod
    def row_to_obj(row) -> dict:
        return {
            'PosX': row['PosX'], 'PosY': row['PosY'], 'PosZ': row['PosZ'],
            'Yaw': row['Yaw'], 'Pitch': row['Pitch'], 'Roll': row['Roll'],
            'Width': row['Width'], 'Length': row['Length'], 'Height': row['Height'],
        }