import json
import numpy as np

def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def quaternion_to_rpy(quaternion: list) -> tuple:
    w, x, y, z = quaternion
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(2*(w*y - z*x))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))

    return roll, pitch, yaw