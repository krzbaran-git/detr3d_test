import numpy as np

class PascalMeasure3D:
    def __init__(self, obj1, obj2, n):
        self.obj1    = obj1
        self.obj2    = obj2
        self.samples = n
        self.iou     = None

    def compute_iou(self):
        pts1 = self._sample_box(self.obj1)
        pts2 = self._sample_box(self.obj2)
        pts  = np.concatenate([pts1, pts2], axis=0)

        in1 = self._is_inside_box(pts, self.obj1)
        in2 = self._is_inside_box(pts, self.obj2)

        intersection = np.sum(in1 & in2)
        union        = np.sum(in1 | in2)

        self.iou = intersection / union if union > 0 else 0.0
        return self.iou

    def _get_rotation_matrix(self, obj):
        roll, pitch, yaw = obj['Roll'], obj['Pitch'], obj['Yaw']
        Rx = np.array([[1,              0,               0],
                       [0,  np.cos(roll), -np.sin(roll)],
                       [0,  np.sin(roll),  np.cos(roll)]])
        Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                       [0,              1,              0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw),  np.cos(yaw), 0],
                       [0,            0,            1]])
        return Rz @ Ry @ Rx

    def _sample_box(self, obj):
        lx = np.random.uniform(-obj['Length'] / 2, obj['Length'] / 2, self.samples)
        ly = np.random.uniform(-obj['Width']  / 2, obj['Width']  / 2, self.samples)
        lz = np.random.uniform(-obj['Height'] / 2, obj['Height'] / 2, self.samples)
        points = np.stack([lx, ly, lz], axis=1)

        R = self._get_rotation_matrix(obj)
        center = np.array([obj['PosX'], obj['PosY'], obj['PosZ']])
        return (points @ R.T) + center

    def _is_inside_box(self, points, obj):
        R = self._get_rotation_matrix(obj)
        center = np.array([obj['PosX'], obj['PosY'], obj['PosZ']])
        local  = (points - center) @ R
        return (
            (np.abs(local[:, 0]) <= obj['Length'] / 2) &
            (np.abs(local[:, 1]) <= obj['Width']  / 2) &
            (np.abs(local[:, 2]) <= obj['Height'] / 2)
        )

