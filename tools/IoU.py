import numpy as np
from scipy.spatial.transform import Rotation


class IoU3D:
    COLS  = ['PosX', 'PosY', 'PosZ', 'Yaw', 'Pitch', 'Roll', 'Width', 'Length', 'Height']
    SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                     dtype=np.float32)

    def __init__(self, obj1, obj2, n):
        self.obj1    = obj1
        self.obj2    = obj2
        self.samples = n
        self.iou     = None

    def compute_iou(self):
        a = np.array([[self.obj1[c] for c in self.COLS]], dtype=float)
        b = np.array([[self.obj2[c] for c in self.COLS]], dtype=float)
        self.iou = float(self.compute_batch(a, b, self.samples)[0])
        return self.iou

    @classmethod
    def compute_batch(cls, A: np.ndarray, B: np.ndarray, n: int = 2000,
                      chunk: int = 256, seed: int = 0) -> np.ndarray:
        P   = len(A)
        out = np.zeros(P)
        u   = np.random.default_rng(seed).random((n, 3), dtype=np.float32)
        for i in range(0, P, chunk):
            out[i:i + chunk] = cls._iou_chunk(A[i:i + chunk], B[i:i + chunk], u)
        return out

    @staticmethod
    def _rotation_matrices(arr: np.ndarray) -> np.ndarray:
        return Rotation.from_euler('ZYX', arr[:, 3:6]).as_matrix().astype(np.float32)

    @classmethod
    def _iou_chunk(cls, A: np.ndarray, B: np.ndarray, u: np.ndarray) -> np.ndarray:
        A, B = A.astype(np.float32), B.astype(np.float32)
        RA, RB = cls._rotation_matrices(A), cls._rotation_matrices(B)
        cA, cB = A[:, :3], B[:, :3]
        hA = A[:, [7, 6, 8]] / 2
        hB = B[:, [7, 6, 8]] / 2

        M_BA = np.einsum('pji,pjk->pik', RA, RB)
        t_BA = np.einsum('pji,pj->pi',   RA, cB - cA)

        cornB_A = np.einsum('pik,pnk->pni', M_BA, cls.SIGNS[None] * hB[:, None, :]) + t_BA[:, None, :]
        lo   = np.minimum(cornB_A.min(axis=1), -hA)
        hi   = np.maximum(cornB_A.max(axis=1),  hA)
        span = hi - lo

        pts  = lo[:, None, :] + u[None] * span[:, None, :]
        in_A = (np.abs(pts) <= hA[:, None, :]).all(axis=2)

        M_AB  = M_BA.transpose(0, 2, 1)
        t_AB  = np.einsum('pji,pj->pi', RB, cA - cB)
        pts_B = np.einsum('pik,pnk->pni', M_AB, pts) + t_AB[:, None, :]
        in_B  = (np.abs(pts_B) <= hB[:, None, :]).all(axis=2)

        inter  = (in_A & in_B).mean(axis=1) * span.prod(axis=1)
        VA, VB = 8 * hA.prod(axis=1), 8 * hB.prod(axis=1)
        return np.clip(inter / (VA + VB - inter), 0.0, 1.0)