import numpy as np
import pandas as pd

from tools.IoU import IoU3D


class ObjectsMatcher:

    def __init__(self, cfg):
        self.class_params   = cfg.class_params
        self.default_params = cfg.default_params
        self.heading_limit  = cfg.heading_limit
        self.iou_samples    = cfg.iou_samples

    def match(self, sys_df: pd.DataFrame, ref_df: pd.DataFrame):
        sys_a = sys_df[IoU3D.COLS].to_numpy(float)
        ref_a = ref_df[IoU3D.COLS].to_numpy(float)
        ns, nr = len(sys_a), len(ref_a)

        if ns == 0 or nr == 0:
            return [], list(range(ns)), list(range(nr))

        s_idx, r_idx = self._candidates(sys_df, sys_a, ref_a)
        if len(s_idx) == 0:
            return [], list(range(ns)), list(range(nr))

        iou = IoU3D.compute_batch(sys_a[s_idx], ref_a[r_idx], self.iou_samples)
        pairs, matched_s, matched_r = self._greedy_by_iou(s_idx, r_idx, iou)

        unmatched_sys = [s for s in range(ns) if s not in matched_s]
        unmatched_ref = [r for r in range(nr) if r not in matched_r]
        return pairs, unmatched_sys, unmatched_ref

    def _candidates(self, sys_df, sys_a, ref_a):
        params = [self.class_params.get(c, self.default_params) for c in sys_df['Class']]
        thr = np.array([p[0] for p in params])
        head = np.array([p[1] for p in params])

        dist = self._distance_matrix(sys_a, ref_a)
        dyaw = self._heading_diff_matrix(sys_a, ref_a)

        ok = (dist <= thr[:, None]) & (~head[:, None] | (dyaw <= self.heading_limit))
        return np.nonzero(ok)

    @staticmethod
    def _distance_matrix(sys_a, ref_a):
        return np.linalg.norm(sys_a[:, None, :3] - ref_a[None, :, :3], axis=2)

    @staticmethod
    def _heading_diff_matrix(sys_a, ref_a):
        d = sys_a[:, None, 3] - ref_a[None, :, 3]
        return np.abs(np.arctan2(np.sin(d), np.cos(d)))

    @staticmethod
    def _greedy_by_iou(s_idx, r_idx, iou):
        matched_s, matched_r, pairs = set(), set(), []
        for k in np.argsort(-iou):
            if iou[k] <= 0:
                break
            s, r = int(s_idx[k]), int(r_idx[k])
            if s in matched_s or r in matched_r:
                continue
            matched_s.add(s)
            matched_r.add(r)
            pairs.append((s, r, float(iou[k])))
        return pairs, matched_s, matched_r