import numpy as np
import pandas as pd
from tools.PascalMeasure import PascalMeasure3D


class ObjectsMatcher:

    CLASS_PARAMS = {
        'car':          (3.0, True),
        'truck':        (3.0, True),
        'bus':          (3.0, True),
        'bicycle':      (2.0, True),
        'motorcycle':   (2.0, True),
        'pedestrian':   (1.0, False),
        'traffic_cone': (1.0, False),
    }
    DEFAULT_PARAMS = (1.5, True)
    IOU_SAMPLES    = 10000

    def match(self, sys_objects: pd.DataFrame, ref_objects: pd.DataFrame):
        sys = sys_objects.reset_index(drop=True)
        ref = ref_objects.reset_index(drop=True)

        candidates = []
        for s in range(len(sys)):
            sys_obj = sys.iloc[s]
            dist_thr, head_on = self.CLASS_PARAMS.get(sys_obj['Class'], self.DEFAULT_PARAMS)

            for r in range(len(ref)):
                ref_obj = ref.iloc[r]

                if not self._distance_filter(sys_obj, ref_obj, dist_thr):
                    continue
                if head_on and not self._heading_filter(sys_obj, ref_obj):
                    continue

                iou = PascalMeasure3D(sys_obj, ref_obj, self.IOU_SAMPLES).compute_iou()
                if iou > 0:
                    candidates.append((iou, s, r))

        candidates = self._sort_by_iou(candidates)

        matched_sys, matched_ref = set(), set()
        pairs = []
        for iou, s, r in candidates:
            if s in matched_sys or r in matched_ref:
                continue
            matched_sys.add(s)
            matched_ref.add(r)
            pairs.append((s, r, iou))

        unmatched_sys = [s for s in range(len(sys)) if s not in matched_sys]
        unmatched_ref = [r for r in range(len(ref)) if r not in matched_ref]

        return pairs, unmatched_sys, unmatched_ref

    def _sort_by_iou(self, candidates: list) -> list:
        return sorted(candidates, key=lambda x: x[0], reverse=True)

    def _heading_filter(self, sys_obj, ref_obj) -> bool:
        diff = sys_obj['Yaw'] - ref_obj['Yaw']
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        return abs(diff) <= np.pi / 2

    def _distance_filter(self, sys_obj, ref_obj, threshold: float) -> bool:
        dist = np.sqrt(
            (sys_obj['PosX'] - ref_obj['PosX'])**2 +
            (sys_obj['PosY'] - ref_obj['PosY'])**2 +
            (sys_obj['PosZ'] - ref_obj['PosZ'])**2
        )
        return dist <= threshold