import numpy as np
import pandas as pd

from DataParsers.AbstractDataParser import AbstractDataParser
from tools.functions import load_json, quaternion_to_rpy


class LabelingDataParser(AbstractDataParser):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.columns = ['Sample token', 'PosX', 'PosY', 'PosZ',
                        'Width', 'Length', 'Height',
                        'Yaw', 'Pitch', 'Roll',
                        'VelX', 'VelY', 'Class']
        self.raw_data = load_json(filepath)
        self.df = pd.DataFrame(columns=self.columns)

    def parse(self, valid_tokens: set = None, timestamp_lookup: dict = None):
        ann_by_token = {a['token']: a for a in self.raw_data}
        records = []

        for annotation in self.raw_data:
            if valid_tokens is not None and annotation['sample_token'] not in valid_tokens:
                continue

            roll, pitch, yaw = quaternion_to_rpy(annotation['rotation'])
            vx, vy = self._box_velocity(annotation, ann_by_token, timestamp_lookup)

            records.append({
                'Sample token': annotation['sample_token'],
                'PosX':   annotation['translation'][0],
                'PosY':   annotation['translation'][1],
                'PosZ':   annotation['translation'][2],
                'Width':  annotation['size'][0],
                'Length': annotation['size'][1],
                'Height': annotation['size'][2],
                'Yaw':    yaw,
                'Pitch':  pitch,
                'Roll':   roll,
                'VelX':   vx,
                'VelY':   vy,
                'Class':  annotation['category_name'],
            })

        self.df = pd.DataFrame(records, columns=self.columns)
        self.df = self.df.sort_values('Sample token').reset_index(drop=True)

    @staticmethod
    def _box_velocity(ann, ann_by_token, ts_lookup, max_time_diff=1.5):
        if ts_lookup is None:
            return np.nan, np.nan

        has_prev = ann['prev'] != ''
        has_next = ann['next'] != ''
        first = ann_by_token[ann['prev']] if has_prev else ann
        last  = ann_by_token[ann['next']] if has_next else ann

        if first is last:
            return np.nan, np.nan

        t_first = ts_lookup.get(first['sample_token'])
        t_last  = ts_lookup.get(last['sample_token'])
        if t_first is None or t_last is None:
            return np.nan, np.nan

        dt = (t_last - t_first) * 1e-6
        limit = max_time_diff * 2 if (has_prev and has_next) else max_time_diff
        if dt <= 0 or dt > limit:
            return np.nan, np.nan

        vx = (last['translation'][0] - first['translation'][0]) / dt
        vy = (last['translation'][1] - first['translation'][1]) / dt
        return vx, vy
