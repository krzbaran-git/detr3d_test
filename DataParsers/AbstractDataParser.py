from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class AbstractDataParser(ABC):
    def __init__(self, path):
        self.filepath = path
        self.data = None

    @abstractmethod
    def parse(self):
        pass

    @staticmethod
    def filter_by_range(df, classes, ego_xy, class_range, default_range):
        if df.empty or ego_xy is None:
            return df
        xy = np.array([ego_xy.get(t, (np.nan, np.nan)) for t in df['Sample token']])
        dist = np.hypot(df['PosX'] - xy[:, 0], df['PosY'] - xy[:, 1])
        limit = classes.map(class_range or {}).fillna(default_range)
        return df[dist <= limit].reset_index(drop=True)