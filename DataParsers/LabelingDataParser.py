import pandas as pd

from DataParsers.AbstractDataParser import AbstractDataParser
from tools.functions import load_json, quaternion_to_rpy


class LabelingDataParser(AbstractDataParser):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.columns = ['Sample token', 'PosX', 'PosY', 'PosZ',
                        'Width', 'Length', 'Height',
                        'Yaw', 'Pitch', 'Roll', 'Class']
        self.raw_data = load_json(filepath)
        self.df = pd.DataFrame(columns=self.columns)

    def parse(self):
        records = []
        for annotation in self.raw_data:
            roll, pitch, yaw = quaternion_to_rpy(annotation['rotation'])
            records.append({
                'Sample token': annotation['sample_token'],
                'PosX': annotation['translation'][0],
                'PosY': annotation['translation'][1],
                'PosZ': annotation['translation'][2],
                'Width': annotation['size'][0],
                'Length': annotation['size'][1],
                'Height': annotation['size'][2],
                'Yaw': yaw,
                'Pitch': pitch,
                'Roll': roll,
                'Class': annotation['category_name'],
            })
        self.df = pd.DataFrame(records, columns=self.columns)
        self.df = self.df.sort_values('Sample token').reset_index(drop=True)
