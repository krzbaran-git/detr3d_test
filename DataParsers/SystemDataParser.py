import pandas as pd

from DataParsers.AbstractDataParser import AbstractDataParser
from tools.functions import load_json, quaternion_to_rpy

class SystemDataParser(AbstractDataParser):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.columns = ['Sample token', 'Sensor ID', 'Timestamp',
                        'PosX', 'PosY', 'PosZ',
                        'Width', 'Length', 'Height',
                        'Yaw', 'Pitch', 'Roll',
                        'VelX', 'VelY',
                        'Class', 'Probability']

        self.raw_data = load_json(filepath)
        self.df = pd.DataFrame(columns=self.columns)

    def parse(self):
        records = []
        for sample_token, detections in self.raw_data['results'].items():
            for detection in detections:
                roll, pitch, yaw = quaternion_to_rpy(detection['rotation'])
                records.append({
                    'Sample token': detection['sample_token'],
                    'Sensor ID': None,
                    'Timestamp': None,
                    'PosX': detection['translation'][0],
                    'PosY': detection['translation'][1],
                    'PosZ': detection['translation'][2],
                    'Width': detection['size'][0],
                    'Length': detection['size'][1],
                    'Height': detection['size'][2],
                    'Yaw': yaw,
                    'Pitch': pitch,
                    'Roll': roll,
                    'VelX': detection['velocity'][0],
                    'VelY': detection['velocity'][1],
                    'Class': detection['detection_name'],
                    'Probability': detection['detection_score'],
                })

        self.df = pd.DataFrame(records, columns=self.columns)
        self._filter_detections()

    def _filter_detections(self):
        self.df = self.df[self.df['Probability'] >= 0.5].reset_index(drop=True)



if __name__ == "__main__":
    filepath = r"D:\Programiki do nauki i inne\Szkolne\Studia\Projekt inzynierski\Logi NuScenes\scene-0103\results_nusc_detr3d.json"
    test_parser = SystemDataParser(filepath)
    test_parser.parse()

    db = 0