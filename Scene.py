import numpy as np
import pandas as pd
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

from tools.functions import load_json
from Sensors.Sensors import LidarSensor, CameraSensor, RadarSensor
from DataParsers.SystemDataParser import SystemDataParser
from DataParsers.LabelingDataParser import LabelingDataParser

class Scene:
    def __init__(self, scene_path):
        # Initialization parameteres (from scene.json file)
        self.scene_path = scene_path
        self.scene_token = None
        self.log_token = None
        self.first_sample_token = None
        self.last_sample_token = None
        self.scene_name = None

        # Data
        self.sensors = None
        self.ego_pose = None
        self.sample_data = None
        self.system_data = None
        self.labeling_data = None

    def build_scene(self):
        self._get_scene_parameters()
        self._get_sensors()
        self._get_ego_pose()
        self._parse_system_data()
        self._parse_labeling_data()
        self.global_to_ego()

    def _parse_system_data(self):
        path = self.scene_path + '/results_nusc_detr3d.json'
        parser = SystemDataParser(path)
        parser.parse()
        self.system_data = parser.df
        self._enrich_with_sample_data()

    def _enrich_with_sample_data(self):
        lookup = {}
        for entry in self.sample_data:
            token = entry['sample_token']
            if token not in lookup:
                lookup[token] = {
                    'Timestamp': entry['timestamp'],
                    'Sensor ID': entry['calibrated_sensor_token'],
                }

        self.system_data['Timestamp'] = self.system_data['Sample token'].map(
            lambda t: lookup.get(t, {}).get('Timestamp')
        )
        self.system_data['Sensor ID'] = self.system_data['Sample token'].map(
            lambda t: lookup.get(t, {}).get('Sensor ID')
        )

    def _parse_labeling_data(self):
        path = self.scene_path + f'/{self.scene_name}' + '/v1.0-trainval/sample_annotation.json'
        parser = LabelingDataParser(path)
        parser.parse()
        self.labeling_data = parser.df

    def _get_sensors(self):
        self.sensors = {}
        sensors_path = self.scene_path + f'/{self.scene_name}' + '/v1.0-trainval/sensor.json'
        calib_path = self.scene_path + f'/{self.scene_name}' + '/v1.0-trainval/calibrated_sensor.json'
        sensors = load_json(sensors_path)
        calib = load_json(calib_path)

        for sensor in sensors:
            sensor_token = sensor['token']
            sensor_channel = sensor['channel']
            sensor_type = sensor['modality']

            sensor_calib = self._find_by_token(calib, 'sensor_token', sensor_token)
            if sensor_calib is not None:
                token = sensor_calib['token']
                translation = sensor_calib['translation']
                rotation = sensor_calib['rotation']
                camera_intrinsic = sensor_calib['camera_intrinsic'] if sensor_type == 'camera' else None

            if sensor_type == 'camera':
                new_sensor = CameraSensor(token, sensor_token, sensor_channel, translation, rotation, camera_intrinsic)
            elif sensor_type == 'lidar':
                new_sensor = LidarSensor(token, sensor_token, sensor_channel, translation, rotation)
            elif sensor_type == 'radar':
                new_sensor = RadarSensor(token, sensor_token, sensor_channel, translation, rotation)
            else:
                new_sensor = None

            self.sensors[sensor_token] = new_sensor

    def _get_ego_pose(self):
        ego_pose_path = self.scene_path + f'/{self.scene_name}' + '/v1.0-trainval/ego_pose.json'
        ego_pose = load_json(ego_pose_path)
        self.ego_pose = {entry['token']: entry for entry in ego_pose}

    def _get_scene_parameters(self):
        self.scene_name = self.scene_path.split('/')[-1]
        params_path = self.scene_path + f'/{self.scene_name}' + '/v1.0-trainval/scene.json'
        params_dict = load_json(params_path)[0]
        self.scene_token = params_dict['token']
        self.log_token = params_dict['log_token']
        self.first_sample_token = params_dict['first_sample_token']
        self.last_sample_token = params_dict['last_sample_token']
        self.sample_data = load_json(self.scene_path + f'/{self.scene_name}' + '/v1.0-trainval/sample_data.json')

    def _find_by_token(self, data: list[dict], token_name: str, token_value: str) -> dict | None:
        return next((item for item in data if item[token_name] == token_value), None)

    def filter_key_frames(self):
        pass

    def global_to_ego(self):
        # lookup: sample_token → ego_pose_token
        ego_token_lookup = {
            entry['sample_token']: entry['ego_pose_token']
            for entry in self.sample_data
        }

        def transform(row):
            ego_pose_token = ego_token_lookup[row['Sample token']]
            ego = self.ego_pose[ego_pose_token]
            r = Rotation.from_quat([*ego['rotation'][1:], ego['rotation'][0]])
            point = np.array([row['PosX'], row['PosY'], row['PosZ']]) - np.array(ego['translation'])
            return pd.Series(r.inv().apply(point), index=['PosX', 'PosY', 'PosZ'])

        for df in [self.system_data, self.labeling_data]:
            if df is not None:
                df[['PosX', 'PosY', 'PosZ']] = df.apply(transform, axis=1)


if __name__ == '__main__':
    test_scene_path = r"D:/Programiki do nauki i inne/Szkolne/Studia/Projekt inzynierski/Logi NuScenes/scene-0103"
    test_scene = Scene(test_scene_path)
    test_scene.build_scene()
    db = 0