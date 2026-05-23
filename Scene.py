import numpy as np
import pandas as pd

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
        self.system_data = None
        self.labeling_data = None

    def build_scene(self):
        self._get_scene_parameters()
        self._get_sensors()
        self._get_ego_pose()

    def _parse_system_data(self):
        pass

    def _parse_labeling_data(self):
        pass

    def _get_sensors(self):
        self.sensors = {}
        sensors_path = self.scene_path + f'/{self.scene_name}' + '/v1.0-trainval/sensor.json'
        calib_path = self.scene_path + f'/{self.scene_name}' + '/v1.0-trainval/calibrated_sensor.json'
        sensors = load_json(sensors_path)
        calib = load_json(calib_path)

        def find_by_token(data: list[dict], token: str) -> dict | None:
            return next((item for item in data if item['sensor_token'] == token), None)

        for sensor in sensors:
            sensor_token = sensor['token']
            sensor_channel = sensor['channel']
            sensor_type = sensor['modality']

            sensor_calib = find_by_token(calib, sensor_token)
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
        self.ego_pose = pd.DataFrame(ego_pose)

    def _get_scene_parameters(self):
        self.scene_name = self.scene_path.split('/')[-1]
        params_path = self.scene_path + f'/{self.scene_name}' + '/v1.0-trainval/scene.json' #TODO: Check for directories structure
        params_dict = load_json(params_path)[0]
        self.scene_token = params_dict['token']
        self.log_token = params_dict['log_token']
        self.first_sample_token = params_dict['first_sample_token']
        self.last_sample_token = params_dict['last_sample_token']
        x = 0

    def filter_key_frames(self):
        pass


if __name__ == '__main__':
    test_scene_path = r"D:/Programiki do nauki i inne/Szkolne/Studia/Projekt inzynierski/Logi NuScenes/scene-0103"
    test_scene = Scene(test_scene_path)
    test_scene.build_scene()
    db = 0