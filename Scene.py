import numpy as np
import pandas as pd
import os
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed

from tools.functions import load_json
from tools.PascalMeasure import PascalMeasure3D
from tools.CameraVisualizer import CameraVisualizer
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
        self.paired_df = None


    # Building data
    def build_scene(self):
        self._get_scene_parameters()
        self._get_sensors()
        self._get_ego_pose()
        self._parse_system_data()
        self._parse_labeling_data()
        self._enrich_with_cameras()
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
            lambda t: lookup.get(t, {}).get('Timestamp'))
        self.system_data['Sensor ID'] = self.system_data['Sample token'].map(
            lambda t: lookup.get(t, {}).get('Sensor ID'))

    def _enrich_with_cameras(self):
        cam_lookup = {}
        for entry in self.sample_data:
            token = entry['sample_token']
            sensor = self.sensors.get(entry['calibrated_sensor_token'])
            if isinstance(sensor, CameraSensor):
                cam_lookup.setdefault(token, {})[sensor.channel] = entry['calibrated_sensor_token']

        self.system_data['Cameras'] = self.system_data['Sample token'].map(cam_lookup)
        self.labeling_data['Cameras'] = self.labeling_data['Sample token'].map(cam_lookup)

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

            self.sensors[token] = new_sensor

    def _get_ego_pose(self):
        ego_pose_path = self.scene_path + f'/{self.scene_name}' + '/v1.0-trainval/ego_pose.json'
        ego_pose = load_json(ego_pose_path)
        self.ego_pose = {entry['token']: entry for entry in ego_pose}

    def _get_scene_parameters(self):
        self.scene_name = os.path.basename(self.scene_path)
        params_path = self.scene_path + f'/{self.scene_name}' + '/v1.0-trainval/scene.json'
        params_dict = load_json(params_path)[0]
        self.scene_token = params_dict['token']
        self.log_token = params_dict['log_token']
        self.first_sample_token = params_dict['first_sample_token']
        self.last_sample_token = params_dict['last_sample_token']
        self.sample_data = load_json(self.scene_path + f'/{self.scene_name}' + '/v1.0-trainval/sample_data.json')

    def _find_by_token(self, data: list[dict], token_name: str, token_value: str) -> dict | None:
        return next((item for item in data if item[token_name] == token_value), None)

    def global_to_ego(self):
        ego_token_lookup = {
            entry['sample_token']: entry['ego_pose_token']
            for entry in self.sample_data
        }

        def transform(row):
            ego_pose_token = ego_token_lookup[row['Sample token']]
            ego = self.ego_pose[ego_pose_token]
            r = Rotation.from_quat([*ego['rotation'][1:], ego['rotation'][0]])

            # pozycja
            point = np.array([row['PosX'], row['PosY'], row['PosZ']]) - np.array(ego['translation'])
            pos_ego = r.inv().apply(point)

            # orientacja
            r_obj = Rotation.from_euler('ZYX', [row['Yaw'], row['Pitch'], row['Roll']])
            r_obj_ego = r.inv() * r_obj
            yaw, pitch, roll = r_obj_ego.as_euler('ZYX')

            return pd.Series([*pos_ego, yaw, pitch, roll], index=['PosX', 'PosY', 'PosZ', 'Yaw', 'Pitch', 'Roll'])

        for df in [self.system_data, self.labeling_data]:
            if df is not None:
                df[['PosX', 'PosY', 'PosZ', 'Yaw', 'Pitch', 'Roll']] = df.apply(transform, axis=1)


    # Data evaluation
    def build_pairs(self, max_match_dist: float = 3.0):
        from scipy.optimize import linear_sum_assignment

        records = []

        for sample_token in self.system_data['Sample token'].unique():
            sys_sample = self.system_data[self.system_data['Sample token'] == sample_token]
            ref_sample = self.labeling_data[self.labeling_data['Sample token'] == sample_token]

            sys_positions = sys_sample[['PosX', 'PosY', 'PosZ']].values
            ref_positions = ref_sample[['PosX', 'PosY', 'PosZ']].values

            diff = sys_positions[:, np.newaxis, :] - ref_positions[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)

            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            matched_sys = set()
            matched_ref = set()

            for sys_idx, ref_idx in zip(row_ind, col_ind):
                sys_row = sys_sample.iloc[sys_idx]
                ref_row = ref_sample.iloc[ref_idx]
                dist = dist_matrix[sys_idx, ref_idx]

                if dist <= max_match_dist:
                    matched_sys.add(sys_idx)
                    matched_ref.add(ref_idx)
                    records.append({
                        'Sample token': sample_token,
                        'Paired': True,
                        'Distance': dist,
                        'Sys_PosX': sys_row['PosX'], 'Sys_PosY': sys_row['PosY'], 'Sys_PosZ': sys_row['PosZ'],
                        'Sys_Yaw': sys_row['Yaw'], 'Sys_Pitch': sys_row['Pitch'], 'Sys_Roll': sys_row['Roll'],
                        'Sys_Width': sys_row['Width'], 'Sys_Length': sys_row['Length'], 'Sys_Height': sys_row['Height'],
                        'Sys_Class': sys_row['Class'], 'Sys_Probability': sys_row['Probability'],
                        'Ref_PosX': ref_row['PosX'], 'Ref_PosY': ref_row['PosY'], 'Ref_PosZ': ref_row['PosZ'],
                        'Ref_Yaw': ref_row['Yaw'], 'Ref_Pitch': ref_row['Pitch'], 'Ref_Roll': ref_row['Roll'],
                        'Ref_Width': ref_row['Width'], 'Ref_Length': ref_row['Length'], 'Ref_Height': ref_row['Height'],
                        'Ref_Class': ref_row['Class'],
                        'Cameras': sys_row['Cameras'],
                    })

            # Sys not in ref
            for sys_idx in range(len(sys_sample)):
                if sys_idx not in matched_sys:
                    sys_row = sys_sample.iloc[sys_idx]
                    records.append({
                        'Sample token': sample_token,
                        'Paired': False,
                        'Distance': None,
                        'Sys_PosX': sys_row['PosX'], 'Sys_PosY': sys_row['PosY'], 'Sys_PosZ': sys_row['PosZ'],
                        'Sys_Yaw': sys_row['Yaw'], 'Sys_Pitch': sys_row['Pitch'], 'Sys_Roll': sys_row['Roll'],
                        'Sys_Width': sys_row['Width'], 'Sys_Length': sys_row['Length'], 'Sys_Height': sys_row['Height'],
                        'Sys_Class': sys_row['Class'], 'Sys_Probability': sys_row['Probability'],
                        'Ref_PosX': None, 'Ref_PosY': None, 'Ref_PosZ': None,
                        'Ref_Yaw': None, 'Ref_Pitch': None, 'Ref_Roll': None,
                        'Ref_Width': None, 'Ref_Length': None, 'Ref_Height': None,
                        'Ref_Class': None,
                        'Cameras': sys_row['Cameras'],
                    })

            # Ref not in Sys
            for ref_idx in range(len(ref_sample)):
                if ref_idx not in matched_ref:
                    ref_row = ref_sample.iloc[ref_idx]
                    records.append({
                        'Sample token': sample_token,
                        'Paired': False,
                        'Distance': None,
                        'Sys_PosX': None, 'Sys_PosY': None, 'Sys_PosZ': None,
                        'Sys_Yaw': None, 'Sys_Pitch': None, 'Sys_Roll': None,
                        'Sys_Width': None, 'Sys_Length': None, 'Sys_Height': None,
                        'Sys_Class': None, 'Sys_Probability': None,
                        'Ref_PosX': ref_row['PosX'], 'Ref_PosY': ref_row['PosY'], 'Ref_PosZ': ref_row['PosZ'],
                        'Ref_Yaw': ref_row['Yaw'], 'Ref_Pitch': ref_row['Pitch'], 'Ref_Roll': ref_row['Roll'],
                        'Ref_Width': ref_row['Width'], 'Ref_Length': ref_row['Length'], 'Ref_Height': ref_row['Height'],
                        'Ref_Class': ref_row['Class'],
                        'Cameras': ref_row['Cameras'],
                    })

        self.paired_df = pd.DataFrame(records)
        self._calculate_pascal_measure()

    def _calculate_pascal_measure(self, n_samples: int = 10_000, n_jobs: int = -1):

        def _compute_row_iou(row):
            obj1 = {
                'PosX': row['Sys_PosX'], 'PosY': row['Sys_PosY'], 'PosZ': row['Sys_PosZ'],
                'Yaw': row['Sys_Yaw'], 'Pitch': row['Sys_Pitch'], 'Roll': row['Sys_Roll'],
                'Width': row['Sys_Width'], 'Length': row['Sys_Length'], 'Height': row['Sys_Height'],
            }
            obj2 = {
                'PosX': row['Ref_PosX'], 'PosY': row['Ref_PosY'], 'PosZ': row['Ref_PosZ'],
                'Yaw': row['Ref_Yaw'], 'Pitch': row['Ref_Pitch'], 'Roll': row['Ref_Roll'],
                'Width': row['Ref_Width'], 'Length': row['Ref_Length'], 'Height': row['Ref_Height'],
            }
            return PascalMeasure3D(obj1, obj2, n_samples).compute_iou()

        paired = self.paired_df[self.paired_df['Paired'] == True].copy()
        rows = [row for _, row in paired.iterrows()]

        ious = Parallel(n_jobs=n_jobs, backend='threading')(delayed(_compute_row_iou)(row) for row in rows)
        self.paired_df.loc[paired.index, 'PascalMeasure'] = ious


    # Visualization
    def visualize_scene(self, output_path: str):
        for sample_token in self.paired_df['Sample token'].unique():
            sample_entries = {
                entry['channel']: entry
                for entry in self.sample_data
                if entry['sample_token'] == sample_token
                   and isinstance(self.sensors.get(entry['calibrated_sensor_token']), CameraSensor)
            }

            paired = self.paired_df[
                (self.paired_df['Sample token'] == sample_token) & (self.paired_df['Paired'] == True)]
            unpaired_sys = self.paired_df[
                (self.paired_df['Sample token'] == sample_token) & (self.paired_df['Paired'] == False) & (
                    self.paired_df['Sys_PosX'].notna())]
            unpaired_ref = self.paired_df[
                (self.paired_df['Sample token'] == sample_token) & (self.paired_df['Paired'] == False) & (
                    self.paired_df['Ref_PosX'].notna())]

            for channel, entry in sample_entries.items():
                camera = self.sensors[entry['calibrated_sensor_token']]
                image_path = os.path.join(self.scene_path, self.scene_name, entry['filename'])
                vis = CameraVisualizer(camera, image_path)
                vis.render(paired, unpaired_sys, unpaired_ref)
                vis.save(os.path.join(output_path, channel, os.path.basename(entry['filename'])))


if __name__ == '__main__':
    test_scene_path = r"D:/Programiki do nauki i inne/Szkolne/Studia/Projekt inzynierski/Logi NuScenes/scene-0103"
    test_scene = Scene(test_scene_path)
    test_scene.build_scene()
    test_scene.build_pairs()
    test_scene.visualize_scene(r'D:\Programiki do nauki i inne\Szkolne\Studia\Projekt inzynierski\Logi NuScenes\scene-0103\scene-0103\output')
    db = 0