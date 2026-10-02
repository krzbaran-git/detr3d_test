import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed

from tools.functions import load_json
from tools.CameraVisualizer import CameraVisualizer
from tools.MapBEVVisualizer import MapBEVVisualizer
from tools.ObjectsMatcher import ObjectsMatcher
from Sensors.Sensors import LidarSensor, CameraSensor, RadarSensor
from DataParsers.DetrDataParser import DetrDataParser
from DataParsers.AdatrackDataParser import AdatrackDataParser
from DataParsers.LabelingDataParser import LabelingDataParser

class Scene:
    def __init__(self, scene_path, cfg):
        self.cfg = cfg

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
        self.detr_data = None
        self.adatrack_data = None
        self.ref_data = None
        self.detr_paired_df = None
        self.adatrack_paired_df = None


    # Building data
    def build_scene(self):
        self._get_scene_parameters()
        self._get_sensors()
        self._get_ego_pose()
        self._parse_detr_data()
        self._parse_adatrack_data()
        self._parse_labeling_data()
        self._enrich_with_sample_data()
        self._enrich_with_cameras()
        # self.global_to_ego()

    def _parse_detr_data(self):
        path = self.scene_path + '/results_nusc_detr3d.json'
        detr_parser = DetrDataParser(path, self.cfg.detr_score_threshold)
        detr_parser.parse()
        self.detr_data = detr_parser.df

    def _parse_adatrack_data(self):
        path = self.scene_path + '/results_nusc.json'
        parser = AdatrackDataParser(path, self.cfg.adatrack_score_threshold)
        parser.parse()
        self.adatrack_data = parser.df

    def _enrich_with_sample_data(self):
        timestamp_lookup = {}
        sensor_lookup = {}

        for entry in self.sample_data:
            token = entry['sample_token']
            if token not in timestamp_lookup:
                timestamp_lookup[token] = entry['timestamp']
                sensor_lookup[token] = entry['calibrated_sensor_token']

        self.detr_data['Timestamp'] = self.detr_data['Sample token'].map(timestamp_lookup)
        self.detr_data['Sensor ID'] = self.detr_data['Sample token'].map(sensor_lookup)

        self.adatrack_data['Timestamp'] = self.adatrack_data['Sample token'].map(timestamp_lookup)
        self.adatrack_data['Sensor ID'] = self.adatrack_data['Sample token'].map(sensor_lookup)

    def _enrich_with_cameras(self):
        cam_lookup = {}
        ego_lookup = {}

        for entry in self.sample_data:
            token = entry['sample_token']
            sensor = self.sensors.get(entry['calibrated_sensor_token'])
            if isinstance(sensor, CameraSensor):
                cam_lookup.setdefault(token, {})[sensor.channel] = entry['calibrated_sensor_token']
            ego_lookup.setdefault(token, {})[entry['calibrated_sensor_token']] = entry['ego_pose_token']

        self.detr_data['Cameras'] = self.detr_data['Sample token'].map(cam_lookup)
        self.detr_data['Ego poses'] = self.detr_data['Sample token'].map(ego_lookup)
        self.adatrack_data['Cameras'] = self.adatrack_data['Sample token'].map(cam_lookup)
        self.adatrack_data['Ego poses'] = self.adatrack_data['Sample token'].map(ego_lookup)
        self.labeling_data['Cameras'] = self.labeling_data['Sample token'].map(cam_lookup)
        self.labeling_data['Ego poses'] = self.labeling_data['Sample token'].map(ego_lookup)

    def _parse_labeling_data(self):
        path = self.scene_path + f'/{self.scene_name}' + '/v1.0-trainval/sample_annotation.json'
        parser = LabelingDataParser(path)

        valid_tokens = set()
        timestamp_lookup = {}
        for entry in self.sample_data:
            if entry['is_key_frame']:
                valid_tokens.add(entry['sample_token'])
                timestamp_lookup.setdefault(entry['sample_token'], entry['timestamp'])

        parser.parse(valid_tokens=valid_tokens, timestamp_lookup=timestamp_lookup)
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

    def _get_map_path(self):
        map_json = load_json(self.scene_path + f'/{self.scene_name}' + '/v1.0-trainval/map.json')
        entry = next(m for m in map_json if self.log_token in m['log_tokens'])
        return os.path.join(self.scene_path, self.scene_name, entry['filename'])

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

    # Data evaluation
    def build_pairs(self, detections: pd.DataFrame) -> pd.DataFrame:
        matcher = ObjectsMatcher(self.cfg)
        tokens = detections['Sample token'].unique()

        sys_samples = {t: detections[detections['Sample token'] == t].reset_index(drop=True)
                       for t in tokens}
        ref_samples = {t: self.labeling_data[self.labeling_data['Sample token'] == t].reset_index(drop=True)
                       for t in tokens}

        results = Parallel(n_jobs=self.cfg.n_jobs)(
            delayed(matcher.match)(sys_samples[t], ref_samples[t]) for t in tokens
        )
        records = []

        for sample_token, (pairs, unmatched_sys, unmatched_ref) in zip(tokens, results):
            sys_sample = sys_samples[sample_token]
            ref_sample = ref_samples[sample_token]

            for s, r, iou in pairs:
                sys_row = sys_sample.iloc[s]
                ref_row = ref_sample.iloc[r]
                dist = np.sqrt(
                    (sys_row['PosX'] - ref_row['PosX']) ** 2 +
                    (sys_row['PosY'] - ref_row['PosY']) ** 2 +
                    (sys_row['PosZ'] - ref_row['PosZ']) ** 2
                )
                records.append({
                    'Sample token': sample_token,
                    'Paired': True,
                    'Distance': dist,
                    'IoU': iou,
                    'Sys_PosX': sys_row['PosX'], 'Sys_PosY': sys_row['PosY'], 'Sys_PosZ': sys_row['PosZ'],
                    'Sys_Yaw': sys_row['Yaw'], 'Sys_Pitch': sys_row['Pitch'], 'Sys_Roll': sys_row['Roll'],
                    'Sys_Width': sys_row['Width'], 'Sys_Length': sys_row['Length'], 'Sys_Height': sys_row['Height'],
                    'Sys_VelX': sys_row['VelX'], 'Sys_VelY': sys_row['VelY'],
                    'Sys_Class': sys_row['Class'], 'Sys_Probability': sys_row['Probability'],
                    'Ref_PosX': ref_row['PosX'], 'Ref_PosY': ref_row['PosY'], 'Ref_PosZ': ref_row['PosZ'],
                    'Ref_Yaw': ref_row['Yaw'], 'Ref_Pitch': ref_row['Pitch'], 'Ref_Roll': ref_row['Roll'],
                    'Ref_Width': ref_row['Width'], 'Ref_Length': ref_row['Length'], 'Ref_Height': ref_row['Height'],
                    'Ref_VelX': ref_row['VelX'], 'Ref_VelY': ref_row['VelY'],
                    'Ref_Class': ref_row['Class'],
                })

            # Unpaired sys detections
            for s in unmatched_sys:
                sys_row = sys_sample.iloc[s]
                records.append({
                    'Sample token': sample_token,
                    'Paired': False,
                    'Distance': None,
                    'IoU': None,
                    'Sys_PosX': sys_row['PosX'], 'Sys_PosY': sys_row['PosY'], 'Sys_PosZ': sys_row['PosZ'],
                    'Sys_Yaw': sys_row['Yaw'], 'Sys_Pitch': sys_row['Pitch'], 'Sys_Roll': sys_row['Roll'],
                    'Sys_Width': sys_row['Width'], 'Sys_Length': sys_row['Length'], 'Sys_Height': sys_row['Height'],
                    'Sys_VelX': sys_row['VelX'], 'Sys_VelY': sys_row['VelY'],
                    'Sys_Class': sys_row['Class'], 'Sys_Probability': sys_row['Probability'],
                    'Ref_PosX': None, 'Ref_PosY': None, 'Ref_PosZ': None,
                    'Ref_Yaw': None, 'Ref_Pitch': None, 'Ref_Roll': None,
                    'Ref_Width': None, 'Ref_Length': None, 'Ref_Height': None,
                    'Ref_VelX': None, 'Ref_VelY': None,
                    'Ref_Class': None,
                })

            # Unpaired reference objects
            for r in unmatched_ref:
                ref_row = ref_sample.iloc[r]
                records.append({
                    'Sample token': sample_token,
                    'Paired': False,
                    'Distance': None,
                    'IoU': None,
                    'Sys_PosX': None, 'Sys_PosY': None, 'Sys_PosZ': None,
                    'Sys_Yaw': None, 'Sys_Pitch': None, 'Sys_Roll': None,
                    'Sys_Width': None, 'Sys_Length': None, 'Sys_Height': None,
                    'Sys_VelX': None, 'Sys_VelY': None,
                    'Sys_Class': None, 'Sys_Probability': None,
                    'Ref_PosX': ref_row['PosX'], 'Ref_PosY': ref_row['PosY'], 'Ref_PosZ': ref_row['PosZ'],
                    'Ref_Yaw': ref_row['Yaw'], 'Ref_Pitch': ref_row['Pitch'], 'Ref_Roll': ref_row['Roll'],
                    'Ref_Width': ref_row['Width'], 'Ref_Length': ref_row['Length'], 'Ref_Height': ref_row['Height'],
                    'Ref_VelX': ref_row['VelX'], 'Ref_VelY': ref_row['VelY'],
                    'Ref_Class': ref_row['Class'],
                })

        return pd.DataFrame(records)

    # Visualization
    def visualize_scene(self, output_path: str, paired_df: pd.DataFrame, visualizer_type = CameraVisualizer):
        frame_lookup = self._build_frame_lookup()

        for sample_token in paired_df['Sample token'].unique():
            frame_name = frame_lookup[sample_token]

            sample_entries = {
                entry['channel']: entry
                for entry in self.sample_data
                if entry['sample_token'] == sample_token
                   and entry['is_key_frame'] == True
                   and isinstance(self.sensors.get(entry['calibrated_sensor_token']), CameraSensor)
            }

            sample_df = paired_df[paired_df['Sample token'] == sample_token]
            paired = sample_df[sample_df['Paired'] == True]
            unpaired_sys = sample_df[(sample_df['Paired'] == False) & (sample_df['Sys_PosX'].notna())]
            unpaired_ref = sample_df[(sample_df['Paired'] == False) & (sample_df['Ref_PosX'].notna())]

            for channel, entry in sample_entries.items():
                camera = self.sensors[entry['calibrated_sensor_token']]
                ego = self.ego_pose[entry['ego_pose_token']]
                image_path = os.path.join(self.scene_path, self.scene_name, entry['filename'])

                vis = visualizer_type(camera, image_path)
                vis.render(paired, unpaired_sys, unpaired_ref, ego)
                vis.save(os.path.join(output_path, channel, f'{frame_name}.jpg'))

        self._visualize_map_bev(output_path, frame_lookup, paired_df)

    def _build_frame_lookup(self):
        ts = {}
        for entry in self.sample_data:
            if entry['is_key_frame'] and entry['sample_token'] not in ts:
                ts[entry['sample_token']] = entry['timestamp']
        ordered = sorted(ts, key=lambda t: ts[t])
        return {token: f'Frame_{i + 1}' for i, token in enumerate(ordered)}

    def _visualize_map_bev(self, output_path: str, frame_lookup: dict, paired_df: pd.DataFrame):
        map_path = self._get_map_path()
        bev = MapBEVVisualizer(paired_df, map_path, self.ego_pose, self.sample_data)
        bev.save_all(os.path.join(output_path, 'MAP_BEV'), frame_lookup)


if __name__ == '__main__':
    test_scene_path = r"D:/Programiki do nauki i inne/Szkolne/Studia/Projekt inzynierski/Logi NuScenes/scene-0103"
    test_scene = Scene(test_scene_path)
    test_scene.build_scene()
    test_scene.build_pairs()
    test_scene.visualize_scene(r'D:\Programiki do nauki i inne\Szkolne\Studia\Projekt inzynierski\Logi NuScenes\scene-0103\scene-0103\output')
    db = 0