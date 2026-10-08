import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from Scene import Scene
from tools.NuScenesVisualizer import NuscCameraVisualizer
from tools.CameraVisualizer import CameraVisualizer
from tools.TrackingEvaluation import TrackingEvaluator
from tools.ThresholdAnalysis import ThresholdAnalysis

class ResultBuilder:

    CLASS_MAP = {
        'vehicle.car':                          'car',
        'vehicle.truck':                        'truck',
        'vehicle.bus.bendy':                    'bus',
        'vehicle.bus.rigid':                    'bus',
        'vehicle.trailer':                      'trailer',
        'vehicle.construction':                 'construction_vehicle',
        'human.pedestrian.adult':               'pedestrian',
        'human.pedestrian.child':               'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer':      'pedestrian',
        'vehicle.motorcycle':                   'motorcycle',
        'vehicle.bicycle':                      'bicycle',
        'movable_object.trafficcone':           'traffic_cone',
        'movable_object.barrier':               'barrier',
    }

    def __init__(self, cfg):
        self.cfg = cfg
        self.path = cfg.path
        self.dataset = []
        self.df = None
        self.adatrack_df = None

    def build(self):
        scene_dirs = [d for d in os.listdir(self.path)
                      if os.path.isdir(os.path.join(self.path, d))
                      and d.startswith('scene-')
                      and os.path.exists(os.path.join(self.path, d, 'results_nusc_detr3d.json'))
                      and os.path.exists(os.path.join(self.path, d, 'results_nusc.json'))]

        output_root = self.cfg.output_dir

        if self.cfg.selected_scenes:
            allowed = set(self.cfg.selected_scenes)
            scene_dirs = [d for d in scene_dirs if d in allowed]

        for scene_dir in tqdm(scene_dirs, desc='Building scenes'):
            scene_path = os.path.join(self.path, scene_dir)

            scene = Scene(scene_path, self.cfg, self.CLASS_MAP)
            scene.build_scene()

            scene.paired_df = scene.build_pairs(scene.detr_data)
            scene.adatrack_paired_df = scene.build_pairs(scene.adatrack_data)

            scene.paired_df['Scene'] = scene_dir
            scene.adatrack_paired_df['Scene'] = scene_dir

            scene_out = os.path.join(output_root, scene_dir)
            detr_out = os.path.join(scene_out, 'DETR')
            ada_out = os.path.join(scene_out, 'ADATRACK')

            self.calculate_metrics(scene.paired_df, detr_out)
            self.calculate_metrics(scene.adatrack_paired_df, ada_out,
                                   extra_sheets=self._tracking_sheets(scene.adatrack_paired_df))

            if self.cfg.visualize:
                scene.visualize_scene(detr_out, scene.paired_df, visualizer_type=self.cfg.visualizer_type)
                scene.visualize_scene(ada_out, scene.adatrack_paired_df, visualizer_type=self.cfg.visualizer_type)

            self.dataset.append(scene)

        self.df = pd.concat([s.paired_df for s in self.dataset], ignore_index=True)
        self.adatrack_df = pd.concat([s.adatrack_paired_df for s in self.dataset], ignore_index=True)

        self.calculate_metrics(self.df, os.path.join(output_root, 'DETR'))
        self.calculate_metrics(self.adatrack_df, os.path.join(output_root, 'ADATRACK'),
                               extra_sheets=self._tracking_sheets(self.adatrack_df))
        # Threshold analysis
        ta = ThresholdAnalysis(self.cfg, self.CLASS_MAP)
        sweeps = {'DETR': ta.sweep(self.df), 'ADATRACK': ta.sweep(self.adatrack_df)}
        ta.export(sweeps, os.path.join(output_root, 'THRESHOLDS'))

    def calculate_metrics(self, df: pd.DataFrame, output_dir: str, iou_threshold=None, extra_sheets=None):
        iou_threshold = self.cfg.iou_threshold if iou_threshold is None else iou_threshold
        df = df.copy()
        df['Ref_Class_Mapped'] = df['Ref_Class'].map(self.CLASS_MAP)

        classes = sorted(set(df['Sys_Class'].dropna()) | set(df['Ref_Class_Mapped'].dropna()))
        labels = classes + ['None']

        cm = pd.DataFrame(0, index=labels, columns=labels, dtype=int)

        for _, row in df.iterrows():
            if row['Paired']:
                ref_c = row['Ref_Class_Mapped']
                sys_c = row['Sys_Class']
                if pd.isna(ref_c):
                    continue
                if row['IoU'] >= iou_threshold:
                    cm.loc[ref_c, sys_c] += 1
                else:
                    cm.loc[ref_c, 'None'] += 1
                    cm.loc['None', sys_c] += 1
            else:
                if pd.notna(row['Sys_Class']):
                    cm.loc['None', row['Sys_Class']] += 1
                elif pd.notna(row['Ref_Class_Mapped']):
                    cm.loc[row['Ref_Class_Mapped'], 'None'] += 1

        tp_mask = (df['Paired'] &
                   (df['IoU'] >= iou_threshold) &
                   (df['Sys_Class'] == df['Ref_Class_Mapped']))
        df_tp = df[tp_mask].copy()
        df_tp['AVE'] = np.sqrt(
            (df_tp['Sys_VelX'] - df_tp['Ref_VelX']) ** 2 +
            (df_tp['Sys_VelY'] - df_tp['Ref_VelY']) ** 2
        )
        ave_by_class = df_tp.groupby('Sys_Class')['AVE'].mean()
        matched = df[df['Paired'] & df['Ref_Class_Mapped'].notna()]
        iou_stats = matched.groupby('Ref_Class_Mapped')['IoU'].agg(['mean', 'sum', 'count'])

        rows = []
        for c in classes:
            tp = cm.loc[c, c]
            fp = cm[c].sum() - tp
            fn = cm.loc[c].sum() - tp
            support = int(cm.loc[c].sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
            rows.append({
                'Class': c, 'TP': tp, 'FP': fp, 'FN': fn, 'Support': support,
                'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy,
                'mAVE': ave_by_class.get(c, np.nan),
                'N_Matched': int(iou_stats['count'].get(c, 0)),
                'mIoU': iou_stats['mean'].get(c, np.nan),
                'mIoU_GT': iou_stats['sum'].get(c, 0.0) / support if support else np.nan,
            })

        metrics_df = pd.DataFrame(rows)
        tp_all, fp_all, fn_all = metrics_df[['TP', 'FP', 'FN']].sum()
        total_support = int(metrics_df['Support'].sum())
        p = tp_all / (tp_all + fp_all) if (tp_all + fp_all) else 0.0
        r = tp_all / (tp_all + fn_all) if (tp_all + fn_all) else 0.0

        metrics_df = pd.concat([metrics_df, pd.DataFrame([{
            'Class': 'OVERALL', 'TP': tp_all, 'FP': fp_all, 'FN': fn_all,
            'Support': total_support,
            'Precision': p, 'Recall': r,
            'F1': 2 * p * r / (p + r) if (p + r) else 0.0,
            'Accuracy': tp_all / (tp_all + fp_all + fn_all) if (tp_all + fp_all + fn_all) else 0.0,
            'mAVE': df_tp['AVE'].mean(),
            'N_Matched': len(matched),
            'mIoU': matched['IoU'].mean(),
            'mIoU_GT': matched['IoU'].sum() / total_support if total_support else np.nan,
        }])], ignore_index=True)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'metrics.xlsx')

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            cm.to_excel(writer, sheet_name='Confusion Matrix')
            for name, sheet in (extra_sheets or {}).items():
                sheet.to_excel(writer, sheet_name=name, index=False)

        return metrics_df, cm

    def _tracking_sheets(self, paired_df):
        return TrackingEvaluator(paired_df, self.CLASS_MAP, self.cfg.iou_threshold).sheets()