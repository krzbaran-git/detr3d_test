import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from Scene import Scene

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

    def __init__(self, path, vis=False):
        self.path      = path
        self.dataset   = []
        self.df        = None
        self.visualize = vis

    def build(self):
        scene_dirs = [d for d in os.listdir(self.path)
                      if os.path.isdir(os.path.join(self.path, d))
                      and d.startswith('scene-')]

        for scene_dir in tqdm(scene_dirs, desc='Building scenes'):
            scene_path = os.path.join(self.path, scene_dir)

            scene = Scene(scene_path)
            scene.build_scene()
            scene.build_pairs()

            scene.paired_df['Scene'] = scene_dir

            if self.visualize:
                output_path = os.path.join(self.path, 'output', scene_dir)
                scene.visualize_scene(output_path)

            self.dataset.append(scene)

        self.df = pd.concat([scene.paired_df for scene in self.dataset], ignore_index=True)

    def calculate_metrics(self, output_dir='.', iou_threshold=0.5):
        df = self.df.copy()
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
                if row['PascalMeasure'] >= iou_threshold:
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
                   (df['PascalMeasure'] >= iou_threshold) &
                   (df['Sys_Class'] == df['Ref_Class_Mapped']))
        df_tp = df[tp_mask].copy()
        df_tp['AVE'] = np.sqrt(
            (df_tp['Sys_VelX'] - df_tp['Ref_VelX']) ** 2 +
            (df_tp['Sys_VelY'] - df_tp['Ref_VelY']) ** 2
        )
        ave_by_class = df_tp.groupby('Sys_Class')['AVE'].mean()

        rows = []
        for c in classes:
            tp = cm.loc[c, c]
            fp = cm[c].sum() - tp
            fn = cm.loc[c].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
            mave = ave_by_class.get(c, np.nan)
            rows.append({
                'Class': c, 'TP': tp, 'FP': fp, 'FN': fn, 'Support': int(cm.loc[c].sum()),
                'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy,
                'mAVE': mave,
            })

        metrics_df = pd.DataFrame(rows)

        tp_all, fp_all, fn_all = metrics_df[['TP', 'FP', 'FN']].sum()
        p = tp_all / (tp_all + fp_all) if (tp_all + fp_all) else 0.0
        r = tp_all / (tp_all + fn_all) if (tp_all + fn_all) else 0.0
        metrics_df = pd.concat([metrics_df, pd.DataFrame([{
            'Class': 'OVERALL', 'TP': tp_all, 'FP': fp_all, 'FN': fn_all,
            'Support': int(metrics_df['Support'].sum()),
            'Precision': p, 'Recall': r,
            'F1': 2 * p * r / (p + r) if (p + r) else 0.0,
            'Accuracy': tp_all / (tp_all + fp_all + fn_all) if (tp_all + fp_all + fn_all) else 0.0,
            'mAVE': df_tp['AVE'].mean(),
        }])], ignore_index=True)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'metrics.xlsx')

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            cm.to_excel(writer, sheet_name='Confusion Matrix')

        return metrics_df, cm