import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial.transform import Rotation


class MapBEVVisualizer:
    def __init__(self, paired_df: pd.DataFrame, map_image_path: str,
                 ego_pose: dict, sample_data: list,
                 resolution: float = 0.1, margin: float = 30.0):
        self.df          = paired_df
        self.map_img     = plt.imread(map_image_path)
        self.ego_pose    = ego_pose
        self.resolution  = resolution
        self.margin      = margin
        self.H           = self.map_img.shape[0]

        self.ego_lookup = {}
        for entry in sample_data:
            if entry['is_key_frame'] and entry['sample_token'] not in self.ego_lookup:
                self.ego_lookup[entry['sample_token']] = self.ego_pose[entry['ego_pose_token']]

    def _to_pixel(self, x, y):
        return x / self.resolution, self.H - y / self.resolution

    def plot_sample(self, sample_token: str, ax=None, title=None):
        df = self.df[self.df['Sample token'] == sample_token]

        xs, ys = [], []
        for pref in ('Sys', 'Ref'):
            xs += df[f'{pref}_PosX'].dropna().tolist()
            ys += df[f'{pref}_PosY'].dropna().tolist()
        if not xs:
            return None

        ego = self.ego_lookup.get(sample_token)
        if ego is not None:
            xs.append(ego['translation'][0])
            ys.append(ego['translation'][1])

        col_min, py_max = self._to_pixel(min(xs) - self.margin, min(ys) - self.margin)
        col_max, py_min = self._to_pixel(max(xs) + self.margin, max(ys) + self.margin)
        col_min, col_max = int(max(0, col_min)), int(min(self.map_img.shape[1], col_max))
        row_min, row_max = int(max(0, py_min)),  int(min(self.H, py_max))

        crop = self.map_img[row_min:row_max, col_min:col_max]

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(crop, cmap='gray', extent=[col_min, col_max, row_max, row_min])

        for _, row in df.iterrows():
            style = '-' if row['Paired'] else '--'
            if pd.notna(row['Sys_PosX']):
                self._draw_box(ax, row, 'Sys', 'red',  style)
            if pd.notna(row['Ref_PosX']):
                self._draw_box(ax, row, 'Ref', 'lime', style)
            if row['Paired']:
                sp = self._to_pixel(row['Sys_PosX'], row['Sys_PosY'])
                rp = self._to_pixel(row['Ref_PosX'], row['Ref_PosY'])
                ax.plot([sp[0], rp[0]], [sp[1], rp[1]], color='yellow', lw=0.8)

        if ego is not None:
            self._draw_ego(ax, ego)

        ax.set_title(title or f'BEV — {sample_token[:8]}')
        ax.axis('off')
        return ax

    def _draw_ego(self, ax, ego):
        cx, cy = ego['translation'][0], ego['translation'][1]
        px, py = self._to_pixel(cx, cy)

        yaw = Rotation.from_quat([*ego['rotation'][1:], ego['rotation'][0]]).as_euler('ZYX')[0]

        ax.plot(px, py, marker='o', color='deepskyblue', markersize=9, zorder=5)

        length = 8.0 / self.resolution
        fx = px + length * np.cos(yaw)
        fy = py - length * np.sin(yaw)   # oś Y obrazu odwrócona
        ax.annotate('', xy=(fx, fy), xytext=(px, py),
                    arrowprops=dict(arrowstyle='->', color='deepskyblue', lw=2), zorder=5)

    def _draw_box(self, ax, row, prefix, color, ls='-'):
        cx, cy = row[f'{prefix}_PosX'], row[f'{prefix}_PosY']
        W, L   = row[f'{prefix}_Width'], row[f'{prefix}_Length']
        yaw    = row[f'{prefix}_Yaw']

        hx, hy  = W / 2, L / 2
        corners = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw),  np.cos(yaw)]])
        corners = corners @ R.T + [cx, cy]

        px = [self._to_pixel(x, y) for x, y in corners]
        ax.add_patch(Polygon(px, closed=True, fill=False,
                             edgecolor=color, lw=1.5, linestyle=ls))

    def save_all(self, output_path: str, frame_lookup: dict):
        os.makedirs(output_path, exist_ok=True)
        for sample_token in self.df['Sample token'].unique():
            frame_name = frame_lookup[sample_token]
            ax = self.plot_sample(sample_token, title=f'BEV — {frame_name}')
            if ax is None:
                continue
            ax.figure.savefig(os.path.join(output_path, f'{frame_name}.png'),
                              dpi=120, bbox_inches='tight')
            plt.close(ax.figure)