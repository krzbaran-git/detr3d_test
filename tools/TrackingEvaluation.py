import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


class TrackingEvaluator:

    def __init__(self, paired_df: pd.DataFrame, class_map: dict, iou_threshold: float = 0.5):
        df = paired_df.copy()
        if 'Scene' not in df.columns:
            df['Scene'] = ''
        df['Ref_Class_Mapped'] = df['Ref_Class'].map(class_map)
        df['Match'] = (df['Paired'].astype(bool)
                       & (df['IoU'] >= iou_threshold)
                       & (df['Sys_Class'] == df['Ref_Class_Mapped']))
        self.df = df
        self.classes = sorted(set(df['Ref_Class_Mapped'].dropna()) | set(df['Sys_Class'].dropna()))

    def evaluate(self, cls=None, df=None):
        d = self.df if df is None else df
        ref = d[d['Ref_Instance'].notna()]
        sys = d[d['Sys_TrackID'].notna()]
        if cls is not None:
            ref = ref[ref['Ref_Class_Mapped'] == cls]
            sys = sys[sys['Sys_Class'] == cls]
        matched = sys[sys['Match']]

        n_gt, n_sys, tp = len(ref), len(sys), len(matched)
        fn, fp = n_gt - tp, n_sys - tp

        gt_tracks = self._gt_tracks(ref)
        sys_tracks = self._sys_tracks(sys)

        idsw = int(gt_tracks['IDSW'].sum()) if len(gt_tracks) else 0
        frag = int(gt_tracks['Frag'].sum()) if len(gt_tracks) else 0
        ratio = gt_tracks['Tracked_Ratio'] if len(gt_tracks) else pd.Series(dtype=float)

        idtp = self._idtp(ref)

        summary = {
            'GT': n_gt, 'Sys': n_sys, 'TP': tp, 'FP': fp, 'FN': fn,
            'IDSW': idsw, 'Frag': frag,
            'Tracks_GT': len(gt_tracks), 'Tracks_Sys': len(sys_tracks),
            'MT': int((ratio >= 0.8).sum()),
            'PT': int(((ratio > 0.2) & (ratio < 0.8)).sum()),
            'ML': int((ratio <= 0.2).sum()),
            'MOTA': 1 - (fn + fp + idsw) / n_gt if n_gt else np.nan,
            'MOTP_dist': ref[ref['Match']]['Distance'].mean(),
            'MOTP_IoU':  ref[ref['Match']]['IoU'].mean(),
            'IDTP': idtp,
            'IDP':  idtp / n_sys if n_sys else np.nan,
            'IDR':  idtp / n_gt if n_gt else np.nan,
            'IDF1': 2 * idtp / (n_gt + n_sys) if (n_gt + n_sys) else np.nan,

            'GT_Len_mean':  gt_tracks['Length'].mean(),
            'GT_Len_median': gt_tracks['Length'].median(),
            'Sys_Len_mean': sys_tracks['Length'].mean(),
            'Sys_Len_median': sys_tracks['Length'].median(),
            'Sys_Len_max':  sys_tracks['Length'].max(),
            'Sys_Short_<=2': int((sys_tracks['Length'] <= 2).sum()),
            'Same_ID_Share_mean': gt_tracks['Same_ID_Share'].mean(),
            'Track_Precision_mean': sys_tracks['Precision'].mean(),
        }
        return summary, gt_tracks, sys_tracks

    @staticmethod
    def _gt_tracks(ref: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (scene, inst), g in ref.groupby(['Scene', 'Ref_Instance']):
            g = g.sort_values('Frame')
            m = g['Match'].to_numpy(dtype=bool)
            ids = g.loc[g['Match'], 'Sys_TrackID'].tolist()

            idsw = sum(a != b for a, b in zip(ids[:-1], ids[1:]))
            starts = m & ~np.r_[False, m[:-1]]
            frag = max(int(starts.sum()) - 1, 0)
            top = pd.Series(ids).value_counts().iloc[0] if ids else 0

            rows.append({
                'Scene': scene, 'Instance': inst, 'Class': g['Ref_Class_Mapped'].iloc[0],
                'First': int(g['Frame'].min()), 'Last': int(g['Frame'].max()),
                'Length': len(g), 'Tracked': int(m.sum()),
                'Tracked_Ratio': m.mean(), 'IDSW': idsw, 'Frag': frag,
                'N_IDs': len(set(ids)),
                'Same_ID_Share': top / len(g),
            })
        return pd.DataFrame(rows, columns=['Scene', 'Instance', 'Class', 'First', 'Last', 'Length',
                                           'Tracked', 'Tracked_Ratio', 'IDSW', 'Frag', 'N_IDs',
                                           'Same_ID_Share'])

    @staticmethod
    def _sys_tracks(sys: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (scene, tid), g in sys.groupby(['Scene', 'Sys_TrackID']):
            fr = g['Frame'].to_numpy()
            rows.append({
                'Scene': scene, 'TrackID': tid, 'Class': g['Sys_Class'].mode().iloc[0],
                'First': int(fr.min()), 'Last': int(fr.max()),
                'Length': len(g),                          # liczba klatek z detekcją
                'Span': int(fr.max() - fr.min() + 1),      # od pierwszej do ostatniej klatki
                'Gaps': int(fr.max() - fr.min() + 1 - len(g)),
                'Matched': int(g['Match'].sum()),
                'Precision': g['Match'].mean(),
                'Mean_Score': g['Sys_Probability'].mean(),
            })
        return pd.DataFrame(rows, columns=['Scene', 'TrackID', 'Class', 'First', 'Last', 'Length',
                                           'Span', 'Gaps', 'Matched', 'Precision', 'Mean_Score'])

    @staticmethod
    def _idtp(ref: pd.DataFrame) -> int:
        m = ref[ref['Match']]
        if m.empty:
            return 0
        ov = (m.assign(G=list(zip(m['Scene'], m['Ref_Instance'])),
                       T=list(zip(m['Scene'], m['Sys_TrackID'])))
                .groupby(['G', 'T']).size())
        gts = {g: i for i, g in enumerate(sorted({k[0] for k in ov.index}, key=str))}
        trs = {t: i for i, t in enumerate(sorted({k[1] for k in ov.index}, key=str))}
        M = np.zeros((len(gts), len(trs)))
        for (g, t), c in ov.items():
            M[gts[g], trs[t]] = c
        r, c = linear_sum_assignment(-M)
        return int(M[r, c].sum())

    def per_scene(self, cls=None) -> pd.DataFrame:
        rows = []
        for scene, g in self.df.groupby('Scene'):
            summary, _, _ = self.evaluate(cls, g)
            if summary['GT'] == 0:
                continue
            rows.append({'Scene': scene, **summary})
        return pd.DataFrame(rows)

    def macro_tables(self):
        mean_rows, std_rows = [], []
        for cls in [None] + self.classes:
            ps = self.per_scene(cls)
            if ps.empty:
                continue
            num = ps.drop(columns='Scene').select_dtypes('number')
            name = 'OVERALL' if cls is None else cls
            mean_rows.append({'Class': name, 'N_Scenes': len(ps), **num.mean().to_dict()})
            std_rows.append({'Class': name, 'N_Scenes': len(ps), **num.std().to_dict()})
        return pd.DataFrame(mean_rows), pd.DataFrame(std_rows)

    def sheets(self) -> dict:
        rows, gt_all, sys_all = [], None, None
        for cls in [None] + self.classes:
            summary, gt_t, sys_t = self.evaluate(cls)
            rows.append({'Class': 'OVERALL' if cls is None else cls, **summary})
            if cls is None:
                gt_all, sys_all = gt_t, sys_t

        mean_df, std_df = self.macro_tables()
        return {
            'Tracking': pd.DataFrame(rows),
            'Tracking_PerScene': self.per_scene(),
            'Tracking_Mean': mean_df,
            'Tracking_Std': std_df,
            'GT_Tracks': gt_all,
            'Sys_Tracks': sys_all,
        }