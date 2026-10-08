import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ThresholdAnalysis:

    def __init__(self, cfg, class_map: dict):
        self.cfg = cfg
        self.class_map = class_map
        self.thresholds = np.array(cfg.sweep_thresholds, dtype=float)

    def _prep(self, paired_df: pd.DataFrame) -> pd.DataFrame:
        df = paired_df.copy()
        df['IoU'] = pd.to_numeric(df['IoU'], errors='coerce')
        df['Prob'] = pd.to_numeric(df['Sys_Probability'], errors='coerce')
        df['Ref_Class_Mapped'] = df['Ref_Class'].map(self.class_map)
        df['Is_Sys'] = df['Sys_Class'].notna()
        df['Is_Ref'] = df['Ref_Class_Mapped'].notna()
        df['Hit'] = (df['Paired'].astype(bool)
                     & (df['IoU'] >= self.cfg.iou_threshold)
                     & (df['Sys_Class'] == df['Ref_Class_Mapped']))
        return df

    def sweep(self, paired_df: pd.DataFrame) -> pd.DataFrame:
        df = self._prep(paired_df)
        keys = ['Scene', 'Sample token'] if 'Scene' in df.columns else ['Sample token']
        n_frames = max(len(df[keys].drop_duplicates()), 1)

        classes = sorted(set(df['Sys_Class'].dropna()) | set(df['Ref_Class_Mapped'].dropna()))
        rows = []
        for name in [None] + classes:
            if name is None:
                ref_mask, sys_mask = df['Is_Ref'], df['Is_Sys']
            else:
                ref_mask = df['Ref_Class_Mapped'] == name
                sys_mask = df['Sys_Class'] == name
            n_ref = int(ref_mask.sum())

            for t in self.thresholds:
                keep = sys_mask & (df['Prob'] >= t)
                n_sys = int(keep.sum())
                tp = int((keep & df['Hit']).sum())
                p = tp / n_sys if n_sys else np.nan
                r = tp / n_ref if n_ref else np.nan
                f1 = 2 * p * r / (p + r) if (p == p and r == r and (p + r) > 0) else np.nan
                rows.append({
                    'Class': 'OVERALL' if name is None else name,
                    'Threshold': t, 'Sys': n_sys, 'GT': n_ref, 'TP': tp,
                    'Precision': p, 'Recall': r, 'F1': f1,
                    'Dets_per_frame': n_sys / n_frames,
                    'GT_per_frame': n_ref / n_frames,
                })
        return pd.DataFrame(rows)

    @staticmethod
    def best_f1(sweeps: dict) -> pd.DataFrame:
        rows = []
        for net, sw in sweeps.items():
            for cls, g in sw.groupby('Class'):
                g = g.dropna(subset=['F1'])
                if g.empty:
                    continue
                b = g.loc[g['F1'].idxmax()]
                rows.append({'Network': net, 'Class': cls, 'Best_Threshold': b['Threshold'],
                             'F1': b['F1'], 'Precision': b['Precision'], 'Recall': b['Recall'],
                             'Dets_per_frame': b['Dets_per_frame']})
        return pd.DataFrame(rows)

    def plot(self, sweeps: dict, output_dir: str):
        classes = sorted({c for sw in sweeps.values() for c in sw['Class'].unique()},
                         key=lambda c: (c != 'OVERALL', c))
        for cls in classes:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            gt_pf = None
            for net, sw in sweeps.items():
                s = sw[sw['Class'] == cls].sort_values('Threshold')
                if s.empty:
                    continue
                gt_pf = s['GT_per_frame'].iloc[0]
                ax1.plot(s['Recall'], s['Precision'], marker='o', ms=3, label=net)
                ax2.plot(s['Threshold'], s['Dets_per_frame'], marker='o', ms=3, label=net)
            if gt_pf is not None:
                ax2.axhline(gt_pf, ls='--', color='k', label='referencja (GT)')
            ax1.set(title=f'Precision–Recall: {cls}', xlabel='Recall', ylabel='Precision',
                    xlim=(0, 1), ylim=(0, 1.02))
            ax2.set(title=f'Detekcje na klatkę: {cls}', xlabel='Próg score',
                    ylabel='detekcji / klatkę', yscale='log')
            for ax in (ax1, ax2):
                ax.grid(alpha=0.3)
                ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'PR_{cls}.png'), dpi=130)
            plt.close(fig)

    def export(self, sweeps: dict, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        with pd.ExcelWriter(os.path.join(output_dir, 'threshold_sweep.xlsx'), engine='openpyxl') as w:
            for net, sw in sweeps.items():
                sw.to_excel(w, sheet_name=net, index=False)
            self.best_f1(sweeps).to_excel(w, sheet_name='Best_F1', index=False)
        self.plot(sweeps, output_dir)