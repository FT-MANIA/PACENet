import ast
import logging
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

def infer_side(left_label, right_label, target_label):
    left_is_target = str(left_label) == target_label
    right_is_target = str(right_label) == target_label

    if left_is_target and right_is_target:
        return "B"
    elif left_is_target:
        return "L"
    elif right_is_target:
        return "R"
    else:
        return "NA"

def label_creater(left_label, right_label):
    """create label according to the disease type"""
    status_map = {
        'Healthy': 0,
        'ACLD': 1,
        'KOA': 2,
    }
    l_code = status_map.get(left_label, 0)
    r_code = status_map.get(right_label, 0)

    if l_code == 1 or r_code == 1: return 1
    elif l_code == 2 or r_code == 2: return 2
    else: return 0

def df_data_extractor(df):
    """Extract feats from dataframe and create (N, 12, T) array"""
    samples, labels, person_ids = [], [], []
    demographics = []  # Store [gender, age, bmi]
    trace_info = []  # Store {'source_file': ..., 'original_id': ...}

    has_trace_info = 'source_file' in df.columns and 'original_id' in df.columns

    groups = df.groupby('person_id')
    for person_id, group in groups:
        if len(group) != 2: continue
        group = group.sort_values('leg')
        if group.iloc[0]['leg'] != 'left': continue

        left_row, right_row = group.iloc[0], group.iloc[1]
        feat_l = ast.literal_eval(left_row['features'])
        feat_l = np.asarray(feat_l, dtype=float)
        feat_r = ast.literal_eval(right_row['features'])
        feat_r = np.asarray(feat_r, dtype=float)

        if feat_l.shape[0] > feat_l.shape[1]:
            feat_l, feat_r = feat_l.T, feat_r.T

        combined = np.concatenate([feat_l, feat_r], axis=0)
        samples.append(combined)
        lbl = label_creater(left_row['label'], right_row['label'])
        labels.append(lbl)
        person_ids.append(int(person_id))

        gender = float(left_row.get('gender', -1))
        age = float(left_row.get('age', 0))
        bmi = float(left_row.get('bmi', 0))
        demographics.append([gender, age, bmi])

        acld_side = infer_side(left_row['label'], right_row['label'], target_label='ACLD')
        koa_side = infer_side(left_row['label'], right_row['label'], target_label='KOA')

        if lbl == 1:
            affected_side = acld_side
            affected_disease = "ACLD"
        elif lbl == 2:
            affected_side = koa_side
            affected_disease = "KOA"
        else:
            affected_side = "NA"
            affected_disease = "Healthy"

        if has_trace_info:
            src = str(left_row.get('source_file', 'unknown'))
            oid = str(left_row.get('original_id', 'unknown'))
            trace_info.append({'source_file': src,
                               'original_id': oid,
                               'acld_side' : acld_side,
                               'koa_side' : koa_side,
                               'affected_side' : affected_side,
                               'affected_disease' : affected_disease})
        else:
            trace_info.append(None)

    return (np.array(samples), np.array(labels), np.array(person_ids),
            np.array(demographics), np.array(trace_info))

logger = logging.getLogger(__name__)

class adaptive_gait_cycle_segmentation:
    def __init__(self):
        self.target_len = 100
        self.flex_idx = 2

    def estimate_gait_period(self, signal, fs=60):
        """Adaptive estimation of fundamental frequency period"""
        min_sec = 0.5
        max_sec = 2.5
        # DC component removal
        centered = signal - np.mean(signal)

        # self-correlation
        corr = np.correlate(centered, centered, mode='full')
        corr = corr[len(corr) // 2:]

        # limit the search scope
        min_lag = int(fs * min_sec)
        max_lag = int(fs * max_sec)
        max_lag = min(max_lag, len(corr) - 1)

        if min_lag >= max_lag:
            return int(fs * 1.2)

        # find the peak with the greatest correlation
        roi = corr[min_lag:max_lag]
        best_lag = np.argmax(roi) + min_lag

        dynamic_range = np.max(signal) - np.min(signal)
        peaks, _ = find_peaks(signal, prominence=max(5, dynamic_range * 0.5))
        if len(peaks) >= 2:
            period = int(600 / len(peaks))
        else:
            period = best_lag

        best_lag = min(best_lag, period)

        return best_lag

    def extract_and_normalize(self, leg_data, flex_idx):
        """Extract gait period of leg"""
        flexion = leg_data[flex_idx, :]
        period = self.estimate_gait_period(flexion, fs=60)

        # find Swing Peak
        dynamic_range = np.max(flexion) - np.min(flexion)
        swing_peaks, _ = find_peaks(flexion, distance=int(period * 0.6),
                                    prominence=max(3, dynamic_range * 0.4))

        cut_points = []
        search_end_offset = int(period * 0.32)

        for p in range(len(swing_peaks) - 1):
            if p == len(swing_peaks):
                e_idx = min(len(flexion), p + search_end_offset)
            else:
                flex = flexion[swing_peaks[p]:swing_peaks[p + 1]]
                low_peaks, _ = find_peaks(flex)
                if len(low_peaks) > 0:
                    e_idx = swing_peaks[p] + low_peaks[0]
                else:
                    e_idx = min(swing_peaks[p + 1], swing_peaks[p] + search_end_offset)
            s_idx = swing_peaks[p] + int(period * 0.08)
            if s_idx < e_idx:
                local_min = np.argmin(flexion[s_idx:e_idx])
                cut_points.append(s_idx + local_min)

        peaks = np.array(cut_points)

        cycles_norm = []
        cycles_raw = []

        if len(peaks) >= 2:
            for j in range(len(peaks) - 1):
                raw_cycle = leg_data[:, peaks[j]:peaks[j + 1]]
                norm_cycle = self.interpolate(raw_cycle)
                cycles_norm.append(norm_cycle)
                cycles_raw.append(raw_cycle)

        return cycles_norm, cycles_raw

    def interpolate(self, array_2d):
        """Interpolate an arbitrary length (6, T) to (6, target_len)"""
        original_len = array_2d.shape[1]
        x_old = np.linspace(0, 1, original_len)
        x_new = np.linspace(0, 1, self.target_len)

        f = interp1d(x_old, array_2d, axis=1, kind='linear')
        return f(x_new)

    def process_subject(self, raw_data):
        left_data = raw_data[:6, :]  # (6, 600)
        right_data = raw_data[6:, :]  # (6, 600)

        left_cycles_norm, left_cycles_raw = self.extract_and_normalize(left_data, self.flex_idx)
        right_cycles_norm, right_cycles_raw = self.extract_and_normalize(right_data, self.flex_idx)

        num_pairs = min(len(left_cycles_norm), len(right_cycles_norm))
        if num_pairs == 0:
            return None, None

        processed_samples_norm = []
        processed_samples_raw = []

        for i in range(num_pairs):
            L = left_cycles_norm[i]  # (6, 100)
            R = right_cycles_norm[i]  # (6, 100)

            combined = np.concatenate([L, R], axis=0)  # (12, 100)
            processed_samples_norm.append(combined)

            raw_pair = {
                'L': left_cycles_raw[i],
                'R': right_cycles_raw[i],
            }
            processed_samples_raw.append(raw_pair)

        return np.array(processed_samples_norm), processed_samples_raw

def filter_cycles_by_similarity(cycles, drop_ratio=0.2):
    """Calculate the correlation of the calculation cycle and eliminate the outliers"""
    K = len(cycles)
    if K <= 3: return np.arange(K), cycles
    flat = cycles.reshape(K, -1)
    corr_mat = np.corrcoef(flat)
    corr_mat = np.nan_to_num(corr_mat, nan=0.0)
    mean_sim = corr_mat.mean(axis=1)
    n_drop = max(2, int(K * drop_ratio))
    n_keep = K - n_drop

    if n_keep < 2: n_keep = 2
    keep_indices = np.argsort(mean_sim)[-n_keep:]
    keep_indices = np.sort(keep_indices)

    return keep_indices, cycles[keep_indices]

def advanced_quality_control(X_raw, y_raw, demo_raw, trace_info_raw, min_cycles=2, drop_ratio=0.2):
    processor = adaptive_gait_cycle_segmentation()
    valid_indices = []
    logger.info(f"The advanced quality control is currently being carried out...")

    total_subjects = len(X_raw)
    for i in range(total_subjects):
        cycles_norm, cycles_raw = processor.process_subject(X_raw[i])
        if cycles_norm is None or len(cycles_norm) == 0:
            continue
        _, cycles_filtered = filter_cycles_by_similarity(cycles_norm, drop_ratio=drop_ratio)
        if len(cycles_filtered) >= min_cycles:
            valid_indices.append(i)

    X_clean = X_raw[valid_indices]
    y_clean = y_raw[valid_indices]
    demo_clean = demo_raw[valid_indices]
    trace_info_clean = trace_info_raw[valid_indices]

    logger.info(f"Filtering completed: {total_subjects} -> {len(X_clean)} (Excluding {total_subjects - len(X_clean)} samples)")
    return X_clean, y_clean, demo_clean, trace_info_clean