import os
import numpy as np
import pandas as pd
import logging
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from Dataset.utils import df_data_extractor, adaptive_gait_cycle_segmentation, filter_cycles_by_similarity, advanced_quality_control
from Models.PACENet import kpf_calculator

logger = logging.getLogger(__name__)

def load_gait_data(config):
    Data = {}
    logger.info("Loading Knee Gait Kinematic Dataset...")
    paths = {
        'dev': os.path.join(config['dataset_path'], 'dev_dataset.csv'),
        'test': os.path.join(config['dataset_path'], 'test_dataset.csv'),
    }

    dfs = {k: pd.read_csv(v) for k, v in paths.items()}

    x_dev, y_dev, _, demo_dev, trace_info_dev = df_data_extractor(dfs['dev'])
    x_test, y_test, _, demo_test, trace_info_test = df_data_extractor(dfs['test'])

    x_dev, y_dev, demo_dev, trace_info_dev = advanced_quality_control(
        x_dev, y_dev, demo_dev, trace_info_dev, min_cycles=1, drop_ratio=0.1
    )
    x_test, y_test, demo_test, trace_info_test = advanced_quality_control(
        x_test, y_test, demo_test, trace_info_test, min_cycles=1, drop_ratio=0.1
    )

    internal_test_size = config['internal_test_size']
    seed = config['seed']
    all_dev_indices = np.arange(len(y_dev))

    dev_indices, internal_test_indices = train_test_split(
        all_dev_indices,
        test_size=internal_test_size,
        random_state=seed,
        stratify=y_dev
    )

    Data['dev_data'] = x_dev[dev_indices]
    Data['dev_label'] = y_dev[dev_indices]
    Data['dev_demo'] = demo_dev[dev_indices]
    Data['dev_trace_info'] = trace_info_dev[dev_indices]

    Data['dev_test_data'] = x_dev[internal_test_indices]
    Data['dev_test_label'] = y_dev[internal_test_indices]
    Data['dev_test_demo'] = demo_dev[internal_test_indices]
    Data['dev_test_trace_info'] = trace_info_dev[internal_test_indices]

    Data['ext_test_data'] = x_test
    Data['ext_test_label'] = y_test
    Data['ext_test_demo'] = demo_test
    Data['ext_test_trace_info'] = trace_info_test

    return Data

class build_dataset(Dataset):
    def __init__(self, raw_data, labels, demographics, trace_info, config, scaler=None, is_train=True):
        self.config = config
        self.is_train = is_train
        self.processor = adaptive_gait_cycle_segmentation()

        temp_kf = []
        valid_indices = []

        for i in range(len(raw_data)):
            cycles_norm, cycles_raw = self.processor.process_subject(raw_data[i])
            if cycles_norm is None or len(cycles_norm) == 0: continue
            keep_indices, _ = filter_cycles_by_similarity(cycles_norm, drop_ratio=0.1)
            if len(keep_indices) == 0: continue
            cycles_raw_kept = [cycles_raw[k] for k in keep_indices]
            raw_kf = kpf_calculator(cycles_raw_kept)
            temp_kf.append(raw_kf)
            valid_indices.append(i)

        self.raw_data = raw_data[valid_indices]
        self.labels = labels[valid_indices]
        self.demographics = demographics[valid_indices]
        self.trace_info = trace_info[valid_indices]
        self.raw_kf = temp_kf

        if self.config.get('Norm', True):
            if is_train and scaler is None:
                self.scalers = []
                if len(self.raw_data) > 0:
                    num_dofs = 6
                    for i in range(num_dofs):
                        sc = StandardScaler()
                        left_data = self.raw_data[:, i, :].flatten().reshape(-1, 1)
                        right_data = self.raw_data[:, i+num_dofs, :].flatten().reshape(-1, 1)
                        combined_data = np.concatenate([left_data, right_data], axis=0)
                        sc.fit(combined_data)
                        self.scalers.append(sc)
                else:
                    logger.warning("There are not enough data to fit the Scaler!")
            else:
                self.scalers = scaler
        else:
            self.scalers = None

        if self.scalers:
            norm_data = np.zeros_like(self.raw_data)
            num_dofs = 6
            N, C, T = self.raw_data.shape
            for i in range(num_dofs):
                sc = self.scalers[i]
                flat_l = self.raw_data[:, i, :].flatten().reshape(-1, 1)
                norm_data[:, i, :] = sc.transform(flat_l).reshape(N, T)

                flat_r = self.raw_data[:, i+num_dofs, :].flatten().reshape(-1, 1)
                norm_data[:, i+num_dofs, :] = sc.transform(flat_r).reshape(N, T)

            self.final_data = norm_data
        else:
            self.final_data = self.raw_data

    def __getitem__(self, index):
        return (torch.from_numpy(self.final_data[index]).float(),
                torch.tensor(self.labels[index]).long(),
                torch.from_numpy(self.demographics[index]).float(),
                self.raw_kf[index],
                self.trace_info[index],
                index)

    def __len__(self):
        return len(self.labels)