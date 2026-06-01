import numpy as np
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(__name__)

class Augmentation:
    def __init__(self, config=None):
        """
        This is the acquisition-aware kinematic data augmentation from paper
        """
        self.config = config
        self.jitter_std = self.config.get('jitter_std', 0.01)
        self.scale_range = self.config.get('scale_range', [0.95, 1.05])
        self.rotation_range = self.config.get('rotation_range', [-5, 5])
        self.time_warp_knots = self.config.get('time_warp_knots', 4)
        self.magnitude_warp_std = self.config.get('magnitude_warp_std', 0.1)
        self.bias_range = self.config.get('bias_range', [-10, 10])
        self.smooth_sigma = self.config.get('smooth_sigma', 1.0)

        self.use_jitter = self.config.get('use_jitter', True)
        self.use_scaling = self.config.get('use_scaling', True)
        self.use_magnitude_warp = self.config.get('use_magnitude_warp', True)
        self.use_time_warp = self.config.get('use_time_warp', True)
        self.use_random_bias = self.config.get('use_random_bias', True)
        self.use_crosstalk = self.config.get('use_crosstalk', True)

    def jitter(self, x):
        """Add Gaussian noise."""
        noise = np.random.normal(0, self.jitter_std, size=x.shape)
        return x + noise

    def scaling(self, x):
        """Random scaling"""
        factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return x * factor

    def magnitude_warp(self, x):
        """Amplitude distortion"""
        C, T = x.shape
        x_new = np.zeros_like(x)

        for i in range(C):
            knots = np.random.normal(1.0, self.magnitude_warp_std, size=self.time_warp_knots)
            knots = np.clip(knots, 0.8, 1.2)

            x_vals = np.linspace(0, T - 1, self.time_warp_knots)
            f = interp1d(x_vals, knots, kind='cubic', fill_value="extrapolate")
            warp_curve = f(np.arange(T))
            x_new[i, :] = x[i, :] * warp_curve

        return x_new

    def time_warp(self, x):
        """Time distortion - Non-uniform changes in simulated walking speed"""
        C, T = x.shape
        x_new = np.zeros_like(x)

        # Generate a random time step
        knots = np.random.normal(0, 1.0, size=self.time_warp_knots)
        # Limit the time offset to prevent the complete destruction of the periodic structure
        knots = np.clip(knots, -2.0, 2.0)

        x_vals = np.linspace(0, T - 1, self.time_warp_knots)
        f = interp1d(x_vals, knots, kind='cubic', fill_value="extrapolate")
        offsets = f(np.arange(T))

        # Normalize and map back to the index
        new_indices = np.arange(T) + offsets
        new_indices = np.clip(new_indices, 0, T - 1)

        for i in range(C):
            x_new[i, :] = np.interp(new_indices, np.arange(T), x[i, :])

        return x_new

    def random_bias(self, x):
        x_new = x.copy()

        # 角度通道：0,1,2,6,7,8
        angle_pairs = [(0, 6), (1, 7), (2, 8)]

        # 平移通道：3,4,5,9,10,11
        trans_pairs = [(3, 9), (4, 10), (5, 11)]

        # 保留“同一采集系统偏置”，但不要彻底打乱左右差异
        for l, r in angle_pairs:
            shared = np.random.uniform(-2.0, 2.0)
            diff = np.random.uniform(-0.5, 0.5)
            x_new[l, :] += shared + diff
            x_new[r, :] += shared - diff

        for l, r in trans_pairs:
            shared = np.random.uniform(-0.5, 0.5)
            diff = np.random.uniform(-0.15, 0.15)
            x_new[l, :] += shared + diff
            x_new[r, :] += shared - diff

        return x_new

    def crosstalk(self, x):
        """
        Simulated Kinematic Crosstalk
        Physical meaning: The coordinate system of the analog sensor/mark point does not align with the skeletal anatomical coordinate system (it is misaligned).
        This will cause large flexion and extension movements to be wrongly recorded as inversion or rotation.
        """
        C, T = x.shape
        x_new = x.copy()

        groups = [
            [0, 1, 2],
            [6, 7, 8]
        ]

        # Rotation range: The sensor error is usually within +/- 10 degrees.
        low, high = self.rotation_range
        for idxs in groups:
            if max(idxs) >= C:
                continue

            angles = np.radians(np.random.uniform(low, high, size=3))
            rx, ry, rz = angles

            # Construct the rotation matrix
            # Rotation around X
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            # Rotation around Y
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            # Rotation around Z
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

            # The combined rotation matrix R = Rz * Ry * Rx
            R = Rz @ Ry @ Rx

            # Extract the original data [3, T]
            original_vec = x[idxs, :]

            # Apply rotation: [3, 3] @ [3, T] -> [3, T]
            rotated_vec = R @ original_vec

            # Assign back to the new array
            x_new[idxs, :] = rotated_vec

        return x_new

    def paired_amplitude_morph(self, x, mode="koa_low_rom"):
        """
        Bilateral relationship-preserving amplitude morphing.

        This is designed to simulate center/domain shifts in KOA:
        - lower bilateral ROM
        - lower AP translation range
        - lower adduction/abduction variation
        - preserve left-right relative pattern as much as possible

        x: [12, T]
           left channels  = 0:6
           right channels = 6:12
        """
        x_new = x.copy()

        # Channel pairs: (left, right)
        # 0 add/abd, 1 rotation, 2 flexion, 3 AP, 4/5 other translations
        pairs = [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)]

        for dof_idx, (l_ch, r_ch) in enumerate(pairs):
            if mode == "koa_low_rom":
                # flexion: strong but plausible ROM compression
                if dof_idx == 2:
                    shared_factor = np.random.uniform(0.60, 0.90)
                # adduction/abduction: moderate compression
                elif dof_idx == 0:
                    shared_factor = np.random.uniform(0.60, 0.90)
                # AP translation: stronger compression
                elif dof_idx == 3:
                    shared_factor = np.random.uniform(0.50, 0.85)
                # other rotation/translation channels: mild compression
                else:
                    shared_factor = np.random.uniform(0.75, 1.00)
            else:
                shared_factor = np.random.uniform(0.85, 1.05)

            # small L/R difference, but not independent distortion
            diff_factor = np.random.uniform(-0.04, 0.04)
            l_factor = np.clip(shared_factor + diff_factor, 0.45, 1.10)
            r_factor = np.clip(shared_factor - diff_factor, 0.45, 1.10)

            for ch, factor in [(l_ch, l_factor), (r_ch, r_factor)]:
                center = np.mean(x_new[ch, :])
                x_new[ch, :] = center + factor * (x_new[ch, :] - center)

        return x_new

    def paired_center_bias(self, x):
        """
        Simulate center-specific zero-offset differences while preserving
        left-right relationship.

        Compared with independent random_bias, this is safer for bilateral gait.
        """
        x_new = x.copy()

        pairs = [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)]

        for dof_idx, (l_ch, r_ch) in enumerate(pairs):
            # angle channels: add/abd, rotation, flexion
            if dof_idx in [0, 1, 2]:
                shared_bias = np.random.uniform(-3.0, 3.0)
                diff_bias = np.random.uniform(-0.5, 0.5)
            # translation channels
            else:
                shared_bias = np.random.uniform(-0.5, 0.5)
                diff_bias = np.random.uniform(-0.1, 0.1)

            x_new[l_ch, :] += shared_bias + diff_bias
            x_new[r_ch, :] += shared_bias - diff_bias

        return x_new

    def temporal_smoothing(self, x):
        """
        Lightweight temporal smoothing to simulate center-specific filtering.
        No scipy dependency needed.
        """
        x_new = x.copy()
        kernel_size = int(self.config.get("smooth_kernel_size", 5))
        kernel_size = max(3, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        pad = kernel_size // 2

        for c in range(x.shape[0]):
            padded = np.pad(x[c], (pad, pad), mode="edge")
            x_new[c] = np.convolve(padded, kernel, mode="valid")

        return x_new

    def koa_domain_shift(self, x):
        """
        KOA-specific domain shift augmentation.

        Purpose:
        simulate external-center KOA cases with lower bilateral motion amplitude,
        lower AP translation, preserved bilateral relationship, and mild center style shift.
        """
        sample = x.copy()

        # 1. Simulate low-ROM / low-translation KOA distribution
        sample = self.paired_amplitude_morph(sample, mode="koa_low_rom")

        # 2. Mild center-specific baseline offset, preserving bilateral relation
        if np.random.rand() < 0.7:
            sample = self.paired_center_bias(sample)

        # 3. Mild smoothing / acquisition filtering difference
        if np.random.rand() < 0.5:
            sample = self.temporal_smoothing(sample)

        # 4. Very light jitter only
        if np.random.rand() < 0.5:
            old_std = self.jitter_std
            self.jitter_std = min(old_std, 0.005)
            sample = self.jitter(sample)
            self.jitter_std = old_std

        return sample

    def augment_batch(self, X, y, demographics, trace_info, augment_ratio=1, class_label=None):
        """Enhance a batch of data"""
        X_aug = [X]
        y_aug = [y]
        demo_aug = [demographics]
        trace_info_aug = [trace_info]

        if augment_ratio <= 1:
            return X, y, demographics, trace_info

        for _ in range(augment_ratio - 1):
            X_new = []

            for i in range(len(X)):
                sample = X[i].copy()

                # KOA-specific augmentation: label 2 = KOA
                if class_label == 2 and self.config.get("use_koa_domain_aug", True):
                    if np.random.rand() < self.config.get("koa_domain_aug_prob", 0.75):
                        sample = self.koa_domain_shift(sample)
                    else:
                        # fallback to conservative general augmentation
                        if self.use_scaling:
                            sample = self.scaling(sample)
                        if self.use_jitter:
                            sample = self.jitter(sample)

                else:
                    # General conservative augmentation
                    rand_choice = np.random.rand()

                    if rand_choice < 0.25:
                        if self.config.get("use_paired_center_bias", True):
                            sample = self.paired_center_bias(sample)
                        elif self.use_random_bias:
                            sample = self.random_bias(sample)

                    elif rand_choice < 0.50:
                        if self.use_scaling:
                            sample = self.scaling(sample)
                        if np.random.rand() < 0.2 and self.use_time_warp:
                            sample = self.time_warp(sample)

                    elif rand_choice < 0.75:
                        if self.use_jitter:
                            sample = self.jitter(sample)
                        if self.use_magnitude_warp:
                            sample = self.magnitude_warp(sample)

                    else:
                        if self.use_crosstalk:
                            sample = self.crosstalk(sample)
                        if self.use_scaling:
                            sample = self.scaling(sample)

                X_new.append(sample)

            X_aug.append(np.array(X_new))
            y_aug.append(y)
            demo_aug.append(demographics)
            trace_info_aug.append(trace_info)

        X_final = np.concatenate(X_aug, axis=0)
        y_final = np.concatenate(y_aug, axis=0)
        demo_final = np.concatenate(demo_aug, axis=0)
        trace_info_final = np.concatenate(trace_info_aug, axis=0)

        return X_final, y_final, demo_final, trace_info_final

def data_augmentor(X, y, demographics, trace_info, config, aug_ratios):
    augmentor = Augmentation(config)

    unique_classes, counts = np.unique(y, return_counts=True)
    logger.info(f"Distribution of categories before data augmentation:{dict(zip(unique_classes, counts))}")

    X_augmented_list = []
    y_augmented_list = []
    demo_augmented_list = []
    trace_info_augmented_list = []

    for label in unique_classes:
        label_mask = (y == label)
        X_label = X[label_mask]
        y_label = y[label_mask]
        demo_label = demographics[label_mask]
        trace_info_label = trace_info[label_mask]

        current_count = label_mask.sum()

        augment_ratio = int(aug_ratios[label])
        augment_ratio = max(1, augment_ratio)
        logger.info(
            f"-> Category {label}: Original {current_count} | Target multiple {augment_ratio}x")

        X_aug, y_aug, demo_aug, trace_info_aug = augmentor.augment_batch(
            X_label,
            y_label,
            demo_label,
            trace_info_label,
            augment_ratio,
            class_label=int(label)
        )
        demo_augmented_list.append(demo_aug)
        trace_info_augmented_list.append(trace_info_aug)

        X_augmented_list.append(X_aug)
        y_augmented_list.append(y_aug)

    X_final = np.concatenate(X_augmented_list, axis=0)
    y_final = np.concatenate(y_augmented_list, axis=0)

    final_unique, final_counts = np.unique(y_final, return_counts=True)
    logger.info(f"Distribution of categories after data augmentation: {dict(zip(final_unique, final_counts))}")

    demo_final = np.concatenate(demo_augmented_list, axis=0)
    trace_info_final = np.concatenate(trace_info_augmented_list, axis=0)

    return X_final, y_final, demo_final,trace_info_final