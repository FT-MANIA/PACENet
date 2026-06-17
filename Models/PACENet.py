import torch
import torch.nn as nn
import numpy as np
from Models.Attention import Attention, AttnPool1D, PatchCrossAttention

class LearnablePositionEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()

        self.pe = nn.Parameter(torch.empty(max_len, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        x = x + self.pe
        return x

class UFE(nn.Module):
    """Unilateral Feature Extractor"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.kernel_size = tuple(config['kernel_size'])
        self.stride = [self.kernel_size[0], config['stride']]
        self.embed_dim = config['embed_dim']
        self.num_patch = (config['ts_dim'] // self.kernel_size[0]) * (
                (config['ts_len'] - self.kernel_size[1]) // self.stride[1] + 1)

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, self.embed_dim, kernel_size=self.kernel_size, stride=self.stride, padding='valid'),
            nn.GroupNorm(num_groups=4, num_channels=self.embed_dim),
            nn.GELU())

        self.flatten = nn.Flatten()

        self.position_embedding = LearnablePositionEncoding(self.embed_dim, max_len=self.num_patch)
        self.attention_layer = Attention(self.embed_dim, 4, dropout=0.0)

        self.ln1 = nn.LayerNorm(self.embed_dim, eps=1e-5)
        self.FeedForward = nn.Sequential(
            nn.Linear(self.embed_dim, config['dim_ff']),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['dim_ff'], self.embed_dim),
            nn.Dropout(config['dropout']))
        self.ln2 = nn.LayerNorm(self.embed_dim, eps=1e-5)
        self.pool = AttnPool1D(self.embed_dim, config['UFE_dim'])

    def forward(self, x):
        x = x.unsqueeze(1)
        patches = self.temporal_conv(x).permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        patches = self.position_embedding(patches)
        if self.config['use_UFE']:
            att_out = patches + self.attention_layer(self.ln1(patches))
            att_out = att_out + self.FeedForward(self.ln2(att_out))
            out = self.pool(att_out)
            return patches, out
        else:
            return patches

class SFE(nn.Module):
    """Spectral Feature Extractor"""
    def __init__(self, config):
        super().__init__()
        self.in_channels = config['ts_dim']
        self.out_dim = config['SFE_dim']
        self.seq_len = config['ts_len']
        self.fft_dim = self.seq_len // 2 + 1
        self.register_buffer('window', torch.hamming_window(self.seq_len))
        self.flatten_dim = self.in_channels * self.fft_dim

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.GELU()
        )

    def forward(self, x):
        x_windowed = x * self.window.view(1, 1, -1)
        x_fft = torch.fft.rfft(x_windowed, dim=-1)
        x_mag = torch.abs(x_fft)
        x_log = torch.log(x_mag + 1e-6)
        out = self.proj(x_log)
        return out

def kpf_calculator(cycles_raw):
    idx_L = {'add': 0, 'rot': 1, 'flex': 2, 'ap': 3}
    idx_R = {'add': 0, 'rot': 1, 'flex': 2, 'ap': 3}

    feats_list = []

    for pair in cycles_raw:
        cycle_L = pair['L']
        cycle_R = pair['R']

        T_L = cycle_L.shape[1]
        T_R = cycle_R.shape[1]

        t_l_stance_end = int(T_L * 0.4)
        t_r_stance_end = int(T_R * 0.4)
        t_l_load_end = int(T_L * 0.2)
        t_r_load_end = int(T_R * 0.2)

        # ROM
        L_rom = np.max(cycle_L[idx_L['flex']]) - np.min(cycle_L[idx_L['flex']])
        R_rom = np.max(cycle_R[idx_R['flex']]) - np.min(cycle_R[idx_R['flex']])

        # ED
        L_ext = np.min(cycle_L[idx_L['flex']])
        R_ext = np.min(cycle_R[idx_R['flex']])

        # StP
        L_peak = np.max(cycle_L[idx_L['flex'], :t_l_stance_end])
        R_peak = np.max(cycle_R[idx_R['flex'], :t_r_stance_end])

        # SwP
        L_swing_peak = np.max(cycle_L[idx_L['flex']])
        R_swing_peak = np.max(cycle_R[idx_R['flex']])

        # EVVE
        L_thrust = np.max(cycle_L[idx_L['add'], :t_l_load_end]) - np.min(cycle_L[idx_L['add'], :t_l_load_end])
        R_thrust = np.max(cycle_R[idx_R['add'], :t_r_load_end]) - np.min(cycle_R[idx_R['add'], :t_r_load_end])

        # APTR
        L_ap = np.max(cycle_L[idx_L['ap']]) - np.min(cycle_L[idx_L['ap']])
        R_ap = np.max(cycle_R[idx_R['ap']]) - np.min(cycle_R[idx_R['ap']])

        feats_list.append([
            L_rom, R_rom, L_ext, R_ext, L_peak, R_peak, L_swing_peak, R_swing_peak,
            L_thrust, R_thrust, L_ap, R_ap
        ])

    feats_arr = np.array(feats_list)
    feat_mean = np.mean(feats_arr, axis=0)
    raw_kpf = np.concatenate([feat_mean])

    scale_factors = np.array([
        180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 30.0, 30.0, 50.0, 50.0
    ])

    raw_kpf_norm = raw_kpf / scale_factors

    return torch.tensor(raw_kpf_norm, dtype=torch.float32)

class PACENet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        leg_config = config.copy()
        leg_config['ts_dim'] = 6

        self.UFE = UFE(leg_config)

        # CFE
        self.cross_attn_r2l = PatchCrossAttention(self.config)
        self.cross_attn_l2r = PatchCrossAttention(self.config)
        self.gamma_l = nn.Parameter(torch.zeros(1))
        self.gamma_r = nn.Parameter(torch.zeros(1))

        self.SFE = SFE(self.config)

        self.KFE = nn.Sequential(
            nn.Linear(self.config['num_KF'], self.config['KFE_dim']),
            nn.BatchNorm1d(self.config['KFE_dim']),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        fusion_input_dim = self.config['UFE_dim'] * 2
        if self.config.get('use_SFE', True):
            fusion_input_dim += self.config['SFE_dim']
        if self.config.get('use_KFE', True):
            fusion_input_dim += self.config['KFE_dim']

        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.6),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, self.config['num_classes'])
        )

    def forward(self, x, raw_kf):
        left_x = x[:, :6, :]
        right_x = x[:, 6:12, :]

        if self.config['use_UFE'] and self.config['use_CFE']:
            # UFE
            left_patches, left_uni = self.UFE(left_x)
            right_patches, right_uni = self.UFE(right_x)
            # CFE
            left_comp = self.cross_attn_r2l(left_patches, right_patches)
            right_comp = self.cross_attn_l2r(right_patches, left_patches)
            # UFE + CFE
            left_feats = left_comp + self.gamma_l * left_uni
            right_feats = right_comp + self.gamma_r * right_uni

        elif self.config['use_UFE'] and not self.config['use_CFE']:
            left_patches, left_feats = self.UFE(left_x)
            right_patches, right_feats = self.UFE(right_x)
        else:
            left_patches = self.UFE(left_x)
            right_patches = self.UFE(right_x)
            left_feats = self.cross_attn_r2l(left_patches, right_patches)
            right_feats = self.cross_attn_l2r(right_patches, left_patches)

        all_features = [left_feats, right_feats]
        if self.config['use_SFE']:
            SFE_feats = self.SFE(x)
            all_features.append(SFE_feats)
        if self.config['use_KFE']:
            KFE_feats = self.KFE(raw_kf)
            all_features.append(KFE_feats)

        combined = torch.cat(all_features, dim=-1)

        final_feature = self.feature_fusion(combined)
        logits = self.classifier(final_feature)
        return logits