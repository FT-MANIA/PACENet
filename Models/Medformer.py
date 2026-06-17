import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
                    torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
                    ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask

class Jitter(nn.Module):
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training:
            x += torch.randn_like(x) * self.scale
        return x


class Scale(nn.Module):
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            x *= 1 + torch.randn(B, C, 1, device=x.device) * self.scale
        return x


class Flip(nn.Module):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if self.training and torch.rand(1) < self.prob:
            return torch.flip(x, [-1])
        return x


class TemporalMask(nn.Module):
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            num_mask = int(T * self.ratio)
            mask_indices = torch.randperm(T)[:num_mask]
            x[:, :, mask_indices] = 0
        return x


class ChannelMask(nn.Module):
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            num_mask = int(C * self.ratio)
            mask_indices = torch.randperm(C)[:num_mask]
            x[:, mask_indices, :] = 0
        return x


class FrequencyMask(nn.Module):
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            x_fft = torch.fft.rfft(x, dim=-1)
            mask = torch.rand(x_fft.shape, device=x.device) > self.ratio
            x_fft = x_fft * mask
            x = torch.fft.irfft(x_fft, n=T, dim=-1)
        return x


def get_augmentation(augmentation):
    if augmentation.startswith("jitter"):
        return Jitter(float(augmentation[6:])) if len(augmentation) > 6 else Jitter()
    elif augmentation.startswith("scale"):
        return Scale(float(augmentation[5:])) if len(augmentation) > 5 else Scale()
    elif augmentation.startswith("drop"):
        return nn.Dropout(float(augmentation[4:])) if len(augmentation) > 4 else nn.Dropout(0.1)
    elif augmentation.startswith("flip"):
        return Flip(float(augmentation[4:])) if len(augmentation) > 4 else Flip()
    elif augmentation.startswith("frequency"):
        return FrequencyMask(float(augmentation[9:])) if len(augmentation) > 9 else FrequencyMask()
    elif augmentation.startswith("mask"):
        return TemporalMask(float(augmentation[4:])) if len(augmentation) > 4 else TemporalMask()
    elif augmentation.startswith("channel"):
        return ChannelMask(float(augmentation[7:])) if len(augmentation) > 7 else ChannelMask()
    elif augmentation == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown augmentation {augmentation}")

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class CrossChannelTokenEmbedding(nn.Module):
    def __init__(self, c_in, l_patch, d_model, stride=None):
        super().__init__()
        if stride is None:
            stride = l_patch
        self.tokenConv = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(c_in, l_patch),
            stride=(1, stride),
            padding=0,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        x = self.tokenConv(x)
        return x


class ListPatchEmbedding(nn.Module):
    def __init__(self, enc_in, d_model, seq_len, patch_len_list, stride_list, dropout, augmentation=["none"],
                 single_channel=False):
        super().__init__()
        self.patch_len_list = patch_len_list
        self.stride_list = stride_list
        self.paddings = [nn.ReplicationPad1d((0, stride)) for stride in stride_list]
        self.single_channel = single_channel

        linear_layers = [
            CrossChannelTokenEmbedding(
                c_in=enc_in if not single_channel else 1,
                l_patch=patch_len,
                d_model=d_model,
            )
            for patch_len in patch_len_list
        ]
        self.value_embeddings = nn.ModuleList(linear_layers)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.channel_embedding = PositionalEmbedding(d_model=seq_len)
        self.dropout = nn.Dropout(dropout)
        self.augmentation = nn.ModuleList([get_augmentation(aug) for aug in augmentation])
        self.learnable_embeddings = nn.ParameterList([nn.Parameter(torch.randn(1, d_model)) for _ in patch_len_list])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.single_channel:
            B, C, L = x.shape
            x = torch.reshape(x, (B * C, 1, L))

        x_list = []
        for padding, value_embedding in zip(self.paddings, self.value_embeddings):
            x_copy = x.clone()
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            x_new = self.augmentation[aug_idx](x_copy)
            x_new = x_new + self.channel_embedding(x_new)
            x_new = padding(x_new).unsqueeze(1)
            x_new = value_embedding(x_new)
            x_new = x_new.squeeze(2).transpose(1, 2)
            x_list.append(x_new)

        x = [x + cxt + self.position_embedding(x) for x, cxt in zip(x_list, self.learnable_embeddings)]
        return x

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask.mask, -np.inf)
        elif self.mask_flag:
            _mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class MedformerLayer(nn.Module):
    def __init__(self, num_blocks, d_model, n_heads, dropout=0.1, output_attention=False, no_inter=False):
        super().__init__()
        self.intra_attentions = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(False, factor=1, attention_dropout=dropout, output_attention=output_attention),
                    d_model, n_heads,
                )
                for _ in range(num_blocks)
            ]
        )
        if no_inter or num_blocks <= 1:
            self.inter_attention = None
        else:
            self.inter_attention = AttentionLayer(
                FullAttention(False, factor=1, attention_dropout=dropout, output_attention=output_attention),
                d_model, n_heads,
            )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attn_mask = attn_mask or ([None] * len(x))
        x_intra = []
        attn_out = []
        for x_in, layer, mask in zip(x, self.intra_attentions, attn_mask):
            _x_out, _attn = layer(x_in, x_in, x_in, attn_mask=mask, tau=tau, delta=delta)
            x_intra.append(_x_out)
            attn_out.append(_attn)

        if self.inter_attention is not None:
            routers = torch.cat([x[:, -1:] for x in x_intra], dim=1)
            x_inter, attn_inter = self.inter_attention(routers, routers, routers, attn_mask=None, tau=tau, delta=delta)
            x_out = [
                torch.cat([x[:, :-1], x_inter[:, i: i + 1]], dim=1)
                for i, x in enumerate(x_intra)
            ]
            attn_out += [attn_inter]
        else:
            x_out = x_intra
        return x_out, attn_out

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = [_x + self.dropout(_nx) for _x, _nx in zip(x, new_x)]
        y = x = [self.norm1(_x) for _x in x]
        y = [self.dropout(self.activation(self.conv1(_y.transpose(-1, 1)))) for _y in y]
        y = [self.dropout(self.conv2(_y).transpose(-1, 1)) for _y in y]
        return [self.norm2(_x + _y) for _x, _y in zip(x, y)], attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        x = torch.cat([x[:, -1, :].unsqueeze(1) for x in x], dim=1)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


class Medformer(nn.Module):
    def __init__(self, config):
        super(Medformer, self).__init__()
        self.model_type = 'Medformer'
        self.task_name = 'classification'

        enc_in = config.get('ts_dim', 12)
        seq_len = config.get('ts_len', 600)
        num_classes = config.get('num_classes', 3)

        self.d_model = config.get('medformer_d_model', 64)
        patch_len_list = config.get('medformer_patch_sizes', [30, 60, 120])
        stride_list = config.get('medformer_strides', [30, 60, 120])
        self.n_heads = config.get('medformer_n_heads', 4)
        e_layers = config.get('medformer_e_layers', 2)
        self.dropout_rate = config.get('medformer_dropout', 0.3)
        d_ff = config.get('medformer_d_ff', self.d_model * 4)
        activation = config.get('medformer_activation', 'gelu')

        output_attention = config.get('output_attention', False)
        no_inter_attn = config.get('no_inter_attn', False)
        self.single_channel = config.get('single_channel', True)
        augmentations = config.get('augmentations', ['none'])

        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]

        self.enc_embedding = ListPatchEmbedding(
            enc_in=enc_in,
            d_model=self.d_model,
            seq_len=seq_len,
            patch_len_list=patch_len_list,
            stride_list=stride_list,
            dropout=self.dropout_rate,
            augmentation=augmentations,
            single_channel=self.single_channel
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    MedformerLayer(
                        len(patch_len_list),
                        self.d_model,
                        self.n_heads,
                        self.dropout_rate,
                        output_attention,
                        no_inter_attn,
                    ),
                    self.d_model,
                    d_ff,
                    dropout=self.dropout_rate,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(self.dropout_rate)

        head_in_dim = self.d_model * len(patch_len_list) * (1 if not self.single_channel else enc_in)
        head_dim = config.get('medformer_head_dim', 128)
        head_dropout = config.get('medformer_head_dropout', self.dropout_rate)

        self.projection = nn.Sequential(
            nn.Linear(head_in_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_dim, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        enc_out = self.enc_embedding(x)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        if self.single_channel:
            enc_out = torch.reshape(enc_out, (-1, x.shape[2], *enc_out.shape[-2:]))

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)

        return output