import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from collections import namedtuple
from functools import wraps
from packaging import version

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.attention import SDPBackend
from hyper_connections import HyperConnections

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)

EfficientAttentionConfig = namedtuple('EfficientAttentionConfig',
                                      ['enable_flash', 'enable_math', 'enable_mem_efficient'])
Statistics = namedtuple('Statistics', ['mean', 'variance', 'gamma', 'beta'])

class RevIN(Module):
    def __init__(self, num_variates, affine=True, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.num_variates = num_variates
        self.gamma = nn.Parameter(torch.ones(num_variates, 1), requires_grad=affine)
        self.beta = nn.Parameter(torch.zeros(num_variates, 1), requires_grad=affine)

    def forward(self, x, return_statistics=False):
        assert x.shape[1] == self.num_variates

        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        var_rsqrt = var.clamp(min=self.eps).rsqrt()
        instance_normalized = (x - mean) * var_rsqrt
        rescaled = instance_normalized * self.gamma + self.beta

        def reverse_fn(scaled_output):
            clamped_gamma = torch.sign(self.gamma) * self.gamma.abs().clamp(min=self.eps)
            unscaled_output = (scaled_output - self.beta) / clamped_gamma
            return unscaled_output * var.sqrt() + mean

        if not return_statistics:
            return rescaled, reverse_fn

        statistics = Statistics(mean, var, self.gamma, self.beta)
        return rescaled, reverse_fn, statistics

class Attend(nn.Module):
    def __init__(self, *, dropout=0., heads=None, scale=None, flash=False, causal=False):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.causal = causal

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse(
            '2.0.0')), '需要 PyTorch 2.0 及以上以支持 Flash Attention'

        self.cpu_config = EfficientAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        major, minor = device_properties.major, device_properties.minor

        if (major, minor) == (8, 0):
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        elif (major, minor) == (9, 0):
            print_once('H100 GPU detected, using flash attention')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device
        config = self.cuda_config if is_cuda else self.cpu_config
        str_to_backend = dict(
            enable_flash=SDPBackend.FLASH_ATTENTION,
            enable_mem_efficient=SDPBackend.EFFICIENT_ATTENTION,
            enable_math=SDPBackend.MATH,
            enable_cudnn=SDPBackend.CUDNN_ATTENTION
        )
        sdpa_backends = [str_to_backend[enable_str] for enable_str, enable in config._asdict().items() if enable]

        with torch.nn.attention.sdpa_kernel(sdpa_backends):
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=self.causal,
                dropout_p=self.dropout if self.training else 0.
            )
        return out

    def forward(self, q, k, v):
        n, heads, kv_heads, device, dtype = q.shape[-2], q.shape[1], k.shape[1], q.device, q.dtype
        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash:
            return self.flash_attn(q, k, v)

        sim = einsum(f'b h i d, b h j d -> b h i j', q, k) * scale

        if self.causal:
            i, j, dtype = *sim.shape[-2:], sim.dtype
            mask_value = -torch.finfo(sim.dtype).max
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim=-1)
        attn = attn.type(dtype)
        attn = self.attn_dropout(attn)
        out = einsum(f'b h i j, b h j d -> b h i d', attn, v)
        return out

class Attention(Module):
    def __init__(self, dim, dim_head=32, heads=4, dropout=0., flash=True, learned_value_residual_mix=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.norm = nn.LayerNorm(dim, bias=False)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv=3, h=heads)
        )

        self.to_value_residual_mix = nn.Sequential(
            nn.Linear(dim, heads, bias=False),
            Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if learned_value_residual_mix else None

        self.to_v_gates = nn.Sequential(
            nn.Linear(dim, heads, bias=False),
            nn.Sigmoid(),
            Rearrange('b n h -> b h n 1', h=heads)
        )

        self.attend = Attend(flash=flash, dropout=dropout)

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x, value_residual=None):
        x = self.norm(x)
        q, k, v = self.to_qkv(x)
        orig_v = v

        if exists(self.to_value_residual_mix):
            assert exists(value_residual)
            mix = self.to_value_residual_mix(x)
            v = v.lerp(value_residual, mix)

        out = self.attend(q, k, v)
        out = out * self.to_v_gates(x)
        return self.to_out(out), orig_v


class GEGLU(Module):
    def forward(self, x):
        x, gate = rearrange(x, '... (r d) -> r ... d', r=2)
        return x * F.gelu(gate)


def FeedForward(dim, mult=4, dropout=0.):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        nn.LayerNorm(dim, bias=False),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_inner, dim)
    )

class iTransformer(nn.Module):
    def __init__(self, config):
        super(iTransformer, self).__init__()
        self.model_type = 'iTransformer'

        self.lookback_len = config['ts_len']
        self.num_variates = config['ts_dim']
        self.num_classes = config['num_classes']

        dim = config.get("itransformer_dim", 128)
        heads = config.get("itransformer_heads", 4)
        depth = config.get("itransformer_depth", 2)
        dim_head = config.get("itransformer_dim_head", 32)
        attn_dropout = config.get("itransformer_attn_dropout", 0.1)
        ff_dropout = config.get("itransformer_ff_dropout", 0.3)
        ff_mult = config.get("itransformer_ff_mult", 4)
        flash_attn = config.get("itransformer_flash_attn", True)

        use_reversible_instance_norm = config.get("itransformer_use_revin", False)
        revin_affine = config.get("itransformer_revin_affine", False)

        if use_reversible_instance_norm and self.num_variates:
            self.reversible_instance_norm = RevIN(self.num_variates, affine=revin_affine)
        else:
            self.reversible_instance_norm = None

        self.mlp_in = nn.Sequential(
            nn.Linear(self.lookback_len, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(config.get("itransformer_input_dropout", 0.1))
        )

        self.use_channel_embedding = config.get("itransformer_use_channel_embedding", True)
        if self.use_channel_embedding:
            self.channel_embedding = nn.Parameter(
                torch.randn(1, self.num_variates, dim) * 0.02
            )

        num_residual_streams = config.get('num_residual_streams', 1)
        if num_residual_streams > 1 and HyperConnections is not None:
            init_hyper_conn, self.expand_streams, self.reduce_streams = HyperConnections.get_init_and_expand_reduce_stream_functions(
                num_residual_streams, disable=False)
        else:
            init_hyper_conn = lambda dim, branch: branch
            self.expand_streams = lambda x: x
            self.reduce_streams = lambda x: x

        self.layers = ModuleList([])
        for i in range(depth):
            is_first = (i == 0)
            self.layers.append(ModuleList([
                init_hyper_conn(
                    dim=dim,
                    branch=Attention(
                        dim,
                        dim_head=dim_head,
                        heads=heads,
                        dropout=attn_dropout,
                        flash=flash_attn,
                        learned_value_residual_mix=not is_first
                    )
                ),
                init_hyper_conn(
                    dim=dim,
                    branch=FeedForward(
                        dim,
                        mult=ff_mult,
                        dropout=ff_dropout
                    )
                ),
            ]))

        self.norm = nn.LayerNorm(dim)

        head_dim = config.get("itransformer_head_dim", 128)
        head_dropout = config.get("itransformer_head_dropout", 0.3)

        self.pooling = config.get("itransformer_pooling", "avgmax")

        if self.pooling == "avgmax":
            head_in_dim = dim * 2
        else:
            head_in_dim = dim

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_dim, self.num_classes)
        )

    def forward(self, x):
        if exists(self.reversible_instance_norm):
            x, reverse_fn = self.reversible_instance_norm(x)

        x = self.mlp_in(x)

        if getattr(self, "use_channel_embedding", False):
            x = x + self.channel_embedding

        x = self.expand_streams(x)

        first_values = None
        for attn, ff in self.layers:
            attn_out, values = attn(x, value_residual=first_values)
            first_values = default(first_values, values)

            x = x + attn_out
            x = ff(x) + x

        x = self.reduce_streams(x)
        x = self.norm(x)

        if self.pooling == "avgmax":
            avg_feat = x.mean(dim=1)
            max_feat = x.max(dim=1)[0]
            x = torch.cat([avg_feat, max_feat], dim=-1)
        elif self.pooling == "max":
            x = x.max(dim=1)[0]
        else:
            x = x.mean(dim=1)

        x = self.head(x)
        return x