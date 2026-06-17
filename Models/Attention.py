import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5

        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        att = torch.matmul(q, k) * self.scale
        att = nn.functional.softmax(att, dim=-1)
        att = self.attn_drop(att)
        self.attn_weights = att.detach()

        out = torch.matmul(att, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, -1)

        out = self.to_out(out)
        return out

class PatchCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.scale = (self.embed_dim // self.num_heads) ** -0.5

        self.key = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.value = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.query = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.attn_drop = nn.Dropout(0.0)
        self.to_out = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Dropout(0.0)
        )
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.FeedForward = nn.Sequential(
            nn.Linear(self.embed_dim, config['dim_ff']),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['dim_ff'], self.embed_dim),
            nn.Dropout(config['dropout']))
        self.ln2 = nn.LayerNorm(self.embed_dim)
        self.norm_s = nn.LayerNorm(self.embed_dim)
        self.d_out = config['CFE_dim']
        self.pool = AttnPool1D(self.embed_dim, self.d_out)

    def forward(self, x, s=None):
        batch_size, num_patch, _ = x.shape
        residual_x = x
        x = self.ln1(x)
        s = self.norm_s(s)
        k = self.key(s).reshape(batch_size, num_patch, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(s).reshape(batch_size, num_patch, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, num_patch, self.num_heads, -1).transpose(1, 2)

        att = torch.matmul(q, k) * self.scale
        att = nn.functional.softmax(att, dim=-1)
        att = self.attn_drop(att)
        self.attn_weights = att.detach()
        out = torch.matmul(att, v)

        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, num_patch, -1)
        out = self.to_out(out)

        x = residual_x + out
        x = x + self.FeedForward(self.ln2(x))
        final_out = self.pool(x)
        return final_out

class AttnPool1D(nn.Module):
    def __init__(self, d, d_out):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d))
        self.proj = nn.Sequential(
            nn.Linear(d, d_out),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        q = self.q.expand(x.size(0), -1, -1)
        attn = torch.softmax((torch.matmul(q, x.transpose(1, 2)) / (x.size(-1) ** 0.5)), dim=-1)
        pooled = torch.matmul(attn, x).squeeze(1)
        out = self.proj(pooled)
        return out