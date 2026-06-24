import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, ctx_n_emb : int, n_heads, device, dropout = 0.2):
        super().__init__()
        
        self.head_size = ctx_n_emb // n_heads
        self.n_heads = n_heads
        
        self.dropout = nn.Dropout(dropout, device=device)
        
        self.qkv_proj = nn.Linear(ctx_n_emb, ctx_n_emb * 3, device=device)

    def forward(self, x):
        # x : [B, T, n_emb]
        B, T, C = x.shape
        Q, K, V = self.qkv_proj(x).chunk(3, dim=-1)

        Q = Q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        wei = Q @ K.transpose(-2, -1) * (self.head_size ** -0.5)
        
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        wei = wei @ V

        wei = wei.transpose(1, 2).contiguous().view(B, T, C)
        return wei
      

class MaskedSelfAttention(nn.Module):
    def __init__(self, ctx_n_emb : int, n_heads, max_len, device, dropout = 0.2):
        super().__init__()
        self.register_buffer('tril', torch.tril(torch.ones((max_len, max_len), device=device)))
        
        self.head_size = ctx_n_emb // n_heads
        self.n_heads = n_heads
        
        self.dropout = nn.Dropout(dropout)
        
        self.qkv_proj = nn.Linear(ctx_n_emb, ctx_n_emb * 3)

    def forward(self, x):
        # x : [B, T, ctx_n_emb]
        B, T, C = x.shape
        Q, K, V = self.qkv_proj(x).chunk(3, dim=-1)

        Q = Q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        wei = Q @ K.transpose(-2, -1) * (self.head_size ** -0.5)
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        wei = wei @ V

        wei = wei.transpose(1, 2).contiguous().view(B, T, C)
        return wei

class CrossAttention(nn.Module):
    def __init__(self, ctx_n_emb, cross_ctx_n_emb, n_heads, device, dropout = 0.2):
        super().__init__()
        
        self.head_size = ctx_n_emb // n_heads
        self.n_heads = n_heads
        
        self.dropout = nn.Dropout(dropout)
        
        self.q_proj = nn.Linear(ctx_n_emb, ctx_n_emb)
        self.kv_proj = nn.Linear(cross_ctx_n_emb, ctx_n_emb * 2)

    def forward(self, x, cross_ctx):
        # x : [B, T, ctx_n_emb]
        B, T, C = x.shape
        Q = self.q_proj(x)
        K, V = self.kv_proj(cross_ctx).chunk(2, dim=-1)

        Q = Q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        wei = Q @ K.transpose(-2, -1) * (self.head_size ** -0.5)
        
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        wei = wei @ V

        wei = wei.transpose(1, 2).contiguous().view(B, T, C)
        return wei
    
class FFN(nn.Module):
    def _init__(self, in_d : int, hidden_d : int, out_d : int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_d, hidden_d),
            nn.ReLU(),
            nn.Linear(hidden_d, out_d)
        )
        
        self.in_d = in_d
        
    def forward(self, x):
        assert x.shape[0] == self.in_d
        return self.layers(x)
    
    
class Transformer(nn.Module):
    def __init__(self, ctx_n_emb, cross_ctx_n_emb, n_heads, device):
        super().__init__()
        
        self.ctx_n_emb = ctx_n_emb
        self.cross_ctx_n_emb = cross_ctx_n_emb
        
        self.layer_norm = nn.LayerNorm(ctx_n_emb)
        self.device = device
        
        d_ffn_hidden = 128
        
        # 1 cross attn + 2 self-attn
        self.cross_attn0 = CrossAttention(ctx_n_emb=ctx_n_emb,
                                          cross_ctx_n_emb=cross_ctx_n_emb,
                                          n_heads=n_heads, device=device)
        self.self_attn0 = SelfAttention(ctx_n_emb=ctx_n_emb, 
                                        n_heads=n_heads, device=device)
        self.self_attn1 = SelfAttention(ctx_n_emb=ctx_n_emb, 
                                        n_heads=n_heads, device=device)
        
        self.ffns = [
            FFN(in_d=ctx_n_emb, hidden_d=d_ffn_hidden, out_d=ctx_n_emb),
            FFN(in_d=ctx_n_emb, hidden_d=d_ffn_hidden, out_d=ctx_n_emb), 
            FFN(in_d=ctx_n_emb, hidden_d=d_ffn_hidden, out_d=ctx_n_emb)
        ]
        
    def forward(self, ctx, cross_ctx):
        x0 = ctx
        x1 = self.ffns[0](self.layer_norm(self.cross_attn0(ctx, cross_ctx))) + x0
        x2 = self.ffns[1](self.layer_norm(self.self_attn0(x1))) + x1
        x3 = self.ffns[2](self.layer_norm(self.self_attn0(x2))) + x2
        return x3
