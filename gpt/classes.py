import torch
import torch.nn as nn
import torch.nn.functional as F


### SELF-ATTENTION
class Head(nn.Module):
    """Attention head using Q,K,V"""

    def __init__(self, n_embd, head_size=64, dropout=0.01, block_size=8):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # We don't want to optimize this triangular matrix, so use register_buffer
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # Attention weights
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        # Mask out the upper triangular part
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


### END SELF-ATTENTION


### MULTI-HEAD ATTENTION
class MultiHead(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size,
                                         dropout, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


### END MULTI-HEAD ATTENTION


### FEEDFORWARD
class FeedForward(nn.Module):
    """Lin Layer + Relu + Lin Layer + Dropout"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


### END FEEDFORWARD


### BLOCK
class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, n_embd, num_heads, dropout, block_size):
        super().__init__()
        head_size = n_embd // num_heads
        self.attn = MultiHead(num_heads, head_size,
                              n_embd, dropout, block_size)
        self.ff = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


### END BLOCK
