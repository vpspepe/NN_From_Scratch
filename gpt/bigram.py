import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt.classes import Head, MultiHead, FeedForward, Block


# Bigram char-level model
class BigramModel(nn.Module):

    def __init__(self, vocab_size, block_size, n_heads, n_layers, n_embd, dropout):
        self.block_size = block_size
        super().__init__()
        assert n_embd % n_heads == 0, "n_embd must be divisible by n_heads"
        # Embedding table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Transformer blocks
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_heads, dropout, self.block_size)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # Last LN
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # Embedding
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx[:, -self.block_size :])
            logits = logits[:, -1, :]  # Only last prediction
            probs = F.softmax(logits, dim=-1)
            # Takes one new token and add it to the idx
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx
