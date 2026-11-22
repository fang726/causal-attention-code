import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

FIG_DIR = r"D:\研究生文件\课堂作业\figures"
os.makedirs(FIG_DIR, exist_ok=True)

os.makedirs(FIG_DIR, exist_ok=True)

torch.manual_seed(123)
torch.set_printoptions(precision=4, sci_mode=False)

TOKENS = ["Your", "journey", "starts", "with", "one", "step"]
INPUTS = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.49],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
)


class CausalAttention(nn.Module):
    """Section 3.5.3 的因果注意力（单头）。"""

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ values


class MultiHeadAttentionWrapper(nn.Module):
    """把 num_heads 个单头模块堆起来，输出拼接。"""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads,
                 qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


print("=" * 70)
print("06 · Section 3.6.1: MultiHeadAttentionWrapper (stack heads)")
print("=" * 70)

batch = torch.stack((INPUTS, INPUTS), dim=0)   # (2, 6, 3)
context_length = batch.shape[1]
print("batch.shape:", batch.shape)

torch.manual_seed(123)
d_in, d_out, num_heads = 3, 2, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0,
                                num_heads=num_heads)
context_vecs = mha(batch)

print("\ncontext_vecs（两个头的输出在最后一维拼接）:")
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape,
      "  # d_out * num_heads = 2 * 2 = 4")

# 打印每个头各自的输出，说明"每个头有独立参数、独立视角"
print("\nHead 1 输出 (b, n, 2):")
print(context_vecs[:, :, :d_out])
print("\nHead 2 输出 (b, n, 2):")
print(context_vecs[:, :, d_out:])

# 可视化：两个头各自的注意力权重（dropout=0 时可直接反推）
with torch.no_grad():
    fig, axes = plt.subplots(1, num_heads, figsize=(9.0, 4.0), dpi=160)
    for hi, head in enumerate(mha.heads):
        q = head.W_query(batch)
        k = head.W_key(batch)
        s = q @ k.transpose(1, 2)
        s = s.masked_fill(head.mask.bool()[:6, :6], -torch.inf)
        w = torch.softmax(s / k.shape[-1] ** 0.5, dim=-1)[0].numpy()
        ax = axes[hi]
        ax.imshow(w, cmap="Blues", vmin=0, vmax=w.max())
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{w[i, j]:.2f}" if w[i, j] > 0 else "0",
                        ha="center", va="center",
                        color="white" if w[i, j] > w.max() * 0.6 else "#1f2937",
                        fontsize=6)
        ax.set_xticks(range(6), labels=TOKENS, fontsize=7)
        ax.set_yticks(range(6), labels=TOKENS, fontsize=7)
        ax.set_title(f"Head {hi + 1}", fontsize=10)
    fig.suptitle("MultiHeadAttentionWrapper: two independent causal heads", fontsize=11)
    fig.tight_layout()
    p = os.path.join(FIG_DIR, "06_two_heads_wrapper.png")
    fig.savefig(p)
    plt.close(fig)
    print(f"\n[figure saved] {p}")
