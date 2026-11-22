import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def save_heatmap(mat, title, fname, tokens=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=160)
    ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > (vmax or mat.max()) * 0.6 else "#1f2937",
                    fontsize=7)
    if tokens:
        ax.set_xticks(range(len(tokens)), labels=tokens, fontsize=8)
        ax.set_yticks(range(len(tokens)), labels=tokens, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path)
    plt.close(fig)
    print(f"[figure saved] {path}")


class SelfAttention_v2(nn.Module):
    """Section 3.4.2：用于生成未掩码的 Q/K/V。"""

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        return attn_weights @ values


print("=" * 70)
print("02 · Section 3.5.1: causal mask (zero-out + renormalize)")
print("=" * 70)

torch.manual_seed(783)
d_in, d_out = INPUTS.shape[1], 2
sa_v2 = SelfAttention_v2(d_in, d_out)

# ---------- Step 0: 先算出未掩码的注意力权重 ----------
queries = sa_v2.W_query(INPUTS)
keys = sa_v2.W_key(INPUTS)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
print("Step 1) attention weights (softmax, 未掩码):")
print(attn_weights)


context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print("\nStep 2) mask (下三角 1):")
print(mask_simple)
masked_simple = attn_weights * mask_simple
print("\nStep 2) masked attention weights (未来位置已置 0):")
print(masked_simple)
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print("\nStep 3) renormalized attention weights (每行和为 1):")
print(masked_simple_norm)

A0 = attn_weights.detach().numpy()
A1 = masked_simple_norm.detach().numpy()
save_heatmap(A0, "Before mask: softmax attention weights",
             "02a_before_mask.png", TOKENS, vmin=0, vmax=float(A0.max()))
save_heatmap(A1, "After causal mask + renormalize",
             "02b_after_mask.png", TOKENS, vmin=0, vmax=float(A0.max()))

print("\n第一行只剩 Your -> 1.00：第一个词只能关注它自己。")
print("行和校验:", masked_simple_norm.sum(dim=-1).detach().numpy())
