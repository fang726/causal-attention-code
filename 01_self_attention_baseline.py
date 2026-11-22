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
    [[0.43, 0.15, 0.89],   # Your     (x^1)
     [0.55, 0.87, 0.66],   # journey  (x^2)
     [0.57, 0.85, 0.64],   # starts   (x^3)
     [0.22, 0.58, 0.49],   # with     (x^4)
     [0.77, 0.25, 0.10],   # one      (x^5)
     [0.05, 0.80, 0.55]]   # step     (x^6)
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
    """带可训练权重的自注意力（书中 3.4.2 节的最终版本）。"""

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
        context_vec = attn_weights @ values
        return context_vec


print("=" * 70)
print("01 · SelfAttention_v2 (Section 3.4.2)")
print("=" * 70)
print("inputs.shape:", INPUTS.shape)
print(INPUTS)

torch.manual_seed(783)          # 与书中一致，保证权重可复现
d_in, d_out = INPUTS.shape[1], 2
sa_v2 = SelfAttention_v2(d_in, d_out)
print("\nSelfAttention_v2 已初始化: d_in=3, d_out=2")

# ---- 未加掩码的注意力权重：每一行都能看到后面的所有 token ----
queries = sa_v2.W_query(INPUTS)
keys = sa_v2.W_key(INPUTS)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

print("\nattention weights (softmax, 未加掩码):")
print(attn_weights)

context_vecs = sa_v2(INPUTS)
print("\ncontext_vecs.shape:", context_vecs.shape)
print(context_vecs)

save_heatmap(attn_weights.detach().numpy(),
             "Self-attention weights (no mask)",
             "01_unmasked_weights.png", TOKENS, vmin=0,
             vmax=float(attn_weights.max()))
print("\n注意：此时第 1 行 'Your' 也能注意到 'step'（未来 token）——")
print("      这就是生成模型必须引入因果掩码的原因。")
