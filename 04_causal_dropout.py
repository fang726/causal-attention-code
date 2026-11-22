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


def save_heatmap(mat, title, fname, tokens=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=160)
    ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}" if v > 0 else "0", ha="center", va="center",
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
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)


print("=" * 70)
print("04 · Section 3.5.2: dropout on attention weights (rate 0.5)")
print("=" * 70)

# ---------- 1) 直观演示：全 1 矩阵过 dropout ----------
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print("dropout(torch.ones(6,6)) —— 约一半被置 0，其余放大 2 倍:")
print(dropout(example))

# ---------- 2) 作用在因果注意力权重上 ----------
torch.manual_seed(783)
sa_v2 = SelfAttention_v2(3, 2)
queries = sa_v2.W_query(INPUTS)
keys = sa_v2.W_key(INPUTS)
attn_scores = queries @ keys.T
mask = torch.triu(torch.ones(6, 6), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)

torch.manual_seed(123)
dropped = dropout(attn_weights)
print("\n因果权重的原始值:")
print(attn_weights)
print("\n经过 dropout(p=0.5) 之后:")
print(dropped)

save_heatmap(dropped.detach().numpy(),
             "Dropout (p=0.5) on causal attention weights",
             "04_dropout.png", TOKENS, vmin=0, vmax=2.0)

print("\n注意：dropout 之后每一行不再严格求和为 1（训练阶段允许），")
print("      推理时 model.eval() 会自动关闭 dropout，权重恢复正常归一化。")
