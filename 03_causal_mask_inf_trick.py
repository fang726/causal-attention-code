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
    """画热力图；-inf 位置用红色 $-\\infty$ 标注。"""
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=160)
    ax.imshow(mat, cmap="Blues", vmin=vmin, vmax=vmax)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if v == float("-inf"):
                ax.text(j, i, r"$-\infty$", ha="center", va="center",
                        color="#b91c1c", fontsize=7)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > (vmax or 0) * 0.6 else "#1f2937",
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
    """Section 3.4.2：提供 W_query / W_key。"""

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)


print("=" * 70)
print("03 · Section 3.5.1: efficient trick (-inf before softmax)")
print("=" * 70)

torch.manual_seed(783)
sa_v2 = SelfAttention_v2(3, 2)
queries = sa_v2.W_query(INPUTS)
keys = sa_v2.W_key(INPUTS)
attn_scores = queries @ keys.T
context_length = attn_scores.shape[0]

# ---------- 关键一步：上三角（不含对角线）填 -inf ----------
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print("masked attention scores（未来位置 = -inf）:")
print(masked)
attn_weights_trick = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
print("\nsoftmax 之后的注意力权重（与三步法结果完全一致）:")
print(attn_weights_trick)

save_heatmap(masked.detach().numpy(), "Masked scores: future = $-\\infty$",
             "03a_inf_mask.png", TOKENS)
save_heatmap(attn_weights_trick.detach().numpy(),
             "softmax of masked scores (rows sum to 1)",
             "03b_softmax_trick.png", TOKENS, vmin=0,
             vmax=float(attn_weights_trick.max()))

# ---------- 等价性验证：与三步法对比 ----------
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
masked_simple = attn_weights * torch.tril(torch.ones(context_length, context_length))
three_step = masked_simple / masked_simple.sum(dim=-1, keepdim=True)
diff = (three_step - attn_weights_trick).abs().max().item()
print(f"\n与三步法的最大绝对误差: {diff:.3e}")
print("→ 数值等价，但 -inf 技巧只需一次 softmax，且不必再做除法归一化。")
