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
    """因果（自回归）自注意力：屏蔽未来 token + dropout。"""

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # 上三角掩码（不含对角线）注册为 buffer：不参与训练，但跟随模型 .to(device)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # (b, num_tokens, d_out) @ (b, d_out, num_tokens) -> (b, n, n)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values      # (b, n, d_out)
        return context_vec


print("=" * 70)
print("05 · Section 3.5.3: CausalAttention class (batch + mask + dropout)")
print("=" * 70)

# 把同一句话复制两份，模拟 batch_size = 2
batch = torch.stack((INPUTS, INPUTS), dim=0)
print("batch.shape:", batch.shape, "  # (batch, num_tokens, d_in)")

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in=3, d_out=2, context_length=context_length, dropout=0.0)
context_vecs = ca(batch)

print("context_vecs.shape:", context_vecs.shape, "  # (2, 6, 2)")
print(context_vecs)

# 用 dropout=0 的模型反推注意力权重，验证因果性
with torch.no_grad():
    q = ca.W_query(batch)
    k = ca.W_key(batch)
    scores = q @ k.transpose(1, 2)
    scores = scores.masked_fill(ca.mask.bool()[:6, :6], -torch.inf)
    w = torch.softmax(scores / k.shape[-1] ** 0.5, dim=-1)[0].numpy()

fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=160)
ax.imshow(w, cmap="Blues", vmin=0, vmax=w.max())
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{w[i, j]:.2f}" if w[i, j] > 0 else "0",
                ha="center", va="center",
                color="white" if w[i, j] > w.max() * 0.6 else "#1f2937",
                fontsize=7)
ax.set_xticks(range(6), labels=TOKENS, fontsize=8)
ax.set_yticks(range(6), labels=TOKENS, fontsize=8)
ax.set_title("CausalAttention weights (upper triangle = 0)", fontsize=10)
fig.tight_layout()
path = os.path.join(FIG_DIR, "05_causal_class_weights.png")
fig.savefig(path)
plt.close(fig)
print(f"[figure saved] {path}")

print("\n对角线以上的权重全部为 0 —— 每个 token 只能看到自己和它前面的 token。")
