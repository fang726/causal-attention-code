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

print("=" * 70)
print("07 · Section 3.6.2: MultiHeadAttention (weight splits, efficient)")
print("=" * 70)

# ======================================================================
# Part A. 批量矩阵乘：一次算完 h 个头
# ======================================================================
print("\n--- Part A: 批量矩阵乘直觉 ---")
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],
                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])
print("a.shape:", a.shape, " -> (b, num_heads, num_tokens, head_dim)")
print("\na @ a.transpose(2,3):")
print(a @ a.transpose(2, 3))

first = a[0, 0]
print("\n第一个头 (a[0,0] @ a[0,0].T):")
print(first @ first.T)
second = a[0, 1]
print("\n第二个头 (a[0,1] @ a[0,1].T):")
print(second @ second.T)
print("\n结果完全一致 -> 一次批量矩阵乘 = 对 h 个头做循环。")


# ======================================================================
# Part B. 高效多头注意力模块
# ======================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads,
                 qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out 必须能被 num_heads 整除"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads      # 每个头的维度

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 融合各头输出
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

        # 切分：(b, n, d_out) -> (b, n, h, head_dim) -> (b, h, n, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)   # (b, h, n, n)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)   # (b, n, h, d)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


print("\n--- Part B: MultiHeadAttention ---")
batch = torch.stack((INPUTS, INPUTS), dim=0)
batch_size, context_length, d_in = batch.shape

torch.manual_seed(123)
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print("context_vecs:")
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape, "  # (b, n, d_out)，输出维度不再暴涨")

# ---- 逐头可视化 ----
with torch.no_grad():
    k = mha.W_key(batch)
    q = mha.W_query(batch)
    b_, n_, _ = k.shape
    hd = mha.head_dim
    k = k.view(b_, n_, mha.num_heads, hd).transpose(1, 2)
    q = q.view(b_, n_, mha.num_heads, hd).transpose(1, 2)
    scores = q @ k.transpose(2, 3)
    scores = scores.masked_fill(mha.mask.bool()[:n_, :n_], -torch.inf)
    w = torch.softmax(scores / hd ** 0.5, dim=-1)   # (b, h, n, n)

heads_np = w[0].numpy()
fig, axes = plt.subplots(1, mha.num_heads, figsize=(9.0, 4.0), dpi=160)
for h in range(mha.num_heads):
    ax = axes[h]
    ax.imshow(heads_np[h], cmap="Blues", vmin=0, vmax=heads_np.max())
    for i in range(n_):
        for j in range(n_):
            v = heads_np[h, i, j]
            ax.text(j, i, f"{v:.2f}" if v > 0 else "0",
                    ha="center", va="center",
                    color="white" if v > heads_np.max() * 0.6 else "#1f2937",
                    fontsize=6)
    ax.set_xticks(range(n_), labels=TOKENS, fontsize=7)
    ax.set_yticks(range(n_), labels=TOKENS, fontsize=7)
    ax.set_title(f"Head {h + 1}", fontsize=10)
fig.suptitle("MultiHeadAttention: different heads, different attention patterns",
             fontsize=11)
fig.tight_layout()
p = os.path.join(FIG_DIR, "07_head_patterns.png")
fig.savefig(p)
plt.close(fig)
print(f"\n[figure saved] {p}")

n_params = sum(x.numel() for x in mha.parameters())
print(f"参数量: {n_params:,}（d_in=3, d_out=2, 2 个头）")
