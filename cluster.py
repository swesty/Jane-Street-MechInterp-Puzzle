import torch
import hashlib

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)
linears = [m for m in model.children() if hasattr(m, "in_features")]

period, body_start, n_blocks = 42, 18, 63

def hashw(t):
    return hashlib.sha1(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:10]

# offset 28: cluster by weight hash
print("=== offset 28 (288→256 LINEAR, weight hash sequence) ===")
seq28 = []
for b in range(n_blocks):
    L = linears[body_start + b*period + 28]
    seq28.append(hashw(L.weight))
# assign cluster IDs in order of first appearance
ids = {}
for h in seq28:
    if h not in ids:
        ids[h] = len(ids)
print(f"#unique clusters: {len(ids)}")
print("cluster sequence:", [ids[h] for h in seq28])

# offset 41: same
print("\n=== offset 41 (256→{336,368} LINEAR) ===")
seq41 = []
for b in range(n_blocks):
    L = linears[body_start + b*period + 41]
    seq41.append((tuple(L.weight.shape), hashw(L.weight), hashw(L.bias)))
ids41 = {}
for s in seq41:
    if s not in ids41:
        ids41[s] = len(ids41)
print(f"#unique clusters: {len(ids41)}")
print("cluster sequence:", [ids41[s] for s in seq41])
print("cluster -> shape:")
for k, v in ids41.items():
    print(f"  cluster {v}: shape={k[0]}, w={k[1]}, b={k[2]}")

# Look at the values / sparsity of offset 28 in block 0
print("\n=== offset 28 block 0 weight stats ===")
w = linears[body_start + 28].weight  # block 0
print(f"shape: {w.shape}")
vals, counts = torch.unique(w, return_counts=True)
for v, c in zip(vals.tolist(), counts.tolist()):
    print(f"  value {v}: {c} ({100*c/w.numel():.2f}%)")

# How many elements are nonzero in EACH block's offset-28 weight?
print("\n=== nonzero count per block (offset 28) ===")
for b in range(n_blocks):
    w = linears[body_start + b*period + 28].weight
    nz = (w != 0).sum().item()
    print(f"  block {b:2d}: nz={nz}, sum={w.sum().item():.0f}", end="")
    if (b+1) % 4 == 0:
        print()
    else:
        print("  |", end=" ")
print()
