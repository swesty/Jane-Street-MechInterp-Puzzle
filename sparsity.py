import torch

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)
linears = [m for m in model.children() if hasattr(m, "in_features")]

period, body_start, n_blocks = 42, 18, 63

# Offset 28: shape (288, 256). 480 nonzeros = 384 ones + 96 -2s.
# Inspect block 0
W = linears[body_start + 28].weight  # [out=288, in=256]
print(f"shape: {W.shape}, dtype: {W.dtype}")
print(f"nonzeros: {(W != 0).sum().item()}, +1s: {(W == 1).sum().item()}, -2s: {(W == -2).sum().item()}")

# How many nonzeros per output row? per input col?
nz_per_row = (W != 0).sum(dim=1)  # [288]
nz_per_col = (W != 0).sum(dim=0)  # [256]
print(f"\nnonzeros per output row: min={nz_per_row.min().item()}, max={nz_per_row.max().item()}, "
      f"unique={torch.unique(nz_per_row, return_counts=True)}")
print(f"nonzeros per input col: min={nz_per_col.min().item()}, max={nz_per_col.max().item()}, "
      f"unique={torch.unique(nz_per_col, return_counts=True)}")

# Per-row nonzero values: each row has how many +1, how many -2?
ones_per_row = (W == 1).sum(dim=1)
neg2_per_row = (W == -2).sum(dim=1)
print(f"\n+1s per row: unique={torch.unique(ones_per_row, return_counts=True)}")
print(f"-2s per row: unique={torch.unique(neg2_per_row, return_counts=True)}")

# What about per-column?
ones_per_col = (W == 1).sum(dim=0)
neg2_per_col = (W == -2).sum(dim=0)
print(f"\n+1s per col: unique={torch.unique(ones_per_col, return_counts=True)}")
print(f"-2s per col: unique={torch.unique(neg2_per_col, return_counts=True)}")

# Are the +1 and -2 positions related? E.g., does each row have a +1 at col i and -2 at col j(i)?
print("\n=== Per-row pattern (first 20 output rows) ===")
for r in range(20):
    plus_cols = (W[r] == 1).nonzero(as_tuple=False).flatten().tolist()
    neg_cols = (W[r] == -2).nonzero(as_tuple=False).flatten().tolist()
    print(f"  row {r:3d}: +1@{plus_cols}  -2@{neg_cols}")

# Look at blocks 0..3 (the inner cluster cycle): row 0's +1 columns
print("\n=== row 0 +1 columns across blocks 0..15 (inner cycle 4 reps) ===")
for b in range(16):
    Wb = linears[body_start + b*period + 28].weight
    plus_cols = (Wb[0] == 1).nonzero(as_tuple=False).flatten().tolist()
    neg_cols = (Wb[0] == -2).nonzero(as_tuple=False).flatten().tolist()
    print(f"  block {b:2d} (cluster {(b//16)*4 + (b%4)}): +1@{plus_cols}  -2@{neg_cols}")
