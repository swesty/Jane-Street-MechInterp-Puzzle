import torch

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)
linears = [m for m in model.children() if hasattr(m, "in_features")]

period, body_start = 42, 18

L = linears[body_start + 28]
W, b = L.weight, L.bias
print(f"bias shape: {b.shape}")
print(f"bias stats: min={b.min().item()} max={b.max().item()} all_zero={(b == 0).all().item()}")
vals, counts = torch.unique(b, return_counts=True)
for v, c in zip(vals.tolist(), counts.tolist()):
    print(f"  bias value {v}: count={c}")
print("nonzero bias positions:", (b != 0).nonzero(as_tuple=False).flatten().tolist())

# Same for offset 41 of block 0
L41 = linears[body_start + 41]
b41 = L41.bias
print(f"\noffset 41 bias: shape={b41.shape}, min={b41.min()}, max={b41.max()}")
vals, counts = torch.unique(b41, return_counts=True)
for v, c in zip(vals.tolist(), counts.tolist()):
    print(f"  bias value {v}: count={c}")
W41 = L41.weight
vals, counts = torch.unique(W41, return_counts=True)
print(f"\noffset 41 weight values:")
for v, c in zip(vals.tolist(), counts.tolist()):
    print(f"  value {v}: count={c} ({100*c/W41.numel():.2f}%)")
