import torch

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)
linears = [m for m in model.children() if hasattr(m, "in_features")]
period, body_start, n_blocks = 42, 18, 63

# Body ends at Linear[body_start + n_blocks*period - 1] = Linear[18 + 63*42 - 1] = Linear[2663]
# Tail = Linear[2664 .. 2720]
tail_start = body_start + n_blocks * period
print(f"#linears total: {len(linears)}, tail starts at Linear[{tail_start}], #tail = {len(linears) - tail_start}")

print("\n=== Tail layer survey ===")
print(f"{'idx':>4} | {'in':>4}→{'out':<4} | {'nz%':>6} | {'#uniq':>6} | val histogram                    | bias")
for i in range(tail_start, len(linears)):
    L = linears[i]
    W, b = L.weight, L.bias
    nz = (W != 0).sum().item()
    pct = 100.0 * nz / W.numel()
    vals, counts = torch.unique(W, return_counts=True)
    n_uniq = vals.numel()
    if n_uniq <= 10:
        hist = ", ".join(f"{v.item():g}:{c.item()}" for v, c in zip(vals, counts))
    else:
        hist = f"min={vals.min().item():g} max={vals.max().item():g} (#={n_uniq})"
    bvals = torch.unique(b)
    if bvals.numel() <= 8:
        bhist = ", ".join(f"{v.item():g}" for v in bvals)
    else:
        bhist = f"#{bvals.numel()} bvals: min={bvals.min().item():g}, max={bvals.max().item():g}"
    print(f"L{i:>3} | {L.in_features:>4}→{L.out_features:<4} | {pct:>5.2f}% | {n_uniq:>5} | {hist:<60} | {bhist}")

# === The very last two layers ===
print("\n=== LAST TWO LAYERS DETAIL ===")
L1 = linears[-2]  # Linear(192, 48)
L2 = linears[-1]  # Linear(48, 1)

print(f"\nLAST-1: Linear({L1.in_features}, {L1.out_features})")
print(f"  weight shape {L1.weight.shape}")
print(f"  weight values: {torch.unique(L1.weight, return_counts=True)}")
print(f"  bias: {L1.bias.tolist()[:24]}...")

# How many nonzeros per output row?
nz_per_row = (L1.weight != 0).sum(dim=1)
print(f"  nz per row: min={nz_per_row.min().item()}, max={nz_per_row.max().item()}, hist={torch.unique(nz_per_row, return_counts=True)}")

print(f"\nLAST: Linear({L2.in_features}, {L2.out_features})")
print(f"  weight shape {L2.weight.shape}")
print(f"  weight: {L2.weight.tolist()}")
print(f"  weight unique: {torch.unique(L2.weight)}")
print(f"  bias: {L2.bias.tolist()}")

# Show the L2 weights side-by-side with L1's bias
print(f"\n=== Last-1 layer breakdown (each of 48 output rows) ===")
print(f"{'r':>3} | bias  | nz | weight cols & vals")
for r in range(L1.out_features):
    cols = (L1.weight[r] != 0).nonzero(as_tuple=False).flatten()
    vals = L1.weight[r, cols].tolist()
    cv = list(zip(cols.tolist(), vals))
    print(f"{r:>3} | {L1.bias[r].item():>5g} | {len(cv):>2} | {cv}")
