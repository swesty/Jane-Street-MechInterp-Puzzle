import torch

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)
linears = [m for m in model.children() if hasattr(m, "in_features")]

period, body_start = 42, 18

# For block 0, dump weight value distribution + sparsity for every offset
print("=== Block 0: ROM layer survey ===")
print(f"{'off':>3} | {'in':>4}→{'out':<4} | {'nz%':>6} | {'#unique':>7} | {'val histogram':<60} | bias_uniq")
for off in range(period):
    L = linears[body_start + off]
    W, b = L.weight, L.bias
    nz = (W != 0).sum().item()
    pct = 100.0 * nz / W.numel()
    vals, counts = torch.unique(W, return_counts=True)
    n_uniq = vals.numel()
    if n_uniq <= 8:
        hist = ", ".join(f"{v.item():g}:{c.item()}" for v, c in zip(vals, counts))
    else:
        hist = f"min={vals.min().item():g} max={vals.max().item():g} (#={n_uniq})"
    bvals = torch.unique(b)
    if bvals.numel() <= 5:
        bhist = ", ".join(f"{v.item():g}" for v in bvals)
    else:
        bhist = f"#{bvals.numel()}"
    star = " *" if off in (28, 41) else ""
    print(f"{off:>3}{star:2} | {L.in_features:>4}→{L.out_features:<4} | {pct:>5.2f}% | {n_uniq:>7} | {hist:<60} | {bhist}")
