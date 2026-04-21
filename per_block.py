import torch

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)
linears = [m for m in model.children() if hasattr(m, "in_features")]

period = 42
body_start = 18
n_blocks = 63

print(f"#blocks = {n_blocks} (body from Linear[{body_start}] to Linear[{body_start+n_blocks*period-1}])")

# Collect per-block (offset 28) and per-block (offset 41) weights/biases
o28 = []
o41 = []
for b in range(n_blocks):
    o28.append(linears[body_start + b*period + 28])
    o41.append(linears[body_start + b*period + 41])

# Sanity: print shape variation
print("\noffset 28 shapes:", set((l.weight.shape) for l in o28))
print("offset 41 shapes:", [tuple(l.weight.shape) for l in o41[:5]], "...")
shape_seq = [l.weight.shape[0] for l in o41]
print("offset 41 out_dim sequence:", shape_seq)

# Look at offset-28 weight value distribution
import torch
all_o28_w = torch.stack([l.weight for l in o28])  # [63, 256, 288]
print("\noffset 28 weight values stats:")
print(f"  min={all_o28_w.min().item()} max={all_o28_w.max().item()}")
unique_vals = torch.unique(all_o28_w)
print(f"  #unique values: {unique_vals.numel()}")
if unique_vals.numel() < 20:
    print(f"  unique: {unique_vals.tolist()}")

# Diff each block from block 0
print("\nPer-block differences in offset 28 (weight L1, bias L1, # nonzero w-diff):")
ref_w = o28[0].weight
ref_b = o28[0].bias
for b in range(n_blocks):
    dw = (o28[b].weight - ref_w)
    db = (o28[b].bias - ref_b)
    nzw = (dw != 0).sum().item()
    nzb = (db != 0).sum().item()
    if nzw > 0 or nzb > 0:
        print(f"  block {b:2d}: nz_w={nzw:5d} max|Δw|={dw.abs().max().item():.2g} | nz_b={nzb:3d} max|Δb|={db.abs().max().item():.2g}")
