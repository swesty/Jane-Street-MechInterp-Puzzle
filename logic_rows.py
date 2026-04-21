import torch

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)
linears = [m for m in model.children() if hasattr(m, "in_features")]

period, body_start, n_blocks = 42, 18, 63

W0 = linears[body_start + 28].weight  # block 0
nz_per_row = (W0 != 0).sum(dim=1)
copy_rows = (nz_per_row == 1).nonzero(as_tuple=False).flatten().tolist()
logic_rows = (nz_per_row == 3).nonzero(as_tuple=False).flatten().tolist()
print(f"copy_rows: {len(copy_rows)} (first/last: {copy_rows[:5]} ... {copy_rows[-5:]})")
print(f"logic_rows: {len(logic_rows)} (first/last: {logic_rows[:5]} ... {logic_rows[-5:]})")

# Are copy_rows = [0..191] and logic_rows = [192..287]?
print(f"\ncopy_rows == 0..191? {copy_rows == list(range(192))}")
print(f"logic_rows == 192..287? {logic_rows == list(range(192, 288))}")

# Print the copy permutation: for each copy row, which input col does it pull?
print("\n=== Copy permutation (copy_rows -> input col) ===")
perm = {}
for r in copy_rows:
    cols = (W0[r] != 0).nonzero(as_tuple=False).flatten().tolist()
    perm[r] = cols[0]
# Show in compact rows of 16
items = sorted(perm.items())
for i in range(0, len(items), 12):
    print("  " + "  ".join(f"r{r:3d}→{c:3d}" for r,c in items[i:i+12]))

# Now logic rows: each has +1@a, +1@b, -2@c
print("\n=== Logic rows for block 0 (first 20) ===")
for ri in logic_rows[:20]:
    plus = (W0[ri] == 1).nonzero(as_tuple=False).flatten().tolist()
    neg = (W0[ri] == -2).nonzero(as_tuple=False).flatten().tolist()
    print(f"  row {ri}: +1@{plus}  -2@{neg}")

# Compare logic row triples across blocks 0..3 (the inner-cycle clusters)
print("\n=== Logic rows: same row index across clusters 0,1,2,3 (block 0,1,2,3) ===")
for ri in logic_rows[:8]:
    print(f"  row {ri}:")
    for b in range(4):
        Wb = linears[body_start + b*period + 28].weight
        plus = (Wb[ri] == 1).nonzero(as_tuple=False).flatten().tolist()
        neg = (Wb[ri] == -2).nonzero(as_tuple=False).flatten().tolist()
        print(f"    block {b} (cluster {(b//16)*4 + (b%4)}): +1@{plus}  -2@{neg}")

# Now compare across outer-cycles: blocks 0, 16, 32, 48 (all cluster {0,4,8,12})
print("\n=== Logic rows: cluster 0, 4, 8, 12 (outer cycle change) ===")
for ri in logic_rows[:6]:
    print(f"  row {ri}:")
    for b in [0, 16, 32, 48]:
        Wb = linears[body_start + b*period + 28].weight
        plus = (Wb[ri] == 1).nonzero(as_tuple=False).flatten().tolist()
        neg = (Wb[ri] == -2).nonzero(as_tuple=False).flatten().tolist()
        print(f"    block {b:2d} (cluster {(b//16)*4 + (b%4):2d}): +1@{plus}  -2@{neg}")
