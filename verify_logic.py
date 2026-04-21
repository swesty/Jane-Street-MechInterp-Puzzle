import torch

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)
linears = [m for m in model.children() if hasattr(m, "in_features")]

period, body_start, n_blocks = 42, 18, 63

W0 = linears[body_start + 28].weight  # block 0
b0 = linears[body_start + 28].bias

# Inferred bias check: is it zero?
print(f"bias stats: max|b|={b0.abs().max().item()}, all zero? {(b0 == 0).all().item()}")

# === 1. VERIFY MODEL ===
# Dump structure for ALL THREE logic-row groups (32-63, 64-95, 96-127)
print("\n=== Logic group structure for block 0 ===")
def describe_group(rows):
    triples = []
    for r in rows:
        plus = (W0[r] == 1).nonzero(as_tuple=False).flatten().tolist()
        neg  = (W0[r] == -2).nonzero(as_tuple=False).flatten().tolist()
        triples.append((r, plus, neg))
    return triples

for grp_start in [32, 64, 96]:
    rows = list(range(grp_start, grp_start + 32))
    print(f"\n  group rows {grp_start}-{grp_start+31}:")
    for r, p, n in describe_group(rows[:6]):
        print(f"    row {r}: +1@{p}  -2@{n}")
    print("    ...")
    for r, p, n in describe_group(rows[-3:]):
        print(f"    row {r}: +1@{p}  -2@{n}")

# === 1. Manual simulation: forward through Linear[body_start+28] ===
# Per-block formulas; let's only verify block 0
# Logic row 32+i:  output = ReLU(in[i] + in[((i+25) mod 32)+96] - 2*in[((i+25) mod 32)+128])
# Need to find formulas for 64-95 and 96-127 too.

# Decode each group by inspecting the structure programmatically
def decode_logic_group(W, grp_start):
    """For rows [grp_start..grp_start+31], find +1@(i, X(i)+B1) -2@(Y(i)+B2)"""
    # First row: identify base
    r0 = grp_start
    plus = (W[r0] == 1).nonzero(as_tuple=False).flatten().tolist()
    neg  = (W[r0] == -2).nonzero(as_tuple=False).flatten().tolist()
    # plus has 2 entries, neg has 1
    a, b = plus[0], plus[1]
    c = neg[0]
    # Determine which of a,b is "row index i" base (i=0 here)
    # and which is the "rotated" pointer
    return (a, b, c)

print("\n=== Decoded base triples (i=0) for block 0 ===")
for grp in [32, 64, 96]:
    a, b, c = decode_logic_group(W0, grp)
    print(f"  group {grp}-{grp+31} row {grp}: +1@({a}, {b}) -2@{c}")

# Verify: predict W * input for random input
def predict_block0(x):
    """x: [256] -> y: [288] via my model"""
    out = torch.zeros(288)
    # copy rows
    for r in range(0, 32):       # out group 0
        out[r] = x[r + 64]
    for r in range(128, 192):    # out groups 4,5 <- in groups 0,1
        out[r] = x[r - 128]
    for r in range(192, 288):    # out groups 6,7,8 <- in groups 5,6,7
        out[r] = x[r - 32]
    # logic rows: each applies ReLU(in[i] + in[((i+R) mod 32) + base_b] - 2*in[((i+R) mod 32) + base_c])
    # Block 0 has R=25. base_b=96, base_c=128 for group 32-63.
    # Group 64-95 and 96-127 have other bases — read from W0
    for grp_start in [32, 64, 96]:
        a0, b0_col, c0 = decode_logic_group(W0, grp_start)
        # base for "in[i]" column is the SMALLER of (a0, b0_col)? for group 32, a0=0 b0_col=121
        # The 'i' column is the one whose row 0 value equals the row index minus offset
        # I'll just enumerate all rows directly
        for i in range(32):
            r = grp_start + i
            plus = (W0[r] == 1).nonzero(as_tuple=False).flatten().tolist()
            neg  = (W0[r] == -2).nonzero(as_tuple=False).flatten().tolist()
            v = x[plus[0]] + x[plus[1]] - 2 * x[neg[0]]
            out[r] = torch.relu(v)
    return out

# Compare
x = torch.randn(256)
linear = linears[body_start + 28]
y_real = torch.relu(linear(x))  # but linear[body_start+28] is followed by a ReLU
y_mine = predict_block0(x)
diff = (y_real - y_mine).abs().max().item()
print(f"\nManual vs real (block 0, random input, post-ReLU): max|Δ| = {diff}")

# Also try without ReLU on real (since I applied ReLU only to the logic rows)
y_real_noR = linear(x)
y_mine_noR = predict_block0(x).clone()
# I did ReLU on logic rows — undo for fair comparison
# Actually for the COPY rows I didn't apply ReLU; for the logic rows I did.
# To compare against bare Linear (no ReLU), recompute logic rows without ReLU:
def predict_block0_noR(x):
    out = torch.zeros(288)
    for r in range(0, 32):
        out[r] = x[r + 64]
    for r in range(128, 192):
        out[r] = x[r - 128]
    for r in range(192, 288):
        out[r] = x[r - 32]
    for grp_start in [32, 64, 96]:
        for i in range(32):
            r = grp_start + i
            plus = (W0[r] == 1).nonzero(as_tuple=False).flatten().tolist()
            neg  = (W0[r] == -2).nonzero(as_tuple=False).flatten().tolist()
            out[r] = x[plus[0]] + x[plus[1]] - 2 * x[neg[0]]
    return out

y_mine_noR = predict_block0_noR(x)
diff2 = (y_real_noR - y_mine_noR).abs().max().item()
print(f"Manual vs real (no ReLU): max|Δ| = {diff2}")
