import torch

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)
linears = [m for m in model.children() if hasattr(m, "in_features")]

period, body_start, n_blocks = 42, 18, 63

# Manual block 0 offset 28: include bias
def predict_block0(x):
    L = linears[body_start + 28]
    W, b = L.weight, L.bias
    out = torch.zeros(288)
    for r in range(0, 32):
        out[r] = x[r + 64] + b[r]
    for r in range(128, 192):
        out[r] = x[r - 128] + b[r]
    for r in range(192, 288):
        out[r] = x[r - 32] + b[r]
    for grp_start in [32, 64, 96]:
        for i in range(32):
            r = grp_start + i
            plus = (W[r] == 1).nonzero(as_tuple=False).flatten().tolist()
            neg  = (W[r] == -2).nonzero(as_tuple=False).flatten().tolist()
            out[r] = x[plus[0]] + x[plus[1]] - 2 * x[neg[0]] + b[r]
    return out

torch.manual_seed(42)
x = torch.randn(256)
y_real_noR = linears[body_start + 28](x)
y_mine_noR = predict_block0(x)
diff = (y_real_noR - y_mine_noR).abs().max().item()
print(f"Manual matches real (no ReLU): max|Δ| = {diff:.2e}")

# === Now look at REAL activations: run the model on "vegetable dog" and inspect
# the input to offset 28 of block 0 (i.e., output of Linear[18+27], post-ReLU)
def encode(s):
    s = str(s)[:55].ljust(55, "\x00")
    return torch.tensor([ord(c) for c in s], dtype=torch.float32)

@torch.no_grad()
def get_activation(x, target_idx):
    """Run model up through model[2*target_idx + 1] (Linear+ReLU pair) and return."""
    state = x
    for i, m in enumerate(model.children()):
        state = m(state)
        if i == target_idx:
            return state
    return state

for inp in ["vegetable dog", "a", "\x00"*55]:
    x_enc = encode(inp)
    # input to offset 28 of block 0 = output of Linear[27]+ReLU at child index 2*27+1 = 55
    # Linear[k] -> child index 2k. ReLU after Linear[k] -> child index 2k+1.
    pre_off28 = get_activation(x_enc, 2*(body_start + 27) + 1)  # post-ReLU before offset 28's linear
    print(f"\n--- input {inp!r} ---")
    print(f"  state into offset 28 of block 0 (size 256): unique vals = {torch.unique(pre_off28).tolist()[:10]}")
    print(f"  L2={pre_off28.norm().item():.2f}, max={pre_off28.max().item()}")
    # Apply offset 28 by hand using the ACTUAL inputs (not random)
    out_post_off28 = torch.relu(linears[body_start + 28](pre_off28))
    print(f"  state OUT of offset 28+ReLU (size 288): unique vals = {torch.unique(out_post_off28).tolist()[:10]}")
    print(f"  L2={out_post_off28.norm().item():.2f}, max={out_post_off28.max().item()}")
