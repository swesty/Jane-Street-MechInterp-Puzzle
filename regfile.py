import torch

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)
linears = [m for m in model.children() if hasattr(m, "in_features")]
period, body_start, n_blocks = 42, 18, 63

def encode(s):
    s = str(s)[:55].ljust(55, "\x00")
    return torch.tensor([ord(c) for c in s], dtype=torch.float32)

@torch.no_grad()
def state_at_block_boundary(x, block_idx):
    """Return state at the START of block `block_idx`.
       block 0 starts after the prefix (after model child 2*body_start - 1 = ReLU at child 35).
       For block b > 0: state at start = output after ReLU at child 2*(body_start + b*period - 1) + 1
       """
    if block_idx == 0:
        target_child = 2*body_start - 1  # last ReLU before body, i.e., after Linear[17]
    else:
        target_child = 2*(body_start + block_idx*period - 1) + 1  # ReLU after last layer of prev block
    state = x
    for i, m in enumerate(model.children()):
        state = m(state)
        if i == target_child:
            return state
    return state

# Sanity: get block 0 input dim (should be 336)
x = encode("\x00"*55)
s = state_at_block_boundary(x, 0)
print(f"block 0 input shape: {s.shape}")  # 336

# Compare states for several inputs
inputs = [
    "\x00"*55,
    "a",
    "ab",
    "abc",
    "vegetable dog",
    "\x01"*55,
    "A"*55,
]

states_b0 = {inp: state_at_block_boundary(encode(inp), 0) for inp in inputs}

# Look at first input: which dimensions are constant across inputs (at block 0 input)?
ref = states_b0["\x00"*55]
print(f"\n=== Block 0 input: dims that are constant across inputs ===")
const_dims = []
for d in range(ref.numel()):
    is_const = all(states_b0[inp][d].item() == ref[d].item() for inp in inputs)
    if is_const:
        const_dims.append((d, ref[d].item()))
print(f"#const dims: {len(const_dims)} / {ref.numel()}")
print(f"unique constant values: {sorted(set(v for _, v in const_dims))}")

# Show the state for "vegetable dog" laid out as 336 values
print("\n=== Block 0 input state for 'vegetable dog' (336 dims) ===")
s = states_b0["vegetable dog"]
# Print in rows of 32
for i in range(0, 336, 32):
    chunk = s[i:i+32].tolist()
    print(f"  [{i:3d}-{i+31:3d}] " + " ".join(f"{int(v):3d}" if v == int(v) else f"{v:5.1f}" for v in chunk))

# Track non-constant dims across all 64 boundaries
print("\n=== State evolution: how does state change across blocks for 'vegetable dog'? ===")
states_per_blk = []
x_veg = encode("vegetable dog")
for b in range(n_blocks + 1):
    if b == n_blocks:
        # state after the last block = right before tail starts = state after ReLU of Linear at body_end
        target_child = 2*(body_start + n_blocks*period - 1) + 1
        if target_child > 2*len(linears) - 1:
            break
    s = state_at_block_boundary(x_veg, b)
    states_per_blk.append(s)

print(f"Got {len(states_per_blk)} block-boundary states")
print(f"shapes: {[s.shape[0] for s in states_per_blk[:5]]} ... {[s.shape[0] for s in states_per_blk[-5:]]}")

# How many dims change between consecutive blocks?
print("\n  block transition | dims_changed / total | max|Δ|")
for b in range(len(states_per_blk) - 1):
    s1, s2 = states_per_blk[b], states_per_blk[b+1]
    if s1.shape == s2.shape:
        diff = (s1 != s2).sum().item()
        mx = (s1 - s2).abs().max().item()
        if b < 5 or b > len(states_per_blk) - 5 or b in {30, 31, 46, 47}:
            print(f"  blk{b:2d}->{b+1:2d}     | {diff:4d} / {s1.shape[0]:4d}      | {mx:.0f}")
    else:
        print(f"  blk{b:2d}->{b+1:2d} SHAPE CHANGE | {s1.shape[0]} -> {s2.shape[0]}")

# Find which dimensions encode the input directly
# For "vegetable dog" the chars are: 118, 101, 103, 101, 116, 97, 98, 108, 101, 32, 100, 111, 103, 0, 0, ...
chars = [ord(c) for c in "vegetable dog".ljust(55, "\x00")]
print(f"\n=== Looking for dims at block 0 input that hold each input char ===")
s0 = states_b0["vegetable dog"]
for ci, c in enumerate(chars[:13]):
    matches = (s0 == c).nonzero(as_tuple=False).flatten().tolist()
    print(f"  char[{ci:2d}] = {c:3d} ('{chr(c) if c >= 32 else '?'}'): found at dims {matches[:8]}")
