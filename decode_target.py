import torch
import hashlib

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)
linears = [m for m in model.children() if hasattr(m, "in_features")]
period, body_start, n_blocks = 42, 18, 63

def encode(s):
    s = str(s)[:55].ljust(55, "\x00")
    return torch.tensor([ord(c) for c in s], dtype=torch.float32)

@torch.no_grad()
def get_activation(x, target_child_idx):
    state = x
    for i, m in enumerate(model.children()):
        state = m(state)
        if i == target_child_idx:
            return state
    return state

# === (i) Confirm byte mapping: get the 192-dim input to L2719 and decode bytes ===
# L2719 is the second-to-last Linear; its input is the post-ReLU output of Linear[2718]
# Child idx for ReLU after Linear[k] = 2k+1; so Linear[2718] -> ReLU at child 2*2718+1 = 5437
in192_idx = 2*2718 + 1
print("=== (i) Decoding 192 bits → 24 bytes for various inputs ===")

for inp in ["vegetable dog", "\x00"*55, "a", "test"]:
    pre_tail = get_activation(encode(inp), in192_idx)
    print(f"\n  --- input {inp!r} ---")
    print(f"  pre_tail shape={pre_tail.shape}, unique={torch.unique(pre_tail).tolist()}, max={pre_tail.max().item()}")
    # Reshape into 24 bytes of 8 bits each, treat as little-endian bit-weight
    bytes_ = []
    for bi in range(24):
        bits = pre_tail[bi*8:(bi+1)*8]
        byte_val = sum(int(bits[k].item()) * (2**k) for k in range(8))
        bytes_.append(byte_val)
    print(f"  24 bytes: {bytes_}")
    print(f"  hex: {' '.join(f'{b:02x}' for b in bytes_)}")
    # Show which conditions PASS
    targets_lit = [(0, 199), (1, 239), (2, 101), (3, 35), (12, 194), (13, 185), (14, 172), (15, 227)]
    targets_diff = [(4, 8, 60), (5, 9, 64), (6, 10, 170), (7, 11, 50),
                    (16, 20, 117), (17, 21, 149), (18, 22, 250), (19, 23, 124)]
    n_pass = 0
    for i, T in targets_lit:
        ok = bytes_[i] == T
        if ok: n_pass += 1
        print(f"    B{i:2d} = {bytes_[i]:3d}  target={T:3d}  {'✓' if ok else '✗'}")
    for a, b, k in targets_diff:
        v = bytes_[a] - 2*bytes_[b]
        ok = v == k
        if ok: n_pass += 1
        print(f"    B{a:2d} - 2·B{b:2d} = {v:4d}  target={k:3d}  {'✓' if ok else '✗'}")
    print(f"  PASS COUNT: {n_pass}/16 (need 16 for output=1)")

# === (ii) Check if the 8 literal target bytes form a recognizable hash ===
print("\n=== (ii) Are the 8 magic bytes a hash of something? ===")
target_lit_bytes = bytes([199, 239, 101, 35, 194, 185, 172, 227])
print(f"  target literal bytes: {target_lit_bytes.hex()}")
# Check various hash candidates
for cand in ["", "vegetable dog", "jane street", "jane-street", "Jane Street", "puzzle", "answer",
             "\x00"*55, " "*55, "0"*55, "a"*55]:
    h_md5 = hashlib.md5(cand.encode()).digest()
    h_sha1 = hashlib.sha1(cand.encode()).digest()
    h_sha256 = hashlib.sha256(cand.encode()).digest()
    if target_lit_bytes in h_md5 or target_lit_bytes in h_sha1 or target_lit_bytes in h_sha256:
        print(f"  MATCH for {cand!r}!")

# === (iii) Trace activation at the END OF THE BODY for various inputs and see how it maps to the 192-bit pre-tail ===
# End of body = ReLU after Linear[body_start + n_blocks*period - 1] = ReLU after Linear[2663]
end_of_body_idx = 2*2663 + 1
print(f"\n=== (iii) End-of-body state (after blk62 last layer + ReLU) ===")
for inp in ["vegetable dog", "\x00"*55]:
    state = get_activation(encode(inp), end_of_body_idx)
    print(f"\n  --- input {inp!r} ---")
    print(f"  shape={state.shape}, unique={torch.unique(state).tolist()[:20]}")
    print(f"  state values per group (32 dims each):")
    for g in range(state.shape[0] // 32 + (1 if state.shape[0] % 32 else 0)):
        chunk = state[g*32:(g+1)*32].tolist()
        # show as ints, with min/max
        as_int = [int(v) for v in chunk]
        print(f"    [{g*32:3d}-{g*32+31:3d}] min={min(as_int)} max={max(as_int)}: {as_int[:24]}{'...' if len(chunk)>24 else ''}")
