import torch

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)
linears = [m for m in model.children() if hasattr(m, "in_features")]

# Encode input the same way the model wrapper does
def encode(s):
    s = str(s)[:55].ljust(55, "\x00")
    return torch.tensor([ord(c) for c in s], dtype=torch.float32)

period, body_start, n_blocks = 42, 18, 63

@torch.no_grad()
def trace(x, label):
    print(f"\n=== TRACE: {label!r} (sum={x.sum().item():.1f}) ===")
    # Walk through model.children() in order, recording activations after each module
    state = x
    rows = []
    for i, m in enumerate(model.children()):
        state = m(state)
        # only record after every Linear+ReLU PAIR (after the ReLU)
        if hasattr(m, 'inplace'):  # ReLU
            nz = (state != 0).sum().item()
            n = state.numel()
            l2 = state.norm().item()
            mx = state.max().item() if n > 0 else 0
            rows.append((i, nz, n, l2, mx))
    return rows

# Trace several inputs
for inp in ["vegetable dog", "", "a"*55, "\x01"*55]:
    rows = trace(encode(inp), inp[:30])
    # print as: pair index (1, 2, 3, ...) which corresponds to Linear[2k+1] = ReLU after Linear[k]
    print("  pair  | i_in_seq | nonzero/total | L2     | max")
    last_alive = -1
    for k, (i, nz, n, l2, mx) in enumerate(rows):
        # block id of this Linear-ReLU pair
        # pair k corresponds to Linear[k]
        if k < body_start:
            tag = f"prefix L{k}"
        elif k < body_start + n_blocks * period:
            b = (k - body_start) // period
            o = (k - body_start) % period
            tag = f"blk{b:02d}.o{o:02d}"
        else:
            tag = f"tail L{k}"
        if nz > 0:
            last_alive = k
            if k % 50 == 0 or k == len(rows)-1 or k < 20 or k > len(rows) - 20:
                print(f"  {k:4d} | {tag:14s} | {nz:5d}/{n:5d}  | {l2:7.2f} | {mx:7.2f}")
        else:
            print(f"  {k:4d} | {tag:14s} | DEAD (all 0)")
            break
    print(f"  -> last alive Linear index: {last_alive}")
