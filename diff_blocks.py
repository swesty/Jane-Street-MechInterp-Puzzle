import torch
from collections import defaultdict

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)

linears = [m for m in model.children() if hasattr(m, "in_features")]
print(f"#Linear: {len(linears)}")

# Period detected: 42 dims -> 42 Linear layers per body block, body starts at Linear index 18
# Group same-position-within-block layers
period = 42
body_start = 18
body_end = len(linears)  # we'll just go until we run out of period chunks

# Index by (offset_in_period, shape) — confirm same offset always has same shape
by_offset = defaultdict(list)  # offset -> list of (block_idx, linear_idx)
for k in range(body_start, len(linears)):
    off = (k - body_start) % period
    block = (k - body_start) // period
    by_offset[off].append((block, k))

# Print offset 0 (the marker layer): are all weights equal?
print("\n=== shape per offset across blocks ===")
shape_by_offset = {}
for off in range(period):
    shapes = set((linears[k].in_features, linears[k].out_features) for _, k in by_offset[off])
    shape_by_offset[off] = shapes
    if len(shapes) > 1:
        print(f"  offset {off:2d}: SHAPES VARY -> {shapes}")
    else:
        print(f"  offset {off:2d}: shape {next(iter(shapes))}, n={len(by_offset[off])}")

print("\n=== weight equality check per offset ===")
for off in range(period):
    entries = by_offset[off]
    # gather all the weight tensors at this offset
    ws = [linears[k].weight for _, k in entries]
    # group by shape (offsets 0/1 may have varying shapes)
    by_shape = defaultdict(list)
    for (b, k), w in zip(entries, ws):
        by_shape[tuple(w.shape)].append((b, k, w))
    msg_parts = []
    for shape, items in by_shape.items():
        ref_b, ref_k, ref_w = items[0]
        n_equal = 0
        n_diff = 0
        max_l1 = 0.0
        for b, k, w in items[1:]:
            if torch.equal(w, ref_w):
                n_equal += 1
            else:
                n_diff += 1
                d = (w - ref_w).abs().max().item()
                if d > max_l1:
                    max_l1 = d
        msg_parts.append(f"shape={shape} n={len(items)} equal_to_first={n_equal} diff_from_first={n_diff} max|Δ|={max_l1:.4g}")
    print(f"  offset {off:2d}: " + " | ".join(msg_parts))
