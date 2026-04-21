import torch
from collections import Counter

model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)

linears = [m for m in model.children() if hasattr(m, "in_features")]
print(f"#Linear layers: {len(linears)}")

dims = [linears[0].in_features] + [l.out_features for l in linears]
print(f"width sequence ({len(dims)} dims):")
print(dims[:50])
print("...")
print(dims[-30:])

# look for repeats
from collections import Counter
c = Counter(dims)
print("\nMost common widths:", c.most_common(15))

# Look for substring patterns: find a short pattern that repeats
def find_repeat(seq):
    n = len(seq)
    for L in range(2, n//2 + 1):
        for start in range(n - 2*L + 1):
            if seq[start:start+L] == seq[start+L:start+2*L]:
                return start, L, seq[start:start+L]
    return None

# Look for the same width transitions repeating
transitions = list(zip(dims[:-1], dims[1:]))
tc = Counter(transitions)
print("\nMost common transitions:", tc.most_common(10))

# Print the full sequence in a structured way (chunks of 20)
print("\nFull width sequence:")
for i in range(0, len(dims), 20):
    print(f"  [{i:4d}] " + " ".join(f"{d:4d}" for d in dims[i:i+20]))
