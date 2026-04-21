import hashlib
import torch

target = "c7ef65233c40aa32c2b9ace37595fa7c"

# Try lots of candidates
candidates = [
    "", "vegetable dog", "jane street", "Jane Street", "jane-street",
    "puzzle", "Puzzle", "answer", "Answer", "neural net",
    "neolithic", "burial mound", "tensor", "tensors", "archaeology",
    "hex-rays", "mechinterp", "mech-interp", "mechanistic interpretability",
    "JANE STREET", "octopus", "Octopus",
    "neural plumber", "the answer", "secret", "key",
    "0", "1", "test", "hello", "hello world",
    "a", "b", "c",
]
for c in candidates:
    h = hashlib.md5(c.encode()).hexdigest()
    mark = "  <-- MATCH" if h == target else ""
    print(f"  {c!r:35s} → {h}{mark}")

# Also try the target itself as some lookup
print(f"\nTarget hex string: {target}")
print(f"As bytes: {bytes.fromhex(target)}")

# Also try inputs of common puzzle forms
import itertools
print("\nSearching short ASCII printable strings...")
charset = "0123456789abcdefghijklmnopqrstuvwxyz"
for L in range(1, 5):
    for combo in itertools.product(charset, repeat=L):
        s = "".join(combo)
        if hashlib.md5(s.encode()).hexdigest() == target:
            print(f"  *** FOUND: {s!r}")
            break

# Confirm decoder works for empty string
print("\nVerifying decoder on \\x00*55 (which the wrapper treats as empty before MD5):")
model = torch.load("model/model_3_11.pt", weights_only=False, map_location="cpu")
model.train(False)
@torch.no_grad()
def get_pre_tail(s):
    s2 = str(s)[:55].ljust(55, "\x00")
    x = torch.tensor([ord(c) for c in s2], dtype=torch.float32)
    state = x
    for i, m in enumerate(model.children()):
        state = m(state)
        if i == 5437:  # ReLU after Linear[2718]
            return state
    return state

def to_bytes(b_array):
    by = []
    for bi in range(24):
        bits = b_array[bi*8:(bi+1)*8]
        by.append(int(sum(bits[k].item() * (2**k) for k in range(8))))
    return by

def decode_md5(bs):
    md5 = []
    for i in range(4):
        md5.append(bs[i])
    for i in range(4):
        md5.append(bs[4+i] - 2*bs[8+i])
    for i in range(4):
        md5.append(bs[12+i])
    for i in range(4):
        md5.append(bs[16+i] - 2*bs[20+i])
    return md5

for s in ["", "a", "vegetable dog"]:
    b = to_bytes(get_pre_tail(s))
    m = decode_md5(b)
    print(f"  input {s!r}: model_md5 = {' '.join(f'{x:02x}' for x in m)}, real_md5 = {hashlib.md5(s.encode()).hexdigest()}")
