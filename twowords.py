"""Brute force two-word combinations from a wordlist against MD5 target."""
import hashlib
import sys
import time

target = "c7ef65233c40aa32c2b9ace37595fa7c"

# Load system wordlist, filter to reasonable words (2-15 chars, lowercase)
words = []
with open("/usr/share/dict/words") as f:
    for w in f:
        w = w.strip().lower()
        if 2 <= len(w) <= 15 and w.isalpha():
            words.append(w)
words = sorted(set(words))
print(f"loaded {len(words)} unique words")

# Also add themed words not in dict
extras = [
    "md5", "hash", "neural", "net", "network", "tensor", "tensors",
    "relu", "mlp", "pytorch", "torch", "model", "puzzle",
    "jane", "street", "janestreet", "ocaml", "quant", "trading",
    "neolithic", "burial", "mound", "hike", "plumber",
    "archaeology", "vegetable", "cipher", "crypto", "secret",
    "mechinterp", "interpretability", "mechanistic",
]
for e in extras:
    if e not in words:
        words.append(e)
print(f"total {len(words)} words after extras")

t0 = time.time()
checked = 0
for i, w1 in enumerate(words):
    for w2 in words:
        s = f"{w1} {w2}"
        if len(s) > 55:
            continue
        if hashlib.md5(s.encode()).hexdigest() == target:
            print(f"\n*** FOUND: '{s}'")
            sys.exit(0)
        checked += 1
    if i % 500 == 0:
        elapsed = time.time() - t0
        rate = checked / elapsed if elapsed > 0 else 0
        print(f"  w1[{i}] '{w1}', checked {checked:,} combos, {rate:,.0f}/s", end="\r")

print(f"\nno match in {checked:,} two-word combos ({time.time()-t0:.1f}s)")
