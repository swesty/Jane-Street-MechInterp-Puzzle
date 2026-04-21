"""Try MD5 preimage with a wordlist — common single words & themed candidates."""
import hashlib
import itertools
import os

target = "c7ef65233c40aa32c2b9ace37595fa7c"

# Themed candidates
themed = [
    "if you do figure it out, please let us know",
    "If you do figure it out, please let us know",
    "Today I went on a hike",
    "model.pt", "model", "the local neural plumber",
    "neural plumber", "the neural plumber",
    "hike", "burial", "neolithic", "tensors", "tensor pile", "pile of tensors",
    "We were amazed by the response",
    "ocaml", "OCaml", "Ocaml",
    "trading", "quant", "puzzle 2025", "puzzle2025",
    "2024-02", "2025-03-10", "2024", "2025",
    "puzzles", "Puzzles", "PUZZLE", "PUZZLES",
    "archaeology@janestreet.com", "archaeology",
    "what does it do", "What does it do?",
    "I figured it out", "I figured it out!",
    "found it", "Found it!", "got it", "Got it!",
    "Eureka", "eureka", "EUREKA", "Eureka!",
    "I solved it", "solved",
    "send help", "help me", "help",
    "vegetable", "dog", "vegetable dog", "vegetabledog",
]
for c in themed:
    if hashlib.md5(c.encode()).hexdigest() == target:
        print(f"*** MATCH: {c!r}")
        exit()
print(f"checked {len(themed)} themed phrases — no match")

# /usr/share/dict/words if available
for path in ["/usr/share/dict/words", "/usr/share/dict/web2"]:
    if os.path.exists(path):
        with open(path) as f:
            words = [w.strip() for w in f]
        print(f"trying {len(words)} words from {path}")
        cnt = 0
        for w in words:
            for v in (w, w.lower(), w.upper(), w.capitalize()):
                if hashlib.md5(v.encode()).hexdigest() == target:
                    print(f"*** MATCH: {v!r}")
                    exit()
                cnt += 1
        print(f"  no match in {cnt} variants")
        break

# Brute-force all 1-4 char printable ASCII
print("brute-forcing 1-4 char ASCII printable...")
charset = bytes(range(32, 127)).decode()
for L in range(1, 5):
    for combo in itertools.product(charset, repeat=L):
        s = "".join(combo)
        if hashlib.md5(s.encode()).hexdigest() == target:
            print(f"*** MATCH: {s!r}")
            exit()
    print(f"  done length {L}")
print("no match in 1-4 char ASCII")
