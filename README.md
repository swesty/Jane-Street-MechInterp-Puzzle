# Jane Street Mechanistic Interpretability Puzzle

Reverse-engineering Jane Street's 2025 mechanistic interpretability puzzle: a 289-million-parameter PyTorch model that, when given the right input, outputs `1`.

> *"Maybe start by looking at the last two layers."* — the only hint.

**Spoiler / TL;DR:** the model is a hand-built implementation of **MD5** using only `Linear` layers and `ReLU` activations. The target preimage is **`bitter lesson`** — a nod to Rich Sutton's essay.

For the full investigation, see [REPORT.md](./REPORT.md).

---

## The Puzzle

A `~1.16 GB` PyTorch `Sequential` (`model.pt`) that:

- Takes a string input (truncated to 55 chars, ASCII-encoded, null-padded).
- Pushes it through **5,442 modules** (2,721 `Linear` + 2,721 `ReLU`, no skips, no attention, no norm).
- Outputs a single scalar — almost always `0`.

Find an input that makes it output something nonzero.

## What's in this repo

The code here is the working set of scripts used during the investigation — exploratory, not polished. Roughly grouped:

| Area | Scripts |
|---|---|
| Architecture survey | `look*.py`, `shapes.py`, `rom_survey.py`, `per_block.py`, `diff_blocks.py` |
| Weight / sparsity analysis | `sparsity.py`, `cluster.py`, `biases.py`, `logic_rows.py`, `verify_logic.py`, `verify2.py` |
| State / activation tracing | `trace.py`, `regfile.py`, `tail.py` |
| Tail decoding | `decode_target.py` (recovers the 128-bit MD5 target) |
| Preimage search | `find_preimage.py`, `wordlist.py`, `themed.py`, `twowords.py`, `brute5.c` |

`model/` holds the original `model.pt` (and a Python 3.11 re-serialized copy); `space/` mirrors the original Gradio Space. Both are excluded from version control because of size.

## Setup

This project uses [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

You'll need to download `model.pt` separately from Jane Street's puzzle release and place it in `model/`.

## The Answer

```python
>>> model("bitter lesson")
tensor([1.])
```

```
MD5("bitter lesson") = c7ef65233c40aa32c2b9ace37595fa7c
```

## Approach in one paragraph

The width sequence of the 2,721 Linear layers reveals an 18-layer prefix, then a 42-layer block repeating 63 times, then a 57-layer tail. Hashing weights across blocks shows 40 of 42 layers per block are bitwise-identical ROM; only two vary — a per-block "instruction" (16 unique matrices in a 4×4 cluster pattern) and a per-block "constant" (63 unique matrices). The ROM cascade halves power-of-two weights from 128 down to 1, doing per-bit decomposition; logic rows compute `ReLU(a + b - 2c - 1)` (boolean AND-NOT). The 4-phase × 16-step structure with bitwise primitives is MD5's `F/G/H/I` rounds. The tail's last two layers form 16 exact-equality indicators that compare the model's output against a fixed 128-bit hash — the target. Once decoded, the puzzle is just an MD5 preimage search; hashcat's combinator attack on the system dictionary finds `bitter lesson` in seconds.

## Credit

Puzzle by [Jane Street](https://www.janestreet.com/). Solution by me, with a lot of help from Claude.
