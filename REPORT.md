# Cracking Jane Street's Mechanistic Interpretability Puzzle

## The Challenge

In early 2025, Jane Street published a mechanistic interpretability puzzle: a PyTorch model (`model.pt`, ~1.16 GB, 289 million parameters) that takes a string input and outputs a single number. For almost every input, the output is `0`. The challenge: find an input that produces a nonzero output.

The only hint: *"Maybe start by looking at the last two layers."*

The example input `"vegetable dog"` produces `0`. The Gradio source code contains a subtle comment on the input box: `# two words?`.

---

## Phase 1: Loading and Initial Inspection

### The Input Encoding

The model is a serialized `torch.nn.Sequential` with a custom `_call_impl` lambda attached via cloudpickle. Disassembling the bytecode revealed:

```python
lambda x: model.forward(torch.Tensor(list(map(ord, str(x)[:55].ljust(55, '\x00')))))
```

The input string is truncated to 55 characters, null-padded, and each character is converted to its ASCII codepoint (0-127). The resulting 55-dimensional float tensor is fed into the Sequential.

### Architecture Overview

The model is a plain `Sequential` of **5,442 modules**: 2,721 Linear layers alternating with 2,721 ReLU activations. No skip connections, no attention, no normalization -- just a massive stack of `Linear + ReLU` pairs.

| Section | Linear layers | Input dim | Output dim |
|---------|-------------|-----------|------------|
| Prefix  | 18          | 55        | 224        |
| Body    | 2,646 (63 blocks x 42) | 336 | 256 |
| Tail    | 57          | 256       | 1          |

---

## Phase 2: Discovering the Block Structure

### The 42-Layer Repeating Block

Plotting the width sequence (output dimension of each Linear layer) revealed a striking pattern: after an 18-layer prefix, a **42-layer block repeats 63 times**. Each block follows the same width trajectory:

```
336 -> 296 -> 340 -> 332 -> 375 -> 399 -> 410 -> 402 -> 412 -> 404 ->
412 -> 404 -> 408 -> 400 -> 352 -> 288 -> 288 -> 256 -> 319 -> 288 ->
318 -> 288 -> 316 -> 288 -> 312 -> 288 -> 304 -> 288 -> 256 -> 288 ->
256 -> 319 -> 288 -> 318 -> 288 -> 316 -> 288 -> 312 -> 288 -> 304 ->
288 -> 256 -> [next block input]
```

The first two widths of each block varied slightly between three "phases":
- Blocks 0-30: `336 -> 296` (state size 336)
- Blocks 31-46: `368 -> 328` (state size 368, +32 extra dims)
- Blocks 47-61: `336 -> 296` (back to 336)
- Block 62: special transition to 256-dim tail

### Weight Sharing: ROM vs RAM

Comparing weights across blocks revealed the key structural insight:

**40 of 42 layers per block have bitwise-identical weights across all 63 blocks.** These are the "ROM" -- a fixed computation shared by every iteration.

**Only 2 layers vary per block:**
- **Offset 28** (256 -> 288): a per-block "instruction selector" with 16 unique weight matrices cycling in a 4x4 pattern
- **Offset 41** (256 -> 336/368): a per-block "constant injector" with 63 unique weight matrices (one per block)

The offset-28 cluster sequence follows a perfect 2-level counter:
```
[0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3,   <- blocks 0-15
 4,5,6,7, 4,5,6,7, 4,5,6,7, 4,5,6,7,   <- blocks 16-31
 8,9,10,11, ...                          <- blocks 32-47
 12,13,14,15, ...]                       <- blocks 48-62
```

Where `cluster_id = (block // 16) * 4 + (block % 4)`.

---

## Phase 3: Decoding the Per-Block Instructions

### Offset 28: The Logic Gate Layer

Each offset-28 weight matrix (288 x 256) has extreme structure:
- **99.35% zeros**, only 480 nonzero entries
- Values restricted to **{-2, 0, +1}**
- Exactly **384 entries of +1** and **96 entries of -2** in every block
- Bias values of **-1** on specific rows

The 288 output rows decompose into:
- **192 "copy" rows** with a single +1 weight: a fixed permutation that routes 8 groups of 32 dimensions around
- **96 "logic" rows** (3 groups of 32) each computing `ReLU(a + b - 2c + bias)`:
  - With bias -1 and binary inputs: this implements **a AND b AND (NOT c)**
  - With bias 0: this implements a conditional copy gate

The per-cluster variation is purely a **rotation offset R** applied to the source column pointers within groups 3 and 4 (cols 96-159). The inner cluster (b mod 4) shifts R by -5 each step; the outer cluster (b // 16) selects a different base rotation.

This is a **parameterized bit-manipulation primitive**: each block rotates a slice of the register file and merges it under a control mask.

---

## Phase 4: The Power-of-2 Cascade

### ROM Layer Analysis

Surveying all 42 layers in a block revealed that every ROM layer is:
- **0.28-0.65% sparse** (extreme sparsity)
- **Integer-quantized** to small values
- Structured with a clear progression

The critical discovery: **layers 0, 2, 4, 6, 8, 10, 12, 14 contain power-of-2 weight values that halve each step:**

| Offset | Weight values | Operation |
|--------|--------------|-----------|
| 0      | {-128, -1, 0, 1, 128} | Extract/use bit 7 (128s place) |
| 2      | {-64, -2, -1, 0, 1, 64} | Extract/use bit 6 |
| 4      | {-32, -1, 0, 1, 32} | Extract/use bit 5 |
| 6      | {-16, -1, 0, 1, 16} | Extract/use bit 4 |
| 8      | {-8, -1, 0, 1, 8} | Extract/use bit 3 |
| 10     | {-4, -1, 0, 1, 4} | Extract/use bit 2 |
| 12     | {-2, -1, 0, 1, 2} | Extract/use bit 1 |
| 14     | {-2, -1, 0, 1} | Extract/use bit 0 |

This is a **binary decomposition cascade** -- each pair of layers peels off one bit from a byte-valued register, operating from the most significant bit (128) down to the least significant (1).

The block's algorithm:
1. **Bit-decompose** the register file (offsets 0-14)
2. **Boolean logic on extracted bits** (offsets 15-28, including the per-block instruction)
3. **Re-compose bits into integers** (offsets 29-41, with big +/-128 biases restoring byte values)

---

## Phase 5: Tracing the Register File

### State Structure at Block Boundaries

Running the model on `"vegetable dog"` and inspecting the 336-dim state at block 0's input revealed:

```
Dims   0-255:  binary {0,1} values  -> 256-BIT WORKING REGISTER
Dims 256-271:  small integers       -> metadata (lengths, indices)
Dims 272-284:  118 101 103 101 116 97 98 108 101 32 100 111 103
               = "vegetable dog" in ASCII  -> THE INPUT STRING
Dim  285:      128                  -> end-of-string sentinel
Dims 286-335:  mostly zero          -> working space
```

Of 336 dimensions, **247 are constant across all inputs** -- the state is highly structured. The input characters sit verbatim in dims 272-326, with a sentinel value of 128 marking the end.

### State Evolution

Tracking the state across all 63 block boundaries:
- **Blocks 0-2**: Large state changes (max|delta| = 128, byte-level setup)
- **Blocks 3-30**: Small changes (max|delta| = 1, bit-level operations)
- **Block 31**: Shape change 336 -> 368 (opens "middle phase" with 32 extra dimensions)
- **Blocks 32-46**: Middle phase processing
- **Block 47**: Shape change 368 -> 336 (closes middle phase)
- **Blocks 48-62**: Final processing and cleanup

This 4-phase structure (setup, bit grind, middle, cleanup) with 63 ~ 64 iterations strongly suggested **MD5**, which has exactly 64 rounds in 4 groups of 16.

---

## Phase 6: Cracking the Tail -- "Look at the Last Two Layers"

### The Indicator Gate Construction

The final two layers implement a brilliant construction:

**Linear(192, 48)**: 48 output rows = **16 groups of 3**, each reading 8 input bits as a binary-weighted byte (weights 1, 2, 4, 8, 16, 32, 64, 128). Within each group, the three rows have biases offset by exactly 1: `(-C, -(C-1), -(C-2))`.

**Linear(48, 1)**: weights `[+1 x 16, -2 x 16, +1 x 16]`, bias `-15`.

The combination `ReLU(x - C) - 2*ReLU(x - (C-1)) + ReLU(x - (C-2))` is an **exact-equality indicator**:

| x value | ReLU(x-C) | -2*ReLU(x-C+1) | ReLU(x-C+2) | Sum |
|---------|-----------|-----------------|-------------|-----|
| < C-1   | 0         | 0               | 0           | 0   |
| = C-1   | 0         | 0               | 1           | **1** |
| = C     | 0         | -2              | 2           | 0   |
| = C+1   | 1         | -4              | 3           | 0   |
| > C+1   | x-C       | -2(x-C+1)      | x-C+2       | 0   |

So each group outputs **1 if and only if the decoded byte equals the target value (C-1)**, and 0 otherwise.

The final layer sums all 16 indicators and subtracts 15. After the final ReLU:
- If all 16 match: output = 16 - 15 = **1**
- If 15 or fewer match: output <= 0, ReLU clips to **0**

### The 16 Target Conditions

8 of the 16 groups check a single byte against a literal target. The other 8 check a difference `Bi - 2*Bj` against a target (encoding the carry structure of MD5's modular addition).

Decoding all 16 conditions yielded the **128-bit target hash**:

```
c7 ef 65 23 3c 40 aa 32 c2 b9 ac e3 75 95 fa 7c
```

---

## Phase 7: Confirming the MD5 Hypothesis

### Verification

We decoded the model's 192-bit pre-tail activation into 16 MD5 bytes using the formula:
- Bytes 0-3: directly from positions B0-B3
- Bytes 4-7: from B4 - 2*B8, B5 - 2*B9, B6 - 2*B10, B7 - 2*B11
- Bytes 8-11: directly from B12-B15
- Bytes 12-15: from B16 - 2*B20, B17 - 2*B21, B18 - 2*B22, B19 - 2*B23

Testing against three inputs produced **exact byte-for-byte matches** with standard MD5:

```
input ''             : model = d41d8cd98f00b204e9800998ecf8427e  (MD5 of empty string)
input 'a'            : model = 0cc175b9c0f1b6a831c399e269772661
input 'vegetable dog': model = ab981aaa62cf6412f3aef1a11cd9b94b
```

All three match `hashlib.md5()` exactly.

**The model is a hand-built, 289-million-parameter implementation of MD5** using only Linear layers and ReLUs. Every architectural detail maps to an MD5 component:

| Architecture | MD5 Component |
|-------------|---------------|
| 63 body blocks | 64 rounds (block 0 = IV setup) |
| 4 outer cluster phases (b // 16) | Round functions F, G, H, I |
| 4 inner cluster phases (b % 4) | State word rotation (A, B, C, D) |
| 63 unique offset-41 weights | Per-round constants K[i] and shift amounts s[i] |
| Power-of-2 cascade | Bit decomposition for 32-bit word arithmetic |
| Logic rows ReLU(a+b-2c-1) | Boolean AND/OR gates for bitwise operations |
| 16-indicator tail | Exact comparison of hash output against target |
| Final ReLU killing gradients | Once any bit is wrong, gradient = 0 everywhere |

---

## Phase 8: Finding the Preimage

The puzzle reduced to: find a string `s` (up to 55 characters) such that `MD5(s) = c7ef65233c40aa32c2b9ace37595fa7c`.

### What We Tried

| Approach | Scope | Result |
|----------|-------|--------|
| Themed phrases | ~150k candidates (puzzle/JS-themed) | No match |
| System dictionary | 236k words, 4 case variants each | No match |
| Python brute force | 1-4 char printable ASCII | No match |
| hashcat brute force | 1-5 char printable ASCII (7.7B candidates) | Exhausted in 27s, no match |
| **hashcat combinator** | **197k words x 197k words with space separator** | **MATCH FOUND** |

### The Answer

hashcat's combinator attack (mode 1), which tries every `word1 + " " + word2` pair from the system dictionary, found:

```
c7ef65233c40aa32c2b9ace37595fa7c : bitter lesson
```

Verified through the model:

```python
>>> model("bitter lesson")
tensor([1.])
```

---

## The Meaning

"The Bitter Lesson" is Rich Sutton's influential 2019 essay arguing that in AI research, **general methods leveraging computation** (search and learning) **always eventually beat hand-crafted, human-engineered approaches**.

This is a deeply ironic and self-aware choice by Jane Street: the puzzle itself is a *hand-engineered* neural network -- 289 million parameters, painstakingly crafted to implement MD5 using nothing but matrix multiplications and ReLUs. The very existence of the puzzle embodies the opposite of the bitter lesson. And yet, the answer whispers back: *in the long run, the hand-crafting won't scale.*

A puzzle about mechanistic interpretability, whose answer is a meditation on whether mechanistic interpretability will ultimately matter.

---

## Summary of Techniques Used

1. **Bytecode disassembly** of cloudpickle lambda to find input encoding
2. **Width sequence analysis** to discover the 42-layer repeating block
3. **Weight hashing** to identify shared vs. per-block layers (40 shared, 2 unique)
4. **Cluster analysis** on per-block weights to find the 4x4 counter structure
5. **Sparsity visualization** of the logic gate layer (offset 28)
6. **Activation tracing** through all 2,721 layers for multiple inputs
7. **Register file mapping** at block boundaries (binary working register + ASCII input ROM)
8. **Power-of-2 weight analysis** revealing the bit-decomposition cascade
9. **ReLU indicator gate decoding** of the final two layers
10. **End-to-end MD5 verification** against `hashlib.md5`
11. **hashcat dictionary combinator attack** for the MD5 preimage
