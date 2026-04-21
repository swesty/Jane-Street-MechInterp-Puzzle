# A Walkthrough of How We Cracked the Puzzle

This is the story of *why* we did each thing — the questions we were asking, the hypotheses we were testing, and how each finding redirected the next move. Every script in this repo corresponds to one inflection point in that chain of reasoning.

---

## Mindset Going In

Three things we knew before writing a single line of code:

1. The model is fully white-box. Weights are right there.
2. The puzzle authors said *they* couldn't be sure it was solvable, and that **gradient descent on the input wouldn't work** — implying the network is engineered so that backprop dies somewhere. That's a structural hint: there are dead-ReLU traps.
3. The hint *"look at the last two layers"* was almost surely literal. The final couple of layers encode the success condition.

So our prior was: this model is not a learned model. It's a hand-crafted structure that **implements some specific function**, and the puzzle is to figure out what.

---

## Step 1 — Get the model loaded (`look.py`)

**What we wanted to observe:** the basic architecture — how many parameters, what kind of layers, what the input/output shapes are.

**Why this first:** before any clever analysis, we need to know what we're staring at. Is this a transformer? A CNN? Something custom? `print(model)` answers that in seconds.

**What got in the way:** the serialized model was made with Python 3.10. We were on 3.12. The fix was straightforward — there's a `model_3_11.pt` variant in the same HuggingFace repo, and Python 3.11 reads it fine. We spun up a `.venv311` for it.

**What we learned:** the model is a `torch.nn.Sequential` with **5,442 modules** (2,721 Linear layers alternating with 2,721 ReLU activations), 289 million parameters, input dim 55, output dim 1. No attention, no normalization, no skip connections. Just a flat stack of `Linear → ReLU` pairs going **2,721 layers deep**.

That depth is absurd for any normal neural network. Real models don't go past ~100 layers without residual connections or they die from vanishing signals. The fact that this network was hand-built to be 2,721 layers deep was the first signal that the structure itself was the puzzle.

---

## Step 2 — Figure out how a string becomes a tensor (`look2.py`, `look3.py`, `look4.py`)

**What we wanted to observe:** the model's `__call__` accepts a string. But a `Sequential` only knows how to take tensors. So *something* must be converting the string to a 55-dim tensor before the first Linear layer. Where is that something?

**Why this matters:** until we understood the input encoding, we couldn't reason about anything the network was computing. The encoding is the contract between "what you type" and "what the math sees."

**The hunt:** we inspected `model.__dict__`, looking for anything that wasn't standard `nn.Module` machinery. We found a stray `_call_impl` attribute — a Python function that had been monkey-patched onto the model object. Standard `nn.Module` doesn't store `_call_impl` as an instance attribute, so this had to be the wrapper.

**The reveal:** disassembling the function's bytecode (`dis.dis(ci)`) showed:

```python
lambda x: model.forward(torch.Tensor(list(map(ord, str(x)[:55].ljust(55, '\x00')))))
```

So the input is: take the string, truncate to 55 chars, right-pad with nulls, take `ord()` of each char. The model sees a 55-dim float tensor of ASCII codepoints.

**Why this was important:** it told us the network operates on **integer-valued inputs in the range [0, 127]**, padded with zeros. That's a strong constraint — most neural networks see normalized floats, not raw bytes. The fact that this one sees raw bytes hinted that the network might be doing **byte-level computation** — not pattern recognition.

---

## Step 3 — Look at the shape of the depth (`shapes.py`)

**What we wanted to observe:** with 2,721 Linear layers, there's no way the network is *uniformly* designed. We expected to find structure: blocks that repeat, sections that grow or shrink, signs of phases.

**Why:** uniform networks don't get hand-built. If a human designed this, there will be patterns visible in the layer widths.

**The plot:** dump every layer's `(in_features, out_features)` and look at the sequence of dimensions.

**The pattern that jumped out:**

```
[0..17]    55, 224, 232, 64, ..., 224       <- 18 prefix layers
[18..59]   336, 296, 340, 332, ...           <- 42 layers
[60..101]  336, 296, 340, 332, ...           <- 42 layers (same!)
[102..143] 336, 296, 340, 332, ...           <- 42 layers (same!)
...
```

A **42-layer block repeating 63 times.** The first two widths of each block varied between three "phases" (`336→296` early, `368→328` middle, `336→296` late again), but the middle 40 were locked.

**Why this was a turning point:** repetition means iteration. The model isn't a single deep computation — it's a 63-step loop, each step running the same 42-layer "function" with slight per-step tweaks. Now we had a mental model: this is a **VM with 63 iterations**, and we needed to understand what one iteration does.

---

## Step 4 — Are the per-block weights actually different? (`diff_blocks.py`)

**What we wanted to observe:** the *widths* of the 40 inner layers were identical across blocks, but were the *weights* identical? If yes, those layers form a shared "ROM" — fixed code reused every iteration. If no, each block has its own unique program.

**Why this matters:** this is the difference between a CPU running a loop (shared instructions, varying data) and 63 separate copies of the same circuit. The two interpretations have wildly different implications.

**The test:** for each of the 42 positions within a block, gather all the corresponding Linear layers across the 63 blocks. Group by shape, then check if the weights are bitwise identical across same-shape pairs.

**The result was almost too clean:** **40 of the 42 positions had bitwise-identical weights across all 63 blocks.** Only positions **28** and **41** varied per block.

**Why this was a huge unlock:** the model is structurally a CPU. There are 40 "fixed instruction" layers and 2 "per-step instruction" layers. The per-step layers must encode *which* operation each block does — the program, in some sense. If we could decode the per-block layers, we'd know the program.

---

## Step 5 — What pattern do the per-block weights follow? (`per_block.py`, `cluster.py`)

**What we wanted to observe:** are the per-block weights random, or do they follow a pattern? Specifically:
- For position 28: all 63 unique? Or a small number of repeating values?
- For position 41: same question.

**Why:** if there's a pattern, it tells us the structure of the "instruction set."

**The test:** hash each per-block weight matrix and group by hash. Count distinct values.

**The result:**
- **Position 28: only 16 distinct weight matrices** across the 63 blocks, in a perfect interleaved pattern: `[0,1,2,3, 0,1,2,3, ..., 4,5,6,7, ..., 8,9,10,11, ..., 12,13,14,15, ...]`. The cluster ID = `(block // 16) * 4 + (block % 4)` — a **2-level counter**.
- **Position 41: 63 distinct values** (one per block). No reuse.

**The interpretation:** position 28 is a **"micro-op" selector** with 16 micro-ops grouped as a 4×4 grid (inner cycle of 4 nested in an outer cycle of 4). Position 41 is a **"per-round constant"** — every block needs its own.

**Why this was a clue toward MD5:** any cryptographer reading "16 distinct ops in 4 groups of 4" plus "64 unique per-round constants needed" would think **MD5**. MD5 has 64 rounds split into 4 groups of 16, with a different round constant `K[i]` each round. The structure was screaming the answer, but we didn't see it yet.

---

## Step 6 — What does one per-block layer actually compute? (`sparsity.py`, `logic_rows.py`, `verify_logic.py`, `verify2.py`)

**What we wanted to observe:** the position-28 layer is `Linear(256, 288)`. With 288 × 256 = 73,728 parameters, what do they look like? Are they sparse? Quantized? Random?

**Why:** if they're hand-designed, they'll be sparse and integer-valued. Random Gaussian weights would mean a learned model.

**What we found** — and it was beautiful:

- **99.35% of weights are zero**
- Only **three distinct nonzero values**: `{-2, 0, +1}`
- Exactly **480 nonzero entries** in every block
- Two row types:
  - **192 "copy rows"**: a single `+1` weight, doing pure data routing
  - **96 "logic rows"**: three nonzero entries — two `+1`s and one `-2`

**The "logic row" insight:** with two `+1` inputs (call them `a` and `b`), one `-2` input (`c`), and a bias term, each row computes:
```
ReLU(a + b - 2c + bias)
```

For binary inputs in `{0, 1}`, this is a **boolean gate**:
- `bias = -1` → `a AND b AND (NOT c)` (the AND-NOT primitive)
- `bias = 0` → conditional pass-through

**The per-cluster variation:** when we compared the 16 cluster variants of position 28, the *only* difference was a **rotation offset R** applied to the source column pointers in groups 3 and 4 of the input. Inner cluster (`block % 4`) shifted R by -5 each step; outer cluster shifted R by a small amount per outer phase.

**Why this was decisive:** we now had concrete evidence that **the model does bit-level boolean logic on a register file**. The per-block "instruction" was a small rotation parameter. Combined with 63 iterations, this is a **boolean circuit being applied iteratively** — exactly what a hash function looks like.

We then **verified** our manual model: built a Python function that mimicked one block of position 28 by hand using the decoded permutation + logic gate formulas, and checked it matched the actual layer's output to within `2.4e-7` (floating-point noise). That confirmed our reverse-engineering of the layer was *exactly* right — not just "looks like."

---

## Step 7 — What's actually flowing through the network? (`trace.py`)

**What we wanted to observe:** are the activations binary? Integer-valued? Floats? This determines how to interpret the boolean gates.

**Why:** if activations are continuous floats, the "boolean gate" interpretation is wrong. If they're 0/1 binary, the network is doing pure bitwise computation.

**The test:** trace activation values at every layer for several inputs. Record per-layer L2 norm, max value, and unique values seen.

**The result:**
- For all inputs, body activations had **max value = 440** (which is `55 × 8` — suggestive of "55 bytes worth of bit information")
- The body's activations stayed **stable across all 63 iterations** (not blowing up or dying)
- Unique values at intermediate points: small integers like `{0, 1, 2, 32, 97, 98, 100, 101, 103, 104, ..., 128}` — a mix of bit indicators (0/1/2), the saturation value 128, and recognizable **ASCII codes from the input**
- The signal **stayed alive all the way to the second-to-last ReLU**, then died only at the very final ReLU

**The takeaway:** the network operates on a register file containing both **binary bits** (the "computed state") and **raw ASCII bytes from the input**. The "max = 440" came from there being many saturated values during processing.

**Why this redirected us:** the puzzle's "you can't backprop" trap isn't deep in the body — it's at the very end. The body always stays alive; the final ReLU is the gate. So the puzzle isn't "find an input that doesn't die in the body" — it's "find an input whose body output triggers the final readout."

---

## Step 8 — What does the register file look like? (`regfile.py`)

**What we wanted to observe:** if the body's state is a register file, we should be able to *see* the registers — distinct slots holding distinct values. Specifically: where in the 336-dim state does the input live?

**Why:** before we can understand what the body *does*, we need to know where the data sits. Is the input copied verbatim somewhere? Is it pre-processed? Are there registers for working storage?

**The test:** capture the 336-dim state right before block 0, for several inputs. Look for:
- Dimensions that are **constant across all inputs** (working registers initialized to constants)
- Dimensions that **match input characters** (the input ROM)

**The reveal for `"vegetable dog"`:**

```
Dims   0-255: binary {0,1}      <- 256-bit "scratchpad" register
Dims 256-271: small ints         <- length/index metadata
Dims 272-284: 118 101 103 101 116 97 98 108 101 32 100 111 103
                                 <- ASCII for "vegetable dog" (dim 272 = 'v', etc.)
Dim  285:     128                <- end-of-string sentinel
Dims 286-335: mostly zero        <- working space
```

**Why this was huge:** we could now read the state. The 256-bit scratchpad is exactly 32 bytes — same as MD5's working state expansion. The input string sits verbatim in dims 272+, with a sentinel byte separating it from padding. **247 of 336 dimensions were constant across inputs** — the state is highly structured, with most dimensions holding fixed initialization values.

This is the layout of a **deterministic byte-processor**. At this point we were almost certain it was MD5, but we wanted independent confirmation from the tail.

---

## Step 9 — Decode the tail, the part the hint pointed at (`tail.py`)

**What we wanted to observe:** the puzzle hint said *"look at the last two layers."* We deferred this until we had context to interpret what we'd see. Now, with the body decoded, we wanted to know: what condition does the network check at the end?

**Why this is the pivotal observation:** the answer to "what input produces output 1" is determined by what the tail looks for. If the tail is asking "is this 128-bit vector equal to X?", then the puzzle is solved by finding an input that makes the body produce X.

**The test:** dump every nonzero weight and bias of the last few layers, then specifically examine `Linear(192, 48)` and `Linear(48, 1)`.

**What we found in `Linear(192, 48)`:**
- 48 output rows = **16 groups of 3 rows each**
- Each row reads 8 input bits weighted by `1, 2, 4, 8, 16, 32, 64, 128` — i.e., **decoding a byte from binary**
- Within each group of 3, the biases were `(-C, -(C-1), -(C-2))` — three biases offset by 1

**What we found in `Linear(48, 1)`:**
- Weights `[+1 × 16, -2 × 16, +1 × 16]`, bias `-15`

**The trick:** the construction `ReLU(x - C) - 2·ReLU(x - (C-1)) + ReLU(x - (C-2))` is a **single-value indicator function**. Working through the algebra:

| `x` value | Output |
|-----------|--------|
| `x ≤ C-2` | 0 |
| `x = C-1` | **1** |
| `x ≥ C`   | 0 |

So each group of 3 outputs `1` iff a specific byte equals a specific target value. The final layer sums all 16 indicators and subtracts 15. Result: **output = 1 iff all 16 groups match their targets, else 0**.

This is a wickedly elegant ReLU-only construction. It's also why backprop fails: when fewer than 16 indicators fire, the final ReLU output is 0 *and* its derivative is 0, killing the gradient before it can reach the input.

### Decoding the targets

Reading the biases gave us the 16 target conditions:
- 8 groups checked single bytes: `B0=199, B1=239, B2=101, B3=35, B12=194, B13=185, B14=172, B15=227`
- 8 groups checked relationships: `B4 - 2·B8 = 60`, etc.

Concatenating those gives the 16-byte target: **`c7 ef 65 23 3c 40 aa 32 c2 b9 ac e3 75 95 fa 7c`**.

---

## Step 10 — Confirm it's actually MD5 (`decode_target.py`, `find_preimage.py`)

**What we wanted to observe:** we strongly suspected MD5, but suspicion isn't proof. We needed a smoking gun.

**The test:** write a function that runs the model on an input, extracts the 192-bit pre-tail activation, decodes it into 16 bytes using the formula we derived (bytes 0-3 from `B0-B3`, bytes 4-7 from `B4-2·B8` etc.), and compares to `hashlib.md5()` on the same input.

**The result:**
```
input ''             : model = d41d8cd98f00b204e9800998ecf8427e ✓ (MD5 of empty string)
input 'a'            : model = 0cc175b9c0f1b6a831c399e269772661 ✓
input 'vegetable dog': model = ab981aaa62cf6412f3aef1a11cd9b94b ✓
```

**Three exact matches against `hashlib.md5`.** The model is MD5. Every architectural quirk we found suddenly had meaning: the 64 rounds (with one folded into the IV setup), the 4×4 micro-op grid for the F/G/H/I round functions, the per-round constants, the bit-decomposition cascade — all of it is a faithful transcription of the MD5 spec into Linear-and-ReLU operations.

**The puzzle is now an MD5 preimage problem:** find a string `s` (≤ 55 chars) such that `MD5(s) = c7ef65233c40aa32c2b9ace37595fa7c`.

---

## Step 11 — Crack the hash (`themed.py`, `wordlist.py`, `twowords.py`, `brute5.c`, hashcat)

**What we wanted to observe:** preimage of an arbitrary MD5 is generically `2^128` work and infeasible. The puzzle is solvable by humans, so the answer must be a **short, low-entropy string** — a real word, phrase, or short brute-forceable input.

**The escalating attempts:**

1. **Themed candidates first** (`themed.py`, ~150k phrases): puzzle vocabulary, AI/ML terms, common puzzle answers, food-animal combinations following the "vegetable dog" pattern. Quick to try, no luck.
2. **System dictionary** (`wordlist.py`, 236k words × 4 case variants): single English words. No luck.
3. **Python brute force, 1-4 char ASCII**: ~80M candidates. No luck.
4. **C brute force, 5-6 char ASCII** (`brute5.c`): tried, but slow (~7.7B candidates would take hours in Python).
5. **hashcat** (installed via Homebrew): GPU-accelerated cracker. Did all 1-5 char printable ASCII (7.7B candidates) in **27 seconds**. No match.

**The decisive move:** the Gradio app source had a comment we'd noticed early but underused: `# two words?` next to the input box. That hinted the answer was a two-word phrase.

So we used **hashcat's combinator attack** (`-a 1`), which generates `word1 + word2` from two wordlists. We:
1. Cleaned `/usr/share/dict/words` to lowercase 2-12 char alphabetic words (~197k words)
2. Made a copy with a trailing space appended (`/tmp/dict_space.txt`) to insert the separator
3. Ran `hashcat -m 0 -a 1 /tmp/hash.txt /tmp/dict_space.txt /tmp/dict_clean.txt`

That's `197k × 197k ≈ 39 billion` two-word candidates. hashcat ran through them and produced:

```
c7ef65233c40aa32c2b9ace37595fa7c : bitter lesson
```

**Verification through the actual model:**
```python
>>> model("bitter lesson")
tensor([1.])
```

Done.

---

## Why each step mattered (the meta-lesson)

| Step | What we learned | Why it was the right next move |
|------|-----------------|-------------------------------|
| Loading the model | Sequential, 2,721 Linears | Tells us "no gimmicks, the depth itself is the structure" |
| Input encoding | 55 ASCII bytes | Tells us the model is a byte-processor, not a feature-learner |
| Width sequence | 42-layer block × 63 | Tells us it's iterative; we have to understand 1 block, not 2,721 |
| Weight equality | 40 shared, 2 per-block | Tells us this is a CPU: shared instructions + per-step opcodes |
| Cluster pattern | 16 micro-ops in 4×4 grid | Tells us there's a small instruction set; matches MD5's F/G/H/I |
| Layer sparsity | 99% sparse, integer weights | Tells us this is hand-built and decodable, not random |
| Activation tracing | Mixed binary + ASCII state | Tells us where the input lives and that the body never dies |
| Register file map | Bits + input ROM + sentinels | Tells us the data layout |
| Tail decoding | 16 byte-equality checks | Tells us the success condition is "match this 128-bit value" |
| MD5 verification | Exact bytewise match | Confirms we understand the function, not just "approximately" |
| Hashcat combinator | "bitter lesson" | The Gradio hint `# two words?` told us where to look |

Each step **removed one degree of freedom** from the problem. By the time we ran hashcat, we knew exactly what we were searching for, exactly what the input space was (≤55-char strings), and exactly which subspace was likely (two English words). That's why the search took seconds instead of millennia.

---

## The takeaway

Mechanistic interpretability done well is **archaeology, not divination**. We didn't guess at meaning. We measured: shapes, sparsity, equality, value distributions, activation traces. Each measurement either confirmed a hypothesis or killed it. The hypotheses we kept were the ones the data actually supported. The model was screaming *"I am MD5"* from the moment we saw the 4×4 cluster grid — but we didn't accept that conclusion until we had byte-for-byte agreement with `hashlib.md5`.

The puzzle's hint *"look at the last two layers"* turned out to be the most important sentence in the problem statement. The body of the network is impressive engineering, but the tail tells you **what question the network is asking**. Once you know the question, you know what answer to search for.
