---
name: gpu-review
description: Reviews deep learning code (.py files and Jupyter notebooks) to ensure GPU is used when available. Identifies anti-patterns like missing device detection, hardcoded CPU usage, tensors not moved to GPU, and performance issues. Supports PyTorch, TensorFlow/Keras, and JAX. Use when the user asks to "review code for GPU usage", "check GPU utilization", "audit device placement", or wants to ensure their DL training code runs on GPU.
---

# GPU Code Review

Review deep learning code for proper GPU utilization. Detect the framework(s) in use, then audit device handling against known anti-patterns.

## Workflow

1. **Detect framework** - Scan imports for `torch`, `tensorflow`/`keras`, or `jax`.
2. **Load reference** - Read `references/gpu-patterns.md` and focus on the detected framework's section.
3. **Audit the code** against the anti-patterns in the reference, checking:
   - Device detection exists and uses dynamic selection (not hardcoded CPU).
   - Model is moved to the detected device.
   - All tensors/data are on the same device as the model before forward pass.
   - DataLoader / data pipeline is GPU-optimized (pin_memory, prefetch, etc.).
   - Eval/inference paths disable gradients.
   - Mixed precision is considered for training code.
   - Multi-GPU is considered when relevant.
   - For notebooks: an early cell verifies GPU availability.
4. **Report findings** as a structured list with severity (Critical / Performance), the problematic code snippet, and a concrete fix.

## Output Format

```
## GPU Review: <filename>

**Framework:** PyTorch | TensorFlow | JAX
**Device setup found:** Yes / No
**Overall:** ✅ Good / ⚠️ Issues found

### Findings

#### [Critical] <title>
- **Line(s):** ...
- **Issue:** ...
- **Fix:** ...

#### [Performance] <title>
- **Line(s):** ...
- **Issue:** ...
- **Fix:** ...

### Summary
- N critical issues, M performance suggestions
- <one-line overall recommendation>
```

## Notebook-Specific Guidance

For `.ipynb` files, also check:
- An early cell runs `!nvidia-smi` or equivalent to confirm GPU runtime.
- Package installs use GPU variants (e.g., PyTorch with CUDA index URL).
- Device variable is printed/verified after setup, not just assigned silently.

## Edge Cases

- If code intentionally forces CPU (e.g., for debugging or small models), note it as informational rather than critical.
- If no DL framework is detected, report that no GPU-relevant code was found.
- For multi-framework files, audit each framework independently.
