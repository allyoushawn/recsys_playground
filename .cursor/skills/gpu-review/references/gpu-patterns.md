# GPU Anti-Patterns by Framework

## PyTorch

### Critical

| Anti-pattern | Fix |
|---|---|
| No device detection | Add `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` |
| Hardcoded `torch.device("cpu")` | Replace with dynamic device selection |
| Model never moved to device | Add `model.to(device)` after construction |
| Tensors created on CPU, never moved | Use `.to(device)` or create directly: `torch.tensor(..., device=device)` |
| `model.cuda()` without availability check | Guard with `if torch.cuda.is_available()` or use `.to(device)` pattern |
| Mixed devices: model on GPU, input on CPU (or vice versa) | Ensure all tensors and model on same device before forward pass |

### Performance

| Anti-pattern | Fix |
|---|---|
| DataLoader missing `pin_memory=True` when GPU available | `DataLoader(..., pin_memory=True)` when using CUDA |
| Not using `torch.cuda.amp` / `torch.amp` for mixed precision | Add `torch.amp.autocast('cuda')` + `GradScaler` for faster training |
| Calling `.item()`, `.cpu()`, or `.numpy()` in hot loop | Batch these calls; keep tensors on GPU during computation |
| Missing `torch.no_grad()` / `torch.inference_mode()` during eval | Wrap eval/inference in context manager |
| Not calling `torch.cuda.empty_cache()` when switching large models | Call between model loads to free VRAM |
| Missing `non_blocking=True` on `.to(device)` transfers | Use `.to(device, non_blocking=True)` with pinned memory |

### Multi-GPU

| Anti-pattern | Fix |
|---|---|
| Not using `DataParallel` / `DistributedDataParallel` when multiple GPUs available | Wrap model in `nn.DataParallel(model)` or use DDP |
| Hardcoded `cuda:0` | Use `torch.cuda.current_device()` or parameterize |

## TensorFlow / Keras

### Critical

| Anti-pattern | Fix |
|---|---|
| No GPU visibility check | Add `tf.config.list_physical_devices('GPU')` check |
| Forcing CPU via `tf.device('/cpu:0')` without fallback | Use dynamic placement or remove forced CPU |
| Not enabling memory growth | `tf.config.experimental.set_memory_growth(gpu, True)` |
| `with tf.device('/GPU:0')` without checking GPU exists | Guard with device availability check |

### Performance

| Anti-pattern | Fix |
|---|---|
| Not using `tf.data` pipeline with `prefetch` | Add `.prefetch(tf.data.AUTOTUNE)` to dataset pipeline |
| Missing mixed precision | `tf.keras.mixed_precision.set_global_policy('mixed_float16')` |
| Eager-only execution for large-scale training | Use `@tf.function` for graph compilation |
| Not using XLA compilation | Add `jit_compile=True` to `model.compile()` |

### Multi-GPU

| Anti-pattern | Fix |
|---|---|
| Not using `tf.distribute.MirroredStrategy` when multiple GPUs | Wrap training in strategy scope |
| Manual device placement on single GPU | Let TF auto-place; only manual-place when needed |

## JAX

### Critical

| Anti-pattern | Fix |
|---|---|
| Not checking backend | `jax.devices()` or `jax.default_backend()` to verify GPU |
| Forcing CPU backend via `JAX_PLATFORM_NAME=cpu` in code | Remove or make configurable |
| Arrays created with `np.array` instead of `jnp.array` | Use `jnp.array` for GPU-resident arrays |
| Not using `jax.device_put` for explicit placement | Place arrays on GPU with `jax.device_put(x, jax.devices('gpu')[0])` |

### Performance

| Anti-pattern | Fix |
|---|---|
| Not using `jax.jit` for compiled execution | Wrap hot functions in `@jax.jit` |
| Transferring between host/device in loop | Batch transfers; keep data on device |
| Not using `jax.pmap` for multi-GPU | Parallelize with `jax.pmap` across devices |

## Colab / Notebook-Specific

| Anti-pattern | Fix |
|---|---|
| No runtime GPU check cell | Add early cell: `!nvidia-smi` and device detection |
| Installing CPU-only package variants | Use GPU variants: `pip install torch --index-url ...cu1xx` |
| Not verifying GPU is actually used after setup | Print `device` variable and confirm with small tensor test |
| Missing `!nvidia-smi` or equivalent diagnostic | Add as first cell for quick verification |
