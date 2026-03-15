# Papermill Compatibility Checklist

When a notebook is run via `papermill` over SSH (instead of the Colab browser),
certain Colab-specific constructs break. Check for and fix these patterns.

## 1. Drive mount without guard

**Breaks because:** `drive.mount()` needs Colab-internal env vars that don't
exist in a papermill kernel, even though Drive is already mounted by the
bootstrap notebook.

**Bad:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Good:**
```python
import os

if os.path.ismount('/content/drive'):
    print('Drive already mounted.')
else:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception:
        print('Skipping drive mount (not in Colab UI or drive unavailable).')
```

## 2. IPython magic commands

**Breaks because:** `%cd`, `%mkdir`, `!command` are IPython magics that may
not work reliably in a papermill kernel.

**Bad:**
```python
%mkdir -p $WORK_DIR
%cd $WORK_DIR
!pip install -q torch pandas
!git clone https://github.com/org/repo.git
```

**Good:**
```python
import os, subprocess

os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)
subprocess.run(['pip', 'install', '-q', 'torch', 'pandas'])
subprocess.run(['git', 'clone', 'https://github.com/org/repo.git'])
```

## 3. Hard assert on Drive path

**Breaks because:** The assert fails if Drive mount was skipped.

**Bad:**
```python
assert os.path.exists('/content/drive')
```

**Good:**
```python
if not os.path.exists('/content/drive'):
    print('Warning: Drive not mounted, some paths may not work.')
```

## 4. IN_COLAB detection via import

This pattern is fine and doesn't need changing — it works in both modes:

```python
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
```

However, don't gate critical logic (like repo cloning) solely on `IN_COLAB`,
since papermill on Colab SSH will still report `IN_COLAB = True` but won't
have the interactive environment.

## Quick Scan

When reviewing a notebook, search for these patterns:
- `drive.mount` without `try/except` or `ismount` guard
- Lines starting with `%` (magic commands)
- Lines starting with `!` (shell commands)
- `assert os.path.exists('/content/drive')`
