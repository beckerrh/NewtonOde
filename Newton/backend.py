# backend.py
import os

# Default backend
_BACKEND = os.getenv("BACKEND", "numpy").lower()

if _BACKEND == "jax":
    import jax.numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    backend_name = "jax"
else:
    import numpy as np
    grad = lambda f: f
    jit = lambda f: f
    backend_name = "numpy"

print(f"[backend] Using {_BACKEND.upper()}")
