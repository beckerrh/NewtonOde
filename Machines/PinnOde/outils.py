import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax import nnx

#==================================================================
def plot_solutions(plot_dict):
    for i, (k,v) in enumerate(plot_dict.items()):
        ax = plt.subplot(len(plot_dict), 1, i+1)
        ax.set_title(k)
        if isinstance(v[1], dict):
            for k2, v2 in v[1].items():
                ax.plot(v[0], v2, label=k2)
            ax.legend()
        else:
            ax.plot(v[0], v[1])
        ax.grid()
    plt.show()

# -------------------- Triangular Dense module --------------------
class TriangularDense(nnx.Module):
    def __init__(self, n, rngs, lower=True, use_bias=True):
        self.n = n
        self.lower = lower
        self.use_bias = use_bias
        w_key = rngs()
        w_init = jax.random.normal(w_key, (n, n)) * jnp.sqrt(2.0 / max(1, n))
        self.weight = nnx.Param(w_init)

        if use_bias:
            self.bias = nnx.Param(jnp.zeros((n,)))
        else:
            self.bias = None
        # if lower:
        #     mask = jnp.tril(jnp.ones((out_dim, in_dim)))
        # else:
        #     mask = jnp.triu(jnp.ones((out_dim, in_dim)))
        # self.mask = nnx.Param(mask, collection='buffers')

    def __call__(self, x):
        if self.lower:
            mask = jnp.tril(jnp.ones((self.n, self.n)))
        else:
            mask = jnp.triu(jnp.ones((self.n, self.n)))
        W = self.weight * mask
        y = x @ W.T
        if self.use_bias:
            y = y + self.bias
        return y

# -------------------- Banded Dense module --------------------
class BandedDense(nnx.Module):
    """
    Square banded dense layer using O(n*k) storage and multiply.

    Args:
      n     : int, input/output dimension
      k     : int, half-bandwidth (1 = tridiagonal)
      rngs  : nnx.RngStream or jax PRNGKey
      use_bias : bool
    """
    def __init__(self, n, k, rngs, use_bias=True):
        if k < 0 or k >= n:
            raise ValueError("k must be between 0 and n-1")
        self.n = n
        self.k = k
        self.use_bias = use_bias

        # Support both RngStream and PRNGKey
        try:
            w_key = rngs()
        except TypeError:
            w_key = rngs

        scale = jnp.sqrt(2.0 / max(1, n))
        # Shape: (n, 2k+1)
        band_init = jax.random.normal(w_key, (n, 2*k + 1)) * scale
        self.band_weights = nnx.Param(band_init)

        if use_bias:
            self.bias = nnx.Param(jnp.zeros((n,)))
        else:
            self.bias = None

    def __call__(self, x):
        """
        x: shape (..., n)
        """
        k = self.k
        n = self.n
        bw = self.band_weights

        # Output y with same batch dims as x
        y = jnp.zeros_like(x)

        # Efficiently add each diagonal
        for offset in range(-k, k+1):
            diag_vals = bw[:, offset + k]  # shape (n,)
            if offset < 0:
                y = y.at[..., :offset].add(diag_vals[-offset:] * x[..., -offset:])
            elif offset > 0:
                y = y.at[..., offset:].add(diag_vals[:-offset] * x[..., :-offset])
            else:
                y = y + diag_vals * x

        if self.use_bias:
            y = y + self.bias

        return y