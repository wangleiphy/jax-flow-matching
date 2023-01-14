import dataclasses
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

@dataclasses.dataclass
class Transformer(hk.Module):
  """A transformer stack."""

  num_heads: int
  num_layers: int
  key_size: int
  L : float
  widening_factor: int = 1
  name: Optional[str] = None

  def __call__(
      self,
      x: jnp.ndarray,  # [T, D]
      t: float
  ) -> jnp.ndarray:  # [T, D]
    """Transforms input embedding sequences to output embedding sequences."""
    
    seq_len, dim = x.shape
    initializer = hk.initializers.VarianceScaling(2 / self.num_layers)

    h = jnp.concatenate([jnp.cos(2*np.pi*x/self.L), 
                         jnp.sin(2*np.pi*x/self.L),
                         jnp.repeat(jnp.array(t).reshape(1, 1), seq_len, axis=0)
                         ], axis=-1)

    model_size = h.shape[-1]
    for _ in range(self.num_layers):
      # First the attention block.
      attn_block = hk.MultiHeadAttention(
          num_heads=self.num_heads,
          key_size=self.key_size,
          model_size=model_size,
          w_init=initializer,
      )
      h_norm = layer_norm(h)
      h_attn = attn_block(h_norm, h_norm, h_norm)
      h = h + h_attn

      # Then the dense block.
      dense_block = hk.Sequential([
          hk.Linear(self.widening_factor * model_size, w_init=initializer),
          jax.nn.gelu,
          hk.Linear(model_size, w_init=initializer),
      ])
      h_norm = layer_norm(h)
      h_dense = dense_block(h_norm)
      h = h + h_dense
        
    return hk.Linear(dim, w_init=initializer)(h) 

def layer_norm(x: jnp.ndarray) -> jnp.ndarray:
  """Applies a unique LayerNorm to x with default settings."""
  return x

def make_transformer(key, n, dim, num_heads, num_layers, key_sizes, L):
    x = jax.random.normal(key, (n, dim))
    t = jax.random.uniform(key)

    def forward_fn(x, t):
        net = Transformer(num_heads, num_layers, key_sizes, L)
        return net(x.reshape(n, dim), t).reshape(n*dim)
    network = hk.without_apply_rng(hk.transform(forward_fn))
    params = network.init(key, x, t)
    return params, network.apply

