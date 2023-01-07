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
  widening_factor: int = 1
  name: Optional[str] = None

  def __call__(
      self,
      embeddings: jnp.ndarray,  # [T, D]
  ) -> jnp.ndarray:  # [T, D]
    """Transforms input embedding sequences to output embedding sequences."""
    
    seq_len, model_size = embeddings.shape
    initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
    
    h = embeddings
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
        
    return layer_norm(h)

def layer_norm(x: jnp.ndarray) -> jnp.ndarray:
  """Applies a unique LayerNorm to x with default settings."""
  return x


if __name__=='__main__':

    def forward_fn(x):
        model = Transformer(num_heads=8, num_layers=4, key_size=16)
        return model(x)

    from jax.config import config   
    config.update("jax_enable_x64", True)
    
    n = 6
    dim = 3
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (n, dim))    

    network = hk.transform(forward_fn)
    params = network.init(key, x)

    v = network.apply(params, None, x)
    P = np.random.permutation(n)
    Pv = network.apply(params, None, x[P, :])

    assert jnp.allclose(Pv, v[P, :])
