import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
from functools import partial
# see https://github.com/google/jax/pull/6143
jax.config.update('jax_default_matmul_precision', 'float32')
from jax_smi import initialise_tracking
initialise_tracking(interval=0.1)

print('device_count = ',jax.device_count())
#> device_count =  8

x_np = np.zeros([48000,40000]).astype(np.complex64)
print(x_np.size / 1000**2, "M-elements")
print(f"{x_np.size * x_np.itemsize /1024**3:.2f} GB")
#> 1920.0 M-elements
#> 14.31 GB

def sharding_for_fft_axis(axis):
    assert axis<=1
    devices = mesh_utils.create_device_mesh((jax.device_count()))
    sharding_shape = [(1,len(devices)),(len(devices),1)][axis]
    sharding = PositionalSharding(devices).reshape(sharding_shape)
    return sharding

fft_axis = 0
x = jax.device_put(
    x_np,
    sharding_for_fft_axis(axis=fft_axis)
).block_until_ready()

@partial(jax.jit, static_argnames=['fft_axis'])
def fft(x,fft_axis):
    y = jax.lax.with_sharding_constraint(
        jnp.fft.fft(x, axis=fft_axis),
        sharding_for_fft_axis(axis=fft_axis)
    )
    return y

y = fft(x,fft_axis)
