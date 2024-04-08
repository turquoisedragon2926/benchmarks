import jax
import jax.numpy as jnp

from jax.sharding import Mesh, SingleDeviceSharding, PositionalSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

_DEVICE_MESH_FOR_FFT_AXIS_0 = mesh_utils.create_device_mesh( (1                 , jax.device_count()) )
_DEVICE_MESH_FOR_FFT_AXIS_1 = mesh_utils.create_device_mesh( (jax.device_count(), 1                 ) )

class MeshForFFT(Enum):
    FFT_AXIS_0 = Mesh(_DEVICE_MESH_FOR_FFT_AXIS_0, axis_names=('i', 'j'))
    FFT_AXIS_1 = Mesh(_DEVICE_MESH_FOR_FFT_AXIS_1, axis_names=('i', 'j'))

class ShardingForFFT(Enum):
    SINGLE_DEVICE0 = SingleDeviceSharding(jax.devices()[0])
    FFT_AXIS_0     = PositionalSharding(_DEVICE_MESH_FOR_FFT_AXIS_0)
    FFT_AXIS_1     = PositionalSharding(_DEVICE_MESH_FOR_FFT_AXIS_1)

def _fft_sharded(a, n=None, axis=None, forward_fft:bool=True):
    """FFT utilizing all available local GPUs.
    Using jax.experimental.shard_map (other solutions available, not implemented here)

    Args:
        a (_type_): Input array (see np.fft.fft)
        n (_type_): Length of output axis (see np.fft.fft)
        axis (_type_): Axis over which to FFT (see np.fft.fft)
        forward_fft (bool): True for fft (forward), False for ifft (inverse)

    Raises:
        ValueError: _description_

    Returns:
        _type_: The truncated or zero-padded input, transformed along the axis indicated by axis (see np.fft.fft)
    """

    # Replicate numpy.fft's original behavior: "If n is not given, ..."
    if n is None:
        # "... the length of the input along the axis specified by axis is used."
        n = a.shape[axis]

    @partial(shard_map, mesh=MeshForFFT.FFT_AXIS_0.value, in_specs=P(None, 'j'), out_specs=P(None, 'j'))
    def _shmap_fft_axis0(a):
        return jnp.fft.fft(a=a, n=n, axis=0)

    @partial(shard_map, mesh=MeshForFFT.FFT_AXIS_1.value, in_specs=P('i', None), out_specs=P('i', None))
    def _shmap_fft_axis1(a):
        return jnp.fft.fft(a=a, n=n, axis=1)

    @partial(shard_map, mesh=MeshForFFT.FFT_AXIS_0.value, in_specs=P(None, 'j'), out_specs=P(None, 'j'))
    def _shmap_ifft_axis0(a):
        return jnp.fft.ifft(a=a, n=n, axis=0)

    @partial(shard_map, mesh=MeshForFFT.FFT_AXIS_1.value, in_specs=P('i', None), out_specs=P('i', None))
    def _shmap_ifft_axis1(a):
        return jnp.fft.ifft(a=a, n=n, axis=1)

    fft_axis = axis
    sharding_axis = 1-fft_axis      # distinguish between the axis for FFT, and the one for sharding (which must be the OTHER axis; thus, for 2D = 1-fft_axis)

    if fft_axis==0:
        if forward_fft:
            _shmap_fft = _shmap_fft_axis0
        else:
            _shmap_fft = _shmap_ifft_axis0
        sharding_by_axis = ShardingForFFT.FFT_AXIS_0.value
    elif fft_axis==1:
        if forward_fft:
            _shmap_fft = _shmap_fft_axis1
        else:
            _shmap_fft = _shmap_ifft_axis1
        sharding_by_axis = ShardingForFFT.FFT_AXIS_1.value
    else: 
        raise ValueError(f"Axis out of range: {fft_axis}. May be only [0,1].")

    jit_fft_shmooped = jax.jit(
        lambda a: _shmap_fft(a),
        in_shardings = sharding_by_axis, out_shardings=sharding_by_axis )
    
    jit_fft_shmooped_lowered  = jit_fft_shmooped.lower(a)
    jit_fft_shmooped_compiled = jit_fft_shmooped_lowered.compile()

    fft_output = jit_fft_shmooped_compiled(a)

    return fft_output

# Example data array
data = jnp.array([[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0]])

# Perform FFT along axis 0 (rows)
fft_first_axis = _fft_sharded(data, axis=0, forward_fft=True)

# Then, perform FFT along axis 1 (columns) on the result
fft_both_axes = _fft_sharded(fft_first_axis, axis=1, forward_fft=True)

# Inverse FFT along axis 1
ifft_first_axis = _fft_sharded(fft_both_axes, axis=1, forward_fft=False)

# Inverse FFT along axis 0
ifft_both_axes = _fft_sharded(ifft_first_axis, axis=0, forward_fft=False)

print(ifft_both_axes)
