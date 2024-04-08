import os
from pathlib import Path
from typing import Union

import jax.numpy as jnp
import jax
import numpy as np
import scipy
from jax import jit
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

import sharded_rfft_general
from utils import Timing, plot_graph

print("jax version", jax.__version__)

num_gpus = int(os.environ.get("SLURM_GPUS"))


# jax.config.update("jax_enable_x64", True)

def host_subset(array: Union[jnp.ndarray, np.ndarray], size: int):
    host_id = jax.process_index()
    start = host_id * size // num_gpus
    end = (host_id + 1) * size // num_gpus
    return array[:, start:end]


def print_subset(x):
    print(x[0, :4, :4])


def compare(a, b):
    is_equal = np.allclose(a, b, rtol=1.e-2, atol=1.e-4)
    print(is_equal)
    diff = a - np.asarray(b)
    max_value = np.max(np.real(out_ref_subs))
    max_diff = np.max(np.abs(diff))
    print("max_value", max_value)
    print("max_diff", max_diff)
    print("max_diff / max_value", max_diff / max_value)


print("distributed initialize")
jax.distributed.initialize()

timing = Timing(print)

print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("devices:", jax.device_count(), jax.devices())
print("local_devices:", jax.local_device_count(), jax.local_devices())
print("process_index", jax.process_index())
print("total number of GPUs:", num_gpus)

timing.log("random ICs")

size = 1024

rng = np.random.default_rng(12345)
x_np_full = rng.random((size, size, size), dtype=np.float32)

x_np = host_subset(x_np_full, size)
print("x_np shape", x_np.shape)
global_shape = (size, size, size)

timing.log("generated")

print(x_np.nbytes / 1024 / 1024 / 1024, "GB")
print(x_np.shape, x_np.dtype)

devices = mesh_utils.create_device_mesh((num_gpus,))
mesh = Mesh(devices, axis_names=('gpus',))
timing.log("start")
with mesh:
    x_single = jax.device_put(x_np)
    xshard = jax.make_array_from_single_device_arrays(
        global_shape,
        NamedSharding(mesh, P(None, "gpus")),
        [x_single])

    rfftn_jit = jit(
        sharded_rfft_general.rfftn,
        donate_argnums=0,  # doesn't help
        in_shardings=(NamedSharding(mesh, P(None, "gpus"))),
        out_shardings=(NamedSharding(mesh, P(None, "gpus")))
    )
    irfftn_jit = jit(
        sharded_rfft_general.irfftn,
        donate_argnums=0,
        in_shardings=(NamedSharding(mesh, P(None, "gpus"))),
        out_shardings=(NamedSharding(mesh, P(None, "gpus")))
    )
    if jax.process_index() == 0:
        with jax.spmd_mode('allow_all'):
            a = Path("compiled.txt")
            a.write_text(rfftn_jit.lower(xshard).compile().as_text())
            z = jax.xla_computation(rfftn_jit)(xshard)
            plot_graph(z)
    sync_global_devices("wait for compiler output")

    with jax.spmd_mode('allow_all'):
        timing.log("warmup")
        rfftn_jit(xshard).block_until_ready()

        timing.log("calculating")
        out_jit: jax.Array = rfftn_jit(xshard).block_until_ready()
        print(out_jit.nbytes / 1024 / 1024 / 1024, "GB")
        print(out_jit.shape, out_jit.dtype)

        timing.log("inverse calculating")
        out_inverse: jax.Array = irfftn_jit(out_jit).block_until_ready()

        timing.log("collecting")
        sync_global_devices("loop")
        local_out_subset = out_jit.addressable_data(0)
        local_inverse_subset = out_inverse.addressable_data(0)

    print(local_out_subset.shape)
    print_subset(local_out_subset)
    # print("JAX output without JIT:")
    # print_subset(out)
    # print("JAX output with JIT:")
    # # print_subset(out_jit)
    # print("out_jit.shape1", out_jit.shape)
    # print(out_jit.dtype)
    timing.log("done")

    out_ref = scipy.fft.rfftn(x_np_full, workers=128)
    timing.log("ref done")

    print("out_ref", out_ref.shape)
    out_ref_subs = host_subset(out_ref, size)
    print("out_ref_subs", out_ref_subs.shape)

    print("JAX output with JIT:")
    print_subset(local_out_subset)
    print("Reference output:")
    print_subset(out_ref_subs)

    print("ref")
    compare(out_ref_subs, local_out_subset)

    print("inverse")

    compare(x_np, local_inverse_subset)

    print_subset(x_np)
    print_subset(local_inverse_subset)
