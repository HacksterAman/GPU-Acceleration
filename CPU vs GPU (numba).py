from __future__ import print_function, absolute_import
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numba import jit, cuda, prange
import time

@cuda.jit
def compute_pi_gpu(rng_states, iterations, out):
    thread_id = cuda.grid(1)

    # Compute pi by drawing random (x, y) points and finding what fraction lie inside a unit circle
    inside = 0
    for i in range(iterations):
        x = xoroshiro128p_uniform_float32(rng_states, thread_id)
        y = xoroshiro128p_uniform_float32(rng_states, thread_id)
        if x**2 + y**2 <= 1.0:
            inside += 1

    out[thread_id] = 4.0 * inside / iterations

@jit(nopython=True, parallel=True)
def compute_pi_cpu(iterations, out):
    """Find the maximum value in values and store in result[0]"""
    for thread_id in prange(out.shape[0]):
        # Compute pi by drawing random (x, y) points and finding what fraction lie inside a unit circle
        inside = 0
        for i in prange(iterations):
            x = np.random.rand()
            y = np.random.rand()
            if x**2 + y**2 <= 1.0:
                inside += 1

        out[thread_id] = 4.0 * inside / iterations

threads_per_block = 1024
blocks = 1024

start_time = time.time()
rng_states_gpu = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
out_gpu = cuda.device_array(threads_per_block * blocks, dtype=np.float32)
compute_pi_gpu[blocks, threads_per_block](rng_states_gpu, 50000, out_gpu)
out_gpu_host = out_gpu.copy_to_host()
gpu_time = time.time() - start_time

print(f"GPU Time: {gpu_time}s")
print("GPU Result:", out_gpu_host.mean())

start_time = time.time()
out_cpu = np.zeros(threads_per_block * blocks, dtype=np.float32)
compute_pi_cpu(50000, out_cpu)
cpu_time = time.time() - start_time

print(f"CPU Time: {cpu_time}s")
print("CPU Result:", out_cpu.mean())

print(f"Boost: GPU is {round(cpu_time/gpu_time)}x better than CPU")
