import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from numba import jit, prange
import time

# CUDA kernel for GPU computation
cuda_code = """
extern "C" {
__device__ float xoroshiro128p_uniform_float32(unsigned int *s);

__global__ void compute_pi_gpu(int iterations, float *out, unsigned int seed) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize random number generator
    unsigned int state[2];
    state[0] = seed;
    state[1] = thread_id;

    // Compute pi by drawing random (x, y) points and finding what fraction lie inside a unit circle
    int inside = 0;
    for (int i = 0; i < iterations; ++i) {
        float x = xoroshiro128p_uniform_float32(state);
        float y = xoroshiro128p_uniform_float32(state);
        if (x * x + y * y <= 1.0) {
            inside += 1;
        }
    }

    out[thread_id] = 4.0 * inside / iterations;
}

__device__ float xoroshiro128p_uniform_float32(unsigned int *s) {
    const float UINT_TO_FLOAT = 2.3283064e-10f; // 1.0 / 2^32
    unsigned int a = s[0];
    unsigned int b = s[1];
    unsigned int result = a + b;
    b ^= a;
    s[0] = ((a << 24) | (a >> 8)) ^ b ^ (b << 16);
    s[1] = (b << 24) | (b >> 8);
    return result * UINT_TO_FLOAT;
}
}
"""

# CPU function using NumPy
@jit(nopython=True, parallel=True)
def compute_pi_cpu(iterations, out):
    """Find the maximum value in values and store in result[0]"""
    for thread_id in prange(out.shape[0]):
        # Compute pi by drawing random (x, y) points and finding what fraction lie inside a unit circle
        inside = 0
        for i in prange(iterations):
            x, y = np.random.rand(), np.random.rand()
            if x**2 + y**2 <= 1.0:
                inside += 1

        out[thread_id] = 4.0 * inside / iterations

start_time = time.time()

# Compile CUDA code
mod = SourceModule(cuda_code)

# Get GPU function
compute_pi_gpu = mod.get_function("compute_pi_gpu")

threads_per_block = 256  # Adjusted for better performance
blocks = 256  # Adjusted for better performance

# Allocate GPU memory only once
out_gpu_host = np.zeros(threads_per_block * blocks, dtype=np.float32)
out_gpu = cuda.mem_alloc(out_gpu_host.nbytes)

# Set random seed for GPU
seed = 1

# GPU execution
compute_pi_gpu(np.int32(50000), out_gpu, np.uint32(seed), block=(threads_per_block, 1, 1), grid=(blocks, 1))
cuda.memcpy_dtoh(out_gpu_host, out_gpu)
gpu_time = time.time() - start_time

print(f"GPU Time: {gpu_time}s")
print("GPU Result:", out_gpu_host.mean())

# CPU execution
start_time = time.time()
out_cpu = np.zeros(threads_per_block * blocks, dtype=np.float32)
compute_pi_cpu(50000, out_cpu)
cpu_time = time.time() - start_time

# Measure CPU time
print(f"CPU Time: {cpu_time}s")
print("CPU Result:", out_cpu.mean())

# Comparing CPU and GPU speed
print(f"Boost: GPU is {round(cpu_time/gpu_time)}x better than CPU")
