# CPU vs GPU Stress Testing and Comparison

## Overview
This program is designed to stress test and compare the performance of CPUs and GPUs using the Monte Carlo method to approximate the value of Pi. It utilizes multicore multithreading to maximize CPU processing power and implements Numba and PyCUDA for GPU acceleration.

### Monte Carlo Method for Approximating Pi
The Monte Carlo method is a statistical technique that uses random sampling to estimate numerical results. In the context of approximating Pi, it involves randomly selecting points within a square and determining how many fall within a quarter circle inscribed within the square. By comparing the ratio of points inside the circle to the total number of points, Pi can be approximated.

![Monte Carlo Method](MonteCarlo.png)

## Extreme Parallelism
The program leverages extreme parallelism by exploiting the massive parallel processing capabilities of GPUs and multicore multithreading on CPUs. With GPU acceleration through CUDA programming and JIT compilation, the workload is divided into thousands of independent tasks, enabling simultaneous execution across multiple cores or threads. This extreme parallelism significantly speeds up computations, resulting in faster performance compared to traditional sequential processing methods.

## Multicore Multithreading
Multicore multithreading is utilized to fully utilize the processing power of modern CPUs. By spawning multiple threads across multiple CPU cores, the program can execute multiple instances of the Monte Carlo simulation concurrently, further increasing computational throughput.

## Just-In-Time Compilation (JIT)
Just-In-Time compilation with Numba translates Python functions into optimized machine code at runtime. This optimization ensures that the program can take full advantage of the available hardware resources, whether running on the CPU or the GPU. JIT compilation, particularly with `prange`, enhances parallelism by optimizing the execution of Python code, allowing for efficient parallelization of loops and other computations.

## Requirements
### Hardware
- CPU: Multicore processor recommended for optimal performance
- GPU: CUDA-compatible GPU for PyCUDA implementation

### Software
- Python 3.x
- Numba
- PyCUDA
- CUDA Toolkit (for PyCUDA)
- NumPy

## Usage
1. Clone the repository.
2. Navigate to the directory containing the scripts.
3. Run `CPU vs GPU (numba).py` for CPU stress testing using Numba.
4. Run `CPU vs GPU (pycuda).py` for GPU stress testing using PyCUDA.
5. The final result will be printed, showing how many times better the GPU is compared to the CPU.

## Disclaimer
This program is for educational and experimental purposes only. Results may vary depending on hardware configuration and other factors.
