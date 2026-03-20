# Parallel Computing Simulation Floods & CUDA Kernels

<div align="center">
  <img src="https://img.shields.io/badge/Language-C-00599C?style=flat-square&logo=c&logoColor=white" alt="C" />
  <img src="https://img.shields.io/badge/Parallel-OpenMP-000000?style=flat-square&logo=openmp&logoColor=white" alt="OpenMP" />
  <img src="https://img.shields.io/badge/Parallel-MPI-008000?style=flat-square" alt="MPI" />
  <img src="https://img.shields.io/badge/GPU-CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA" />
  <img src="https://img.shields.io/badge/Visualization-Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
</div>

> **About this project:**
> This repository contains a collection of high-performance computing projects developed for the **Parallel Computing** course (3rd Year, Computer Engineering) at the **University of Valladolid**.
> 
> The core of the repository is a **cellular automata simulation of rainwater flooding**, parallelized using three different paradigms. Additionally, it includes a suite of highly optimized **CUDA kernels** for matrix operations.

<div align="center">
  <img width="473" height="417" alt="image" src="https://github.com/user-attachments/assets/150ae4a7-e745-4109-b35d-c1d59c927c19" />
</div> 

---

## 1. Rainwater Flooding Simulation

The simulation models rainwater falling from moving clouds onto a 2D topographical grid. It calculates water levels and spillage across neighboring cells based on height differences, ensuring mass conservation.

### Parallel Implementations:
* **Sequential (Reference):** Baseline implementation used for validation.
* **OpenMP (Shared Memory):** Uses `#pragma omp parallel for` with a **guided schedule** to handle load imbalance in flooded areas and `atomic` operations for thread-safe rainfall updates.
* **MPI (Distributed Memory):** Implements **Domain Decomposition** using a **Ghost Rows (Halo)** strategy. Processes exchange boundary rows using non-blocking communications (`MPI_Isend`/`MPI_Irecv`) to ensure memory scalability and high performance.

### Performance Benchmarks (Stress Test: 4M Cells)
*Tested on a 2000x2000 grid for 200 simulation minutes using WSL2.*

| Version | Configuration | Execution Time | Speedup |
| :--- | :--- | :--- | :--- |
| **Sequential** | 1 Core | 161.63 s | 1.00x |
| **OpenMP** | 8 Threads | **36.62 s** | **4.41x** |
| **MPI** | 4 Processes | **57.92 s** | **2.79x** |

> **Mathematical Consistency:** Both parallel versions (OpenMP and MPI) produced identical physical results (Total Rain, Water Level, and Water Loss) down to the 6th decimal place, confirming the robustness of the boundary exchange logic (Ghost Rows) and the thread-safe accumulation via reduction clauses.

---

## 2. Optimized CUDA Kernels

A set of laboratories focused on leveraging GPU architecture for massive parallelism. *(Note: These benchmarks were developed and tested on NVIDIA hardware provided by the university laboratories. They require a CUDA-capable GPU and the NVIDIA CUDA Toolkit to run locally).*

### Key Projects & Results:
1.  **Tiled Matrix Multiplication:** Implements **Shared Memory Tiling** to reduce Global Memory traffic.
    * **Result:** Achieved **477.84 GFLOPS** throughput and **4.49 ms** latency on a 1024x1024 matrix.
2.  **Vector Dot Product (Reduction):** Uses a tree-based parallel reduction in Shared Memory to avoid non-coalesced memory access.
    * **Result:** Processed 1M elements in just **1.28 ms** with 99.99% accuracy.
3.  **Optimized Matrix Transpose:** Uses Shared Memory with **Padding** to avoid Bank Conflicts, maximizing effective bandwidth.

---

## Visualization & Demo

The project includes a Python script to visualize the simulation results. It uses `Matplotlib` to render the terrain and an animated water overlay.

### How to run the animation:
1. Compile the simulation with animation flags:
   ```bash
   make clean && make animation
Run the simulation and redirect output to a file:

Bash
./flood_seq 100 100 M 0.0 50 1 15 20 20 0 10 5 30 10 45 123 > data.txt
Play the visualizer:

Bash
python3 visualization/animation.py data.txt

Build Instructions
A unified Makefile is provided for the main simulation project.

Bash
# Compile all CPU versions (Seq, OMP, MPI)
make all

# Run OpenMP version (e.g., with 8 threads)
export OMP_NUM_THREADS=8
./flood_omp <args...>

# Run MPI version (e.g., with 4 processes)
mpirun -np 4 ./flood_mpi <args...>

# Compile CUDA kernels (NVIDIA GPU required)
nvcc CUDA-Labs/matrix_mul.cu -o matmul
📂 Project Structure
Plaintext
.
├── Flood-Simulation/
│   ├── src/                 # C source codes (flood.c, flood_omp.c, flood_mpi.c, rng.c)
│   ├── visualization/       # Python animation script (animation.py)
│   └── Makefile             # Unified build system
├── CUDA-Labs/               # Specialized GPU kernels
│   ├── matrix_mul.cu
│   ├── dot_product.cu
│   └── matrix_transpose.cu
└── README.md
