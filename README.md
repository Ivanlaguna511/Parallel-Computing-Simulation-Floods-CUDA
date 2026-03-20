# Parallel Computing: Rainwater Flooding Simulation & CUDA Kernels

<div align="center">
  <img src="https://img.shields.io/badge/Language-C-00599C?style=flat-square&logo=c&logoColor=white" alt="C" />
  <img src="https://img.shields.io/badge/Parallel-OpenMP-000000?style=flat-square&logo=openmp&logoColor=white" alt="OpenMP" />
  <img src="https://img.shields.io/badge/Parallel-MPI-008000?style=flat-square" alt="MPI" />
  <img src="https://img.shields.io/badge/GPU-CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA" />
  <img src="https://img.shields.io/badge/Visualization-Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
</div>

> **About this project:**
> This repository contains a collection of High-Performance Computing (HPC) projects developed for the **Parallel Computing** course (3rd Year, Computer Engineering) at the **University of Valladolid**.
> 
> The core of the repository is a **cellular automata simulation of rainwater flooding**, parallelized from scratch using three different computing paradigms (Sequential, Shared Memory, and Distributed Memory). Additionally, it includes a suite of highly optimized **CUDA kernels** for low-level GPU acceleration.

<div align="center">
  <img width="600" alt="Simulation Preview" src="https://github.com/user-attachments/assets/150ae4a7-e745-4109-b35d-c1d59c927c19" />
</div> 

---

## Project Structure

```text
.
├── src/                         # C source codes
│   ├── cuda/                    # Specialized GPU kernels
│   │   ├── matrix_mul.cu        # Tiled Matrix Multiplication
│   │   ├── dot_product.cu       # Parallel Tree Reduction
│   │   └── matrix_transpose.cu  # Bank-Conflict free Transpose
│   ├── flood.c                  # Sequential Implementation
│   ├── flood_omp.c              # OpenMP (Shared Memory)
│   ├── flood_mpi.c              # MPI (Distributed Memory)
│   └── rng.c                    # Random Number Generator
├── visualization/       
│   └── animation.py             # Python/Matplotlib animation script
├── Makefile                     # Unified build system
└── README.md
```

---

## 1. Rainwater Flooding Simulation

### The Physics & Logic (Cellular Automata)
The simulation models a dynamic weather system over a 2D topographical grid. 
1. **Rainfall:** Moving clouds drop specific volumes of water onto the grid cells they cover during each discrete time step (minute).
2. **Spillage:** Water flows to adjacent cells based on gravity. If the total height (terrain elevation + water level) of a cell is greater than its neighbors, water spills outwards proportionally to the height difference.
3. **Conservation:** The algorithm ensures strict mass conservation—no water is created or destroyed during the spillage calculation.
4. **Termination:** The simulation ends either when the `max_minutes` are reached, or when the water system reaches a steady state (spillage falls below a specified `threshold`).

### Command Line Parameters Explained
To run the simulation, you must provide a specific set of physical parameters defining the scenario.

**Usage:**
`./flood_seq <rows> <cols> <terrain> <threshold> <max_mins> <num_clouds> [cloud_1_params...] <seed>`

* `rows` / `cols`: The dimensions of the topographical grid (e.g., 1000 1000).
* `terrain`: A character defining the geography (e.g., `M` for Mountain, `V` for Valley, `D` for Valley with Dam).
* `threshold`: Minimum water spillage required to keep the simulation active. Set to `0.0` to force the simulation to run all minutes.
* `max_mins`: Maximum duration of the storm in simulation minutes.
* `num_clouds`: How many clouds exist in the scenario.
* `[cloud_params]`: An array of values defining the size, speed, position, and rain volume for each cloud.
* `seed`: Random seed for reproducibility.

**Example Command:**
```bash
./flood_seq 100 100 M 0.0 50 1 15 20 20 0 10 5 30 10 45 123
```

### Parallel Implementations & Technical Challenges
* **Sequential (Baseline):** The standard C implementation used to validate the math.
* **OpenMP (Shared Memory):** Uses `#pragma omp parallel for` across the grid. **Challenge solved:** Handled race conditions when multiple clouds rain over the same cell using `atomic` operations, and utilized `guided` scheduling to balance the CPU load dynamically (since dry cells compute much faster than flooded cells).
* **MPI (Distributed Memory):** **Challenge solved:** Implemented a 1D Domain Decomposition using the **Ghost Rows (Halo)** strategy. The grid is sliced horizontally, and boundary rows are exchanged between isolated processes at every time step using non-blocking communications (`MPI_Isend`/`MPI_Irecv`), ensuring high scalability across multiple physical nodes.

### Performance Benchmarks (Stress Test)
*Tested on a 4-Million cell grid (`2000x2000`) for 200 simulation minutes using WSL2.*

| Paradigm | Configuration | Execution Time | Speedup | Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **Sequential** | 1 Core CPU | 161.63 s | 1.00x | 100% |
| **OpenMP** | 8 Threads | **36.62 s** | **4.41x** | 55% |
| **MPI** | 4 Processes | **57.92 s** | **2.79x** | 70% |

> **Validation:** Both parallel versions produced identical physical results (Total Rain, Water Level, and Water Loss) down to the 6th decimal place, confirming the robustness of the boundary exchange logic and thread-safe reductions.

---

## 2. Optimized CUDA Kernels

A set of specialized laboratories focused on leveraging NVIDIA GPU architecture for massive parallelism. *(Note: These benchmarks require a CUDA-capable GPU and the NVIDIA CUDA Toolkit).*

### Key Optimizations & Results:
1. **Matrix Multiplication (`matrix_mul.cu`):**
   * **Technique:** Implemented **Shared Memory Tiling**. Instead of pulling data directly from slow Global Memory, threads load data chunks (tiles) into ultra-fast L1 Shared Memory, drastically improving the arithmetic intensity.
   * **Result:** Achieved **477.84 GFLOPS** throughput and **4.49 ms** latency on a 1024x1024 matrix.
2. **Vector Dot Product (`dot_product.cu`):**
   * **Technique:** Avoided CPU bottlenecks by executing an in-place **Parallel Tree Reduction** inside the GPU's Shared Memory, combined with a Grid-Stride Loop to handle arbitrary vector sizes.
   * **Result:** Processed 1 Million elements in just **1.28 ms** with 99.999% mathematical accuracy.
3. **Matrix Transpose (`matrix_transpose.cu`):**
   * **Technique:** Solved the classic memory uncoalescing problem. Added **Padding** to the shared memory array (`TILE_DIM + 1`) to completely eliminate **Bank Conflicts**, allowing the GPU to write transposed columns at maximum hardware bandwidth.

---

## Visualization & Demo

A Python script is provided to visualize the raw numerical output of the simulation, using `Matplotlib` to render the terrain and an animated water overlay.

### How to run the animation:
1. Compile the simulation with the animation preprocessor flags:
   ```bash
   make clean && make animation
   ```
2. Run the simulation and pipe the stdout output to a text file (keep grid size small, e.g., `100x100` for smooth rendering):
   ```bash
   ./flood_seq 100 100 M 0.0 50 1 15 20 20 0 10 5 30 10 45 123 > data.txt
   ```
3. Launch the visualizer:
   ```bash
   python3 visualization/animation.py data.txt
   ```

---

## Build & Run Instructions

A unified `Makefile` is provided in the root directory.

```bash
# Compile all CPU versions (Seq, OMP, MPI)
make all

# Run OpenMP version (e.g., forcing 8 threads)
export OMP_NUM_THREADS=8
./flood_omp <args...>

# Run MPI version (e.g., forcing 4 distributed processes)
mpirun -np 4 ./flood_mpi <args...>

# Compile CUDA kernels (NVIDIA GPU required)
nvcc src/cuda/matrix_mul.cu -o matmul
./matmul
```

**Original Authors**

-Iván Moro Cienfuegos and Gonzalo Sánchez Maroto
