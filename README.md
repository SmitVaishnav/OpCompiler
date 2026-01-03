# OpCompiler 🚀

**A High-Performance, Heterogeneous AOT Compiler for Machine Learning Operators.**

OpCompiler is a custom compiler built from scratch in **Modern C++17**. It reads high-level Model Definition Files (`.mdf`), performs static analysis to detect memory bottlenecks, and generates optimized machine code for both **CPUs (C++)** and **NVIDIA GPUs (CUDA)**.

It demonstrates advanced systems engineering concepts including **Semantics Engineering**, **Kernel Fusion**, and **Automated Granular Profiling**.

---

## ⚡ The Problem: The "Memory Wall"
In standard deep learning execution (Naive Mode), operators are executed sequentially. This kills performance due to the **Memory Wall**:
1. `MatMul` reads Input → Computes → **Writes to RAM**.
2. `ReLU` **Reads from RAM** → Computes → Writes to Output.

This "Ping-Pong" effect wastes 50% of memory bandwidth on intermediate data that doesn't need to persist.

## 🛠️ The Solution: Kernel Fusion
OpCompiler's **Static Optimizer** analyzes the Intermediate Representation (IR) graph. It detects fuseable subgraphs (e.g., `MatMul + ReLU` or `ReLU + MatMul`) and generates a **Single Fused Kernel**.

* **Result:** Intermediate data lives in **CPU Registers / GPU Local Memory**.
* **Benefit:** Zero Global Memory writes for intermediate results.

---

## 📊 Performance Benchmarks (100M Elements)

Benchmarks run on an Intel Core CPU (Single Thread) and verified on **NVIDIA Tesla T4 GPU**.

| Metric | Naive Execution | OpCompiler Optimized | Improvement |
| :--- | :--- | :--- | :--- |
| **Execution Time** | 0.973 s | **0.696 s** | **~28.5% Speedup** |
| **Memory Writes** | 200 Million | **100 Million** | **50% Reduction** |
| **Kernel Count** | 2 Kernels | **1 Fused Kernel** | Better Cache Locality |

---

## ✨ Key Features

### 1. Custom Frontend & IR
* **Parser:** Reads `.mdf` files into a custom C++ AST (Abstract Syntax Tree).
* **Flexibility:** Supports variable model structures and input sizes.

### 2. Static Optimizer ("The Brain")
* **Pattern Matching:** Bi-directional fusion detection.
    * Detects `MatMul` → `ReLU`
    * Detects `ReLU` → `MatMul`
* **Graph Rewriting:** Replaces multiple nodes with a single `FusedOp` node before code generation.

### 3. Heterogeneous Backend ("The Muscle")
* **CPU Backend:** Generates standard C++17 code with `<chrono>` based granular profiling.
* **GPU Backend:** Generates **CUDA (`.cu`)** files with:
    * `__global__` fused kernels.
    * Automatic Host-to-Device (`cudaMemcpy`) management.
    * `CHECK_CUDA` macros for robust error handling.

---

## 🚀 Quick Start

### Prerequisites
* **CMake** (3.10+)
* **G++** (or any C++17 compiler)
* **NVIDIA CUDA Toolkit** (Optional, for GPU mode)

### 1. Build the Compiler
```bash
mkdir build && cd build
cmake ..
make
```

### 2. Run: CPU Mode
The default mode generates optimized C++ code.
```bash
# 1. Compile the Model
./opc ../examples/simple_model.mdf cpu

# 2. Compile and Run the Generated Binary
g++ generated_model.cpp -o run_cpu -O3
./run_cpu
```

###3. Run: GPU Mode (CUDA)
Target an NVIDIA GPU (or run on Google Colab).
```bash
# 1. Compile the Model for GPU
./opc ../examples/simple_model.mdf gpu

# 2. Compile the Generated CUDA Kernel
# Note: Adjust -arch=sm_75 for your specific GPU (sm_75 = Tesla T4)
nvcc -arch=sm_75 generated_model.cu -o run_gpu

# 3. Execute on GPU
./run_gpu
```

###📂 Project Structure
```plaintext

OpCompiler/
├── src/
│   ├── Parser.cpp        # Converts Text -> IR
│   ├── Optimizer.cpp     # Graph fusion algorithms
│   ├── CodeGen.cpp       # Generates C++ (CPU)
│   ├── CudaCodeGen.cpp   # Generates CUDA (GPU)
│   └── main.cpp          # Driver code
├── include/
│   └── IR.hpp            # Shared Data Structures
├── examples/
│   └── simple_model.mdf  # Test Model
└── CMakeLists.txt        # Build System
```
