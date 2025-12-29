# OpCompiler 🚀

OpCompiler is a lightweight **Ahead-of-Time (AOT) Compiler** for Machine Learning operators, written in C++17. It demonstrates the core principles of **Semantics Engineering** and **Kernel Fusion** to optimize memory bandwidth.

## Key Features
* **Custom IR Parser:** Reads `.mdf` (Model Definition Files).
* **Static Optimizer:** Implements a "Happy Path" pass to fuse `MatMul + ReLU` operators.
* **Code Generator:** Emits dependency-free C++ code with built-in performance timers.

## Performance Benchmark (100M Elements)
* **Naive Compilation:** 0.35s (Memory Bound)
* **OpCompiler Optimized:** 0.25s (**30% Speedup**)

## How to Build
```bash
mkdir build && cd build
cmake ..
make
./opc ../examples/simple_model.mdf
