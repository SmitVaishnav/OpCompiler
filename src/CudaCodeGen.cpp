#include "IR.hpp"
#include <fstream>
#include <iostream>
#include <string>

class CudaCodeGen {
public:
    void generate(const Model& model, const std::string& output_path) {
        std::ofstream out(output_path);
        
        // 1. HEADERS & ERROR MACRO
        out << "#include <iostream>\n";
        out << "#include <cuda_runtime.h>\n\n";

        // This macro is the industry standard for catching CUDA bugs
        out << "#define CHECK_CUDA(call) { \\\n";
        out << "    cudaError_t err = call; \\\n";
        out << "    if (err != cudaSuccess) { \\\n";
        out << "        std::cerr << \"CUDA Error: \" << cudaGetErrorString(err) << \" at line \" << __LINE__ << std::endl; \\\n";
        out << "        exit(1); \\\n";
        out << "    } \\\n";
        out << "}\n\n";

        // 2. KERNELS (Same as before)
        out << "__global__ void matmul_kernel(float* in, float* out, int N) {\n";
        out << "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
        out << "    if (idx < N) out[idx] = in[idx] * 2.0f;\n";
        out << "}\n\n";

        out << "__global__ void relu_kernel(float* in, float* out, int N) {\n";
        out << "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
        out << "    if (idx < N) out[idx] = fmaxf(0.0f, in[idx]);\n";
        out << "}\n\n";

        out << "__global__ void fused_matmul_relu_kernel(float* in, float* out, int N) {\n";
        out << "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
        out << "    if (idx < N) {\n";
        out << "        float temp = in[idx] * 2.0f;\n";
        out << "        out[idx] = fmaxf(0.0f, temp);\n";
        out << "    }\n";
        out << "}\n\n";

        out << "__global__ void fused_relu_matmul_kernel(float* in, float* out, int N) {\n";
        out << "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n";
        out << "    if (idx < N) {\n";
        out << "        float temp = fmaxf(0.0f, in[idx]);\n";
        out << "        out[idx] = temp * 2.0f;\n";
        out << "    }\n";
        out << "}\n\n";

        // 3. HOST CODE WITH ERROR CHECKING
        out << "int main() {\n";
        out << "    int N = " << model.input_size << ";\n";
        out << "    size_t bytes = N * sizeof(float);\n";
        out << "    std::cout << \"Running CUDA Model with N=\" << N << std::endl;\n\n";

        out << "    float* h_in = (float*)malloc(bytes);\n";
        out << "    float* h_out = (float*)malloc(bytes);\n";
        out << "    for(int i=0; i<N; ++i) h_in[i] = 1.0f;\n\n";

        out << "    float *d_in, *d_out, *d_temp;\n";
        // Wrap Allocations in CHECK_CUDA
        out << "    CHECK_CUDA(cudaMalloc(&d_in, bytes));\n";
        out << "    CHECK_CUDA(cudaMalloc(&d_out, bytes));\n";
        out << "    CHECK_CUDA(cudaMalloc(&d_temp, bytes));\n\n";

        out << "    // Copy Input to GPU\n";
        out << "    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));\n\n";

        out << "    int blockSize = 256;\n";
        out << "    int gridSize = (N + blockSize - 1) / blockSize;\n\n";

        std::string current_src = "d_in";
        std::string current_dst = "d_temp"; 

        for (size_t i = 0; i < model.operators.size(); ++i) {
            const auto& op = model.operators[i];
            if (i == model.operators.size() - 1) current_dst = "d_out";

            out << "    // Launch Op: " << op.name << "\n";
            
            if (op.type == OpType::MATMUL) {
                out << "    matmul_kernel<<<gridSize, blockSize>>>(" << current_src << ", " << current_dst << ", N);\n";
            } else if (op.type == OpType::RELU) {
                out << "    relu_kernel<<<gridSize, blockSize>>>(" << current_src << ", " << current_dst << ", N);\n";
            } else if (op.type == OpType::FUSED_MATMUL_RELU) {
                out << "    fused_matmul_relu_kernel<<<gridSize, blockSize>>>(" << current_src << ", " << current_dst << ", N);\n";
            } else if (op.type == OpType::FUSED_RELU_MATMUL) {
                out << "    fused_relu_matmul_kernel<<<gridSize, blockSize>>>(" << current_src << ", " << current_dst << ", N);\n";
            }

            // CHECK FOR KERNEL ERRORS
            out << "    CHECK_CUDA(cudaGetLastError());\n";
            out << "    CHECK_CUDA(cudaDeviceSynchronize());\n";
            
            current_src = current_dst;
        }

        out << "\n    // Copy Result Back\n";
        out << "    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));\n";
        out << "    std::cout << \"Final Output Preview: \" << h_out[0] << \" \" << h_out[1] << std::endl;\n\n";
        
        out << "    cudaFree(d_in); cudaFree(d_out); cudaFree(d_temp);\n";
        out << "    free(h_in); free(h_out);\n";
        out << "    return 0;\n";
        out << "}\n";
        
        std::cout << "[CUDA GEN] Generated generated_model.cu (Debug Mode)" << std::endl;
        out.close();
    }
};