#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

__global__ void matmul_kernel(float* in, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = in[idx] * 2.0f;
}

__global__ void relu_kernel(float* in, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = fmaxf(0.0f, in[idx]);
}

__global__ void fused_matmul_relu_kernel(float* in, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float temp = in[idx] * 2.0f;
        out[idx] = fmaxf(0.0f, temp);
    }
}

__global__ void fused_relu_matmul_kernel(float* in, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float temp = fmaxf(0.0f, in[idx]);
        out[idx] = temp * 2.0f;
    }
}

int main() {
    int N = 100000000;
    size_t bytes = N * sizeof(float);
    std::cout << "Running CUDA Model with N=" << N << std::endl;

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    for(int i=0; i<N; ++i) h_in[i] = 1.0f;

    float *d_in, *d_out, *d_temp;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMalloc(&d_temp, bytes));

    // Copy Input to GPU
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch Op: fused_relu_matmul
    fused_relu_matmul_kernel<<<gridSize, blockSize>>>(d_in, d_out, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy Result Back
    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    std::cout << "Final Output Preview: " << h_out[0] << " " << h_out[1] << std::endl;

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_temp);
    free(h_in); free(h_out);
    return 0;
}
