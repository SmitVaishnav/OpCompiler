#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

int main() {
    // 1. Initialize Memory
    int N = 100000000;
    std::vector<float> input_tensor(N, 1.0f); // Init with 1.0
    auto start = std::chrono::high_resolution_clock::now();


    // --- Op: matmul ---
    std::vector<float> matmul_out(N);
    // Simulating MatMul (y = x * 2.0 for demo)
    for (int i = 0; i < N; ++i) {
        matmul_out[i] = input_tensor[i] * 2.0f;
    }

    // --- Op: relu ---
    std::vector<float> final_out(N);
    for (int i = 0; i < N; ++i) {
        final_out[i] = std::max(0.0f, matmul_out[i]);
    }

    // Print first 5 results to verify
    std::cout << "Final Output Preview: ";
    for(int i=0; i<5; ++i) std::cout << final_out[i] << " ";
    std::cout << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time taken: " << diff.count() << " s" << std::endl;
    return 0;
}
