#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

int main() {
    // 1. Initialize Memory
    int N = 100000000;
    std::vector<float> input_tensor(N, 1.0f); // Init with 1.0
    double total_time = 0.0;

    // --- Op: relu ---
    std::vector<float> mid(N);
    auto start_0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) mid[i] = std::max(0.0f, input_tensor[i]);
    auto end_0 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_0 = end_0 - start_0;
    std::cout << "  Op relu: " << diff_0.count() << " s" << std::endl;
    total_time += diff_0.count();

    // --- Op: matmul ---
    std::vector<float> out(N);
    auto start_1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) out[i] = mid[i] * 2.0f;
    auto end_1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_1 = end_1 - start_1;
    std::cout << "  Op matmul: " << diff_1.count() << " s" << std::endl;
    total_time += diff_1.count();

    // Print first 5 results to verify
    std::cout << "Final Output Preview: ";
    for(int i=0; i<5; ++i) std::cout << out[i] << " ";
    std::cout << std::endl;

    std::cout << "-------------------------" << std::endl;
    std::cout << "TOTAL TIME: " << total_time << " s" << std::endl;
    return 0;
}
