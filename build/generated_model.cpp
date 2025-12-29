#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

int main() {
    // 1. Initialize Memory
    int N = -2021496688;
    std::vector<float> input_tensor(N, 1.0f); // Init with 1.0
    double total_time = 0.0;

    // Print first 5 results to verify
    std::cout << "Final Output Preview: ";
    for(int i=0; i<5; ++i) std::cout << input_tensor[i] << " ";
    std::cout << std::endl;

    std::cout << "-------------------------" << std::endl;
    std::cout << "TOTAL TIME: " << total_time << " s" << std::endl;
    return 0;
}
