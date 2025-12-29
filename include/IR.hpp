#pragma once  // 1. Prevents this file from being included twice
#include <string>
#include <vector>
#include <iostream>

// 2. Enum: Makes code safer. Instead of checking strings "relu", 
// we check specific ID numbers.
enum class OpType {
    RELU,
    MATMUL,
    FUSED_MATMUL_RELU,
    UNKNOWN
};

// 3. struct Operator: Represents a single line in your MDF file
// Example: op: relu : input_tensor -> relu_out
struct Operator {
    OpType type;
    std::string name;        // "relu" or "matmul"
    std::string input_name;  // "input_tensor"
    std::string output_name; // "relu_out"
};

// 4. struct Model: Represents the whole graph
struct Model {
    std::string name;
    int input_size;          // e.g., 1024
    std::vector<Operator> operators; // A dynamic list of ops
};

// Helper: Converts string "relu" -> OpType::RELU
inline OpType stringToOpType(const std::string& s) {
    if (s == "relu") return OpType::RELU;
    if (s == "matmul") return OpType::MATMUL;
    return OpType::UNKNOWN;
}