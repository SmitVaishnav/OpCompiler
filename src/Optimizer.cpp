#include "IR.hpp"
#include <vector>
#include <iostream>

class Optimizer {
public:
    // Takes a Model, returns a BETTER Model
    Model optimize(const Model& input_model) {
        Model optimized_model = input_model; 
        optimized_model.operators.clear();   

        const auto& old_ops = input_model.operators;
        
        for (size_t i = 0; i < old_ops.size(); ++i) {
            // If it's the last op, just push it and finish
            if (i + 1 >= old_ops.size()) {
                optimized_model.operators.push_back(old_ops[i]);
                continue;
            }

            const auto& current_op = old_ops[i];
            const auto& next_op = old_ops[i + 1];

            // Common Condition: Data must flow from A -> B
            bool is_connected = (current_op.output_name == next_op.input_name);

            if (!is_connected) {
                optimized_model.operators.push_back(current_op);
                continue;
            }

            // PATTERN 1: MatMul -> ReLU
            if (current_op.type == OpType::MATMUL && next_op.type == OpType::RELU) {
                std::cout << "[OPTIMIZER] Fusing " << current_op.name << " >> " << next_op.name << std::endl;
                
                Operator fused_op;
                fused_op.type = OpType::FUSED_MATMUL_RELU;
                fused_op.name = "fused_matmul_relu";
                fused_op.input_name = current_op.input_name;
                fused_op.output_name = next_op.output_name;
                
                optimized_model.operators.push_back(fused_op);
                i++; // Skip next
            }
            // PATTERN 2: ReLU -> MatMul (The new flexibility)
            else if (current_op.type == OpType::RELU && next_op.type == OpType::MATMUL) {
                std::cout << "[OPTIMIZER] Fusing " << current_op.name << " >> " << next_op.name << std::endl;
                
                Operator fused_op;
                fused_op.type = OpType::FUSED_RELU_MATMUL;
                fused_op.name = "fused_relu_matmul";
                fused_op.input_name = current_op.input_name;
                fused_op.output_name = next_op.output_name;
                
                optimized_model.operators.push_back(fused_op);
                i++; // Skip next
            }
            else {
                // Connected but not fusable (e.g., ReLU -> ReLU)
                optimized_model.operators.push_back(current_op);
            }
        }
        return optimized_model;
    }
};