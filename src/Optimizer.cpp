#include "IR.hpp"
#include <vector>
#include <iostream>

class Optimizer {
public:
    // Takes a Model, returns a BETTER Model
    Model optimize(const Model& input_model) {
        Model optimized_model = input_model; // Copy metadata (name, input_size)
        optimized_model.operators.clear();   // Clear ops so we can rebuild the list

        const auto& old_ops = input_model.operators;
        
        for (size_t i = 0; i < old_ops.size(); ++i) {
            // Check if we are at the end
            if (i + 1 >= old_ops.size()) {
                optimized_model.operators.push_back(old_ops[i]);
                continue;
            }

            const auto& current_op = old_ops[i];
            const auto& next_op = old_ops[i + 1];

            // --- THE PATTERN MATCHER ---
            // Pattern: MatMul -> ReLU
            // Condition 1: Current is MATMUL
            // Condition 2: Next is RELU
            // Condition 3: Data flows directly (Output of A == Input of B)
            bool is_fusion_candidate = 
                (current_op.type == OpType::MATMUL) &&
                (next_op.type == OpType::RELU) &&
                (current_op.output_name == next_op.input_name);

            if (is_fusion_candidate) {
                std::cout << "[OPTIMIZER] Fusing " << current_op.name 
                          << " + " << next_op.name << std::endl;

                // Create the Super Operator
                Operator fused_op;
                fused_op.type = OpType::FUSED_MATMUL_RELU;
                fused_op.name = "fused_matmul_relu";
                fused_op.input_name = current_op.input_name; // Input of first
                fused_op.output_name = next_op.output_name;  // Output of second
                
                optimized_model.operators.push_back(fused_op);

                // SKIP the next operator because we merged it!
                i++; 
            } else {
                // No optimization found, just copy the operator
                optimized_model.operators.push_back(current_op);
            }
        }

        return optimized_model;
    }
};