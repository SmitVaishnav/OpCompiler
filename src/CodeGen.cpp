#include "IR.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

class CodeGen {
public:
    void generate(const Model& model, const std::string& output_path) {
        std::ofstream out(output_path);
        if (!out.is_open()) {
            std::cerr << "Error: Could not write to " << output_path << std::endl;
            return;
        }

        // 1. Write the Header (Standard C++ boilerplate)
        out << "#include <iostream>\n";
        out << "#include <vector>\n";
        out << "#include <cmath>\n";
        out << "#include <algorithm>\n";
        out << "#include <chrono>\n\n";

        out << "int main() {\n";
        out << "    // 1. Initialize Memory\n";
        out << "    int N = " << model.input_size << ";\n";
        
        // We need to track which variables exist so we can chain them
        // For simplicity in the Happy Path, we assume a linear chain:
        // Input -> Op1 -> Op2 -> Output
        
        // Create Input Buffer
        out << "    std::vector<float> input_tensor(N, 1.0f); // Init with 1.0\n";
        
        std::string current_input = "input_tensor";

        // --- START TIMER ---
        out << "    auto start = std::chrono::high_resolution_clock::now();\n\n";

        // 2. Loop through operators and generate code for each
        for (const auto& op : model.operators) {
            out << "\n    // --- Op: " << op.name << " ---\n";
            
            // Create the output buffer for this op
            out << "    std::vector<float> " << op.output_name << "(N);\n";

            if (op.type == OpType::RELU) {
                // Generate ReLU Loop
                out << "    for (int i = 0; i < N; ++i) {\n";
                out << "        " << op.output_name << "[i] = std::max(0.0f, " << current_input << "[i]);\n";
                out << "    }\n";
            } 
            else if (op.type == OpType::MATMUL) {
                // For this simple project, we are faking a MatMul as a 
                // "Matrix-Vector Multiplication" with a fake 1.0 weight.
                // In a real compiler, we would load weights from a file.
                out << "    // Simulating MatMul (y = x * 2.0 for demo)\n";
                out << "    for (int i = 0; i < N; ++i) {\n";
                out << "        " << op.output_name << "[i] = " << current_input << "[i] * 2.0f;\n";
                out << "    }\n";
            }
            // --- NEW: Handle the Fused Op ---
            else if (op.type == OpType::FUSED_MATMUL_RELU) {
                out << "    // [OPTIMIZED] Fused MatMul + ReLU Kernel\n";
                out << "    for (int i = 0; i < N; ++i) {\n";
                out << "        // 1. Perform MatMul (temp variable, in register)\n";
                out << "        float temp = " << current_input << "[i] * 2.0f;\n";
                out << "        // 2. Perform ReLU immediately\n";
                out << "        " << op.output_name << "[i] = std::max(0.0f, temp);\n";
                out << "    }\n";
            }

            // Update current input for the next op
            current_input = op.output_name;
        }

        // 3. Print Result (Verification)
        out << "\n    // Print first 5 results to verify\n";
        out << "    std::cout << \"Final Output Preview: \";\n";
        out << "    for(int i=0; i<5; ++i) std::cout << " << current_input << "[i] << \" \";\n";
        out << "    std::cout << std::endl;\n";

        // --- STOP TIMER ---
        out << "\n    auto end = std::chrono::high_resolution_clock::now();\n";
        out << "    std::chrono::duration<double> diff = end - start;\n";
        out << "    std::cout << \"Time taken: \" << diff.count() << \" s\" << std::endl;\n";

        out << "    return 0;\n";
        out << "}\n";
        
        std::cout << "[CODEGEN] Successfully generated: " << output_path << std::endl;
        out.close();
    }
};