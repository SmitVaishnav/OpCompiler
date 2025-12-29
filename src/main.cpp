// #include <iostream>
// #include <string>
// #include "../include/IR.hpp"

// // Forward declare the Parser class logic
// // (In a real project, we would put Parser in a Parser.hpp header file)
// // For this tutorial, we will rely on the linker finding the class.
// #include "../src/Parser.cpp"

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: ./opc <model_file.mdf>" << std::endl;
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::cout << "--- OpCompiler Phase 2 ---" << std::endl;
//     std::cout << "Compiling: " << model_path << std::endl;

//     // 1. Instantiate the Parser
//     Parser parser;
    
//     // 2. Parse the file into the Model struct
//     Model model = parser.parse(model_path);

//     // 3. Print the result to prove we stored it in C++ memory
//     std::cout << "\n[PARSER SUCCESS]" << std::endl;
//     std::cout << "Model Name: " << model.name << std::endl;
//     std::cout << "Input Size: " << model.input_size << std::endl;
//     std::cout << "Operator Count: " << model.operators.size() << std::endl;

//     for (const auto& op : model.operators) {
//         std::cout << " - Found Op: " << op.name 
//                   << " (" << (op.type == OpType::RELU ? "RELU" : "MATMUL") << ")"
//                   << "\n   Input: " << op.input_name 
//                   << "\n   Output: " << op.output_name << std::endl;
//     }

//     return 0;
// }


#include <iostream>
#include <string>
#include "IR.hpp"

// Include sources
#include "../src/Parser.cpp"
#include "../src/Optimizer.cpp" // <--- Add this
#include "../src/CodeGen.cpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./opc <model_file.mdf>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::cout << "--- OpCompiler Phase 4 (Optimized) ---" << std::endl;

    // 1. Parse
    Parser parser;
    Model raw_model = parser.parse(model_path);
    std::cout << "[PARSER] Found " << raw_model.operators.size() << " raw ops.\n";

    // 2. Optimize
    Optimizer optimizer;
    Model optimized_model = optimizer.optimize(raw_model);
    std::cout << "[OPTIMIZER] Reduced to " << optimized_model.operators.size() << " ops.\n";

    // 3. Generate
    CodeGen codegen;
    std::string output_file = "generated_model.cpp";
    // codegen.generate(raw_model, output_file);
    codegen.generate(optimized_model, output_file);

    std::cout << "Done! Compile with: \n";
    std::cout << "  g++ " << output_file << " -o run_model && ./run_model" << std::endl;

    return 0;
}