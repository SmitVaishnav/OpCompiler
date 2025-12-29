#include "../include/IR.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

class Parser {
public:
    // The main function: Takes a filename, returns a Model struct
    Model parse(const std::string& filepath) {
        std::ifstream file(filepath);
        Model model;
        std::string line;

        if (!file.is_open()) {
            std::cerr << "Error: Failed to open " << filepath << std::endl;
            exit(1);
        }

        while (std::getline(file, line)) {
            // Skip comments (#) and empty lines
            if (line.empty() || line[0] == '#') continue;

            // Logic to identify what kind of line this is
            if (line.find("model_name:") != std::string::npos) {
                model.name = extractValue(line);
            }
            else if (line.find("input:") != std::string::npos) {
                // Parsing "input: tensor (1024)"
                size_t open_paren = line.find('(');
                size_t close_paren = line.find(')');
                if (open_paren != std::string::npos) {
                    std::string val = line.substr(open_paren + 1, close_paren - open_paren - 1);
                    model.input_size = std::stoi(val);
                }
            }
            else if (line.find("op:") != std::string::npos) {
                Operator op = parseOperator(line);
                model.operators.push_back(op);
            }
        }
        return model;
    }

private:
    // Helper: extracts "value" from "key: value"
    std::string extractValue(const std::string& line) {
        size_t colon_pos = line.find(':');
        return line.substr(colon_pos + 1); // Returns everything after ':'
    }

    // Helper: Parses a complex op line
    // Format: "op: relu : input -> output"
    Operator parseOperator(const std::string& line) {
        Operator op;
        
        // Find positions of delimiters
        size_t first_colon = line.find(':');
        size_t second_colon = line.find(':', first_colon + 1);
        size_t arrow_pos = line.find("->");

        // Extract the strings between the delimiters
        std::string type_str = line.substr(first_colon + 1, second_colon - first_colon - 1);
        std::string in_str = line.substr(second_colon + 1, arrow_pos - second_colon - 1);
        std::string out_str = line.substr(arrow_pos + 2);

        // Clean up spaces (Trim)
        auto trim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t"));
            s.erase(s.find_last_not_of(" \t") + 1);
        };

        trim(type_str);
        trim(in_str);
        trim(out_str);

        // Fill the struct
        op.type = stringToOpType(type_str);
        op.name = type_str;
        op.input_name = in_str;
        op.output_name = out_str;

        return op;
    }
};