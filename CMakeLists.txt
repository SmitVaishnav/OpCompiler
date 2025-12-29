# CMakeLists.txt - Build recipe for OpCompiler
cmake_minimum_required(VERSION 3.10)

# Project Name
project(OpCompiler VERSION 1.0)

# We want Modern C++ (C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Where are the header files?
include_directories(include)

# Find all .cpp files in src/
file(GLOB SOURCES "src/*.cpp")

# Create the executable named 'opc' (OpCompiler)
add_executable(opc ${SOURCES})

# Standard compiler warnings (Treat warnings as errors - best practice)
if(MSVC)
    target_compile_options(opc PRIVATE /W4 /WX)
else()
    target_compile_options(opc PRIVATE -Wall -Wextra -Werror)
endif()
