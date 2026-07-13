cmake_minimum_required(VERSION 3.14)
project(MNIST_C__Tests LANGUAGES CXX)

# --- Fetch Catch2 directly from GitHub so it compiles with Clang ---
include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.4.0 # Or whatever v3 version you prefer
)
FetchContent_MakeAvailable(Catch2)
# -------------------------------------------------------------------

# Compile the test executable
add_executable(tests
        test_matrix.cpp
        test_activations.cpp
        ../src/matrix.cpp
)

set_target_properties(tests PROPERTIES
        CXX_STANDARD 26
        CXX_STANDARD_REQUIRED ON
)

# Link Catch2 (The FetchContent target name is exactly the same)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(tests)