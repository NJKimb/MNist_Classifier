#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_approx.hpp>
#include "../include/network.h"
#include "../include/matrix.h"
#include "../include/activations.h"

TEST_CASE("NeuralNetwork construction and forward pass") {
    // 3 layers: input 2, hidden 3, output 1
    std::vector<size_t> layers = {2, 3, 1};
    NeuralNetwork net(layers, Activations::ActivationType::ReLU);

    // Create a 1x2 input matrix
    Matrix input(1, 2, {0.5, -0.2});

    // Run forward pass
    Matrix output = net.forward(input);

    // The output matrix should be 1x1
    REQUIRE(output.getRows() == 1);
    REQUIRE(output.getColumns() == 1);

    // Run forward pass with Sigmoid
    NeuralNetwork netSigmoid(layers, Activations::ActivationType::Sigmoid);
    Matrix outputSigmoid = netSigmoid.forward(input);
    REQUIRE(outputSigmoid.getRows() == 1);
    REQUIRE(outputSigmoid.getColumns() == 1);
    // Sigmoid outputs should be between 0 and 1
    REQUIRE(outputSigmoid(0, 0) >= 0.0);
    REQUIRE(outputSigmoid(0, 0) <= 1.0);
}
