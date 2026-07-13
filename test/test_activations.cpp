#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_approx.hpp>
#include "../include/activations.h"
#include <cmath>

TEST_CASE("Sigmoid Activation") {
    // Sigmoid of 0 is exactly 0.5
    REQUIRE(Activations::sigmoid(0.0) == Catch::Approx(0.5));
    
    // Asymptotic values
    REQUIRE(Activations::sigmoid(100.0) == Catch::Approx(1.0));
    REQUIRE(Activations::sigmoid(-100.0) == Catch::Approx(0.0).margin(1e-9));
    
    // Derivative test: sigmoid'(x) = x * (1 - x) where x is already sigmoid output
    // If output value is 0.5 (from input 0), derivative is 0.5 * 0.5 = 0.25
    REQUIRE(Activations::sigmoidDerivative(0.5) == Catch::Approx(0.25));
}

TEST_CASE("ReLU Activation") {
    REQUIRE(Activations::relu(5.0) == Catch::Approx(5.0));
    REQUIRE(Activations::relu(0.0) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(Activations::relu(-3.0) == Catch::Approx(0.0).margin(1e-9));
    
    REQUIRE(Activations::reluDerivative(5.0) == Catch::Approx(1.0));
    REQUIRE(Activations::reluDerivative(-3.0) == Catch::Approx(0.0).margin(1e-9));
}

TEST_CASE("Softmax Activation (Scalar)") {
    // Since Activations::softmax is currently identical to sigmoid, test it similarly
    REQUIRE(Activations::softmax(0.0) == Catch::Approx(0.5));
    REQUIRE(Activations::softmax(100.0) == Catch::Approx(1.0));
    REQUIRE(Activations::softmax(-100.0) == Catch::Approx(0.0).margin(1e-9));
    
    REQUIRE(Activations::softmaxDerivative(0.5) == Catch::Approx(0.25));
}

TEST_CASE("Tanh Activation") {
    // Tanh of 0 is 0
    REQUIRE(Activations::tanh(0.0) == Catch::Approx(0.0).margin(1e-9));
    
    // Standard tanh outputs
    REQUIRE(Activations::tanh(1.0) == Catch::Approx(std::tanh(1.0)));
    
    // Derivative test: tanh'(x) = 1 - tanh(x)^2
    REQUIRE(Activations::tanhDerivative(0.0) == Catch::Approx(1.0));
    REQUIRE(Activations::tanhDerivative(1.0) == Catch::Approx(1.0 - std::tanh(1.0) * std::tanh(1.0)));
}
