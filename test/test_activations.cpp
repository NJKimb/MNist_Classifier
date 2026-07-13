#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_approx.hpp>
#include "../include/activations.h"
#include <cmath>

TEST_CASE("Sigmoid Activation") {
    // Sigmoid of 0 is exactly 0.5
    REQUIRE(Activations::Sigmoid::activate(0.0) == Catch::Approx(0.5));
    
    // Asymptotic values
    REQUIRE(Activations::Sigmoid::activate(100.0) == Catch::Approx(1.0));
    REQUIRE(Activations::Sigmoid::activate(-100.0) == Catch::Approx(0.0).margin(1e-9));
    
    // Derivative test: sigmoid'(x) = x * (1 - x) where x is already sigmoid output
    // If output value is 0.5 (from input 0), derivative is 0.5 * 0.5 = 0.25
    REQUIRE(Activations::Sigmoid::derivative(0.5) == Catch::Approx(0.25));
}

TEST_CASE("ReLU Activation") {
    REQUIRE(Activations::RelU::activate(5.0) == Catch::Approx(5.0));
    REQUIRE(Activations::RelU::activate(0.0) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(Activations::RelU::activate(-3.0) == Catch::Approx(0.0).margin(1e-9));
    
    REQUIRE(Activations::RelU::derivative(5.0) == Catch::Approx(1.0));
    REQUIRE(Activations::RelU::derivative(-3.0) == Catch::Approx(0.0).margin(1e-9));
}

TEST_CASE("Softmax Activation (Scalar)") {
    // Since Activations::softmax is currently identical to sigmoid, test it similarly
    REQUIRE(Activations::Softmax::activate(0.0) == Catch::Approx(0.5));
    REQUIRE(Activations::Softmax::activate(100.0) == Catch::Approx(1.0));
    REQUIRE(Activations::Softmax::activate(-100.0) == Catch::Approx(0.0).margin(1e-9));
    
    REQUIRE(Activations::Softmax::derivative(0.5) == Catch::Approx(0.25));
}

TEST_CASE("Tanh Activation") {
    // Tanh of 0 is 0
    REQUIRE(Activations::Tanh::activate(0.0) == Catch::Approx(0.0).margin(1e-9));
    
    // Standard tanh outputs
    REQUIRE(Activations::Tanh::activate(1.0) == Catch::Approx(std::tanh(1.0)));
    
    // Derivative test: tanh'(x) = 1 - tanh(x)^2
    REQUIRE(Activations::Tanh::derivative(0.0) == Catch::Approx(1.0));
    REQUIRE(Activations::Tanh::derivative(1.0) == Catch::Approx(1.0 - std::tanh(1.0) * std::tanh(1.0)));
}
