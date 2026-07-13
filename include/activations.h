#pragma once
#include "matrix.h"
#include <cmath>

namespace Activations {
   inline double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    inline double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

    inline double relu(double x) {
        return x > 0 ? x : 0;
    }
    inline double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    inline double softmax(double x) {
        return std::exp(x) / (1.0 + std::exp(x));
    }
    inline double softmaxDerivative(double x) {
        return x * (1.0 - x);
    }

    inline double tanh(double x) {
        return std::tanh(x);
    }
    inline double tanhDerivative(double x) {
        return 1.0 - std::tanh(x) * std::tanh(x);
    }

    inline double linear(double x) {
        return x;
    }
    inline double linearDerivative(double x) {
        return 1.0;
    }

    inline double leakyRelu(double x) {
        return x > 0 ? x : 0.01 * x;
    }
    inline double leakyReluDerivative(double x) {
        return x > 0 ? 1 : 0.01;
    }
}