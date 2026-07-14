#pragma once
#include <cmath>
#include <stdexcept>

namespace Activations {
    enum class ActivationType {
        Sigmoid,
        ReLU,
        Softmax,
        Tanh,
        Linear,
        LeakyReLU
    };

    struct Sigmoid {
        static inline double activate(double x) {
            return 1.0 / (1.0 + std::exp(-x));
        }
        static inline double derivative(double x) {
            return x * (1.0 - x);
        }
    };

    struct RelU {
        static inline double activate(double x) {
            return x > 0 ? x : 0;
        }
        static inline double derivative(double x) {
            return x > 0 ? 1 : 0;
        }
    };

    struct Softmax {
        static inline double activate(double x) {
            return std::exp(x) / (1.0 + std::exp(x));
        }
        static inline double derivative(double x) {
            return x * (1.0 - x);
        }
    };

    struct Tanh {
        static inline double activate(double x) {
            return std::tanh(x);
        }
        static inline double derivative(double x) {
            return 1.0 - std::tanh(x) * std::tanh(x);
        }
    };

    struct Linear {
        static inline double activate(double x) {
            return x;
        }
        static inline double derivative(double x) {
            return 1.0;
        }
    };

    struct LeakyRelU {
        static inline double activate(double x) {
            return x > 0 ? x : 0.01 * x;
        }
        static inline double derivative(double x) {
            return x > 0 ? 1 : 0.01;
        }
    };

    // Stores the function pointers for activation and its derivative
    struct FunctionPair{
        double (*activation)(double);
        double (*derivative)(double);
    };

    // Returns activation functions for separation of concerns
    inline FunctionPair getActivation(ActivationType type) {
    switch (type) {
        case ActivationType::Sigmoid:
            return {Sigmoid::activate, Sigmoid::derivative};
        case ActivationType::ReLU:
            return {RelU::activate, RelU::derivative};
        case ActivationType::Softmax:
            return {Softmax::activate, Softmax::derivative};
        case ActivationType::Tanh:
            return {Tanh::activate, Tanh::derivative};
        case ActivationType::Linear:
            return {Linear::activate, Linear::derivative};
        case ActivationType::LeakyReLU:
            return {LeakyRelU::activate, LeakyRelU::derivative};
        default:
            throw std::invalid_argument("Invalid activation type");
    }
}
}
