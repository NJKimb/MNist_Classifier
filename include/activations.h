#pragma once
#include <cmath>

namespace Activations {
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
}
