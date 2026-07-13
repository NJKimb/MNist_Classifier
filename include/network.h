#pragma once
#include "matrix.h"

template <typename ActivationFunction>
class NeuralNetwork {
private:
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;

    std::vector<Matrix> activations;
    std::vector<Matrix> preActivations;

    std::function<double(double)> activation;
    std::function<double(double)> activationDerivative;
public:
    NeuralNetwork(const std::vector<int>& layerSizes);
    Matrix forward(const Matrix& input);
    void backpropagate(const Matrix& input, const Matrix& target, double lr);
};
