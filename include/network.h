#pragma once
#include "matrix.h"
#include "activations.h"

class NeuralNetwork {
private:
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;

    std::vector<Matrix> activations;
    std::vector<Matrix> preActivations;

    std::function<double(double)> activationFunction;
    std::function<double(double)> d_activationFunction;
public:
    NeuralNetwork(const std::vector<size_t>& layerSizes, Activations::ActivationType actType);
    Matrix forward(const Matrix& input);
    void backpropagate(const Matrix& input, const Matrix& target, double lr);
};