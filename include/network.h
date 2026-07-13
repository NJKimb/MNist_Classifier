#pragma once
#include "matrix.h"

class NeuralNetwork {
public:
    explicit NeuralNetwork(const std::vector<int>& layerSizes);
    Matrix forward(const Matrix&);
    void backpropagate(const Matrix&, const Matrix&, double lr);
};
