#include "network.h"
#include "activations.h"
#include <cmath>

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& layerSizes, Activations::ActivationType actType) {
    // Map the activation type to the corresponding static functions
    auto ActFns = Activations::getActivation(actType);
    this->activationFunction = ActFns.activation;
    this->d_activationFunction = ActFns.derivative;

    // LayerSizes is a vector containing the number of neurons in each layer,
    // so the size of the vector is number of layers
    size_t numLayers = layerSizes.size();

    // Resize vectors to numLayers - 1 to account for transitions between layers
    // being equal to number of layers - 1
    this->weights.resize(numLayers - 1);
    this->biases.resize(numLayers - 1);
    this->activations.resize(numLayers);
    this->preActivations.resize(numLayers - 1);

    // Initialize weight matricies (layerSizes[i] is post-activation
    // from last layer, layerSizes[i+1] is pre-activation for next layer)
    for (size_t i = 0; i < numLayers - 1; i++) {
        this->weights[i] = Matrix(layerSizes[i], layerSizes[i + 1]);
        this->weights[i].Randomize();

        // Glorot initialization: Scale weights to be centered around zero to
        // eliminate vanishing gradient/exploding gradient problem 
        // (no idea how this works)
        double limit = std::sqrt(6.0 / (layerSizes[i] + layerSizes[i + 1]));
        for (size_t r = 0; r < layerSizes[i]; r++) {
            for (size_t c = 0; c < layerSizes[i + 1]; c++) {
                this->weights[i](r, c) = (this->weights[i](r, c) - 0.5) * 2.0 * limit;
            }
        }
    }

    // Initialize biases to zero for first forward pass
    for (size_t i = 0; i < numLayers - 1; i++) {
        biases[i] = Matrix(1, layerSizes[i + 1]);
        biases[i].SetZero();
    }
}

Matrix NeuralNetwork::forward(const Matrix& input) {
    activations[0] = input;

    for (size_t i = 0; i < weights.size(); i++) {
        // z = activations_prev * W + b
        Matrix z = Matrix::dot(activations[i], weights[i]);
        z.add(biases[i]);
        preActivations[i] = z;

        // Apply activation function to z to get activation for next layer
        activations[i + 1] = Matrix::map(z, activationFunction);
    }

    // Return the activation of the last layer (output)
    return activations.back();
}

void NeuralNetwork::backpropagate(const Matrix& input, const Matrix& target, double learningRate) {
    
}
