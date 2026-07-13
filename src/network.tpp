#include "network.h"

template <typename ActivationFunction>
NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes) {
    //LayerSizes is a vector containing the number of neurons in each layer,
    //so the size of the vector is number of layers
    int numLayers = layerSizes.size();

    // Resize vectors to numLayers -1 to account for transitions between layers
    // being equal to number of layers -1
    this->weights.resize(numLayers - 1);
    this->biases.resize(numLayers - 1);
    this->activations.resize(numLayers - 1);
    this->preActivations.resize(numLayers-1);

    // Initialize weight matricies (layerSizes[i] is post-activation
    // from last layer, layerSizes[i+1] is pre-activation for next layer)
    for (int i = 0; i < numLayers - 1; i++) {
        this->weights[i] = Matrix(layerSizes[i], layerSizes[i + 1]);
        this->weights[i].Randomize();

    // Glorot initialization: Scale weights to be centered around zero to
    // eliminate vanishing gradient/exploding gradient problem 
    // (no idea how this works)
    double limit = std::sqrt(6.0 / (layerSizes[i] + layerSizes[i+1]));
        for (int r = 0; r < layerSizes[i]; r++) {
            for (int c = 0; c < layerSizes[i+1]; c++) {
                this->weights[i](r, c) = (this->weights[i](r, c) - 0.5) * 2.0 * limit;
            }
        }
    }

    // Initialize biases to zero for first forward pass
    for (int i = 0; i < numLayers - 1; i++) {
        biases[i] = Matrix(1, layerSizes[i+1]);
        biases[i].SetZero();
    }
}

template <typename ActivationFunction>
Matrix NeuralNetwork::forward(const Matrix& input) {
    activations[0] = input;

    for (int i = 0; i < weights.size(); i++) {
        // z = activations_prev * W + b
        Matrix z = Matrix::dot(activations[i], weights[i]);
        z.add(biases[i]);
        preActivations[i] = z;

        // Apply activation function to z to get activation for next layer
        activations[i + 1] = Matrix::map(z, ActivationFunction::activate);
    }

    // Return the activation of the last layer (output)
    return activations.back();
}

template <typename ActivationFunction>
void NeuralNetwork::backpropagate(const Matrix& input, const Matrix& target, double learningRate) {
    
}