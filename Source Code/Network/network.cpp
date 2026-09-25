// network.cpp
#include "network.h"
#include <cstdint>

namespace TAPA {
namespace Network {

Network::Network() {
    // Initialize weights and biases with simple fixed values for testing.
    // We'll set weights to 1 and biases to 0.
    // Layer 1
    for (int i = 0; i < InputSize; ++i) {
        for (int j = 0; j < HiddenSize1; ++j) {
            layer1.weights[i][j] = 1;
        }
    }
    for (int j = 0; j < HiddenSize1; ++j) {
        layer1.bias[j] = 0;
    }
    // Layer 2
    for (int i = 0; i < HiddenSize1; ++i) {
        for (int j = 0; j < HiddenSize2; ++j) {
            layer2.weights[i][j] = 1;
        }
    }
    for (int j = 0; j < HiddenSize2; ++j) {
        layer2.bias[j] = 0;
    }
    // Output layer
    for (int i = 0; i < HiddenSize2; ++i) {
        for (int j = 0; j < OutputSize; ++j) {
            output_layer.weights[i][j] = 1;
        }
    }
    for (int j = 0; j < OutputSize; ++j) {
        output_layer.bias[j] = 0;
    }
}

int32_t Network::forward_pass(const TAPA::Accumulator::Accumulator& acc_white,
                              const TAPA::Accumulator::Accumulator& acc_black) {
    // Concatenate accumulators
    std::array<int32_t, InputSize> input{};
    const auto& white_data = acc_white.get_data();
    const auto& black_data = acc_black.get_data();
    for (int i = 0; i < AccumulatorNeuronCount; ++i) {
        input[i] = white_data[i];
        input[i + AccumulatorNeuronCount] = black_data[i];
    }

    std::array<int32_t, HiddenSize1> hidden1{};
    layer1.forward(input, hidden1, activation_limit);

    std::array<int32_t, HiddenSize2> hidden2{};
    layer2.forward(hidden1, hidden2, activation_limit);

    std::array<int32_t, OutputSize> output{};
    output_layer.forward(hidden2, output, activation_limit);

    return output[0];
}

} // namespace Network
} // namespace TAPA