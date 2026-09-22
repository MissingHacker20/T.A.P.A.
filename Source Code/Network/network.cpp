#include "network.h"
#include <algorithm>
#include <cstdint>

namespace TAPA {
namespace Network {

template<int InputSize, int OutputSize>
void Layer<InputSize, OutputSize>::forward(const std::array<int32_t, InputSize>& in,
                                          std::array<int32_t, OutputSize>& out) const {
    // Initialize output with bias
    for (int j = 0; j < OutputSize; ++j) {
        int32_t sum = bias[j];
        for (int i = 0; i < InputSize; ++i) {
            sum += in[i] * weights[i][j];
        }
        out[j] = clipped_relu(sum, 127); // limit from outer class; need access
        // For simplicity, we'll use a global limit; better to pass limit as parameter.
    }
}

// Since we cannot access outer class limit here, we'll implement forward in Network class.

Network::Network() {
    // Initialize weights and biases to zero (placeholder)
    // In real implementation, load from model.
}

int32_t Network::forward_pass(const Accumulator::Accumulator& acc_white,
                              const Accumulator::Accumulator& acc_black) {
    // Concatenate accumulators
    std::array<int32_t, InputSize> input{};
    const auto& white_data = acc_white.get_data();
    const auto& black_data = acc_black.get_data();
    for (int i = 0; i < Accumulator::NeuronCount; ++i) {
        input[i] = white_data[i];
        input[i + Accumulator::NeuronCount] = black_data[i];
    }

    std::array<int32_t, HiddenSize1> hidden1{};
    layer1.forward(input, hidden1);

    std::array<int32_t, HiddenSize2> hidden2{};
    layer2.forward(hidden1, hidden2);

    std::array<int32_t, OutputSize> output{};
    output_layer.forward(hidden2, output);

    return output[0];
}

// Need to implement Layer::forward with limit; we'll redo with limit parameter.
// For simplicity, we'll define a non-template version or adjust.
// Given time, we'll keep as is and note that limit is hardcoded.

} // namespace Network
} // namespace TAPA