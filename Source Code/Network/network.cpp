// network.cpp
#include "network.h"
#include "network_model.h"
#include <cstdint>
#include <limits>
#include <algorithm>

namespace TAPA {
namespace Network {

Network::Network() = default;

Network::Network(const NetworkWeights& weights) : Network() { load_weights(weights); }

void Network::load_weights(const NetworkWeights& weights) noexcept {
    layer1 = weights.layer1;
    layer2 = weights.layer2;
    output_layer = weights.output;
    activation_limit = weights.activation_limit;
    output_scale = weights.output_scale;
}

int32_t Network::forward_pass(const TAPA::Accumulator::Accumulator& acc_white,
                              const TAPA::Accumulator::Accumulator& acc_black) const noexcept {
    return forward_pass(acc_white.get_data(), acc_black.get_data());
}

int32_t Network::forward_pass(const std::array<std::int32_t, AccumulatorNeuronCount>& white,
                              const std::array<std::int32_t, AccumulatorNeuronCount>& black) const noexcept {
    std::array<int32_t, InputSize> input{};
    for (std::size_t i = 0; i < AccumulatorNeuronCount; ++i) {
        input[i] = white[i];
        input[i + AccumulatorNeuronCount] = black[i];
    }
    std::array<int32_t, HiddenSize1> hidden1{};
    layer1.forward(input, hidden1, activation_limit);
    std::array<int32_t, HiddenSize2> hidden2{};
    layer2.forward(hidden1, hidden2, activation_limit);
    std::array<int32_t, OutputSize> output{};
    output_layer.forward_linear(hidden2, output);
    const std::int64_t scaled = static_cast<std::int64_t>(output[0]) * output_scale;
    return static_cast<std::int32_t>(std::clamp<std::int64_t>(scaled,
        std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max()));
}

} // namespace Network
} // namespace TAPA