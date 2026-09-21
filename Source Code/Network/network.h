// network.h
#pragma once

#include <cstdint>
#include <array>
#include "accumulator.h"

namespace TAPA {
namespace Network {

// Placeholder for layer parameters
template<int InputSize, int OutputSize>
struct Layer {
    std::array<std::array<int32_t, OutputSize>, InputSize> weights{}; // [input][output]
    std::array<int32_t, OutputSize> bias{};

    // Forward pass: input -> output
    void forward(const std::array<int32_t, InputSize>& in, std::array<int32_t, OutputSize>& out) const;
};

// Define network architecture (to be filled)
constexpr int InputSize = 2 * Accumulator::NeuronCount; // White + Black accumulators concatenated
constexpr int HiddenSize1 = 32;
constexpr int HiddenSize2 = 32;
constexpr int OutputSize = 1;

// Activation function: clipped ReLU
inline int32_t clipped_relu(int32_t x, int32_t limit) {
    if (x < 0) return 0;
    if (x > limit) return limit;
    return x;
}

// Network class encapsulating layers
class Network {
public:
    Network();
    int32_t forward_pass(const Accumulator::Accumulator& acc_white,
                         const Accumulator::Accumulator& acc_black);

private:
    Layer<InputSize, HiddenSize1> layer1;
    Layer<HiddenSize1, HiddenSize2> layer2;
    Layer<HiddenSize2, OutputSize> output_layer;
    int32_t activation_limit = 127; // placeholder
};

} // namespace Network
} // namespace TAPA