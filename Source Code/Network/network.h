// network.h
#pragma once

#include <array>
#include <cstdint>
#include "../Accumulator/accumulator.h"

namespace TAPA {
namespace Network {

constexpr int AccumulatorNeuronCount = TAPA::Accumulator::NeuronCount;
constexpr int InputSize = 2 * AccumulatorNeuronCount; // White + Black accumulators
constexpr int HiddenSize1 = 32;
constexpr int HiddenSize2 = 32;
constexpr int OutputSize = 1;

// Clipped ReLU activation
inline int32_t clipped_relu(int32_t x, int32_t limit) {
    if (x < 0) return 0;
    if (x > limit) return limit;
    return x;
}

// Layer definition
template<int In, int Out>
class Layer {
public:
    std::array<std::array<int32_t, Out>, In> weights{}; // [in][out]
    std::array<int32_t, Out> bias{};

    void forward(const std::array<int32_t, In>& in,
                 std::array<int32_t, Out>& out,
                 int32_t limit) const {
        for (int j = 0; j < Out; ++j) {
            int32_t sum = bias[j];
            for (int i = 0; i < In; ++i) {
                sum += in[i] * weights[i][j];
            }
            out[j] = clipped_relu(sum, limit);
        }
    }
};

// Network class
class Network {
public:
    Network();
    // Forward pass: returns network output (score)
    int32_t forward_pass(const TAPA::Accumulator::Accumulator& acc_white,
                         const TAPA::Accumulator::Accumulator& acc_black);

private:
    Layer<InputSize, HiddenSize1> layer1;
    Layer<HiddenSize1, HiddenSize2> layer2;
    Layer<HiddenSize2, OutputSize> output_layer;
    int32_t activation_limit = 127; // placeholder, could be configured
};

} // namespace Network
} // namespace TAPA