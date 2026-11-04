// network.h
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>
#include "../Accumulator/accumulator.h"

namespace TAPA::Network { struct NetworkWeights; }

namespace TAPA {
namespace Network {

constexpr std::size_t AccumulatorNeuronCount = TAPA::Accumulator::NeuronCount;
constexpr int InputSize = 2 * AccumulatorNeuronCount; // White + Black accumulators
constexpr int HiddenSize1 = 32;
constexpr int HiddenSize2 = 32;
constexpr int OutputSize = 1;
static_assert(AccumulatorNeuronCount == 256, "Update the versioned model architecture and input schema when changing the embedding width");

// Clipped ReLU for hidden layers; output layer is linear.
inline int32_t clipped_relu(int32_t x, int32_t limit) {
    if (x < 0) return 0;
    if (x > limit) return limit;
    return x;
}

// Preserve the complete dot product, including cancellation, before saturating it to
// int32. A fan-in of 512 with unrestricted int32 inputs and weights can exceed int64.
template<int In, int Out>
class Layer {
public:
    std::array<std::array<int32_t, Out>, In> weights{}; // [in][out]
    std::array<int32_t, Out> bias{};

    void forward(const std::array<int32_t, In>& in,
                 std::array<int32_t, Out>& out,
                 int32_t limit) const {
        const std::int32_t safe_limit = limit < 0 ? 0 : limit;
        for (int j = 0; j < Out; ++j) {
            WideAccumulator sum;
            sum.add(bias[j]);
            for (int i = 0; i < In; ++i)
                sum.add(static_cast<std::int64_t>(in[i]) * weights[i][j]);
            out[j] = clipped_relu(sum.saturate_int32(), safe_limit);
        }
    }

    void forward_linear(const std::array<int32_t, In>& in,
                        std::array<int32_t, Out>& out) const {
        for (int j = 0; j < Out; ++j) {
            WideAccumulator sum;
            sum.add(bias[j]);
            for (int i = 0; i < In; ++i)
                sum.add(static_cast<std::int64_t>(in[i]) * weights[i][j]);
            out[j] = sum.saturate_int32();
        }
    }

private:
    // Two-word two's-complement sum; individual int32 products fit int64, but their
    // complete fan-in sum may not.
    class WideAccumulator {
    public:
        void add(std::int64_t value) noexcept {
            const std::uint64_t old_low = low_;
            low_ += static_cast<std::uint64_t>(value);
            high_ += (value < 0 ? -1 : 0) + (low_ < old_low ? 1 : 0);
        }

        std::int32_t saturate_int32() const noexcept {
            constexpr std::uint64_t MinInt32Bits = 0xffffffff80000000ULL;
            if (high_ > 0 || (high_ == 0 &&
                low_ > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())))
                return std::numeric_limits<std::int32_t>::max();
            if (high_ < -1 || (high_ == -1 && low_ < MinInt32Bits))
                return std::numeric_limits<std::int32_t>::min();
            if (high_ == 0) return static_cast<std::int32_t>(low_);
            const std::uint64_t magnitude = (~low_) + 1;
            if (magnitude == (std::uint64_t{1} << 31))
                return std::numeric_limits<std::int32_t>::min();
            return -static_cast<std::int32_t>(magnitude);
        }

    private:
        std::uint64_t low_ = 0;
        std::int64_t high_ = 0;
    };
};

// Network is immutable during forward passes once configured; no global mutable weights are used.
class Network {
public:
    Network();
    explicit Network(const NetworkWeights& weights);
    // Forward pass: returns network output (score)
    int32_t forward_pass(const TAPA::Accumulator::Accumulator& acc_white,
                         const TAPA::Accumulator::Accumulator& acc_black) const noexcept;
    int32_t forward_pass(const std::array<std::int32_t, AccumulatorNeuronCount>& white,
                         const std::array<std::int32_t, AccumulatorNeuronCount>& black) const noexcept;
    template<typename Value, typename = std::enable_if_t<
        std::is_same_v<Value, TAPA::Accumulator::AccumulatorValue> &&
        !std::is_same_v<Value, std::int32_t>>>
    int32_t forward_pass(const std::array<Value, AccumulatorNeuronCount>& white,
                         const std::array<Value, AccumulatorNeuronCount>& black) const noexcept {
        std::array<std::int32_t, AccumulatorNeuronCount> white_input{};
        std::array<std::int32_t, AccumulatorNeuronCount> black_input{};
        for (std::size_t i = 0; i < AccumulatorNeuronCount; ++i) {
            white_input[i] = static_cast<std::int32_t>(white[i]);
            black_input[i] = static_cast<std::int32_t>(black[i]);
        }
        return forward_pass(white_input, black_input);
    }
    void load_weights(const NetworkWeights& weights) noexcept;
    std::int32_t activation_limit_value() const noexcept { return activation_limit; }
    std::int32_t output_scale_value() const noexcept { return output_scale; }

private:
    Layer<InputSize, HiddenSize1> layer1;
    Layer<HiddenSize1, HiddenSize2> layer2;
    Layer<HiddenSize2, OutputSize> output_layer;
    int32_t activation_limit = 127;
    int32_t output_scale = 1;
};

} // namespace Network
} // namespace TAPA