#include "../Source Code/Network/network.h"
#include "../Source Code/Network/network_model.h"
#include <array>
#include <cstdint>
#include <limits>

int main() {
    using namespace TAPA::Network;
    NetworkWeights weights{};
    weights.activation_limit = 7;
    weights.output_scale = 2;
    Network network(weights);
    std::array<std::int32_t, AccumulatorNeuronCount> white{};
    std::array<std::int32_t, AccumulatorNeuronCount> black{};

    weights.layer1.bias[0] = 1;
    weights.layer2.bias[0] = 1;
    weights.output.bias[0] = 5;
    network.load_weights(weights);
    if (network.forward_pass(white, black) != 10) return 1;
    weights.output.bias[0] = -5;
    network.load_weights(weights);
    if (network.forward_pass(white, black) != -10)
        return 2;

    // The output is a complete dot product: intermediate overflow must not
    // clamp the accumulator before later, cancelling terms are added.
    weights.layer1 = {};
    weights.layer2 = {};
    weights.output = {};
    weights.output.bias[0] = std::numeric_limits<std::int32_t>::max();
    weights.output.weights[0][0] = std::numeric_limits<std::int32_t>::max();
    weights.output.weights[1][0] = -std::numeric_limits<std::int32_t>::max();
    network.load_weights(weights);
    white[0] = 2;
    white[1] = 2;
    if (network.forward_pass(white, black) != std::numeric_limits<std::int32_t>::max())
        return 7;
    white[0] = 0;
    white[1] = 0;

    // output_scale is applied after the linear sum and saturates to int32.
    weights.layer1 = {};
    weights.layer2 = {};
    weights.output = {};
    weights.output.bias[0] = -5;
    weights.output_scale = 2;
    network.load_weights(weights);
    if (network.forward_pass(white, black) != -10)
        return 8;
    weights.output.bias[0] = std::numeric_limits<std::int32_t>::max();
    network.load_weights(weights);
    if (network.forward_pass(white, black) != std::numeric_limits<std::int32_t>::max())
        return 9;
    weights.output.bias[0] = std::numeric_limits<std::int32_t>::min();
    network.load_weights(weights);
    if (network.forward_pass(white, black) != std::numeric_limits<std::int32_t>::min())
        return 10;

    weights.output.bias[0] = 0;
    weights.output_scale = 2;
    network.load_weights(weights);
    if (network.forward_pass(white, black) != 0) return 3;

    weights.output.bias[0] = std::numeric_limits<std::int32_t>::max();
    network.load_weights(weights);
    if (network.forward_pass(white, black) != std::numeric_limits<std::int32_t>::max()) return 4;
    weights.output.bias[0] = std::numeric_limits<std::int32_t>::min();
    network.load_weights(weights);
    if (network.forward_pass(white, black) != std::numeric_limits<std::int32_t>::min()) return 5;

    Layer<4, 1> layer{};
    layer.bias[0] = 0;
    layer.weights[0][0] = 100;
    layer.weights[1][0] = 200;
    layer.weights[2][0] = -300;
    layer.weights[3][0] = 400;
    const std::array<std::int32_t, 4> input{1, 1, 1, 1};
    std::array<std::int32_t, 1> output{};
    layer.forward_linear(input, output);
    if (output[0] != 400) return 6;

    layer.weights[0][0] = std::numeric_limits<std::int32_t>::max();
    layer.weights[1][0] = std::numeric_limits<std::int32_t>::max();
    layer.weights[2][0] = 0;
    layer.weights[3][0] = 0;
    layer.forward_linear(input, output);
    if (output[0] != std::numeric_limits<std::int32_t>::max()) return 11;

    layer.weights[0][0] = std::numeric_limits<std::int32_t>::min();
    layer.weights[1][0] = std::numeric_limits<std::int32_t>::min();
    layer.forward_linear(input, output);
    if (output[0] != std::numeric_limits<std::int32_t>::min()) return 12;

    Layer<6, 1> wide_cancellation{};
    const std::array<std::int32_t, 6> wide_input{
        std::numeric_limits<std::int32_t>::max(), std::numeric_limits<std::int32_t>::max(),
        std::numeric_limits<std::int32_t>::max(), std::numeric_limits<std::int32_t>::max(),
        std::numeric_limits<std::int32_t>::max(), std::numeric_limits<std::int32_t>::max()
    };
    for (int index = 0; index < 3; ++index)
        wide_cancellation.weights[static_cast<std::size_t>(index)][0] =
            std::numeric_limits<std::int32_t>::max();
    for (int index = 3; index < 6; ++index)
        wide_cancellation.weights[static_cast<std::size_t>(index)][0] =
            -std::numeric_limits<std::int32_t>::max();
    wide_cancellation.forward_linear(wide_input, output);
    if (output[0] != 0) return 13;

    // --- Extended coverage (TODO #9) -------------------------------------

    // Clipped ReLU via the public path: negative -> 0, in-range -> value,
    // above limit -> limit. Note the limit applies to BOTH hidden layers.
    {
        NetworkWeights relu_weights{};
        relu_weights.activation_limit = 100;
        relu_weights.output_scale = 1;
        relu_weights.layer1.weights[0][0] = 5;   // in = 1 -> pre = 5+2 = 7 (in range)
        relu_weights.layer1.bias[0] = 2;
        relu_weights.layer1.weights[0][1] = -5;  // in = 1 -> pre = -5 -> clip 0
        relu_weights.layer1.weights[0][2] = 200; // in = 1 -> pre = 200 -> clip to 100
        relu_weights.layer2.bias[0] = 0;
        relu_weights.layer2.weights[0][0] = 1;   // h1 = 7
        relu_weights.layer2.weights[1][0] = 0;   // h1 = 0
        relu_weights.layer2.weights[2][0] = 1;   // h1 = 100
        relu_weights.output.weights[0][0] = 1;   // 7 + 0 + 100 = 107 -> clip to 100
        relu_weights.output.bias[0] = 0;
        Network relu_network(relu_weights);
        std::array<std::int32_t, AccumulatorNeuronCount> one{};
        one[0] = 1;
        const std::int32_t relu_result = relu_network.forward_pass(one, black);
        if (relu_result != 100) return 14;
        // Negative input: neuron1 passes 5, but layer2 multiplies it by 0; the
        // other two neurons pre-activate negative and clip to 0.
        std::array<std::int32_t, AccumulatorNeuronCount> minus{};
        minus[0] = -1;
        const std::int32_t negative_result = relu_network.forward_pass(minus, black);
        if (negative_result != 0) return 15;
    }

    // Full fan-in of 512 on layer1 with a reference computed in int64.
    {
        NetworkWeights full_weights{};
        full_weights.activation_limit = 32767;
        full_weights.output_scale = 1;
        std::array<std::int32_t, AccumulatorNeuronCount> full_white{};
        std::array<std::int32_t, AccumulatorNeuronCount> full_black{};
        std::int64_t reference = 0;
        for (std::size_t i = 0; i < AccumulatorNeuronCount; ++i) {
            const std::int32_t w = (static_cast<std::int32_t>(i % 11) - 5);
            full_white[i] = (static_cast<std::int32_t>(i % 7) - 3);
            full_black[i] = (static_cast<std::int32_t>(i % 13) - 6);
            // neuron 0 of layer1 receives white+black inputs at i and i+256.
            full_weights.layer1.weights[i][0] = w;
            full_weights.layer1.weights[i + AccumulatorNeuronCount][0] = w;
            reference += static_cast<std::int64_t>(full_white[i]) * w +
                         static_cast<std::int64_t>(full_black[i]) * w;
        }
        full_weights.layer1.bias[0] = 3;
        reference += 3;
        Network full_network(full_weights);
        const std::int32_t full_result = full_network.forward_pass(full_white, full_black);
        std::int64_t saturated = reference;
        if (saturated > std::numeric_limits<std::int32_t>::max()) saturated = std::numeric_limits<std::int32_t>::max();
        if (saturated < std::numeric_limits<std::int32_t>::min()) saturated = std::numeric_limits<std::int32_t>::min();
        const std::int32_t expected = clipped_relu(static_cast<std::int32_t>(saturated),
            full_weights.activation_limit);
        if (full_result != expected) return 16;
    }

    // The complete 512-term dot product can exceed int64 before later terms
    // cancel it. This exercises the production fan-in, not only a helper layer.
    {
        NetworkWeights cancellation_weights{};
        cancellation_weights.activation_limit = 32767;
        cancellation_weights.output_scale = 1;
        std::array<std::int32_t, AccumulatorNeuronCount> max_white;
        std::array<std::int32_t, AccumulatorNeuronCount> max_black;
        max_white.fill(std::numeric_limits<std::int32_t>::max());
        max_black.fill(std::numeric_limits<std::int32_t>::max());
        for (std::size_t input = 0; input < InputSize; ++input) {
            cancellation_weights.layer1.weights[input][0] =
                input < InputSize / 2
                    ? std::numeric_limits<std::int32_t>::max()
                    : -std::numeric_limits<std::int32_t>::max();
        }
        Network cancellation_network(cancellation_weights);
        if (cancellation_network.forward_pass(max_white, max_black) != 0) return 20;
    }

    // Output stays linear and may be negative; saturation limits both signs.
    {
        NetworkWeights sign_weights{};
        sign_weights.activation_limit = 32767;
        sign_weights.output_scale = 1;
        sign_weights.output.bias[0] = -1000000;
        Network sign_network(sign_weights);
        if (sign_network.forward_pass(white, black) != -1000000) return 17;
        sign_weights.output.bias[0] = std::numeric_limits<std::int32_t>::min();
        sign_network.load_weights(sign_weights);
        if (sign_network.forward_pass(white, black) != std::numeric_limits<std::int32_t>::min()) return 18;
        sign_weights.output.bias[0] = std::numeric_limits<std::int32_t>::max();
        sign_network.load_weights(sign_weights);
        if (sign_network.forward_pass(white, black) != std::numeric_limits<std::int32_t>::max()) return 19;
    }
    return 0;
}
