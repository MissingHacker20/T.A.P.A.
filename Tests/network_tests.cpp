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
    return 0;
}
