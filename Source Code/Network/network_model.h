#pragma once
#include "network.h"

namespace TAPA::Network
{
    struct NetworkWeights
    {
        Layer<InputSize, HiddenSize1> layer1{};
        Layer<HiddenSize1, HiddenSize2> layer2{};
        Layer<HiddenSize2, OutputSize> output{};
        std::int32_t activation_limit = 127;
        // Positive integer multiplier applied after the signed linear output; the
        // final centipawn-like score saturates to int32_t. Raw output may be signed.
        // Positive multiplier applied after signed linear output; score saturates to int32.
        std::int32_t output_scale = 1;
    };
}
