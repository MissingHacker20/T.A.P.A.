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
        // Integer multiplier that maps the raw network output to centipawns.
        std::int32_t output_scale = 1;
    };
}
