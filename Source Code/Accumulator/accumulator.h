// accumulator.h
#pragma once

#include <cstdint>
#include <array>
#include "features.h"

namespace TAPA {
namespace Accumulator {

constexpr int NeuronCount = 256; // placeholder, should match first layer size
constexpr int AccumulatorSize = NeuronCount;

using AccumulatorValue = int32_t; // maybe int16_t

struct Accumulator {
    std::array<AccumulatorValue, AccumulatorSize> data{};
    AccumulatorValue bias = 0;

    // Initialize accumulator with bias
    void initialize(AccumulatorValue bias_val) {
        bias = bias_val;
        data.fill(bias_val);
    }

    // Add feature weight vector (to be implemented)
    // void add_feature(const Feature& f, const WeightVector& weights);

    // Remove feature weight vector
    // void remove_feature(const Feature& f, const WeightVector& weights);

    // Get reference to accumulator array for network
    const std::array<AccumulatorValue, AccumulatorSize>& get_data() const { return data; }

    // Full reconstruction from feature set (reference)
    void reconstruct(const Features::FeatureSet& fs, /* weights */);
    // Incremental update given added/removed features
    void update_incremental(const Features::FeatureSet& added, const Features::FeatureSet& removed, /* weights */);
};

} // namespace Accumulator
} // namespace TAPA