// accumulator.h
#pragma once

#include <array>
#include <cstdint>
#include "../Features/features.h"

namespace TAPA {
namespace Accumulator {

constexpr int NeuronCount = 256; // should match first layer size
using AccumulatorValue = int32_t;

struct Accumulator {
    std::array<AccumulatorValue, NeuronCount> data{};
    AccumulatorValue bias = 0;

    // Initialize accumulator with bias value
    void initialize(AccumulatorValue bias_val) {
        bias = bias_val;
        data.fill(bias_val);
    }

    // Add contribution of a feature (weights pointer to NeuronCount values)
    void add_feature(const Features::Feature& f, const AccumulatorValue* weights);
    // Remove contribution of a feature
    void remove_feature(const Features::Feature& f, const AccumulatorValue* weights);

    // Full reconstruction from a feature set and weights
    void reconstruct(const Features::FeatureSet& fs, const AccumulatorValue* weights);
    // Incremental update given added/removed feature sets
    void update_incremental(const Features::FeatureSet& added,
                            const Features::FeatureSet& removed,
                            const AccumulatorValue* weights);

    // Get reference to accumulator data for network
    const std::array<AccumulatorValue, NeuronCount>& get_data() const { return data; }
};

} // namespace Accumulator
} // namespace TAPA