// accumulator.cpp
#include "accumulator.h"

namespace TAPA {
namespace Accumulator {

void Accumulator::add_feature(const Features::Feature& f, const AccumulatorValue* weights) {
    for (int i = 0; i < NeuronCount; ++i) {
        data[i] += weights[i];
    }
}

void Accumulator::remove_feature(const Features::Feature& f, const AccumulatorValue* weights) {
    for (int i = 0; i < NeuronCount; ++i) {
        data[i] -= weights[i];
    }
}

void Accumulator::reconstruct(const Features::FeatureSet& fs, const AccumulatorValue* weights) {
    // Start with bias
    data.fill(bias);
    for (std::size_t i = 0; i < fs.size; ++i) {
        add_feature(fs.features[i], weights);
    }
}

void Accumulator::update_incremental(const Features::FeatureSet& added,
                                     const Features::FeatureSet& removed,
                                     const AccumulatorValue* weights) {
    // Subtract removed features
    for (std::size_t i = 0; i < removed.size; ++i) {
        remove_feature(removed.features[i], weights);
    }
    // Add added features
    for (std::size_t i = 0; i < added.size; ++i) {
        add_feature(added.features[i], weights);
    }
}

} // namespace Accumulator
} // namespace TAPA