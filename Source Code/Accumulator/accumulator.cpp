// accumulator.cpp
#include "accumulator.h"
#include <limits>

namespace TAPA {
namespace Accumulator {

void Accumulator::add_feature(const Features::Feature& f, const AccumulatorValue* weights) {
    if (weights == nullptr || f.value < -1 || f.value > 1) return;
    for (std::size_t i = 0; i < NeuronCount; ++i) {
        const std::int64_t sum = static_cast<std::int64_t>(data[i]) +
            static_cast<std::int64_t>(f.value) * weights[i];
        data[i] = static_cast<AccumulatorValue>(sum < std::numeric_limits<AccumulatorValue>::min()
            ? std::numeric_limits<AccumulatorValue>::min()
            : sum > std::numeric_limits<AccumulatorValue>::max()
                ? std::numeric_limits<AccumulatorValue>::max() : sum);
    }
}

void Accumulator::remove_feature(const Features::Feature& f, const AccumulatorValue* weights) {
    if (weights == nullptr || f.value < -1 || f.value > 1) return;
    for (std::size_t i = 0; i < NeuronCount; ++i) {
        const std::int64_t sum = static_cast<std::int64_t>(data[i]) -
            static_cast<std::int64_t>(f.value) * weights[i];
        data[i] = static_cast<AccumulatorValue>(sum < std::numeric_limits<AccumulatorValue>::min()
            ? std::numeric_limits<AccumulatorValue>::min()
            : sum > std::numeric_limits<AccumulatorValue>::max()
                ? std::numeric_limits<AccumulatorValue>::max() : sum);
    }
}

void Accumulator::reconstruct(const Features::FeatureSet& fs, const AccumulatorValue* weights) {
    data.fill(bias);
    const std::size_t count = fs.size < fs.features.size() ? fs.size : fs.features.size();
    for (std::size_t i = 0; i < count; ++i) {
        add_feature(fs.features[i], weights);
    }
}

void Accumulator::update_incremental(const Features::FeatureSet& added,
                                     const Features::FeatureSet& removed,
                                     const AccumulatorValue* weights) {
    const std::size_t removed_count = removed.size < removed.features.size() ? removed.size : removed.features.size();
    for (std::size_t i = 0; i < removed_count; ++i) {
        remove_feature(removed.features[i], weights);
    }
    const std::size_t added_count = added.size < added.features.size() ? added.size : added.features.size();
    for (std::size_t i = 0; i < added_count; ++i) {
        add_feature(added.features[i], weights);
    }
}

} // namespace Accumulator
} // namespace TAPA