#include "piece_accumulator.h"
#include <algorithm>
#include <limits>

namespace TAPA::Accumulator
{

    namespace {
        AccumulatorValue saturate_accumulator_value(std::int64_t value) noexcept
        {
            return static_cast<AccumulatorValue>(std::clamp<std::int64_t>(value,
                std::numeric_limits<AccumulatorValue>::min(), std::numeric_limits<AccumulatorValue>::max()));
        }
    }

    void PieceKingAccumulator::reset(AccumulatorValue bias) noexcept
    {
        exact_data_.fill(bias);
        data_.fill(bias);
    }

    std::size_t PieceKingAccumulator::slot(const Features::Feature &feature) noexcept
    {
        return feature.index;
    }

    void PieceKingAccumulator::add(const Features::Feature &feature, const PieceWeights &weights) noexcept
    {
        const std::size_t index = slot(feature);
        if (!weights.valid() || index >= PieceFeatureCount || index >= weights.feature_weights.size() ||
            feature.value < -1 || feature.value > 1) return;
        const auto &row = weights.feature_weights[index];
        for (std::size_t neuron = 0; neuron < data_.size(); ++neuron) {
            exact_data_[neuron] += static_cast<std::int64_t>(row[neuron]) * feature.value;
            data_[neuron] = saturate_accumulator_value(exact_data_[neuron]);
        }
    }

    void PieceKingAccumulator::remove(const Features::Feature &feature, const PieceWeights &weights) noexcept
    {
        const std::size_t index = slot(feature);
        if (!weights.valid() || index >= PieceFeatureCount || index >= weights.feature_weights.size() ||
            feature.value < -1 || feature.value > 1) return;
        const auto &row = weights.feature_weights[index];
        for (std::size_t neuron = 0; neuron < data_.size(); ++neuron) {
            exact_data_[neuron] -= static_cast<std::int64_t>(row[neuron]) * feature.value;
            data_[neuron] = saturate_accumulator_value(exact_data_[neuron]);
        }
    }

    void PieceKingAccumulator::reconstruct(const Features::FeatureSet &features, const PieceWeights &weights,
                                           Features::PieceColor perspective) noexcept
    {
        reset(weights.bias);
        if (!weights.valid()) return;
        const std::size_t count = std::min(features.size, features.features.size());
        for (std::size_t i = 0; i < count; ++i)
            if (features.features[i].perspective == perspective) add(features.features[i], weights);
    }

    void PieceKingAccumulator::update(const Features::FeatureSet &added, const Features::FeatureSet &removed,
                                      const PieceWeights &weights, Features::PieceColor perspective) noexcept
    {
        if (!weights.valid()) return;
        const std::size_t removed_count = std::min(removed.size, removed.features.size());
        const std::size_t added_count = std::min(added.size, added.features.size());
        for (std::size_t i = 0; i < removed_count; ++i)
            if (removed.features[i].perspective == perspective) remove(removed.features[i], weights);
        for (std::size_t i = 0; i < added_count; ++i)
            if (added.features[i].perspective == perspective) add(added.features[i], weights);
    }

} // namespace TAPA::Accumulator
