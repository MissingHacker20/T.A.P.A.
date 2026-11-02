#include "piece_accumulator.h"

namespace TAPA::Accumulator
{

    void PieceKingAccumulator::reset(AccumulatorValue bias) noexcept { data_.fill(bias); }

    std::size_t PieceKingAccumulator::slot(const Features::Feature &feature) noexcept
    {
        return feature.index;
    }

    void PieceKingAccumulator::add(const Features::Feature &feature, const PieceWeights &weights) noexcept
    {
        const std::size_t index = slot(feature);
        if (index >= PieceFeatureCount) return;
        const auto &row = weights.feature_weights[index];
        for (std::size_t neuron = 0; neuron < data_.size(); ++neuron)
            data_[neuron] += static_cast<AccumulatorValue>(row[neuron]) * feature.value;
    }

    void PieceKingAccumulator::remove(const Features::Feature &feature, const PieceWeights &weights) noexcept
    {
        const std::size_t index = slot(feature);
        if (index >= PieceFeatureCount) return;
        const auto &row = weights.feature_weights[index];
        for (std::size_t neuron = 0; neuron < data_.size(); ++neuron)
            data_[neuron] -= static_cast<AccumulatorValue>(row[neuron]) * feature.value;
    }

    void PieceKingAccumulator::reconstruct(const Features::FeatureSet &features, const PieceWeights &weights,
                                           Features::PieceColor perspective) noexcept
    {
        reset(weights.bias);
        for (std::size_t i = 0; i < features.size; ++i)
            if (features.features[i].perspective == perspective) add(features.features[i], weights);
    }

    void PieceKingAccumulator::update(const Features::FeatureSet &added, const Features::FeatureSet &removed,
                                      const PieceWeights &weights, Features::PieceColor perspective) noexcept
    {
        for (std::size_t i = 0; i < removed.size; ++i)
            if (removed.features[i].perspective == perspective) remove(removed.features[i], weights);
        for (std::size_t i = 0; i < added.size; ++i)
            if (added.features[i].perspective == perspective) add(added.features[i], weights);
    }

} // namespace TAPA::Accumulator
