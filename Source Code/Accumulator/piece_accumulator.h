#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "accumulator.h"

namespace TAPA::Accumulator {

constexpr std::size_t PieceNeuronCount = NeuronCount;
using PieceWeight = std::int16_t;
constexpr std::size_t PieceFeatureCount = 64 * 2 * 6 * 64;
static_assert(PieceFeatureCount == 49152);
// Maximum active features per perspective on a 64-square board is 63. A single feature
// contributes int16_weight * {-1,0,1}; the sum plus int32 bias fits int64.
constexpr std::size_t MaxActivePieceFeatures = 63;
struct PieceWeights {
    AccumulatorValue bias = 0;
    // Canonical order is the Feature::index order.
    std::vector<std::array<PieceWeight, NeuronCount>> feature_weights;
    PieceWeights() : feature_weights(PieceFeatureCount) {}
    bool valid() const noexcept { return feature_weights.size() == PieceFeatureCount; }
};

class PieceKingAccumulator {
public:
    void reset(AccumulatorValue bias = 0) noexcept;
    void reconstruct(const Features::FeatureSet& features, const PieceWeights& weights,
                     Features::PieceColor perspective) noexcept;
    void update(const Features::FeatureSet& added, const Features::FeatureSet& removed,
                const PieceWeights& weights, Features::PieceColor perspective) noexcept;
    void add(const Features::Feature& feature, const PieceWeights& weights) noexcept;
    void remove(const Features::Feature& feature, const PieceWeights& weights) noexcept;
    const std::array<AccumulatorValue, NeuronCount>& data() const noexcept { return data_; }

private:
    std::array<AccumulatorValue, NeuronCount> data_{};
    std::array<std::int64_t, NeuronCount> exact_data_{};
    static std::size_t slot(const Features::Feature& feature) noexcept;
};

} // namespace TAPA::Accumulator
