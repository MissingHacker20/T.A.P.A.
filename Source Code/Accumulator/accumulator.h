// accumulator.h
#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include "../Features/features.h"

namespace TAPA {
namespace Accumulator {

// Architectural embedding width v3; also stored in the model header so a
// mismatching model is rejected at load time.
constexpr std::size_t NeuronCount = 256;
// Stored values saturate to int32_t. Products/sums are formed in int64_t;
// each feature coefficient is restricted to {-1,0,1}.
using AccumulatorValue = std::int32_t;

struct Accumulator {
    // DEPRECATED / legacy path: unlike PawnAccumulator and PieceKingAccumulator,
    // which keep exact int64_t sums and saturate only when handing data to the
    // network, this legacy accumulator saturates to int32_t after EVERY
    // add/remove operation. Cancellation is therefore not preserved across a
    // sequence of updates. NNUE::evaluate() does NOT use this type; it is kept
    // only for small helper tests and legacy callers. For production NNUE state
    // use PawnAccumulator / PieceKingAccumulator.
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
    const std::array<AccumulatorValue, NeuronCount>& get_data() const noexcept { return data; }
    std::array<AccumulatorValue, NeuronCount>& get_data() noexcept { return data; }
    void reset(AccumulatorValue bias_value = 0) noexcept { initialize(bias_value); }
};

} // namespace Accumulator
} // namespace TAPA