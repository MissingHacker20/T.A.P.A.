// accumulator.cpp
#include "accumulator.h"

namespace TAPA {
namespace Accumulator {

void Accumulator::accumulator.h:1:1: error: expected unqualified-id before '.' token
 // accumulator.cpp
#include "accumulator.h"

namespace TAPA {
namespace Accumulator {

void Accumulator::initialize(AccumulatorValue bias_val) {
    bias = bias_val;
    data.fill(bias_val);
}

// Placeholder: full reconstruction
void Accumulator::reconstruct(const Features::FeatureSet& fs, /* weights */) {
    // TODO: set data to bias, then for each feature add weights[feature.index]
}

// Placeholder: incremental update
void Accumulator::update_incremental(const Features::FeatureSet& added, const Features::FeatureSet& removed, /* weights */) {
    // TODO: for each removed feature subtract weights, for each added feature add weights
}

} // namespace Accumulator
} // namespace TAPA