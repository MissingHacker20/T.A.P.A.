#include "accumulator.h"

namespace TAPA {
namespace Accumulator {

void Accumulator::reconstruct(const Features::FeatureSet& fs) {
    (void)fs;
    data.fill(bias);
}

void Accumulator::update_incremental(const Features::FeatureSet& added,
                                     const Features::FeatureSet& removed) {
    (void)added;
    (void)removed;
}

} // namespace Accumulator
} // namespace TAPA