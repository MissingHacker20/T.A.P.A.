// features.cpp
#include "features.h"

namespace TAPA {
namespace Features {

void FeatureSet::add(const Feature& f) {
    if (size < MaxFeatures) {
        features[size++] = f;
    }
    // else: handle overflow (assert or ignore)
}

void FeatureSet::remove(int index) {
    // linear search; could be optimized
    for (int i = 0; i < size; ++i) {
        if (features[i].index == index) {
            // swap with last and decrement size
            features[i] = features[size - 1];
            --size;
            return;
        }
    }
    // not found
}

} // namespace Features
} // namespace TAPA