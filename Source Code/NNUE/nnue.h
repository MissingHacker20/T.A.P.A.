// nnue.h
#pragma once

#include "features.h"
#include "accumulator.h"
#include "network.h"

namespace TAPA {

class NNUE {
public:
    // Initialize with model parameters (to be implemented)
    bool initialize();

    // Evaluate current position
    int32_t evaluate(const Features::FeatureSet& features,
                     const Accumulator::Accumulator& acc_white,
                     const Accumulator::Accumulator& acc_black);

    // Update accumulators incrementally given added/removed features
    void update(const Features::FeatureSet& added,
                const Features::FeatureSet& removed,
                Accumulator::Accumulator& acc_white,
                Accumulator::Accumulator& acc_black);

    // Push current state (for undo)
    void push();

    // Pop to previous state
    void pop();

    // Full reconstruction from scratch
    void reconstruct(const Features::FeatureSet& features,
                     Accumulator::Accumulator& acc_white,
                     Accumulator::Accumulator& acc_black);

private:
    // Internal state: features, accumulators, network, history
    Features::FeatureSet current_features_;
    Accumulator::Accumulator acc_white_;
    Accumulator::Accumulator acc_black_;
    Network::Network network_;
    // TODO: history stack for push/pop
};

} // namespace TAPA