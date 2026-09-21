// nnue.cpp
#include "nnue.h"

namespace TAPA {

bool NNUE::initialize() {
    // Load model weights, biases, etc.
    // For now, return true.
    return true;
}

int32_t NNUE::evaluate(const Features::FeatureSet& features,
                       const Accumulator::Accumulator& acc_white,
                       const Accumulator::Accumulator& acc_black) {
    // For simplicity, ignore passed features and use internal accumulators
    // In real usage, we would update accumulators based on features delta.
    return network_.forward_pass(acc_white, acc_black);
}

void NNUE::update(const Features::FeatureSet& added,
                  const Features::FeatureSet& removed,
                  Accumulator::Accumulator& acc_white,
                  Accumulator::Accumulator& acc_black) {
    // For each removed feature: subtract weights from accumulators
    // For each added feature: add weights to accumulators
    // Placeholder:
    (void)added; (void)removed; (void)acc_white; (void)acc_black;
}

void NNUE::push() {
    // Save current state to history stack
}

void NNUE::pop() {
    // Restore previous state from history stack
}

void NNUE::reconstruct(const Features::FeatureSet& features,
                       Accumulator::Accumulator& acc_white,
                       Accumulator::Accumulator& acc_black) {
    // Recompute accumulators from scratch using features and weights
    // Placeholder:
    (void)features; (void)acc_white; (void)acc_black;
}

} // namespace TAPA