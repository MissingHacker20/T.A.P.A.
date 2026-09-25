// nnue.h
#pragma once

#include "../Features/features.h"
#include "../Accumulator/accumulator.h"
#include "../Network/network.h"

namespace TAPA {

// Forward declaration for weight access
const int32_t* get_weights_for_feature(std::size_t feature_index);

class NNUE {
public:
    NNUE();
    // Initialize with bias value (could be loaded from model)
    void initialize(int32_t bias_value = 0);
    // Evaluate a position and return score
    int32_t evaluate(const Features::Position& pos);
    // Update internal state after a move (incremental)
    // For simplicity, we accept previous and current positions
    void update(const Features::Position& previous,
                const Features::Position& current);
    // Push current state onto stack (for undo)
    void push();
    // Pop previous state and restore
    void pop();
    // Reconstruct accumulators from scratch for a position
    void reconstruct(const Features::Position& pos,
                     TAPA::Accumulator::Accumulator& acc_white,
                     TAPA::Accumulator::Accumulator& acc_black);

private:
    // Internal state for push/pop
    struct State {
        Features::FeatureSet features;
        TAPA::Accumulator::Accumulator acc_white;
        TAPA::Accumulator::Accumulator acc_black;
    };
    State current_state_;
    std::array<State, 64> history_; // fixed size stack
    std::size_t history_size = 0;
    int32_t bias_ = 0;
    TAPA::Network::Network network_;
};

} // namespace TAPA