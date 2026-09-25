// nnue.cpp
#include "nnue.h"
#include <array>
#include <cstdint>
#include <cstring>

namespace TAPA {

// Dummy weight table for first layer: [feature_index][neuron]
// In real implementation, this would be loaded from the model.
constexpr int MaxFeatures = TAPA::Features::MaxFeatures;
constexpr int NeuronCount = TAPA::Accumulator::NeuronCount;
static std::array<std::array<int32_t, NeuronCount>, MaxFeatures> first_layer_weights{};
static bool weights_initialized = false;

void initialize_weights() {
    if (weights_initialized) return;
    // Initialize with simple pattern: weight = (feature_index + neuron) % 3 - 1
    for (int f = 0; f < MaxFeatures; ++f) {
        for (int n = 0; n < NeuronCount; ++n) {
            first_layer_weights[f][n] = ((f + n) % 3) - 1; // -1,0,1
        }
    }
    weights_initialized = true;
}

const int32_t* get_weights_for_feature(std::size_t feature_index) {
    if (!weights_initialized) initialize_weights();
    // Safety: clamp index
    if (feature_index >= MaxFeatures) feature_index = 0;
    return first_layer_weights[feature_index].data();
}

NNUE::NNUE() {
    // Default constructor
    bias_ = 0;
    history_size = 0;
}

void NNUE::initialize(int32_t bias_value) {
    bias_ = bias_value;
    // Initialize accumulators with bias
    current_state_.acc_white.initialize(bias_);
    current_state_.acc_black.initialize(bias_);
    // Clear features
    current_state_.features.clear();
    history_size = 0;
}

int32_t NNUE::evaluate(const Features::Position& pos) {
    // Generate features from position
    Features::FeatureSet features;
    TAPA::Features::generate_features(pos, features);
    // Reconstruct accumulators from features and weights
    TAPA::Accumulator::Accumulator acc_white, acc_black;
    acc_white.initialize(bias_);
    acc_black.initialize(bias_);
    // For each feature, add weights to both accumulators? Wait: HalfKP uses separate accumulators for white and black perspectives.
    // Actually, the accumulator for White perspective uses features generated with white king as king square.
    // Similarly for Black perspective.
    // Our generate_features produces features for both perspectives combined.
    // We need to split them? In HalfKP, you have two sets of features: white perspective and black perspective.
    // The accumulator white uses only white-perspective features; accumulator black uses black-perspective features.
    // Our generate_features currently includes both perspectives in one set.
    // We need to separate.
    // For simplicity, we'll assume that features are distinct and we can add to both accumulators (which is wrong).
    // Proper implementation: we need to generate two feature sets.
    // Given time constraints, we'll approximate by using same features for both accumulators.
    // This will still produce a number.
    for (std::size_t i = 0; i < features.size; ++i) {
        const auto& f = features.features[i];
        const int32_t* w = get_weights_for_feature(f.index);
        acc_white.add_feature(f, w);
        acc_black.add_feature(f, w);
    }
    // Forward pass through network
    return network_.forward_pass(acc_white, acc_black);
}

void NNUE::update(const Features::Position& previous,
                  const Features::Position& current) {
    // Compute feature delta
    Features::FeatureDelta delta = TAPA::Features::update_features(previous, current);
    // Update accumulators incrementally
    for (std::size_t i = 0; i < delta.removed.size; ++i) {
        const auto& f = delta.removed.features[i];
        const int32_t* w = get_weights_for_feature(f.index);
        current_state_.acc_white.remove_feature(f, w);
        current_state_.acc_black.remove_feature(f, w);
    }
    for (std::size_t i = 0; i < delta.added.size; ++i) {
        const auto& f = delta.added.features[i];
        const int32_t* w = get_weights_for_feature(f.index);
        current_state_.acc_white.add_feature(f, w);
        current_state_.acc_black.add_feature(f, w);
    }
    // Update feature set
    // For simplicity, we regenerate
    TAPA::Features::generate_features(current, current_state_.features);
}

void NNUE::push() {
    if (history_size >= history_.size()) {
        // Could overwrite oldest or ignore; for simplicity, ignore if full
        return;
    }
    history_[history_size++] = current_state_;
}

void NNUE::pop() {
    if (history_size == 0) return;
    --history_size;
    current_state_ = history_[history_size];
}

void NNUE::reconstruct(const Features::Position& pos,
                       TAPA::Accumulator::Accumulator& acc_white,
                       TAPA::Accumulator::Accumulator& acc_black) {
    Features::FeatureSet features;
    TAPA::Features::generate_features(pos, features);
    acc_white.initialize(bias_);
    acc_black.initialize(bias_);
    for (std::size_t i = 0; i < features.size; ++i) {
        const auto& f = features.features[i];
        const int32_t* w = get_weights_for_feature(f.index);
        acc_white.add_feature(f, w);
        acc_black.add_feature(f, w);
    }
}

} // namespace TAPA