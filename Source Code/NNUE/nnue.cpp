// nnue.cpp
#include "nnue.h"
#include "../Model/model.h"
#include <array>
#include <cstdint>

namespace TAPA {

struct Weights {
    TAPA::Accumulator::PawnWeights pawn{};
    TAPA::Accumulator::PieceWeights piece{};
};

Weights make_default_weights() {
    Weights weights;
    for (std::size_t bit = 0; bit < 64; ++bit)
        for (std::size_t neuron = 0; neuron < TAPA::Accumulator::PawnNeuronCount; ++neuron)
            weights.pawn.bit_weights[bit][neuron] =
                static_cast<TAPA::Accumulator::PawnWeight>(static_cast<int>((bit + neuron) % 5) - 2);
    for (std::size_t feature = 0; feature < TAPA::Accumulator::PieceFeatureCount; ++feature)
        for (std::size_t neuron = 0; neuron < TAPA::Accumulator::NeuronCount; ++neuron)
            weights.piece.feature_weights[feature][neuron] =
                static_cast<TAPA::Accumulator::PieceWeight>(static_cast<int>((feature + neuron) % 7) - 3);
    return weights;
}

void NNUE::load_model(const TAPA::Model::Model& model) {
    if (model.input_size != TAPA::Model::InputSize || model.hidden1_size != TAPA::Model::Hidden1Size ||
        model.hidden2_size != TAPA::Model::Hidden2Size || model.output_size != TAPA::Model::OutputSize ||
        model.feature_index_version != TAPA::Model::FeatureIndexVersion ||
        model.pawn_structure_version != TAPA::Model::PawnStructureEncodingVersion ||
        model.network_weights.activation_limit <= 0 || model.network_weights.output_scale <= 0)
        return;
    auto replacement = std::make_unique<Weights>();
    replacement->pawn = model.pawn_weights;
    replacement->piece = model.piece_weights;
    auto replacement_cache = std::make_unique<TAPA::Accumulator::PawnAccumulatorCache>(replacement->pawn);
    auto replacement_network = TAPA::Network::Network(model.network_weights);
    weights_ = std::move(replacement);
    pawn_cache_ = std::move(replacement_cache);
    network_ = replacement_network;
    current_state_.valid = false;
    history_size = 0;
    history_overflowed_ = false;
}

TAPA::Accumulator::PawnCacheStats NNUE::pawn_cache_stats() const noexcept {
    return pawn_cache_->stats();
}

std::size_t NNUE::pawn_cache_bytes() const noexcept {
    return sizeof(TAPA::Accumulator::PawnAccumulatorCache);
}

std::size_t NNUE::weights_bytes() noexcept {
    return sizeof(Weights);
}

std::size_t NNUE::cache_entry_bytes() noexcept {
    return sizeof(TAPA::Accumulator::PawnAccumulator) + sizeof(TAPA::Features::PawnStructureKey) +
           sizeof(std::uint64_t) + sizeof(bool);
}

NNUE::NNUE()
    : history_(std::make_unique<State[]>(HistoryCapacity)),
      weights_(std::make_unique<Weights>(make_default_weights())),
      pawn_cache_(std::make_unique<TAPA::Accumulator::PawnAccumulatorCache>(weights_->pawn)) {
    history_size = 0;
}

NNUE::~NNUE() = default;
NNUE::NNUE(NNUE&&) noexcept = default;
NNUE& NNUE::operator=(NNUE&&) noexcept = default;

void NNUE::initialize(int32_t bias_value) {
    bias_ = bias_value;
    // Initialize accumulators with bias
    pawn_cache_->clear();
    current_state_.pawn_accumulator.reset(weights_->pawn.bias);
    current_state_.piece_white.reset(weights_->piece.bias);
    current_state_.piece_black.reset(weights_->piece.bias);
    // Clear features
    current_state_.features.clear();
    current_state_.pawn_structure = 0;
    current_state_.position = {};
    current_state_.valid = false;
    history_size = 0;
    history_overflowed_ = false;
}

int32_t NNUE::evaluate(const Features::Position& pos) {
    if (!pos.is_valid()) {
        current_state_.valid = false;
        return 0;
    }
    TAPA::Features::generate_features(pos, current_state_.features);
    current_state_.pawn_structure = TAPA::Features::make_pawn_structure_key(pos);
    current_state_.pawn_accumulator = pawn_cache_->get(current_state_.pawn_structure);
    current_state_.piece_white.reconstruct(current_state_.features, weights_->piece, Features::PieceColor::White);
    current_state_.piece_black.reconstruct(current_state_.features, weights_->piece, Features::PieceColor::Black);
    refresh_network_accumulators();
    current_state_.position = pos;
    current_state_.valid = true;
    return evaluate_current();
}

int32_t NNUE::evaluate_current() const noexcept {
    if (!current_state_.valid) return 0;
    return network_.forward_pass(current_state_.network_white, current_state_.network_black);
}

void NNUE::refresh_network_accumulators() noexcept {
    std::array<TAPA::Accumulator::AccumulatorValue, TAPA::Accumulator::NeuronCount> white{};
    std::array<TAPA::Accumulator::AccumulatorValue, TAPA::Accumulator::NeuronCount> black{};
    white.fill(bias_);
    black.fill(bias_);
    const auto& pawn = current_state_.pawn_accumulator.data();
    const auto& piece_white = current_state_.piece_white.data();
    const auto& piece_black = current_state_.piece_black.data();
    for (std::size_t i = 0; i < white.size(); ++i) {
        white[i] += pawn[i] + piece_white[i];
        black[i] += pawn[i] + piece_black[i];
    }
    current_state_.network_white = white;
    current_state_.network_black = black;
}

void NNUE::update(const Features::Position& previous,
                  const Features::Position& current) {
    if (!current_state_.valid || current_state_.position != previous) {
        (void)evaluate(previous);
        return;
    }
    const Features::FeatureDelta delta = TAPA::Features::update_features(previous, current);
    if (!delta.valid) return;
    apply_delta(delta, current);
}

void NNUE::apply_delta(const Features::FeatureDelta& delta, const Features::Position& current) {
    Features::FeatureSet next_features = current_state_.features;
    for (std::size_t i = 0; i < delta.removed.size; ++i)
        if (!next_features.remove(delta.removed.features[i])) return;
    for (std::size_t i = 0; i < delta.added.size; ++i)
        if (!next_features.add(delta.added.features[i])) return;

    if (current_state_.pawn_structure != delta.current_pawn_structure) {
        const TAPA::Accumulator::PawnAccumulator* cached = pawn_cache_->touch(delta.current_pawn_structure);
        if (cached != nullptr) {
            current_state_.pawn_accumulator = *cached;
        } else {
            current_state_.pawn_accumulator.update(current_state_.pawn_structure,
                                                   delta.current_pawn_structure, weights_->pawn);
        }
    } else {
        (void)pawn_cache_->touch(current_state_.pawn_structure); // reuse, record as a cache hit
    }
    for (std::size_t i = 0; i < delta.removed.size; ++i) {
        const auto& feature = delta.removed.features[i];
        if (feature.perspective == Features::PieceColor::White)
            current_state_.piece_white.remove(feature, weights_->piece);
        else
            current_state_.piece_black.remove(feature, weights_->piece);
    }
    for (std::size_t i = 0; i < delta.added.size; ++i) {
        const auto& feature = delta.added.features[i];
        if (feature.perspective == Features::PieceColor::White)
            current_state_.piece_white.add(feature, weights_->piece);
        else
            current_state_.piece_black.add(feature, weights_->piece);
    }
    current_state_.pawn_structure = delta.current_pawn_structure;
    current_state_.features = next_features;
    refresh_network_accumulators();
    current_state_.position = current;
    current_state_.valid = true;
}

void NNUE::update(const Features::Position& previous, const Features::Move& move,
                  const Features::Position& current) {
    if (!current_state_.valid || current_state_.position != previous) {
        (void)evaluate(previous);
        return;
    }
    const Features::FeatureDelta delta = Features::update_features(previous, move, current);
    if (!delta.valid) return;
    apply_delta(delta, current);
}

void NNUE::push() {
    if (history_size >= HistoryCapacity) {
        history_overflowed_ = true;
        return;
    }
    history_[history_size++] = current_state_;
}

void NNUE::pop() {
    if (history_overflowed_) {
        history_overflowed_ = false;
        history_size = 0;
        current_state_.valid = false;
        return;
    }
    if (history_size == 0) return;
    --history_size;
    current_state_ = history_[history_size];
}

} // namespace TAPA