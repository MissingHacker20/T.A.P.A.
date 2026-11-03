// nnue.cpp
#include "nnue.h"
#include "../Model/model.h"
#include <array>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <exception>
#include <new>

namespace TAPA {

NNUEModel::NNUEModel(const Model::Model& model)
    : pawn(model.pawn_weights), piece(model.piece_weights), network(model.network_weights) {}

bool NNUE::load_model(const TAPA::Model::Model& model, std::string* error) {
    if (!model.piece_weights.valid() || model.input_size != TAPA::Model::InputSize || model.hidden1_size != TAPA::Model::Hidden1Size ||
        model.hidden2_size != TAPA::Model::Hidden2Size || model.output_size != TAPA::Model::OutputSize ||
        model.feature_index_version != TAPA::Model::FeatureIndexVersion ||
        model.pawn_structure_version != TAPA::Model::PawnStructureEncodingVersion ||
        model.network_schema_version != TAPA::Model::NetworkSchemaVersion ||
        model.network_weights.activation_limit <= 0 || model.network_weights.output_scale <= 0) {
        if (error) *error = "invalid model dimensions, weights, or numeric parameters";
        return false;
    }
    try {
        return load_model(std::make_shared<const NNUEModel>(model), error);
    } catch (const std::bad_alloc&) {
        if (error) *error = "out of memory while loading model";
        return false;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

bool NNUE::load_model(std::shared_ptr<const NNUEModel> model, std::string* error) {
    if (!model || !model->piece.valid() || model->network.activation_limit <= 0 || model->network.output_scale <= 0) {
        if (error) *error = "null or invalid NNUE model";
        return false;
    }
    std::unique_ptr<TAPA::Accumulator::PawnAccumulatorCache> replacement_cache;
    try {
        replacement_cache = std::make_unique<TAPA::Accumulator::PawnAccumulatorCache>(model->pawn);
    } catch (const std::bad_alloc&) {
        if (error) *error = "out of memory while preparing model cache";
        return false;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
    model_ = std::move(model);
    pawn_cache_ = std::move(replacement_cache);
    network_.load_weights(model_->network);
    current_state_ = State{};
    history_size = 0;
    history_overflowed_ = false;
    if (error) error->clear();
    return true;
}

TAPA::Accumulator::PawnCacheStats NNUE::pawn_cache_stats() const noexcept {
    return pawn_cache_ ? pawn_cache_->stats() : TAPA::Accumulator::PawnCacheStats{};
}

std::size_t NNUE::pawn_cache_insertions() const noexcept {
    return pawn_cache_ ? pawn_cache_->insertions() : 0;
}

std::size_t NNUE::pawn_cache_bytes() const noexcept {
    return sizeof(TAPA::Accumulator::PawnAccumulatorCache);
}

std::size_t NNUE::worker_state_bytes() const noexcept {
    return sizeof(current_state_) + sizeof(State) * HistoryCapacity + pawn_cache_bytes();
}

std::size_t NNUE::weights_bytes() noexcept {
    return sizeof(NNUEModel) + sizeof(std::array<TAPA::Accumulator::PieceWeight, TAPA::Accumulator::NeuronCount>) *
        TAPA::Accumulator::PieceFeatureCount;
}

std::size_t NNUE::owned_weight_bytes() const noexcept {
    return model_ ? weights_bytes() : 0;
}

std::size_t NNUE::model_bytes(const NNUEModel& model) noexcept {
    return sizeof(model) + model.piece.feature_weights.capacity() *
        sizeof(std::array<TAPA::Accumulator::PieceWeight, TAPA::Accumulator::NeuronCount>);
}

std::size_t NNUE::cache_entry_bytes() noexcept {
    return sizeof(TAPA::Accumulator::PawnAccumulator) + sizeof(TAPA::Features::PawnStructureKey) +
           sizeof(std::uint64_t) + sizeof(bool);
}

NNUE::NNUE()
    : history_(std::make_unique<State[]>(HistoryCapacity)) {
    history_size = 0;
}

NNUE::~NNUE() = default;
NNUE::NNUE(NNUE&&) noexcept = default;
NNUE& NNUE::operator=(NNUE&&) noexcept = default;

void NNUE::initialize(int32_t bias_value) {
    bias_ = bias_value;
    // Initialize accumulators with bias
    if (pawn_cache_) pawn_cache_->clear();
    current_state_.pawn_accumulator.reset(model_ ? model_->pawn.bias : 0);
    current_state_.piece_white.reset(model_ ? model_->piece.bias : 0);
    current_state_.piece_black.reset(model_ ? model_->piece.bias : 0);
    // Clear features
    current_state_.features.clear();
    current_state_.pawn_structure = 0;
    current_state_.position = {};
    current_state_.valid = false;
    history_size = 0;
    history_overflowed_ = false;
}

int32_t NNUE::evaluate(const Features::Position& pos) {
    if (!model_ || !pawn_cache_) { current_state_.valid = false; return 0; }
    if (!pos.is_valid()) {
        current_state_.valid = false;
        return 0;
    }
    TAPA::Features::generate_features(pos, current_state_.features);
    current_state_.pawn_structure = TAPA::Features::make_pawn_structure_key(pos);
    current_state_.pawn_accumulator = pawn_cache_->get(current_state_.pawn_structure);
    current_state_.piece_white.reconstruct(current_state_.features, model_->piece, Features::PieceColor::White);
    current_state_.piece_black.reconstruct(current_state_.features, model_->piece, Features::PieceColor::Black);
    refresh_network_accumulators();
    current_state_.position = pos;
    current_state_.valid = true;
    return evaluate_current();
}

int32_t NNUE::evaluate_current() const noexcept {
    if (!model_ || !current_state_.valid) return 0;
    return network_.forward_pass(current_state_.network_white, current_state_.network_black);
}

void NNUE::refresh_network_accumulators() noexcept {
    std::array<TAPA::Accumulator::AccumulatorValue, TAPA::Accumulator::NeuronCount> white{};
    std::array<TAPA::Accumulator::AccumulatorValue, TAPA::Accumulator::NeuronCount> black{};
    const std::int64_t combine_bias = model_ ? static_cast<std::int64_t>(model_->piece.bias) + bias_ : bias_;
    white.fill(static_cast<TAPA::Accumulator::AccumulatorValue>(std::clamp<std::int64_t>(combine_bias,
        std::numeric_limits<TAPA::Accumulator::AccumulatorValue>::min(),
        std::numeric_limits<TAPA::Accumulator::AccumulatorValue>::max())));
    black.fill(static_cast<TAPA::Accumulator::AccumulatorValue>(std::clamp<std::int64_t>(combine_bias,
        std::numeric_limits<TAPA::Accumulator::AccumulatorValue>::min(),
        std::numeric_limits<TAPA::Accumulator::AccumulatorValue>::max())));
    const auto& pawn = current_state_.pawn_accumulator.data();
    const auto& piece_white = current_state_.piece_white.data();
    const auto& piece_black = current_state_.piece_black.data();
    for (std::size_t i = 0; i < white.size(); ++i) {
        const auto sum_white = combine_bias + pawn[i] + piece_white[i];
        const auto sum_black = combine_bias + pawn[i] + piece_black[i];
        white[i] = static_cast<TAPA::Accumulator::AccumulatorValue>(std::clamp<std::int64_t>(sum_white,
            std::numeric_limits<TAPA::Accumulator::AccumulatorValue>::min(),
            std::numeric_limits<TAPA::Accumulator::AccumulatorValue>::max()));
        black[i] = static_cast<TAPA::Accumulator::AccumulatorValue>(std::clamp<std::int64_t>(sum_black,
            std::numeric_limits<TAPA::Accumulator::AccumulatorValue>::min(),
            std::numeric_limits<TAPA::Accumulator::AccumulatorValue>::max()));
    }
    current_state_.network_white = white;
    current_state_.network_black = black;
}

NNUE::UpdateStatus NNUE::update(const Features::Position& previous,
                                const Features::Position& current) noexcept {
    if (!model_) return UpdateStatus::InvalidState;
    if (!previous.is_valid() || !current.is_valid()) {
        current_state_.valid = false;
        return UpdateStatus::InvalidPosition;
    }
    if (!current_state_.valid || current_state_.position != previous) {
        (void)evaluate(previous);
        return UpdateStatus::InvalidState;
    }
    if (previous == current) return UpdateStatus::Unchanged;
    const Features::FeatureDelta delta = TAPA::Features::update_features(previous, current);
    if (!delta.valid) {
        current_state_.valid = false;
        return UpdateStatus::InvalidDelta;
    }
    return apply_delta(delta, current);
}

NNUE::UpdateStatus NNUE::apply_delta(const Features::FeatureDelta& delta, const Features::Position& current) noexcept {
    if (!delta.valid || !current.is_valid() || !pawn_cache_ || !model_) {
        current_state_.valid = false;
        return UpdateStatus::InvalidDelta;
    }
    if (current_state_.position == current) return UpdateStatus::Unchanged;
    if (delta.removed.size > delta.removed.features.size() || delta.added.size > delta.added.features.size()) {
        current_state_.valid = false;
        return UpdateStatus::InvalidDelta;
    }
    Features::FeatureSet next_features = current_state_.features;
    for (std::size_t i = 0; i < delta.removed.size; ++i)
        if (!next_features.remove(delta.removed.features[i])) {
            current_state_.valid = false;
            return UpdateStatus::InvalidDelta;
        }
    for (std::size_t i = 0; i < delta.added.size; ++i)
        if (!next_features.add(delta.added.features[i])) {
            current_state_.valid = false;
            return UpdateStatus::InvalidDelta;
        }

    if (current_state_.pawn_structure != delta.current_pawn_structure) {
        const TAPA::Accumulator::PawnAccumulator* cached = pawn_cache_->touch(delta.current_pawn_structure);
        if (cached != nullptr) {
            current_state_.pawn_accumulator = *cached;
        } else {
            current_state_.pawn_accumulator.update(current_state_.pawn_structure,
                                                   delta.current_pawn_structure, model_->pawn);
            pawn_cache_->insert(delta.current_pawn_structure, current_state_.pawn_accumulator);
        }
    } else {
        (void)pawn_cache_->touch(current_state_.pawn_structure); // reuse, record as a cache hit
    }
    for (std::size_t i = 0; i < delta.removed.size; ++i) {
        const auto& feature = delta.removed.features[i];
        if (feature.perspective == Features::PieceColor::White)
            current_state_.piece_white.remove(feature, model_->piece);
        else
            current_state_.piece_black.remove(feature, model_->piece);
    }
    for (std::size_t i = 0; i < delta.added.size; ++i) {
        const auto& feature = delta.added.features[i];
        if (feature.perspective == Features::PieceColor::White)
            current_state_.piece_white.add(feature, model_->piece);
        else
            current_state_.piece_black.add(feature, model_->piece);
    }
    current_state_.pawn_structure = delta.current_pawn_structure;
    current_state_.features = next_features;
    refresh_network_accumulators();
    current_state_.position = current;
    current_state_.valid = true;
    return UpdateStatus::Updated;
}

NNUE::UpdateStatus NNUE::update(const Features::Position& previous, const Features::Move& move,
                                const Features::Position& current) noexcept {
    if (!model_) return UpdateStatus::InvalidState;
    if (!previous.is_valid() || !current.is_valid()) {
        current_state_.valid = false;
        return UpdateStatus::InvalidPosition;
    }
    if (!current_state_.valid || current_state_.position != previous) {
        (void)evaluate(previous);
        return UpdateStatus::InvalidState;
    }
    const Features::FeatureDelta delta = Features::update_features(previous, move, current);
    if (!delta.valid) {
        current_state_.valid = false;
        return UpdateStatus::InvalidDelta;
    }
    return apply_delta(delta, current);
}

bool NNUE::push() noexcept {
    if (history_size >= HistoryCapacity) {
        history_overflowed_ = true;
        return false;
    }
    history_[history_size++] = current_state_;
    return true;
}

bool NNUE::pop() noexcept {
    if (history_overflowed_) {
        history_overflowed_ = false;
        history_size = 0;
        current_state_.valid = false;
        return false;
    }
    if (history_size == 0) return false;
    --history_size;
    current_state_ = history_[history_size];
    return true;
}

} // namespace TAPA