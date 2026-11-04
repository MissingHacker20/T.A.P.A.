// nnue.h
#pragma once

#include "../Features/features.h"
#include "../Features/move.h"
#include "../Accumulator/pawn_accumulator.h"
#include "../Accumulator/accumulator.h"
#include "../Accumulator/piece_accumulator.h"
#include "../Network/network.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include "../Network/network_model.h"
#include "../Model/model.h"

namespace TAPA {
struct NNUEModel {
    Accumulator::PawnWeights pawn;
    Accumulator::PieceWeights piece;
    Network::NetworkWeights network;
    explicit NNUEModel(const Model::Model& model);
};

class NNUE {
public:
    NNUE();
    ~NNUE();
    NNUE(const NNUE&) = delete;
    NNUE& operator=(const NNUE&) = delete;
    NNUE(NNUE&&) noexcept;
    NNUE& operator=(NNUE&&) noexcept;
    // Initialize the bias used when combining the feature embeddings.
    void initialize(int32_t bias_value = 0); // Sets the embedding-combination bias; does not load a model.
    bool load_model(const TAPA::Model::Model& model, std::string* error = nullptr);
    // Failed loads leave the current model and position state unchanged.
    bool load_model(std::shared_ptr<const NNUEModel> model, std::string* error = nullptr);
    bool model_loaded() const noexcept { return static_cast<bool>(model_); }
    // Reconstruct the owned state from a position and return its score.
    // Returns zero without a model or when a position cannot be represented by the feature schema.
    // Chess legality is the host engine's responsibility.
    int32_t evaluate(const Features::Position& pos);
    // Evaluate the currently owned state without rebuilding it.
    int32_t evaluate_current() const noexcept;
    enum class UpdateStatus { Updated, Unchanged, InvalidState, InvalidPosition, InvalidDelta, ResourceFailure };
    // Incrementally update; status distinguishes a no-op from rejected or failed updates.
    UpdateStatus update(const Features::Position& previous,
                        const Features::Position& current) noexcept;
    UpdateStatus update(const Features::Position& previous, const Features::Move& move,
                        const Features::Position& current) noexcept;
    // Push current state onto stack (for undo)
    bool push() noexcept;
    // Pop previous state and restore; false means empty history or prior overflow.
    bool pop() noexcept;
    TAPA::Accumulator::PawnCacheStats pawn_cache_stats() const noexcept;
    std::size_t pawn_cache_insertions() const noexcept;
    std::size_t pawn_cache_bytes() const noexcept;
    std::size_t worker_state_bytes() const noexcept;
    static std::size_t weights_bytes() noexcept;
    std::size_t owned_weight_bytes() const noexcept;
    static std::size_t model_bytes(const NNUEModel& model) noexcept;
    static std::size_t cache_entry_bytes() noexcept;

private:
    // Internal state for push/pop
    struct State {
        Features::FeatureSet features;
        Features::PawnStructureKey pawn_structure = 0;
        Features::Position position{};
        bool valid = false;
        TAPA::Accumulator::PawnAccumulator pawn_accumulator;
        std::array<TAPA::Accumulator::AccumulatorValue, TAPA::Accumulator::NeuronCount> network_white{};
        std::array<TAPA::Accumulator::AccumulatorValue, TAPA::Accumulator::NeuronCount> network_black{};
        TAPA::Accumulator::PieceKingAccumulator piece_white;
        TAPA::Accumulator::PieceKingAccumulator piece_black;
    };
    State current_state_;
    static constexpr std::size_t HistoryCapacity = 64;
    bool history_overflowed_ = false;
    std::unique_ptr<State[]> history_;
    std::size_t history_size = 0;
    int32_t bias_ = 0;
    std::shared_ptr<const NNUEModel> model_;
    std::unique_ptr<TAPA::Accumulator::PawnAccumulatorCache> pawn_cache_;
    void refresh_network_accumulators() noexcept;
    UpdateStatus apply_delta(const Features::FeatureDelta& delta, const Features::Position& current) noexcept;
    TAPA::Network::Network network_;
};

} // namespace TAPA