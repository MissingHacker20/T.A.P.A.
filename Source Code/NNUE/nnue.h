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

namespace TAPA {
namespace Model { struct Model; }
struct Weights;

class NNUE {
public:
    NNUE();
    ~NNUE();
    NNUE(const NNUE&) = delete;
    NNUE& operator=(const NNUE&) = delete;
    NNUE(NNUE&&) noexcept;
    NNUE& operator=(NNUE&&) noexcept;
    // Initialize the bias used when combining the feature embeddings.
    void initialize(int32_t bias_value = 0);
    void load_model(const TAPA::Model::Model& model);
    // Reconstruct the owned state from a position and return its score.
    int32_t evaluate(const Features::Position& pos);
    // Evaluate the currently owned state without rebuilding it.
    int32_t evaluate_current() const noexcept;
    // Incrementally update from the supplied before/after position snapshots.
    void update(const Features::Position& previous,
                const Features::Position& current);
    void update(const Features::Position& previous, const Features::Move& move,
                const Features::Position& current);
    // Push current state onto stack (for undo)
    void push();
    // Pop previous state and restore
    void pop();
    TAPA::Accumulator::PawnCacheStats pawn_cache_stats() const noexcept;
    std::size_t pawn_cache_bytes() const noexcept;
    static std::size_t weights_bytes() noexcept;
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
    std::unique_ptr<Weights> weights_;
    std::unique_ptr<TAPA::Accumulator::PawnAccumulatorCache> pawn_cache_;
    void refresh_network_accumulators() noexcept;
    void apply_delta(const Features::FeatureDelta& delta, const Features::Position& current);
    TAPA::Network::Network network_;
};

} // namespace TAPA