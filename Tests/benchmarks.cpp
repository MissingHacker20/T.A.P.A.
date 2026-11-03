#include "../Source Code/Features/features.h"
#include "../Source Code/Accumulator/pawn_accumulator.h"
#include "../Source Code/Accumulator/piece_accumulator.h"
#include "../Source Code/Accumulator/accumulator.h"
#include "../Source Code/Network/network.h"
#include "../Source Code/Network/network_model.h"
#include "../Source Code/NNUE/nnue.h"
#include "../Source Code/Model/model.h"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <array>
#include <algorithm>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t Iterations = 20000;
constexpr std::size_t BenchmarkRepeats = 7;
volatile std::int64_t benchmark_sink = 0;

struct TimingSummary { long long median_us = 0; long long min_us = 0; long long max_us = 0; };

template <typename Function>
long long measure_us(Function&& function) {
    const auto begin = Clock::now();
    function();
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - begin).count();
}

template <typename Function>
TimingSummary measure_repeated(Function&& function) {
    std::array<long long, BenchmarkRepeats> samples{};
    for (auto& sample : samples) sample = measure_us(function);
    std::sort(samples.begin(), samples.end());
    return {samples[samples.size() / 2], samples.front(), samples.back()};
}


TAPA::Features::Position benchmark_position() {
    using namespace TAPA::Features;
    Position position{};
    position.pieces[0] = {PieceType::King, PieceColor::White, 4};
    position.pieces[1] = {PieceType::King, PieceColor::Black, 60};
    position.pieces[2] = {PieceType::Pawn, PieceColor::White, 12};
    position.pieces[3] = {PieceType::Pawn, PieceColor::White, 27};
    position.pieces[4] = {PieceType::Pawn, PieceColor::Black, 52};
    position.pieces[5] = {PieceType::Pawn, PieceColor::Black, 36};
    position.pieces[6] = {PieceType::Knight, PieceColor::White, 18};
    position.pieces[7] = {PieceType::Bishop, PieceColor::Black, 45};
    position.num_pieces = 8;
    return position;
}

TAPA::Features::PawnStructureKey benchmark_key(std::size_t index) {
    const auto value = static_cast<TAPA::Features::PawnStructureKey>(index + 1);
    const auto other = static_cast<TAPA::Features::PawnStructureKey>(index * 17 + 3);
    return (value * 0x9e3779b97f4a7c15ULL) ^ (other << 32);
}
}

int main() {
    using namespace TAPA::Features;
    using namespace TAPA::Accumulator;

    PawnWeights weights;
    for (std::size_t bit = 0; bit < 64; ++bit)
        for (std::size_t neuron = 0; neuron < PawnNeuronCount; ++neuron)
            weights.bit_weights[bit][neuron] = static_cast<PawnWeight>((bit + neuron) % 7) - 3;

    const Position position = benchmark_position();
    FeatureSet features;
    const auto feature_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i) {
            generate_features(position, features);
            benchmark_sink += static_cast<std::int64_t>(features.size);
        }
    });
    FeatureSet move_added, move_removed;
    const Move knight_move{18, 33, PieceType::Knight, PieceColor::White};
    Position moved_position{};
    apply_move(position, knight_move, moved_position);
    const auto incremental_delta_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i) {
            const FeatureDelta move_delta = update_features(position, knight_move, moved_position);
            benchmark_sink += static_cast<std::int64_t>(move_delta.added.size + move_delta.removed.size);
        }
    });

    std::array<DecodedPawn, 32> decoded{};
    measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i) {
            const PawnStructureKey key = make_pawn_structure_key(position);
            benchmark_sink += static_cast<std::int64_t>(decode_pawn_structure_key(key, decoded));
        }
    });
    const PawnStructureKey old_key = make_pawn_structure_key(position);
    const PawnStructureKey new_key = old_key ^ (PawnStructureKey{1} << 4) ^ (PawnStructureKey{1} << 28);
    PawnAccumulator full;
    PawnAccumulator incremental;
    const auto rebuild_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i) {
            full.reconstruct(benchmark_key(i), weights);
            benchmark_sink += full.data()[i % PawnNeuronCount];
        }
    });
    incremental.reconstruct(old_key, weights);
    const auto incremental_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i) {
            incremental.update(old_key, new_key, weights);
            incremental.update(new_key, old_key, weights);
            benchmark_sink += incremental.data()[i % PawnNeuronCount];
        }
    });
    PawnAccumulatorCache incremental_cache(weights);
    incremental_cache.get(old_key);
    const auto cache_hit_plus_miss_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i) {
            const PawnStructureKey next_key = (i & 1) == 0 ? new_key : old_key;
            incremental_cache.get(next_key);
        }
    });

    PawnAccumulatorCache hit_cache(weights);
    hit_cache.get(old_key);
    const auto cache_hit_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i)
            benchmark_sink += hit_cache.get(old_key).data()[i % PawnNeuronCount];
    });
    PawnAccumulatorCache miss_cache(weights);
    const auto cache_miss_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i)
            benchmark_sink += miss_cache.get(benchmark_key(i)).data()[i % PawnNeuronCount];
    });
    static TAPA::Accumulator::PieceWeights piece_weights;
    TAPA::Accumulator::PieceKingAccumulator piece_accumulator;
    const auto piece_reconstruct_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i) {
            piece_accumulator.reconstruct(features, piece_weights, PieceColor::White);
            benchmark_sink += piece_accumulator.data()[i % TAPA::Accumulator::NeuronCount];
        }
    });
    FeatureSet empty_features;
    const auto piece_update_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i) {
            piece_accumulator.update(features, empty_features, piece_weights, PieceColor::White);
            piece_accumulator.update(empty_features, features, piece_weights, PieceColor::White);
        }
    });
    const FeatureDelta representative_delta = update_features(position, knight_move, moved_position);
    const auto full_delta_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i) {
            benchmark_sink += static_cast<std::int64_t>(update_features(position, moved_position).added.size);
        }
    });
    const auto delta_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i)
            benchmark_sink += static_cast<std::int64_t>(update_features(position, moved_position).added.size);
    });
    const auto piece_delta_update_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i) {
            piece_accumulator.update(representative_delta.added, representative_delta.removed, piece_weights, PieceColor::White);
            piece_accumulator.update(representative_delta.removed, representative_delta.added, piece_weights, PieceColor::White);
        }
    });

    Accumulator white;
    Accumulator black;
    white.initialize(0);
    black.initialize(0);
    TAPA::Network::Network network;
    const auto network_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i)
            benchmark_sink += network.forward_pass(white, black);
    });

    TAPA::Model::Model benchmark_model;
    std::shared_ptr<const TAPA::NNUEModel> shared_model = std::make_shared<const TAPA::NNUEModel>(benchmark_model);
    TAPA::NNUE nnue;
    (void)nnue.load_model(shared_model);
    nnue.initialize(0);
    const auto nnue_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i)
            benchmark_sink += nnue.evaluate(position);
    });
    (void)nnue.evaluate(position);
    const auto evaluate_current_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i)
            benchmark_sink += nnue.evaluate_current();
    });
    Position updated_position{};
    const bool update_possible = apply_move(position, knight_move, updated_position);
    const auto nnue_incremental_us = update_possible ? measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i) {
            nnue.update(position, knight_move, updated_position);
            nnue.update(updated_position, knight_move, position);
        }
    }) : 0;
    const auto forward_only = measure_repeated([&] {
        for (std::size_t i = 0; i < Iterations; ++i)
            benchmark_sink += network.forward_pass(white, black);
    });
    const auto reconstruct_only = measure_repeated([&] {
        for (std::size_t i = 0; i < Iterations; ++i)
            full.reconstruct(benchmark_key(i), weights);
    });
    const auto current_only = measure_repeated([&] {
        for (std::size_t i = 0; i < Iterations; ++i)
            benchmark_sink += nnue.evaluate_current();
    });
    std::vector<long long> worker_times;
    for (std::size_t worker_count : {1u, 2u, 4u}) {
        std::vector<std::thread> workers;
        const auto elapsed = measure_us([&] {
            for (std::size_t worker = 0; worker < worker_count; ++worker)
                workers.emplace_back([&] {
                    TAPA::NNUE local;
                    for (std::size_t i = 0; i < Iterations / worker_count; ++i)
                        benchmark_sink += local.evaluate(position);
                });
            for (auto& worker : workers) worker.join();
        });
        worker_times.push_back(elapsed);
    }

    const auto hit_stats = hit_cache.stats();
    std::cout << "benchmark_iterations=" << Iterations << " repeats=" << BenchmarkRepeats << '\n'
              << "network_forward_us=" << network_us << " median/min/max=" << forward_only.median_us << '/' << forward_only.min_us << '/' << forward_only.max_us << '\n'
              << "nnue_evaluate_us=" << nnue_us << '\n'
              << "nnue_evaluate_current_us=" << evaluate_current_us << " median/min/max=" << current_only.median_us << '/' << current_only.min_us << '/' << current_only.max_us << '\n'
              << "nnue_incremental_update_pair_us=" << nnue_incremental_us << '\n'
              << "pawn_full_reconstruction_us=" << rebuild_us << " median/min/max=" << reconstruct_only.median_us << '/' << reconstruct_only.min_us << '/' << reconstruct_only.max_us << '\n'
              << "pawn_cache_hit_us=" << cache_hit_us << " hits=" << hit_stats.hits << " misses=" << hit_stats.misses << '\n'
              << "pawn_cache_miss_us=" << cache_miss_us << " hits=" << miss_cache.stats().hits << " misses=" << miss_cache.stats().misses << '\n'
              << "workers_1_2_4_elapsed_us=" << worker_times[0] << '/' << worker_times[1] << '/' << worker_times[2] << '\n'
              << "full_position_delta_us=" << full_delta_us << '\n'
              << "features_reconstruction_us=" << feature_us << '\n'
              << "pawn_incremental_us=" << incremental_us << '\n'
              << "piece_reconstruction_us=" << piece_reconstruct_us << '\n'
              << "piece_incremental_us=" << piece_update_us << '\n'
              << "full_feature_delta_comparison_us=" << delta_us << '\n'
              << "move_incremental_feature_delta_us=" << incremental_delta_us << '\n'
              << "piece_move_delta_incremental_roundtrip_us=" << piece_delta_update_us << '\n'
              << "pawn_cache_hit_miss_pair_us=" << cache_hit_plus_miss_us << '\n'
              << "sizeof(PawnStructureKey)=" << sizeof(PawnStructureKey)
              << " sizeof(Feature)=" << sizeof(TAPA::Features::Feature)
              << " sizeof(PawnAccumulator)=" << sizeof(PawnAccumulator)
              << " sizeof(PawnCache)=" << sizeof(PawnAccumulatorCache)
              << " pawn_weights_bytes=" << sizeof(PawnWeights)
              << " piece_weights_bytes=" << sizeof(TAPA::Accumulator::PieceWeights) +
                    TAPA::Accumulator::PieceFeatureCount * sizeof(std::array<TAPA::Accumulator::PieceWeight, TAPA::Accumulator::NeuronCount>)
              << " sizeof(Accumulator)=" << sizeof(Accumulator)
              << " sizeof(NNUE)=" << sizeof(TAPA::NNUE)
              << " nnue_worker_state_bytes=" << nnue.worker_state_bytes()
              << " nnue_owned_weight_estimate_bytes=" << nnue.owned_weight_bytes()
              << " nnue_shared_model_bytes=" << TAPA::NNUE::model_bytes(*shared_model)
              << " network_weights_bytes=" << sizeof(TAPA::Network::NetworkWeights)
              << " nnue_feature_weights_bytes=" << TAPA::NNUE::weights_bytes()
              << " nnue_cache_bytes=" << nnue.pawn_cache_bytes()
              << " cache_entry_bytes=" << TAPA::NNUE::cache_entry_bytes()
              << " sink=" << benchmark_sink << '\n';
    return 0;
}
