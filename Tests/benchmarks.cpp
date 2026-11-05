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
#include <atomic>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t Iterations = 20000;
constexpr std::size_t BenchmarkRepeats = 7;
std::atomic<std::int64_t> benchmark_sink{0};
std::atomic<bool> benchmark_failure{false};

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

// Fill a default model with small deterministic non-zero values so NNUE
// evaluations return real numbers instead of a constant zero.
void fill_benchmark_model(TAPA::Model::Model& model) {
    using namespace TAPA::Accumulator;
    model.pawn_weights.bias = 7;
    model.piece_weights.bias = -5;
    for (std::size_t bit = 0; bit < 64; ++bit)
        for (std::size_t neuron = 0; neuron < PawnNeuronCount; ++neuron)
            model.pawn_weights.bit_weights[bit][neuron] =
                static_cast<PawnWeight>(static_cast<int>((bit * 31 + neuron * 7) % 11) - 5);
    for (std::size_t feature = 0; feature < PieceFeatureCount; ++feature)
        for (std::size_t neuron = 0; neuron < NeuronCount; ++neuron)
            model.piece_weights.feature_weights[feature][neuron] =
                static_cast<PieceWeight>(static_cast<int>((feature * 13 + neuron * 3) % 9) - 4);
    auto& layer1 = model.network_weights.layer1;
    std::size_t value_index = 0;
    for (auto& row : layer1.weights)
        for (auto& value : row) value = static_cast<std::int32_t>(value_index++ % 7) - 3;
    value_index = 0;
    for (auto& value : layer1.bias) value = static_cast<std::int32_t>(value_index++ % 5) - 2;
    auto& layer2 = model.network_weights.layer2;
    value_index = 0;
    for (auto& row : layer2.weights)
        for (auto& value : row) value = static_cast<std::int32_t>(value_index++ % 7) - 3;
    value_index = 0;
    for (auto& value : layer2.bias) value = static_cast<std::int32_t>(value_index++ % 5) - 2;
    auto& output = model.network_weights.output;
    for (auto& row : output.weights)
        for (auto& value : row) value = 64;
    output.bias[0] = 3;
    model.network_weights.activation_limit = 32767;
}

bool same_feature_set(const TAPA::Features::FeatureSet& lhs,
                      const TAPA::Features::FeatureSet& rhs) {
    if (lhs.size != rhs.size) return false;
    for (std::size_t i = 0; i < lhs.size; ++i) {
        bool found = false;
        for (std::size_t j = 0; j < rhs.size; ++j) {
            const auto& left = lhs.features[i];
            const auto& right = rhs.features[j];
            if (left.index == right.index && left.value == right.value &&
                left.perspective == right.perspective) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

bool same_delta(const TAPA::Features::FeatureDelta& lhs,
                const TAPA::Features::FeatureDelta& rhs) {
    return lhs.valid && rhs.valid &&
        lhs.previous_pawn_structure == rhs.previous_pawn_structure &&
        lhs.current_pawn_structure == rhs.current_pawn_structure &&
        same_feature_set(lhs.added, rhs.added) && same_feature_set(lhs.removed, rhs.removed);
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
    const FeatureDelta full_delta = update_features(position, moved_position);
    const FeatureDelta move_delta = update_features(position, knight_move, moved_position);
    if (!same_delta(full_delta, move_delta)) {
        std::cerr << "ERROR: full-position and move-based feature deltas differ\n";
        return 1;
    }
    const FeatureDelta representative_delta = move_delta;
    // Compare full reconstruction against the move-descriptor incremental path.
    const auto full_delta_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i) {
            benchmark_sink += static_cast<std::int64_t>(update_features(position, moved_position).added.size);
        }
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
    fill_benchmark_model(benchmark_model);
    std::shared_ptr<const TAPA::NNUEModel> shared_model = std::make_shared<const TAPA::NNUEModel>(benchmark_model);
    TAPA::NNUE nnue;
    if (!nnue.load_model(shared_model)) {
        std::cerr << "ERROR: failed to load benchmark model\n";
        return 1;
    }
    nnue.initialize(0);
    if (nnue.evaluate(position) == 0) {
        std::cerr << "ERROR: benchmark model produced a zero evaluation\n";
        return 1;
    }
    const auto nnue_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i)
            benchmark_sink += nnue.evaluate(position);
    });
    const auto evaluate_current_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i)
            benchmark_sink += nnue.evaluate_current();
    });
    // #7: correct reversed move descriptor for the backward update; status of
    // each update is verified and the benchmark aborts on failure.
    const Move knight_move_back{33, 18, PieceType::Knight, PieceColor::White};
    Position updated_position{};
    const bool update_possible = apply_move(position, knight_move, updated_position);
    bool roundtrip_verified = false;
    if (update_possible) {
        TAPA::NNUE verify;
        if (verify.load_model(shared_model)) {
            verify.initialize(0);
            verify.evaluate(position);
            const std::int32_t base_score = verify.evaluate_current();
            const bool forward_ok =
                verify.update(position, knight_move, updated_position) == TAPA::NNUE::UpdateStatus::Updated;
            const bool back_ok =
                verify.update(updated_position, knight_move_back, position) == TAPA::NNUE::UpdateStatus::Updated;
            // After a forward+backward pair the state must return to the
            // starting position and reproduce the full reconstruction score.
            TAPA::NNUE reference;
            if (reference.load_model(shared_model)) {
                reference.initialize(0);
                reference.evaluate(position);
                roundtrip_verified = forward_ok && back_ok &&
                    verify.evaluate_current() == reference.evaluate_current() &&
                    verify.evaluate_current() == base_score;
            }
        }
    }
    if (!roundtrip_verified) {
        std::cerr << "ERROR: incremental update round-trip failed verification\n";
        return 1;
    }
    const auto nnue_incremental_us = measure_us([&] {
        for (std::size_t i = 0; i < Iterations; ++i) {
            if (nnue.update(position, knight_move, updated_position) != TAPA::NNUE::UpdateStatus::Updated ||
                nnue.update(updated_position, knight_move_back, position) != TAPA::NNUE::UpdateStatus::Updated) {
                benchmark_failure.store(true, std::memory_order_relaxed);
                break;
            }
            benchmark_sink += nnue.evaluate_current();
        }
    });
    if (benchmark_failure.load(std::memory_order_relaxed)) {
        std::cerr << "ERROR: incremental update failed during measurement\n";
        return 1;
    }
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
    benchmark_failure.store(false, std::memory_order_relaxed);
    for (std::size_t worker_count : {1u, 2u, 4u}) {
        std::vector<std::thread> workers;
        const auto elapsed = measure_us([&] {
            for (std::size_t worker = 0; worker < worker_count; ++worker)
                workers.emplace_back([&] {
                    // Each worker owns its own NNUE state and cache but shares
                    // one immutable model.
                    TAPA::NNUE local;
                    if (!local.load_model(shared_model)) {
                        benchmark_failure.store(true, std::memory_order_relaxed);
                        return;
                    }
                    local.initialize(0);
                    const std::int64_t sample = local.evaluate(position);
                    if (sample == 0) {
                        benchmark_failure.store(true, std::memory_order_relaxed);
                        return;
                    }
                    std::int64_t worker_sink = sample;
                    for (std::size_t i = 0; i < Iterations / worker_count; ++i)
                        worker_sink += local.evaluate(position);
                    benchmark_sink.fetch_add(worker_sink, std::memory_order_relaxed);
                });
            for (auto& worker : workers) worker.join();
        });
        worker_times.push_back(elapsed);
    }
    if (benchmark_failure.load(std::memory_order_relaxed)) {
        std::cerr << "ERROR: one or more NNUE benchmark workers failed to evaluate\n";
        return 1;
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
              << "features_reconstruction_us=" << feature_us << '\n'
              << "pawn_incremental_us=" << incremental_us << '\n'
              << "piece_reconstruction_us=" << piece_reconstruct_us << '\n'
              << "piece_incremental_us=" << piece_update_us << '\n'
              << "full_position_delta_us=" << full_delta_us << '\n'
              << "move_based_feature_delta_us=" << incremental_delta_us << '\n'
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
              << " nnue_owned_weight_bytes=" << nnue.owned_weight_bytes()
              << " nnue_shared_model_bytes=" << TAPA::NNUE::model_bytes(*shared_model)
              << " network_weights_bytes=" << sizeof(TAPA::Network::NetworkWeights)
              << " nnue_feature_weights_bytes=" << TAPA::NNUE::weights_bytes()
              << " nnue_cache_bytes=" << nnue.pawn_cache_bytes()
              << " cache_entry_bytes=" << TAPA::NNUE::cache_entry_bytes()
              << " sink=" << benchmark_sink.load(std::memory_order_relaxed) << '\n';
    return 0;
}
