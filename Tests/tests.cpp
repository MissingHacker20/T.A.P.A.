#include "tests.h"
#include "../Source Code/Features/features.h"
#include "../Source Code/Accumulator/accumulator.h"
#include "../Source Code/NNUE/nnue.h"
#include "../Source Code/Accumulator/pawn_accumulator.h"
#include "../Source Code/Accumulator/piece_accumulator.h"
#include "../Source Code/Features/move.h"
#include "../Source Code/Model/model.h"
#include "../Source Code/Network/network_model.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <iostream>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

namespace TAPA::Tests
{
    namespace
    {
        using namespace TAPA::Features;

        Position kings_only()
        {
            Position position{};
            position.pieces[0] = {PieceType::King, PieceColor::White, 4};
            position.pieces[1] = {PieceType::King, PieceColor::Black, 60};
            position.num_pieces = 2;
            return position;
        }

        void add(Position &position, PieceType type, PieceColor color, int square)
        {
            position.pieces[position.num_pieces++] = {type, color, square};
        }

        bool check(bool condition) { return condition; }

        bool feature_tests()
        {
            auto require = [](bool ok, int marker) {
                if (!ok) std::printf("feature_tests: check failed near line marker %d\n", marker);
                return ok;
            };
            if (!require(sizeof(PawnStructureKey) == sizeof(std::uint64_t), 1)) return false;
            if (!check(is_valid_square(0) && is_valid_square(63) && !is_valid_square(64)))
                return false;
            for (int square = 0; square < 64; ++square)
            {
                if (square_rank(square) >= 1 && square_rank(square) <= 4)
                {
                    if (!require(is_structural_pawn(PieceColor::White, square), 5))
                        return false;
                }
                if (square_rank(square) >= 3 && square_rank(square) <= 6)
                {
                    if (!require(is_structural_pawn(PieceColor::Black, square), 6))
                        return false;
                }
                if (square_rank(square) >= 1 && square_rank(square) <= 6)
                {
                    if (!require(is_structural_pawn(PieceColor::White, square) != is_advanced_pawn(PieceColor::White, square), 7))
                        return false;
                    if (!require(is_structural_pawn(PieceColor::Black, square) != is_advanced_pawn(PieceColor::Black, square), 8))
                        return false;
                }
            }

            if (!require([&] {
                    for (PieceColor color : {PieceColor::White, PieceColor::Black})
                        for (int square = 0; square < 64; ++square) {
                            const int rank = square_rank(square);
                            const bool on_promotion_rank = rank == (color == PieceColor::White ? 7 : 0);
                            const bool behind_pawn_start = color == PieceColor::White ? rank == 0 : rank == 7;
                            if (!on_promotion_rank && !behind_pawn_start &&
                                (is_structural_pawn(color, square) == is_advanced_pawn(color, square))) return false;
                        }
                    return true;
                }(), 2)) return false;

            Position position = kings_only();
            add(position, PieceType::Pawn, PieceColor::White, 12); // e2, bit 4
            add(position, PieceType::Pawn, PieceColor::White, 36); // e5, bit 28
            add(position, PieceType::Pawn, PieceColor::White, 44); // e6, advanced
            add(position, PieceType::Pawn, PieceColor::Black, 52); // e7, black level 0
            add(position, PieceType::Pawn, PieceColor::Black, 28); // e4, black level 3
            add(position, PieceType::Pawn, PieceColor::Black, 20); // e3, advanced
            Position black_position = kings_only();
            add(black_position, PieceType::Pawn, PieceColor::Black, 52);
            // Black pawn e7: black half starts at bit 32, level 0, file 4 → bit 36.
            if (!require((make_pawn_structure_key(black_position) & (PawnStructureKey{1} << 36)) != 0, 2)) return false;
            if (!require((make_pawn_structure_key(black_position) >> 32) == (PawnStructureKey{1} << 4), 3)) return false;
            Position white_advanced = kings_only();
            add(white_advanced, PieceType::Pawn, PieceColor::White, 44);
            if (!require(make_pawn_structure_key(white_advanced) == 0, 9)) return false;
            FeatureSet advanced_features;
            generate_features(white_advanced, advanced_features);
            if (!require(advanced_features.size == 4, 10)) return false;

            const PawnStructureKey expected = (PawnStructureKey{1} << 4) |
                                              (PawnStructureKey{1} << 28) |
                                              (PawnStructureKey{1} << 36) |
                                              (PawnStructureKey{1} << 60);
            if (!require(compute_pawn_structure_key(position) == expected, 4)) return false;
            std::array<DecodedPawn, 32> decoded{};
            const std::size_t decoded_count = decode_pawn_structure_key(expected, decoded);
            if (!require(decoded_count == 4, 11)) return false;
            PawnStructureKey roundtrip = 0;
            for (std::size_t i = 0; i < decoded_count; ++i) {
                if (!is_structural_pawn(decoded[i].color, decoded[i].square)) return false;
                const int level = decoded[i].color == PieceColor::White
                    ? square_rank(decoded[i].square) - 1 : 6 - square_rank(decoded[i].square);
                const int bit = (decoded[i].color == PieceColor::White ? 0 : 32) + level * 8 + square_file(decoded[i].square);
                const PawnStructureKey mask = PawnStructureKey{1} << bit;
                if (roundtrip & mask) return false;
                roundtrip |= mask;
            }
            if (!require(roundtrip == expected, 12)) return false;
            Position changed_file = kings_only();
            add(changed_file, PieceType::Pawn, PieceColor::White, 11);
            Position changed_rank = kings_only();
            add(changed_rank, PieceType::Pawn, PieceColor::White, 19);
            Position changed_color = kings_only();
            add(changed_color, PieceType::Pawn, PieceColor::Black, 12);
            Position added_pawn = changed_file;
            add(added_pawn, PieceType::Pawn, PieceColor::Black, 52);
            Position removed_pawn = kings_only();
            add(removed_pawn, PieceType::Pawn, PieceColor::White, 12);
            Position empty_key = kings_only();
            if (!require(make_pawn_structure_key(changed_file) != make_pawn_structure_key(changed_rank) &&
                make_pawn_structure_key(added_pawn) != make_pawn_structure_key(changed_file) &&
                make_pawn_structure_key(removed_pawn) != make_pawn_structure_key(changed_file) &&
                make_pawn_structure_key(kings_only()) == 0 &&
                make_pawn_structure_key(changed_color) == 0 && // black e2 is an advanced pawn, not structural
                make_pawn_structure_key(empty_key) == 0 &&
                make_pawn_structure_key(kings_only()) != expected, 13)) return false;

            Position reordered = position;
            for (std::size_t left = 0, right = reordered.num_pieces - 1; left < right; ++left, --right) {
                const Piece tmp = reordered.pieces[left];
                reordered.pieces[left] = reordered.pieces[right];
                reordered.pieces[right] = tmp;
            }
            FeatureSet features, reordered_features;
            generate_features(position, features);
            generate_features(reordered, reordered_features);
            if (!require(features.size == reordered_features.size, 14)) return false;
            for (std::size_t i = 0; i < features.size; ++i)
                if (!require(features.features[i].index == reordered_features.features[i].index &&
                    features.features[i].perspective == reordered_features.features[i].perspective, 15)) return false;
            // Each perspective sees all non-own-king pieces: 2 advanced pawns × 2 perspectives = 4
            // plus opposing king per perspective = 2 more → 6.
            // Both perspectives see the advanced pawn and the opposing king:
            // white persp sees (black king, white pawn), black persp sees (white king, white pawn).
            if (!require(features.size == 6, 16))
                return false;
            bool found_white_advanced = false;
            bool found_black_advanced = false;
            for (std::size_t i = 0; i < features.size; ++i) {
                const Feature& feature = features.features[i];
                if (feature.index == encode_index(4, 0, 0, 44) && feature.perspective == PieceColor::White)
                    found_white_advanced = true;
                if (feature.index == encode_index(60, 1, 0, 20) && feature.perspective == PieceColor::Black)
                    found_black_advanced = true;
            }
            if (!require(found_white_advanced && found_black_advanced, 17)) return false;

            Position invalid = position;
            invalid.pieces[0].square = 64;
            if (!require(!invalid.is_valid(), 18))
                return false;
            return position.is_valid();
        }

        bool accumulator_tests()
        {
            using Accumulator = TAPA::Accumulator::Accumulator;
            std::array<std::int32_t, TAPA::Accumulator::NeuronCount> extreme_weights{};
            extreme_weights.fill(std::numeric_limits<std::int32_t>::max());
            Accumulator extreme;
            extreme.initialize(std::numeric_limits<std::int32_t>::max());
            Feature positive{0, 1, PieceColor::White};
            Feature negative{1, -1, PieceColor::White};
            Feature invalid_coefficient{2, 2, PieceColor::White};
            extreme.add_feature(invalid_coefficient, extreme_weights.data());
            if (extreme.get_data()[0] != std::numeric_limits<std::int32_t>::max()) return false;
            extreme.add_feature(positive, extreme_weights.data());
            if (extreme.get_data()[0] != std::numeric_limits<std::int32_t>::max()) return false;
            extreme.initialize(std::numeric_limits<std::int32_t>::min());
            extreme.add_feature(negative, extreme_weights.data());
            if (extreme.get_data()[0] != std::numeric_limits<std::int32_t>::min()) return false;
            FeatureSet full_extremes;
            full_extremes.add(positive);
            full_extremes.add(negative);
            extreme.initialize(0);
            extreme.reconstruct(full_extremes, extreme_weights.data());
            if (extreme.get_data()[0] != 0) return false;
            Accumulator incremental_extremes;
            incremental_extremes.initialize(0);
            FeatureSet first_add, second_add;
            first_add.add(positive);
            second_add.add(negative);
            incremental_extremes.update_incremental(first_add, {}, extreme_weights.data());
            incremental_extremes.update_incremental(second_add, {}, extreme_weights.data());
            if (extreme.get_data() != incremental_extremes.get_data()) return false;

            Accumulator full;
            Accumulator incremental;
            full.initialize(7);
            incremental.initialize(7);
            std::array<int32_t, TAPA::Accumulator::NeuronCount> weights{};
            for (std::size_t i = 0; i < weights.size(); ++i)
                weights[i] = static_cast<int32_t>(i % 5) - 2;
            Feature a{1, 1, PieceColor::White};
            Feature b{2, -1, PieceColor::Black};
            FeatureSet all, added, removed;
            all.add(a);
            all.add(b);
            added.add(b);
            removed.add(a);
            full.reconstruct(all, weights.data());
            incremental.add_feature(a, weights.data());
            incremental.update_incremental(added, removed, weights.data());
            // Start from bias, apply a then replace it by b.
            Accumulator expected;
            expected.initialize(7);
            expected.add_feature(b, weights.data());
            return full.get_data() != incremental.get_data() && incremental.get_data() == expected.get_data();
        }

        bool pawn_accumulator_tests()
        {
            TAPA::Accumulator::PawnWeights weights;
            for (std::size_t bit = 0; bit < 64; ++bit)
                for (std::size_t neuron = 0; neuron < TAPA::Accumulator::PawnNeuronCount; ++neuron)
                    weights.bit_weights[bit][neuron] = static_cast<TAPA::Accumulator::PawnWeight>((bit + neuron) % 5 - 2);
            TAPA::Accumulator::PawnAccumulator full, incremental;
            const PawnStructureKey old_key = (PawnStructureKey{1} << 4) | (PawnStructureKey{1} << 40);
            const PawnStructureKey new_key = (PawnStructureKey{1} << 28) | (PawnStructureKey{1} << 60);
            full.reconstruct(new_key, weights);
            incremental.reconstruct(old_key, weights);
            incremental.update(old_key, new_key, weights);
            if (full.data() != incremental.data()) return false;
            TAPA::Accumulator::PawnAccumulatorCache cache(weights);
            cache.get(new_key);
            if (cache.touch(new_key) == nullptr || cache.stats().hits != 1 || cache.stats().misses != 1) return false;
            const auto cached_accumulator = cache.get(new_key);
            const auto insertion_count = cache.insertions();
            cache.insert(old_key, incremental);
            if (cache.insertions() != insertion_count + 1 || !cache.contains(old_key) ||
                cache.get(old_key) != full || cached_accumulator != full) return false;
            const auto old_insertions = cache.insertions();
            cache.insert(old_key, full);
            if (cache.insertions() != old_insertions + 1) return false;
            for (std::size_t i = 0; i < TAPA::Accumulator::PawnAccumulatorCache::Capacity; ++i)
                cache.get(static_cast<PawnStructureKey>(i + 1000));
            return cache.find(new_key) == nullptr;
        }

        bool model_tests()
        {
            auto fail = [](int step) { std::printf("model_tests: failed at step %d\n", step); return false; };
            TAPA::Model::Model malformed;
            malformed.piece_weights.feature_weights.clear();
            std::string malformed_error;
            if (TAPA::Model::save("tapa_invalid_piece_model.bin", malformed, &malformed_error)) return fail(1);
            NNUE reject_malformed;
            if (reject_malformed.load_model(malformed, &malformed_error) || malformed_error.empty() || reject_malformed.model_loaded()) return fail(2);
            TAPA::Model::Model valid_piece_model;
            valid_piece_model.piece_weights.feature_weights.resize(TAPA::Accumulator::PieceFeatureCount - 1);
            if (TAPA::Model::save("tapa_invalid_piece_model.bin", valid_piece_model, &malformed_error)) return fail(3);
            valid_piece_model.piece_weights.feature_weights.resize(TAPA::Accumulator::PieceFeatureCount + 1);
            if (TAPA::Model::save("tapa_invalid_piece_model.bin", valid_piece_model, &malformed_error)) return fail(4);
            static TAPA::Model::Model model;
            model.pawn_weights.bias = 17;
            model.pawn_weights.bit_weights[3][7] = -11;
            model.network_weights.layer2.bias[4] = -9;
            model.network_weights.output.weights[7][0] = 31;
            model.piece_weights.bias = -13;
            model.piece_weights.feature_weights[19][2] = -7;
            model.network_weights.layer1.weights[0][0] = 1;
            model.network_weights.layer1.weights[256][0] = 1;
            model.network_weights.layer2.weights[0][0] = 1;
            model.network_weights.output.weights[0][0] = 1;
            model.network_weights.output_scale = 1;
            model.piece_weights.feature_weights[3836][0] = 30;
            model.piece_weights.feature_weights[46162][0] = 5;
            const std::string path = "tapa_test_model.bin";
            std::string error;
            if (!TAPA::Model::save(path, model, &error)) { std::printf("model save failed: %s\n", error.c_str()); return fail(5); }
            const auto load_path = std::string("tapa_test_model_loaded.bin");
            const auto loaded = TAPA::Model::load(path);
            if (!loaded) { std::printf("model load failed: %s\n", loaded.error.c_str()); std::remove(path.c_str()); std::remove(load_path.c_str()); return fail(6); }
            if (loaded.model.pawn_weights.bias != 17 ||
                loaded.model.pawn_weights.bit_weights[3][7] != -11 ||
                loaded.model.network_weights.layer1.weights[0][0] != 1 ||
                loaded.model.network_weights.layer2.bias[4] != -9 ||
                loaded.model.network_weights.output.weights[7][0] != 31 ||
                loaded.model.piece_weights.bias != -13 ||
                loaded.model.piece_weights.feature_weights[19][2] != -7) { std::remove(path.c_str()); std::remove(load_path.c_str()); return fail(7); }
            NNUE loaded_nnue;
            if (!loaded_nnue.load_model(loaded.model)) { std::remove(path.c_str()); std::remove(load_path.c_str()); return fail(8); }
            loaded_nnue.initialize(0);
            if (!loaded_nnue.model_loaded() || loaded_nnue.evaluate_current() != 0) { std::remove(path.c_str()); return fail(9); }
            NNUE shared_first, shared_second;
            std::shared_ptr<const TAPA::NNUEModel> shared_model;
            try { shared_model = std::make_shared<const TAPA::NNUEModel>(loaded.model); }
            catch (...) { std::remove(path.c_str()); return fail(10); }
            if (!shared_first.load_model(shared_model) || !shared_second.load_model(shared_model)) { std::remove(path.c_str()); return fail(11); }
            Position shared_start = kings_only();
            add(shared_start, PieceType::Knight, PieceColor::White, 18);
            const auto initial_first = shared_first.evaluate(shared_start);
            const auto initial_second = shared_second.evaluate(shared_start);
            if (initial_first == 0) { std::remove(path.c_str()); return fail(12); }
            Position shared_next = shared_start;
            shared_next.pieces[2].square = 19;
            if (shared_first.update(shared_start, shared_next) != NNUE::UpdateStatus::Updated ||
                shared_second.evaluate_current() != initial_second) {
                std::remove(path.c_str()); return fail(13);
            }
            const auto prior_load_model = shared_model;
            const auto score_before_failed_load = shared_first.evaluate_current();
            if (shared_first.load_model(std::shared_ptr<const TAPA::NNUEModel>{}, &error) ||
                !shared_first.model_loaded() || shared_first.evaluate_current() != score_before_failed_load ||
                prior_load_model.use_count() < 2 || error.empty()) {
                std::remove(path.c_str()); return fail(14);
            }
            std::array<std::int32_t, 4> worker_scores{};
            std::vector<std::thread> workers;
            for (std::size_t worker = 0; worker < worker_scores.size(); ++worker) {
                workers.emplace_back([&, worker] {
                    NNUE local;
                    if (!local.load_model(shared_model)) return;
                    Position pos = kings_only();
                    add(pos, PieceType::Knight, PieceColor::White, 18);
                    worker_scores[worker] = local.evaluate(pos);
                });
            }
            for (auto& worker : workers) worker.join();
            for (auto score : worker_scores)
                if (score != worker_scores[0]) { std::remove(path.c_str()); return fail(15); }
            (void)loaded_nnue.evaluate(kings_only());
            {
                std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
                file.seekg(64, std::ios::beg);
                const int original = file.get();
                file.clear();
                file.seekp(64, std::ios::beg);
                file.put(static_cast<char>(original ^ 0x55));
            }
            if (TAPA::Model::load(path)) { std::remove(path.c_str()); std::remove(load_path.c_str()); return fail(16); }
            model.input_size = 513;
            if (TAPA::Model::save(path, model, &error)) { std::remove(path.c_str()); std::remove(load_path.c_str()); return fail(17); }
            model.input_size = TAPA::Model::InputSize;
            if (!TAPA::Model::save(path, model, &error)) { std::printf("save18 error: %s\n", error.c_str()); return fail(18); }
            {
                std::ifstream input(path, std::ios::binary);
                std::ofstream output(load_path, std::ios::binary | std::ios::trunc);
                output << input.rdbuf();
                output.put('X');
            }
            if (TAPA::Model::load(load_path)) { std::remove(path.c_str()); std::remove(load_path.c_str()); return fail(19); }
            std::remove(path.c_str());
            std::remove(load_path.c_str());
            return true;
        }

        bool random_pawn_key_roundtrip_tests()
        {
            std::mt19937 generator(0x5041574eu);
            for (int sample = 0; sample < 500; ++sample) {
                Position position = kings_only();
                std::array<bool, 64> occupied{};
                occupied[4] = occupied[60] = true;
                const int pawn_count = static_cast<int>(generator() % 17);
                for (int i = 0; i < pawn_count; ++i) {
                    const PieceColor color = (generator() & 1u) ? PieceColor::White : PieceColor::Black;
                    int square = 0;
                    do {
                        const int rank = 1 + static_cast<int>(generator() % 6);
                        square = rank * 8 + static_cast<int>(generator() % 8);
                    } while (occupied[square] ||
                        (color == PieceColor::White && square_rank(square) == 0) ||
                        (color == PieceColor::Black && square_rank(square) == 7));
                    occupied[square] = true;
                    add(position, PieceType::Pawn, color, square);
                }
                const PawnStructureKey key = make_pawn_structure_key(position);
                std::array<DecodedPawn, 32> decoded{};
                const std::size_t decoded_count = decode_pawn_structure_key(key, decoded);
                std::size_t expected_count = 0;
                for (std::size_t i = 0; i < position.num_pieces; ++i) {
                    const Piece& piece = position.pieces[i];
                    if (piece.type == PieceType::Pawn && is_structural_pawn(piece.color, piece.square)) ++expected_count;
                }
                if (decoded_count != expected_count) return false;
                PawnStructureKey rebuilt = 0;
                for (std::size_t i = 0; i < decoded_count; ++i) {
                    const int level = decoded[i].color == PieceColor::White
                        ? square_rank(decoded[i].square) - 1 : 6 - square_rank(decoded[i].square);
                    const int bit = (decoded[i].color == PieceColor::White ? 0 : 32) + level * 8 + square_file(decoded[i].square);
                    const PawnStructureKey mask = PawnStructureKey{1} << bit;
                    if (rebuilt & mask) return false;
                    rebuilt |= mask;
                }
                if (rebuilt != key) return false;
            }
            return true;
        }

        bool piece_accumulator_tests()
        {
            TAPA::Accumulator::PieceWeights invalid_weights;
            invalid_weights.feature_weights.resize(0);
            TAPA::Accumulator::PieceKingAccumulator guarded;
            Feature invalid_feature{0, 1, PieceColor::White};
            guarded.reset(123);
            guarded.add(invalid_feature, invalid_weights);
            guarded.remove(invalid_feature, invalid_weights);
            FeatureSet invalid_set;
            invalid_set.add(invalid_feature);
            guarded.reconstruct(invalid_set, invalid_weights, PieceColor::White);
            guarded.update(invalid_set, {}, invalid_weights, PieceColor::White);
            if (invalid_weights.valid() || guarded.data()[0] != 0) return false;
            TAPA::Accumulator::PieceWeights oversized_weights;
            oversized_weights.feature_weights.push_back({});
            if (oversized_weights.valid()) return false;
            static TAPA::Accumulator::PieceWeights weights;
            for (std::size_t index = 0; index < weights.feature_weights.size(); ++index)
                for (std::size_t neuron = 0; neuron < weights.feature_weights[index].size(); ++neuron)
                    weights.feature_weights[index][neuron] = static_cast<TAPA::Accumulator::PieceWeight>((index + neuron) % 7 - 3);
            Position position = kings_only();
            add(position, PieceType::Knight, PieceColor::White, 18);
            FeatureSet set;
            generate_features(position, set);
            TAPA::Accumulator::PieceKingAccumulator full, incremental;
            full.reconstruct(set, weights, PieceColor::White);
            incremental.reset(weights.bias);
            FeatureSet added, removed;
            for (std::size_t i = 0; i < set.size; ++i) added.add(set.features[i]);
            incremental.update(added, removed, weights, PieceColor::White);
            return full.data() == incremental.data();
        }

        bool move_tests()
        {
            Position before = kings_only();
            add(before, PieceType::Pawn, PieceColor::White, 36);
            Features::Move move{36, 44, PieceType::Pawn, PieceColor::White};
            Position after{};
            if (!Features::apply_move(before, move, after)) return false;
            if (make_pawn_structure_key(before) == make_pawn_structure_key(after)) return false;
            const FeatureDelta delta = Features::update_features(before, move, after);
            if (!delta.valid || delta.previous_pawn_structure == delta.current_pawn_structure) return false;
            if (delta.removed.size != 0 || delta.added.size != 2) return false;
            bool saw_white_advanced_pawn = false;
            for (std::size_t i = 0; i < delta.added.size; ++i) {
                const Feature& feature = delta.added.features[i];
                if (feature.index == encode_index(4, 0, static_cast<int>(PieceType::Pawn), 44) &&
                    feature.perspective == PieceColor::White) saw_white_advanced_pawn = true;
            }
            if (!saw_white_advanced_pawn) return false;
            Move invalid_promotion{36, 44, PieceType::Pawn, PieceColor::White};
            invalid_promotion.promotion = PieceType::Queen;
            // A promotion must land on the promotion rank (e6 -> e7 is not one).
            if (Features::apply_move(before, invalid_promotion, after)) return false;
            Move castling{};
            castling.from = 4; castling.to = 6; castling.moving_type = PieceType::King;
            castling.color = PieceColor::White; castling.castling = true;
            castling.rook_from = 7; castling.rook_to = 5;
            Position castle_position = kings_only();
            add(castle_position, PieceType::Rook, PieceColor::White, 7);
            if (!Features::apply_move(castle_position, castling, after)) return false;
            if (after.pieces[0].square != 6 || after.pieces[2].square != 5) return false;
            return delta.added.size == 2;
        }

        bool differential_random_sequence_tests()
        {
            std::mt19937 generator(0x504f5349u);
            for (int game = 0; game < 40; ++game) {
                Position position = kings_only();
                add(position, PieceType::Knight, PieceColor::White, 8 + static_cast<int>(generator() % 8));
                add(position, PieceType::Knight, PieceColor::Black, 48 + static_cast<int>(generator() % 8));
                TAPA::NNUE incremental, rebuilt;
                incremental.initialize(); rebuilt.initialize();
                (void)incremental.evaluate(position);
                std::array<Position, 65> positions{};
                positions[0] = position;
                int plies = 0;
                for (; plies < 64; ++plies) {
                    std::array<Move, 128> candidates{};
                    std::size_t candidate_count = 0;
                    const PieceColor side = (plies & 1) == 0 ? PieceColor::White : PieceColor::Black;
                    for (std::size_t piece_index = 0; piece_index < position.num_pieces; ++piece_index) {
                        const Piece& piece = position.pieces[piece_index];
                        if (piece.color != side || piece.type != PieceType::Knight) continue;
                        constexpr int offsets[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
                        for (const auto& offset : offsets) {
                            const int file = square_file(piece.square) + offset[0];
                            const int rank = square_rank(piece.square) + offset[1];
                            if (file < 0 || file >= 8 || rank < 0 || rank >= 8) continue;
                            Move move{}; move.from = piece.square; move.to = rank * 8 + file;
                            move.moving_type = PieceType::Knight; move.color = side;
                            bool occupied = false;
                            for (std::size_t target_index = 0; target_index < position.num_pieces; ++target_index) {
                                const Piece& target = position.pieces[target_index];
                                if (target.square != move.to) continue;
                                if (target.color == side || target.type == PieceType::King) { occupied = true; break; }
                                move.capture = true; move.captured_type = target.type;
                            }
                            if (!occupied && candidate_count < candidates.size()) candidates[candidate_count++] = move;
                        }
                    }
                    if (candidate_count == 0) break;
                    const Move move = candidates[generator() % candidate_count];
                    Position next{};
                    if (!Features::apply_move(position, move, next)) {
                        std::printf("random apply_move rejected: from=%d to=%d type=%u color=%u capture=%d\n",
                            move.from, move.to, static_cast<unsigned>(move.moving_type),
                            static_cast<unsigned>(move.color), move.capture ? 1 : 0);
                        return false;
                    }
                    const FeatureDelta delta = Features::update_features(position, move, next);
                    if (!delta.valid) {
                        std::printf("random delta invalid: from=%d to=%d type=%u\n", move.from, move.to,
                            static_cast<unsigned>(move.moving_type));
                        return false;
                    }
                    incremental.update(position, move, next);
                    const std::int32_t expected_score = rebuilt.evaluate(next);
                    if (incremental.evaluate_current() != expected_score) {
                        std::printf("random differential mismatch game=%d ply=%d side=%u move=%d-%d score=%d expected=%d\n",
                            game, plies, static_cast<unsigned int>(side), move.from, move.to,
                            incremental.evaluate_current(), expected_score);
                        std::printf("random state: whiteKing=%d blackKing=%d pieces=%zu", next.king_square(PieceColor::White),
                                    next.king_square(PieceColor::Black), next.num_pieces);
                        for (std::size_t diagnostic = 0; diagnostic < next.num_pieces; ++diagnostic)
                            std::printf(" [%u,%u,%d]", static_cast<unsigned int>(next.pieces[diagnostic].color),
                                static_cast<unsigned int>(next.pieces[diagnostic].type), next.pieces[diagnostic].square);
                        std::printf("\n");
                        return false;
                    }
                    positions[plies + 1] = next;
                    position = next;
                }
                while (plies > 0) {
                    const Position current = positions[plies];
                    const Position prior = positions[plies - 1];
                    incremental.push();
                    incremental.update(current, prior);
                    if (incremental.evaluate_current() != rebuilt.evaluate(prior)) {
                        std::printf("random unmake mismatch game=%d ply=%d\n", game, plies);
                        return false;
                    }
                    incremental.pop();
                    --plies;
                }
            }
            return true;
        }

        bool incremental_move_delta_tests()
        {
            auto compare_move = [](const Position& before, const Move& move) {
                Position after{};
                if (!Features::apply_move(before, move, after)) {
                    std::printf("compare_move: apply rejected from=%d to=%d\n", move.from, move.to);
                    return false;
                }
                const FeatureDelta delta = Features::update_features(before, move, after);
                if (!delta.valid) {
                    std::printf("compare_move: delta invalid from=%d to=%d\n", move.from, move.to);
                    return false;
                }
                FeatureSet expected_removed, expected_added;
                const FeatureSet old_set = [&]() { FeatureSet set; generate_features(before, set); return set; }();
                const FeatureSet new_set = [&]() { FeatureSet set; generate_features(after, set); return set; }();
                auto has = [](const FeatureSet& set, const Feature& f) {
                    for (std::size_t i = 0; i < set.size; ++i)
                        if (set.features[i].index == f.index && set.features[i].perspective == f.perspective) return true;
                    return false;
                };
                for (std::size_t i = 0; i < old_set.size; ++i)
                    if (!has(new_set, old_set.features[i])) expected_removed.add(old_set.features[i]);
                for (std::size_t i = 0; i < new_set.size; ++i)
                    if (!has(old_set, new_set.features[i])) expected_added.add(new_set.features[i]);
                auto same_set = [](const FeatureSet& a, const FeatureSet& b) {
                    if (a.size != b.size) return false;
                    for (std::size_t i = 0; i < a.size; ++i) {
                        bool found = false;
                        for (std::size_t j = 0; j < b.size; ++j)
                            if (a.features[i].index == b.features[j].index && a.features[i].perspective == b.features[j].perspective) found = true;
                        if (!found) return false;
                    }
                    return true;
                };
                if (!same_set(delta.removed, expected_removed) || !same_set(delta.added, expected_added)) {
                    std::printf("compare_move: delta sets differ for move %d-%d\n", move.from, move.to);
                    return false;
                }
                TAPA::NNUE incremental, rebuilt;
                incremental.initialize(); rebuilt.initialize(0);
                (void)incremental.evaluate(before);
                incremental.update(before, move, after);
                const std::int32_t incremental_score = incremental.evaluate_current();
                const std::int32_t full_score = rebuilt.evaluate(after);
                if (incremental_score != full_score) {
                    std::printf("compare_move: nnue mismatch for move %d-%d\n", move.from, move.to);
                    return false;
                }
                incremental.push();
                incremental.update(after, before);
                const std::int32_t unmade_score = incremental.evaluate_current();
                incremental.pop();
                if (incremental.evaluate_current() != incremental_score || unmade_score != full_score) {
                    std::printf("compare_move: push/pop mismatch for move %d-%d\n", move.from, move.to);
                    return false;
                }
                return true;
            };

            Position kings = kings_only();
            add(kings, PieceType::Knight, PieceColor::White, 18);
            if (!compare_move(kings, Move{18, 33, PieceType::Knight, PieceColor::White})) return false;

            Position border = kings_only();
            add(border, PieceType::Pawn, PieceColor::White, 36);
            Move white_cross{36, 44, PieceType::Pawn, PieceColor::White};
            if (!compare_move(border, white_cross)) return false;
            border.pieces[2] = {PieceType::Pawn, PieceColor::Black, 28};
            Move black_cross{28, 20, PieceType::Pawn, PieceColor::Black};
            if (!compare_move(border, black_cross)) return false;

            Position capture = kings_only();
            add(capture, PieceType::Rook, PieceColor::White, 8);
            add(capture, PieceType::Knight, PieceColor::Black, 16);
            Move capture_move{}; capture_move.from = 8; capture_move.to = 16;
            capture_move.moving_type = PieceType::Rook; capture_move.color = PieceColor::White;
            capture_move.capture = true; capture_move.captured_type = PieceType::Knight;
            if (!compare_move(capture, capture_move)) return false;

            Position king_move = kings_only();
            add(king_move, PieceType::Queen, PieceColor::White, 12);
            Move king_step{}; king_step.from = 4; king_step.to = 3; // conflicts with enemy king? no: enemy king is on 60
            king_step.moving_type = PieceType::King; king_step.color = PieceColor::White;
            if (!compare_move(king_move, king_step)) return false;

            Position en_passant = kings_only();
            add(en_passant, PieceType::Pawn, PieceColor::White, 36);
            add(en_passant, PieceType::Pawn, PieceColor::Black, 35);
            Move ep{}; ep.from = 36; ep.to = 43; ep.moving_type = PieceType::Pawn; // e5xd6 en passant
            ep.color = PieceColor::White; ep.capture = true; ep.en_passant = true;
            ep.captured_square = 35; ep.captured_type = PieceType::Pawn;
            if (!compare_move(en_passant, ep)) return false;
            Position ep_boundary = kings_only();
            add(ep_boundary, PieceType::Pawn, PieceColor::White, 12); // e2, own pawn elsewhere
            add(ep_boundary, PieceType::Pawn, PieceColor::White, 36);
            add(ep_boundary, PieceType::Pawn, PieceColor::Black, 51);
            Move black_ep{}; black_ep.from = 51; black_ep.to = 35; black_ep.moving_type = PieceType::Pawn;
            black_ep.color = PieceColor::Black; // d7xd5 double push past white e5
            Position ep_after_double{};
            if (!Features::apply_move(ep_boundary, black_ep, ep_after_double)) return false;
            Move white_ep{}; white_ep.from = 36; white_ep.to = 43; white_ep.moving_type = PieceType::Pawn;
            white_ep.color = PieceColor::White; white_ep.capture = true; white_ep.en_passant = true;
            white_ep.captured_square = 35; white_ep.captured_type = PieceType::Pawn;
            if (!compare_move(ep_after_double, white_ep)) return false;

            Position castle = kings_only();
            add(castle, PieceType::Rook, PieceColor::White, 7);
            Move castle_move{}; castle_move.from = 4; castle_move.to = 6; castle_move.moving_type = PieceType::King;
            castle_move.color = PieceColor::White; castle_move.castling = true;
            castle_move.rook_from = 7; castle_move.rook_to = 5;
            if (!compare_move(castle, castle_move)) return false;

            Position promotion = kings_only();
            add(promotion, PieceType::Pawn, PieceColor::White, 54);
            Move promote{}; promote.from = 54; promote.to = 62; promote.moving_type = PieceType::Pawn;
            promote.color = PieceColor::White; promote.promotion = PieceType::Queen;
            if (!compare_move(promotion, promote)) return false;

            Position promotion_capture = kings_only();
            add(promotion_capture, PieceType::Pawn, PieceColor::White, 54);
            add(promotion_capture, PieceType::Rook, PieceColor::Black, 61);
            Move capture_promotion{}; capture_promotion.from = 54; capture_promotion.to = 61;
            capture_promotion.moving_type = PieceType::Pawn; capture_promotion.color = PieceColor::White;
            capture_promotion.capture = true; capture_promotion.captured_type = PieceType::Rook;
            capture_promotion.promotion = PieceType::Knight;
            if (!compare_move(promotion_capture, capture_promotion)) return false;

            Position pawn_capture = kings_only();
            add(pawn_capture, PieceType::Pawn, PieceColor::White, 36);
            add(pawn_capture, PieceType::Pawn, PieceColor::Black, 45);
            Move pawn_take{}; pawn_take.from = 36; pawn_take.to = 45;
            pawn_take.moving_type = PieceType::Pawn; pawn_take.color = PieceColor::White;
            pawn_take.capture = true; pawn_take.captured_type = PieceType::Pawn;
            if (!compare_move(pawn_capture, pawn_take)) return false;

            // Boundary-crossing en passant: white pawn from e5 captures black d5,
            // landing on the rank where the pawn leaves the structural zone.
            Position advanced_ep = kings_only();
            add(advanced_ep, PieceType::Pawn, PieceColor::White, 36); // e5
            add(advanced_ep, PieceType::Pawn, PieceColor::Black, 35); // d5
            Move ep_cross{}; ep_cross.from = 36; ep_cross.to = 43; ep_cross.moving_type = PieceType::Pawn; // e5xd6 ep
            ep_cross.color = PieceColor::White; ep_cross.capture = true; ep_cross.en_passant = true;
            ep_cross.captured_square = 35; ep_cross.captured_type = PieceType::Pawn;
            return compare_move(advanced_ep, ep_cross);
        }

        bool differential_feature_tests()
        {
            std::mt19937 generator(0x54415041u);
            for (int iteration = 0; iteration < 200; ++iteration) {
                Position previous = kings_only();
                Position current = previous;
                int from = 8 + static_cast<int>(generator() % 48);
                int to = 8 + static_cast<int>(generator() % 48);
                if (from == to) to = to == 55 ? 8 : to + 1;
                add(previous, PieceType::Knight, PieceColor::White, from);
                current.pieces[2] = previous.pieces[2];
                current.pieces[2].square = to;
                current.num_pieces = previous.num_pieces;
                FeatureSet old_features, new_features;
                generate_features(previous, old_features);
                generate_features(current, new_features);
                const FeatureDelta delta = update_features(previous, current);
                if (!delta.valid) return false;
                TAPA::Accumulator::Accumulator incremental, full;
                std::array<int32_t, TAPA::Accumulator::NeuronCount> weights{};
                for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<int32_t>(i % 11) - 5;
                incremental.initialize(3); full.initialize(3);
                for (std::size_t feature = 0; feature < new_features.size; ++feature)
                    full.add_feature(new_features.features[feature], weights.data());
                for (std::size_t feature = 0; feature < old_features.size; ++feature)
                    incremental.add_feature(old_features.features[feature], weights.data());
                for (std::size_t feature = 0; feature < delta.removed.size; ++feature)
                    incremental.remove_feature(delta.removed.features[feature], weights.data());
                for (std::size_t feature = 0; feature < delta.added.size; ++feature)
                    incremental.add_feature(delta.added.features[feature], weights.data());
                if (incremental.get_data() != full.get_data()) return false;
            }
            return true;
        }

        bool nnue_tests()
        {
            NNUE no_model;
            if (no_model.model_loaded() || no_model.evaluate(kings_only()) != 0 || no_model.evaluate_current() != 0) return false;
            TAPA::Model::Model test_model;
            NNUE nnue;
            if (!nnue.load_model(test_model)) return false;
            nnue.initialize(0);
            Position before = kings_only();
            add(before, PieceType::Knight, PieceColor::White, 18);
            const int32_t initial_score = nnue.evaluate(before);
            if (nnue.update(before, before) != NNUE::UpdateStatus::Unchanged ||
                nnue.update(before, Position{}) != NNUE::UpdateStatus::InvalidPosition) return false;
            if (nnue.evaluate(before) != initial_score) return false;

            Position after = before;
            after.pieces[2].square = 21;
            nnue.push();
            nnue.update(before, after);
            TAPA::NNUE reconstructed;
            if (!reconstructed.load_model(test_model)) return false;
            reconstructed.initialize(0);
            const int32_t incremental_score = nnue.evaluate_current();
            const int32_t reconstructed_score = reconstructed.evaluate(after);
            if (incremental_score != reconstructed_score) {
                std::printf("nnue: incremental=%d reconstructed=%d\n", incremental_score, reconstructed_score);
                return false;
            }
            nnue.pop();
            if (nnue.evaluate_current() != initial_score) {
                std::printf("nnue: pop restore mismatch\n");
                return false;
            }

            Position pawn_before = kings_only();
            add(pawn_before, PieceType::Pawn, PieceColor::White, 12);
            Position pawn_after = pawn_before;
            pawn_after.pieces[2].square = 20;
            nnue.evaluate(pawn_before);
            nnue.update(pawn_before, pawn_after);
            const int32_t pawn_incremental = nnue.evaluate_current();
            const int32_t pawn_reconstructed = reconstructed.evaluate(pawn_after);
            if (pawn_incremental != pawn_reconstructed) {
                std::printf("nnue: pawn incremental=%d reconstructed=%d\n", pawn_incremental, pawn_reconstructed);
                return false;
            }

            Position repeat_before = pawn_before;
            add(repeat_before, PieceType::Pawn, PieceColor::Black, 52);
            const int black_pawn_index = static_cast<int>(repeat_before.num_pieces - 1);
            Position repeat_after = repeat_before;
            repeat_after.pieces[black_pawn_index].square = 44;
            (void)nnue.evaluate(repeat_before);
            const auto misses_before = nnue.pawn_cache_stats().misses;
            const auto hits_before = nnue.pawn_cache_stats().hits;
            const auto insertions_before = nnue.pawn_cache_insertions();
            nnue.update(repeat_before, repeat_after);
            const auto misses_after = nnue.pawn_cache_stats().misses;
            const auto hits_after = nnue.pawn_cache_stats().hits;
            const int repeated_score = reconstructed.evaluate(repeat_after);
            const bool cache_consistent = nnue.evaluate_current() == repeated_score && misses_after == misses_before &&
                hits_after == hits_before && nnue.pawn_cache_insertions() == insertions_before + 1;
            if (!cache_consistent) std::printf("nnue: cache current=%d reconstructed=%d misses=%zu/%zu hits=%zu\\n",
                nnue.evaluate_current(), repeated_score, misses_before, misses_after, hits_after);
            return cache_consistent;
        }

        bool memory_layout_tests()
        {
            if (sizeof(TAPA::Accumulator::PawnWeights) !=
                64 * TAPA::Accumulator::PawnNeuronCount * sizeof(TAPA::Accumulator::PawnWeight) +
                sizeof(TAPA::Accumulator::PawnAccumulatorValue)) return false;
            if (sizeof(TAPA::Accumulator::PawnAccumulator) !=
                TAPA::Accumulator::PawnNeuronCount * sizeof(TAPA::Accumulator::PawnAccumulatorValue)) return false;
            if (TAPA::Accumulator::PawnAccumulatorCache::capacity() != 64) return false;
            std::cout << "sizeof(PawnWeights)=" << sizeof(TAPA::Accumulator::PawnWeights)
                      << " sizeof(PieceWeights struct)=" << sizeof(TAPA::Accumulator::PieceWeights)
                      << " piece_weight_table_bytes=" << (TAPA::Accumulator::PieceFeatureCount *
                          TAPA::Accumulator::NeuronCount * sizeof(TAPA::Accumulator::PieceWeight))
                      << " sizeof(NetworkWeights)=" << sizeof(TAPA::Network::NetworkWeights)
                      << " sizeof(PawnAccumulator)=" << sizeof(TAPA::Accumulator::PawnAccumulator)
                      << " sizeof(PawnAccumulatorCache)=" << sizeof(TAPA::Accumulator::PawnAccumulatorCache)
                      << "\n";
            return true;
        }
    }

    bool run_all_tests()
    {
        const bool features_ok = feature_tests();
        const bool accumulator_ok = accumulator_tests();
        const bool nnue_ok = nnue_tests();
        const bool pawn_ok = pawn_accumulator_tests();
        const bool model_ok = model_tests();
        const bool differential_ok = differential_feature_tests();
        const bool move_delta_ok = incremental_move_delta_tests();
        const bool random_sequences_ok = differential_random_sequence_tests();
        const bool random_keys_ok = random_pawn_key_roundtrip_tests();
        const bool piece_ok = piece_accumulator_tests();
        const bool move_ok = move_tests();
        const bool memory_ok = memory_layout_tests();
        if (!features_ok)
            std::printf("feature tests failed\n");
        if (!accumulator_ok)
            std::printf("accumulator tests failed\n");
        if (!nnue_ok)
            std::printf("nnue tests failed\n");
        if (!pawn_ok)
            std::printf("pawn accumulator tests failed\n");
        if (!model_ok)
            std::printf("model tests failed\n");
        if (!differential_ok)
            std::printf("differential tests failed\n");
        if (!move_delta_ok)
            std::printf("move delta tests failed\n");
        if (!random_sequences_ok)
            std::printf("random sequence differential tests failed\n");
        if (!random_keys_ok)
            std::printf("random pawn key roundtrip tests failed\n");
        if (!piece_ok)
            std::printf("piece accumulator tests failed\n");
        if (!move_ok)
            std::printf("move tests failed\n");
        if (!memory_ok)
            std::printf("memory layout tests failed\n");
        return features_ok && accumulator_ok && nnue_ok && pawn_ok && model_ok && differential_ok &&
               move_delta_ok && random_sequences_ok && random_keys_ok && piece_ok && move_ok && memory_ok;
    }
}

int main()
{
    return TAPA::Tests::run_all_tests() ? 0 : 1;
}

