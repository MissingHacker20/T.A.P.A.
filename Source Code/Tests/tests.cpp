#include "tests.h"
#include <cassert>
#include <iostream>

namespace TAPA {
namespace Tests {

namespace {
using namespace Features;

void test_pawn_structure() {
    Position position{};
    position.pawns[0] = 1ULL << 8;
    position.pawns[1] = 1ULL << 55;
    const auto key = pawn_structure_key(position);
    assert(key.white == (1ULL << 8));
    assert(key.black == (1ULL << 55));
    const PawnStructureKey swapped{key.black, key.white};
    assert(key != swapped);
}

void test_feature_generation() {
    Position position{};
    position.kings[0] = 4;
    position.kings[1] = 60;
    position.pawns[0] = 1ULL << 8;
    position.pawns[1] = 1ULL << 55;
    position.pieces[0][static_cast<int>(PieceType::Knight)] = 1ULL << 21;
    position.pieces[0][static_cast<int>(PieceType::Bishop)] = 1ULL << 2;
    position.pieces[1][static_cast<int>(PieceType::Rook)] = 1ULL << 56;
    position.pieces[1][static_cast<int>(PieceType::Queen)] = 1ULL << 59;

    assert(is_valid_position(position));
    FeatureSet features;
    generate_features(position, features);
    assert(features.size == 8);
    bool found_knight = false;
    bool found_bishop = false;
    bool found_rook = false;
    bool found_queen = false;
    for (std::size_t i = 0; i < features.size; ++i) {
        const auto& feature = features.features[i];
        assert(feature.piece.type != PieceType::Knight || feature.piece.square == 21);
        assert(feature.piece.type != PieceType::Bishop || feature.piece.square == 2);
        assert(feature.piece.type != PieceType::Rook || feature.piece.square == 56);
        assert(feature.piece.type != PieceType::Queen || feature.piece.square == 59);
        found_knight |= feature.piece.type == PieceType::Knight;
        found_bishop |= feature.piece.type == PieceType::Bishop;
        found_rook |= feature.piece.type == PieceType::Rook;
        found_queen |= feature.piece.type == PieceType::Queen;
    }
    assert(found_knight && found_bishop && found_rook && found_queen);
}

void test_feature_delta() {
    Position previous{};
    previous.kings[0] = 4;
    previous.kings[1] = 60;
    previous.pieces[0][static_cast<int>(PieceType::Knight)] = 1ULL << 21;

    Position current = previous;
    current.pieces[0][static_cast<int>(PieceType::Knight)] = 1ULL << 22;
    const auto delta = update_features(previous, current);
    assert(!delta.pawn_structure_changed());
    assert(delta.added.size == 2);
    assert(delta.removed.size == 2);
}

} // namespace
bool run_all_tests() {
    test_pawn_structure();
    test_feature_generation();
    test_feature_delta();
    return true;
}

} // namespace Tests
} // namespace TAPA
int main() {
    try {
        return TAPA::Tests::run_all_tests() ? 0 : 1;
    } catch (...) {
        std::cerr << "T.A.P.A. tests failed\n";
        return 1;
    }
}
