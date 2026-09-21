// features.h
#pragma once

#include <cstdint>
#include <array>

namespace TAPA {
namespace Features {

constexpr int BoardSquares = 64;
constexpr int PieceColors = 2; // White, Black
constexpr int PieceTypes = 6; // Pawn, Knight, Bishop, Rook, Queen, King
constexpr int KingSquares = BoardSquares;

// Feature represents a single HalfKP feature
struct Feature {
    int index;   // feature index
    int value;   // feature value (typically 1)
};

// FeatureSet stores active features without dynamic allocation in typical case
// For simplicity, we use a fixed-size container; max features = 2 * number of pieces * ?
// We'll define a reasonable max.
constexpr int MaxFeatures = 256; // placeholder

struct FeatureSet {
    std::array<Feature, MaxFeatures> features{};
    int size = 0;

    void add(const Feature& f);
    void remove(int index); // remove by index (if needed)
    void clear() { size = 0; }
};

// HalfKP index encoding
inline int encode_index(int king_square, int piece_color, int piece_type, int piece_square) {
    // king_square: 0..63
    // piece_color: 0..1
    // piece_type: 0..5
    // piece_square: 0..63
    return (((king_square * PieceColors + piece_color)
            * PieceTypes + piece_type)
            * BoardSquares) + piece_square;
}

// Generate full feature set from a position (to be implemented)
// void generate_features(const Position& pos, FeatureSet& out);

// Update features incrementally (to be implemented)
// void update_features(const Move& mv, FeatureSet& out);

} // namespace Features
} // namespace TAPA