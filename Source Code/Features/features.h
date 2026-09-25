// features.h
#pragma once

#include <array>
#include <cstdint>
#include <cstddef>

namespace TAPA {
namespace Features {

constexpr int BoardSquares = 64;
constexpr int PieceColors = 2; // White=0, Black=1
constexpr int PieceTypes = 6; // Pawn, Knight, Bishop, Rook, Queen, King
constexpr int KingSquares = BoardSquares;

// Index encoding: ((king_square * PieceColors + piece_color) * PieceTypes + piece_type) * BoardSquares + piece_square
inline std::size_t encode_index(int king_square, int piece_color, int piece_type, int piece_square) {
    return (((king_square * PieceColors + piece_color)
            * PieceTypes + piece_type)
            * BoardSquares) + piece_square;
}

// Feature representation
struct Feature {
    std::size_t index; // computed via encode_index
    int value;         // typically 1
};

// FeatureSet: stores active features without dynamic allocation in typical case
constexpr int MaxFeatures = 512; // conservative upper bound
struct FeatureSet {
    std::array<Feature, MaxFeatures> features{};
    std::size_t size = 0;

    bool add(const Feature& f);
    bool remove(std::size_t index); // remove by index
    void clear() { size = 0; }
};

// Position representation (simplified for demonstration)
enum class PieceType : uint8_t { Pawn = 0, Knight, Bishop, Rook, Queen, King };
enum class PieceColor : uint8_t { White = 0, Black = 1 };

struct Piece {
    PieceType type;
    PieceColor color;
    int square; // 0-63
};

struct Position {
    // For simplicity, we store a list of pieces; in real engine this would be bitboards.
    Piece pieces[32]; // max 32 pieces
    std::size_t num_pieces = 0;

    // Helper to get king square for a color
    int king_square(PieceColor color) const;
};

// Generate full feature set from a position
void generate_features(const Position& pos, FeatureSet& out);

// Compute feature delta between two positions (for incremental update)
struct FeatureDelta {
    FeatureSet added;
    FeatureSet removed;
};

FeatureDelta update_features(const Position& previous, const Position& current);

} // namespace Features
} // namespace TAPA