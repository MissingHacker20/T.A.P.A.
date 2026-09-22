#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace TAPA {
namespace Features {

constexpr int BoardSquares = 64;
constexpr int ColorCount = 2;

using Bitboard = std::uint64_t;
using Square = std::uint8_t;

enum class Color : std::uint8_t { White = 0, Black = 1 };
enum class PieceType : std::uint8_t { Knight = 0, Bishop, Rook, Queen };

constexpr Color opposite(Color color) {
    return color == Color::White ? Color::Black : Color::White;
}

struct PawnStructureKey {
    Bitboard white = 0;
    Bitboard black = 0;

    friend constexpr bool operator==(const PawnStructureKey&, const PawnStructureKey&) = default;
};

struct PawnStructureKeyHash {
    std::size_t operator()(const PawnStructureKey& key) const noexcept {
        // The two bitboards are deliberately kept distinguishable.
        const std::size_t h1 = std::hash<Bitboard>{}(key.white);
        const std::size_t h2 = std::hash<Bitboard>{}(key.black);
        return h1 ^ (h2 + static_cast<std::size_t>(0x9e3779b9) + (h1 << 6) + (h1 >> 2));
    }
};

struct Piece {
    Color color = Color::White;
    PieceType type = PieceType::Knight;
    Square square = 0;

    friend constexpr bool operator==(const Piece&, const Piece&) = default;
};

struct King {
    Color color = Color::White;
    Square square = 0;

    friend constexpr bool operator==(const King&, const King&) = default;
};

// A position view used by the feature generator. MissingPawn can populate this
// view without making Features depend on the engine's internal board classes.
struct Position {
    Bitboard pawns[ColorCount]{};
    Bitboard pieces[ColorCount][4]{}; // Knight, Bishop, Rook, Queen
    Square kings[ColorCount]{};
};

struct Feature {
    PawnStructureKey pawn_structure;
    Piece piece;
    Square king_square = 0; // king belonging to perspective
    Color perspective = Color::White;

    friend constexpr bool operator==(const Feature&, const Feature&) = default;
};

// A legal chess position has at most 15 non-king, non-pawn pieces per side
// (seven original pieces plus eight promoted pieces): 30 features total.
constexpr int MaxPiecesPerSide = 15;
constexpr int MaxFeatures = ColorCount * MaxPiecesPerSide;

struct FeatureSet {
    std::array<Feature, MaxFeatures> features{};
    std::size_t size = 0;

    bool add(const Feature& feature);
    bool remove(const Feature& feature);
    void clear() noexcept { size = 0; }
};

PawnStructureKey pawn_structure_key(const Position& position) noexcept;
void generate_features(const Position& position, FeatureSet& out);

struct FeatureDelta {
    FeatureSet added;
    FeatureSet removed;
    PawnStructureKey old_pawn_structure;
    PawnStructureKey new_pawn_structure;

    bool pawn_structure_changed() const noexcept {
        return !(old_pawn_structure == new_pawn_structure);
    }
};

// Computes the piece/king delta and exposes pawn-key changes separately for the
// Pawn Accumulator. The caller may pass a new Position after making a move.
FeatureDelta update_features(const Position& previous, const Position& current);

} // namespace Features
} // namespace TAPA
namespace std {
template<> struct hash<TAPA::Features::PawnStructureKey> {
    std::size_t operator()(const TAPA::Features::PawnStructureKey& key) const noexcept {
        return TAPA::Features::PawnStructureKeyHash{}(key);
    }
};
}