// features.h
#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <limits>

namespace TAPA {
namespace Features {

constexpr int BoardSquares = 64;
constexpr int PieceColors = 2; // White=0, Black=1
constexpr int PieceTypes = 6; // Pawn, Knight, Bishop, Rook, Queen, King
constexpr int KingSquares = BoardSquares;

// Feature-index schema v1. A perspective's king square conditions every piece feature.
// Index order: king square, piece color, piece type, piece square; perspective is a separate tag.
inline std::size_t encode_index(int king_square, int piece_color, int piece_type, int piece_square) {
    if (king_square < 0 || king_square >= BoardSquares || piece_color < 0 || piece_color >= PieceColors ||
        piece_type < 0 || piece_type >= PieceTypes || piece_square < 0 || piece_square >= BoardSquares)
        return std::numeric_limits<std::size_t>::max();
    return (((static_cast<std::size_t>(king_square) * PieceColors + static_cast<std::size_t>(piece_color))
            * PieceTypes + static_cast<std::size_t>(piece_type))
            * BoardSquares) + static_cast<std::size_t>(piece_square);
}

// Position representation (simplified for demonstration)
enum class PieceType : uint8_t { Pawn = 0, Knight, Bishop, Rook, Queen, King };
enum class PieceColor : uint8_t { White = 0, Black = 1 };

// Feature representation
struct Feature {
    std::size_t index; // computed via encode_index
    int value = 1;     // signed contribution
    PieceColor perspective = PieceColor::White;
};

// FeatureSet: stores active features without dynamic allocation in typical case
constexpr int MaxFeatures = 2 * BoardSquares;
struct FeatureSet {
    std::array<Feature, MaxFeatures> features{};
    std::size_t size = 0;

    bool add(const Feature& f);
    bool remove(std::size_t index); // remove by index
    bool remove(const Feature& feature);
    void clear() { size = 0; }
};

struct Piece {
    PieceType type;
    PieceColor color;
    int square; // 0-63
};

struct Position {
    // A chessboard has at most one piece per square, including after promotions.
    Piece pieces[BoardSquares]{};
    std::size_t num_pieces = 0;

    int king_square(PieceColor color) const;
    bool is_valid() const noexcept;
    // Checks only that the position can be represented safely by the feature schema.
    // It does not check piece overlap or chess legality.
    bool is_feature_compatible() const noexcept;
};

using PieceBitboards = std::array<std::uint64_t, PieceColors * PieceTypes>;

// Bitboard order is White Pawn/Knight/Bishop/Rook/Queen/King, then Black.
// The output is unchanged when the bitboards do not describe a valid position.
bool position_from_bitboards(const PieceBitboards& bitboards, Position& output) noexcept;

inline bool operator==(const Position& lhs, const Position& rhs) noexcept {
    if (lhs.num_pieces != rhs.num_pieces) return false;
    const std::size_t count = lhs.num_pieces <= std::size(lhs.pieces) ? lhs.num_pieces : std::size(lhs.pieces);
    for (std::size_t i = 0; i < count; ++i) {
        if (lhs.pieces[i].type != rhs.pieces[i].type ||
            lhs.pieces[i].color != rhs.pieces[i].color ||
            lhs.pieces[i].square != rhs.pieces[i].square) return false;
    }
    return true;
}

inline bool operator!=(const Position& lhs, const Position& rhs) noexcept {
    return !(lhs == rhs);
}

// Canonical PawnStructureKey v1: white ranks 2..5 use bits 0..31, black ranks 7..4 use 32..63.
// Within each rank byte files a..h map to increasing bit positions, independent of host endianness.
using PawnStructureKey = uint64_t;

constexpr int PawnStructRanksPerColor = 4;
constexpr int FilesPerRank = 8;
constexpr int PawnStructBitsPerColor = PawnStructRanksPerColor * FilesPerRank; // 32

// Helper to convert square to rank and file (0-based)
constexpr bool is_valid_square(int square) noexcept {
    return square >= 0 && square < BoardSquares;
}

constexpr int square_rank(int sq) { return sq / 8; }
constexpr int square_file(int sq) { return sq % 8; }

// Determine if a pawn is structural (part of pawn structure) based on color and square
inline bool is_valid_color(PieceColor color) noexcept {
    return static_cast<unsigned int>(color) < PieceColors;
}

inline bool is_structural_pawn(PieceColor color, int square) noexcept {
    if (!is_valid_color(color) || !is_valid_square(square)) return false;
    const int rank = square_rank(square);
    // White: ranks 2..5; Black: ranks 7..4 (towards promotion).
    return color == PieceColor::White ? (rank >= 1 && rank <= 4)
                                      : (rank >= 3 && rank <= 6);
}

inline bool is_pawn_structure_square(PieceColor color, int square) noexcept {
    return is_structural_pawn(color, square);
}

// Determine if a pawn is advanced (treated as piece)
inline bool is_advanced_pawn(PieceColor color, int square) noexcept {
    if (!is_valid_color(color) || !is_valid_square(square)) return false;
    const int rank = square_rank(square);
    // Promotion ranks (1 and 8) contain no pawn in a valid chess position.
    return color == PieceColor::White ? (rank >= 5 && rank <= 6)
                                      : (rank >= 1 && rank <= 2);
}

inline bool is_piece_pawn(PieceColor color, int square) noexcept {
    return is_advanced_pawn(color, square);
}

// Compute pawn structure key from position
inline PawnStructureKey make_pawn_structure_key(const Position& pos) noexcept {
    PawnStructureKey key = 0;
    const std::size_t count = pos.num_pieces < std::size(pos.pieces) ? pos.num_pieces : std::size(pos.pieces);
    for (std::size_t i = 0; i < count; ++i) {
        const Piece& pc = pos.pieces[i];
        if (pc.type != PieceType::Pawn || !is_structural_pawn(pc.color, pc.square)) continue;
        const int rank = square_rank(pc.square);
        const int file = square_file(pc.square);
        // Canonical v1 mapping is relative to each color's forward direction; see docs/architecture.md.
        const int level = pc.color == PieceColor::White ? rank - 1 : 6 - rank;
        const int bit = (pc.color == PieceColor::White ? 0 : 32) + level * 8 + file;
        key |= PawnStructureKey{1} << bit;
    }
    return key;
}

inline PawnStructureKey compute_pawn_structure_key(const Position& pos) noexcept {
    return make_pawn_structure_key(pos);
}

struct DecodedPawn {
    PieceColor color;
    int square;
};

inline std::size_t decode_pawn_structure_key(PawnStructureKey key,
                                             std::array<DecodedPawn, 32>& output) noexcept {
    std::size_t count = 0;
    for (std::size_t bit = 0; bit < 64 && count < output.size(); ++bit) {
        if ((key & (PawnStructureKey{1} << bit)) == 0) continue;
        const bool white = bit < 32;
        const std::size_t local = white ? bit : bit - 32;
        const int level = static_cast<int>(local / 8);
        const int file = static_cast<int>(local % 8);
        const int rank = white ? level + 1 : 6 - level;
        output[count++] = {white ? PieceColor::White : PieceColor::Black, rank * 8 + file};
    }
    return count;
}

// Generate full feature set from a position (excluding structural pawns)
bool generate_features(const Position& pos, FeatureSet& out);

// Compute feature delta between two positions (for incremental update).
// Structural pawn changes are exposed separately so an accumulator/cache can
// update its pawn representation without regenerating all piece features.
struct FeatureDelta {
    FeatureSet added;
    FeatureSet removed;
    PawnStructureKey previous_pawn_structure = 0;
    PawnStructureKey current_pawn_structure = 0;
    bool valid = false;
};

FeatureDelta update_features(const Position& previous, const Position& current);

} // namespace Features
} // namespace TAPA