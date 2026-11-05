// features.cpp
#include "features.h"
#include <algorithm>
#include <bit>

namespace TAPA {
namespace Features {

int Position::king_square(PieceColor color) const {
    const std::size_t count = num_pieces <= std::size(pieces) ? num_pieces : std::size(pieces);
    for (std::size_t i = 0; i < count; ++i) {
        if (pieces[i].color == color && pieces[i].type == PieceType::King) {
            return pieces[i].square;
        }
    }
    return -1;
}

bool Position::is_valid() const noexcept {
    if (num_pieces > std::size(pieces)) return false;
    bool kings[PieceColors] = {false, false};
    for (std::size_t i = 0; i < num_pieces; ++i) {
        const Piece& piece = pieces[i];
        if (!is_valid_square(piece.square)) return false;
        for (std::size_t j = 0; j < i; ++j)
            if (pieces[j].square == piece.square) return false;
        const auto color_value = static_cast<unsigned int>(piece.color);
        const auto type_value = static_cast<unsigned int>(piece.type);
        if (color_value >= PieceColors || type_value >= PieceTypes) return false;
        const int color = static_cast<int>(color_value);
        if (piece.type == PieceType::King) {
            if (kings[color]) return false;
            kings[color] = true;
        }
    }
    return kings[0] && kings[1];
}

bool Position::is_feature_compatible() const noexcept {
    // A position with more pieces than representable slots cannot be encoded
    // safely; reject it instead of reading past the end of pieces[].
    if (num_pieces > std::size(pieces)) return false;
    bool kings[PieceColors] = {false, false};
    for (std::size_t i = 0; i < num_pieces; ++i) {
        const Piece& piece = pieces[i];
        const auto color = static_cast<unsigned int>(piece.color);
        const auto type = static_cast<unsigned int>(piece.type);
        if (!is_valid_square(piece.square) || color >= PieceColors || type >= PieceTypes) return false;
        if (piece.type == PieceType::King) kings[color] = true;
    }
    return kings[0] && kings[1];
}

bool position_from_bitboards(const PieceBitboards& bitboards, Position& output) noexcept {
    Position candidate{};
    std::uint64_t occupied = 0;
    for (int color = 0; color < PieceColors; ++color) {
        for (int type = 0; type < PieceTypes; ++type) {
            std::uint64_t squares = bitboards[static_cast<std::size_t>(color * PieceTypes + type)];
            if ((occupied & squares) != 0) return false;
            occupied |= squares;
            while (squares != 0) {
                if (candidate.num_pieces >= std::size(candidate.pieces)) return false;
                const int square = static_cast<int>(std::countr_zero(squares));
                candidate.pieces[candidate.num_pieces++] = {
                    static_cast<PieceType>(type), static_cast<PieceColor>(color), square
                };
                squares &= squares - 1;
            }
        }
    }
    if (!candidate.is_valid()) return false;
    output = candidate;
    return true;
}

bool FeatureSet::add(const Feature& f) {
    // Avoid duplicate indices
    for (std::size_t i = 0; i < size; ++i) {
        if (features[i].index == f.index && features[i].perspective == f.perspective) {
            return false; // already present
        }
    }
    if (size >= MaxFeatures) {
        return false; // overflow
    }
    features[size++] = f;
    return true;
}

bool FeatureSet::remove(std::size_t index) {
    for (std::size_t i = 0; i < size; ++i) {
        if (features[i].index == index) {
            features[i] = features[size - 1];
            --size;
            return true;
        }
    }
    return false;
}

bool FeatureSet::remove(const Feature& feature) {
    for (std::size_t i = 0; i < size; ++i) {
        if (features[i].index == feature.index && features[i].perspective == feature.perspective) {
            features[i] = features[size - 1];
            --size;
            return true;
        }
    }
    return false;
}

bool generate_features(const Position& pos, FeatureSet& out) {
    out.clear();
    if (!pos.is_feature_compatible()) return false;
    std::array<Feature, MaxFeatures> canonical{};
    std::size_t count = 0;
    const int king_squares[PieceColors] = {
        pos.king_square(PieceColor::White), pos.king_square(PieceColor::Black)
    };
    for (int perspective_value = 0; perspective_value < PieceColors; ++perspective_value) {
        const auto perspective = static_cast<PieceColor>(perspective_value);
        for (std::size_t i = 0; i < pos.num_pieces; ++i) {
            const Piece& piece = pos.pieces[i];
            if (piece.type == PieceType::King && piece.color == perspective) continue;
            if (piece.type == PieceType::Pawn && is_structural_pawn(piece.color, piece.square)) continue;
            canonical[count++] = Feature{encode_index(king_squares[perspective_value], static_cast<int>(piece.color),
                static_cast<int>(piece.type), piece.square), 1, perspective};
        }
    }
    std::sort(canonical.begin(), canonical.begin() + count, [](const Feature& lhs, const Feature& rhs) {
        if (lhs.index != rhs.index) return lhs.index < rhs.index;
        return static_cast<unsigned int>(lhs.perspective) < static_cast<unsigned int>(rhs.perspective);
    });
    std::copy_n(canonical.begin(), count, out.features.begin());
    out.size = count;
    return true;
}

FeatureDelta update_features(const Position& previous, const Position& current) {
    FeatureDelta delta;
    FeatureSet prev_set, curr_set;
    if (!generate_features(previous, prev_set) || !generate_features(current, curr_set)) return delta;

    delta.previous_pawn_structure = compute_pawn_structure_key(previous);
    delta.current_pawn_structure = compute_pawn_structure_key(current);
    // Feature indices are canonical and bounded by the encoder. A sorted
    // merge gives linear delta construction without allocating a map.
    auto by_key = [](const Feature& lhs, const Feature& rhs) {
        if (lhs.index != rhs.index) return lhs.index < rhs.index;
        return static_cast<int>(lhs.perspective) < static_cast<int>(rhs.perspective);
    };
    std::array<Feature, MaxFeatures> previous_sorted = prev_set.features;
    std::array<Feature, MaxFeatures> current_sorted = curr_set.features;
    std::sort(previous_sorted.begin(), previous_sorted.begin() + prev_set.size, by_key);
    std::sort(current_sorted.begin(), current_sorted.begin() + curr_set.size, by_key);
    std::size_t previous_index = 0;
    std::size_t current_index = 0;
    while (previous_index < prev_set.size || current_index < curr_set.size) {
        if (previous_index == prev_set.size) { delta.added.add(current_sorted[current_index++]); continue; }
        if (current_index == curr_set.size) { delta.removed.add(previous_sorted[previous_index++]); continue; }
        if (by_key(previous_sorted[previous_index], current_sorted[current_index])) { delta.removed.add(previous_sorted[previous_index++]); continue; }
        if (by_key(current_sorted[current_index], previous_sorted[previous_index])) { delta.added.add(current_sorted[current_index++]); continue; }
        ++previous_index;
        ++current_index;
    }
    delta.valid = true;
    return delta;
}

} // namespace Features
} // namespace TAPA