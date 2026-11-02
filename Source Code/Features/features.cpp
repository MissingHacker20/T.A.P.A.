// features.cpp
#include "features.h"
#include <algorithm>

namespace TAPA {
namespace Features {

int Position::king_square(PieceColor color) const {
    for (std::size_t i = 0; i < num_pieces; ++i) {
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

void generate_features(const Position& pos, FeatureSet& out) {
    out.clear();
    if (!pos.is_valid()) return;
    constexpr std::size_t MaxGeneratedFeatures = MaxFeatures;
    static_assert(MaxGeneratedFeatures == MaxFeatures);
    std::array<Feature, MaxGeneratedFeatures> canonical{};
    std::size_t count = 0;
    for (int perspective_value = 0; perspective_value < PieceColors; ++perspective_value) {
        const auto perspective = static_cast<PieceColor>(perspective_value);
        const int king_square = pos.king_square(perspective);
        for (std::size_t i = 0; i < pos.num_pieces; ++i) {
            const Piece& piece = pos.pieces[i];
            if (piece.type == PieceType::King && piece.color == perspective) continue;
            if (piece.type == PieceType::Pawn && is_structural_pawn(piece.color, piece.square)) continue;
            if (count >= canonical.size()) { out.clear(); return; }
            canonical[count++] = Feature{encode_index(king_square, static_cast<int>(piece.color),
                static_cast<int>(piece.type), piece.square), 1, perspective};
        }
    }
    if (count > MaxFeatures) return;
    std::sort(canonical.begin(), canonical.begin() + count, [](const Feature& lhs, const Feature& rhs) {
        if (lhs.index != rhs.index) return lhs.index < rhs.index;
        return static_cast<unsigned int>(lhs.perspective) < static_cast<unsigned int>(rhs.perspective);
    });
    for (std::size_t i = 0; i < count; ++i)
        if (!out.add(canonical[i])) {
            out.clear();
            return;
        }
}

FeatureDelta update_features(const Position& previous, const Position& current) {
    FeatureDelta delta;
    if (!previous.is_valid() || !current.is_valid()) return delta;
    FeatureSet prev_set, curr_set;
    generate_features(previous, prev_set);
    generate_features(current, curr_set);
    if ((prev_set.size == 0 && previous.num_pieces > 2) ||
        (curr_set.size == 0 && current.num_pieces > 2)) return delta;

    delta.previous_pawn_structure = compute_pawn_structure_key(previous);
    delta.current_pawn_structure = compute_pawn_structure_key(current);
    if (prev_set.size > MaxFeatures || curr_set.size > MaxFeatures) return FeatureDelta{};

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