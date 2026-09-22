#include "features.h"
#include <cassert>

namespace TAPA {
namespace Features {

bool FeatureSet::add(const Feature& feature) {
    if (size >= features.size()) {
        assert(false && "FeatureSet capacity exceeded");
        return false;
    }
    features[size++] = feature;
    return true;
}

bool FeatureSet::remove(const Feature& feature) {
    for (std::size_t i = 0; i < size; ++i) {
        if (features[i] == feature) {
            features[i] = features[size - 1];
            --size;
            return true;
        }
    }
    return false;
}

bool is_valid_position(const Position& position) noexcept {
    for (int color = 0; color < ColorCount; ++color) {
        if (!is_valid_square(position.kings[color])) return false;
        for (int type = 0; type < 4; ++type) {
            if ((position.pieces[color][type] & position.pawns[color]) != 0) return false;
        }
    }
    return true;
}

PawnStructureKey pawn_structure_key(const Position& position) noexcept {
    return {position.pawns[static_cast<int>(Color::White)],
            position.pawns[static_cast<int>(Color::Black)]};
}

void generate_features(const Position& position, FeatureSet& out) {
    out.clear();
    const PawnStructureKey pawns = pawn_structure_key(position);

    for (int perspective = 0; perspective < ColorCount; ++perspective) {
        const auto side = static_cast<Color>(perspective);
        const auto king = position.kings[perspective];
        for (int piece_color = 0; piece_color < ColorCount; ++piece_color) {
            const auto color = static_cast<Color>(piece_color);
            for (int type = 0; type < 4; ++type) {
                Bitboard occupied = position.pieces[piece_color][type];
                while (occupied != 0) {
                    const Square square = static_cast<Square>(std::countr_zero(occupied));
                    occupied &= occupied - 1;
                    out.add({pawns, {color, static_cast<PieceType>(type), square}, king, side});
                }
            }
        }
    }
}

FeatureDelta update_features(const Position& previous, const Position& current) {
    FeatureDelta delta;
    delta.old_pawn_structure = pawn_structure_key(previous);
    delta.new_pawn_structure = pawn_structure_key(current);

    FeatureSet old_features;
    FeatureSet new_features;
    generate_features(previous, old_features);
    generate_features(current, new_features);

    for (std::size_t i = 0; i < old_features.size; ++i) {
        bool still_present = false;
        for (std::size_t j = 0; j < new_features.size; ++j) {
            if (old_features.features[i] == new_features.features[j]) {
                still_present = true;
                break;
            }
        }
        if (!still_present) delta.removed.add(old_features.features[i]);
    }
    for (std::size_t i = 0; i < new_features.size; ++i) {
        bool was_present = false;
        for (std::size_t j = 0; j < old_features.size; ++j) {
            if (new_features.features[i] == old_features.features[j]) {
                was_present = true;
                break;
            }
        }
        if (!was_present) delta.added.add(new_features.features[i]);
    }
    return delta;
}

} // namespace Features
} // namespace TAPA