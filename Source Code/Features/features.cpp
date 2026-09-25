// features.cpp
#include "features.h"

namespace TAPA {
namespace Features {

int Position::king_square(PieceColor color) const {
    for (std::size_t i = 0; i < num_pieces; ++i) {
        if (pieces[i].color == color && pieces[i].type == PieceType::King) {
            return pieces[i].square;
        }
    }
    // Should not happen in a valid position
    return -1;
}

bool FeatureSet::add(const Feature& f) {
    // Avoid duplicate indices
    for (std::size_t i = 0; i < size; ++i) {
        if (features[i].index == f.index) {
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
            // swap with last and shrink
            features[i] = features[size - 1];
            --size;
            return true;
        }
    }
    return false; // not found
}

void generate_features(const Position& pos, FeatureSet& out) {
    out.clear();
    for (int persp_int = 0; persp_int < PieceColors; ++persp_int) {
        PieceColor persp = static_cast<PieceColor>(persp_int);
        int king_sq = pos.king_square(persp);
        if (king_sq < 0) continue; // skip if king missing (should not happen)
        for (std::size_t i = 0; i < pos.num_pieces; ++i) {
            const Piece& pc = pos.pieces[i];
            // Skip the king of the perspective side
            if (pc.color == persp && pc.type == PieceType::King) {
                continue;
            }
            Feature f;
            f.index = encode_index(king_sq,
                                   static_cast<int>(pc.color),
                                   static_cast<int>(pc.type),
                                   pc.square);
            f.value = 1;
            out.add(f);
        }
    }
}

FeatureDelta update_features(const Position& previous, const Position& current) {
    FeatureSet prev_set, curr_set;
    generate_features(previous, prev_set);
    generate_features(current, curr_set);

    FeatureDelta delta;
    // Added: in curr_set but not in prev_set
    for (std::size_t i = 0; i < curr_set.size; ++i) {
        const Feature& f = curr_set.features[i];
        bool found = false;
        for (std::size_t j = 0; j < prev_set.size; ++j) {
            if (prev_set.features[j].index == f.index) {
                found = true;
                break;
            }
        }
        if (!found) {
            delta.added.add(f);
        }
    }
    // Removed: in prev_set but not in curr_set
    for (std::size_t i = 0; i < prev_set.size; ++i) {
        const Feature& f = prev_set.features[i];
        bool found = false;
        for (std::size_t j = 0; j < curr_set.size; ++j) {
            if (curr_set.features[j].index == f.index) {
                found = true;
                break;
            }
        }
        if (!found) {
            delta.removed.add(f);
        }
    }
    return delta;
}

} // namespace Features
} // namespace TAPA