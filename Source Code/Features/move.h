#pragma once
#include "features.h"

namespace TAPA::Features
{

    // Host-supplied move descriptor. This API checks geometry and state coherence,
    // but is not a legal-move generator and does not check attacks/check legality.
    struct Move
    {
        int from = -1;
        int to = -1;
        PieceType moving_type = PieceType::Pawn;
        PieceColor color = PieceColor::White;
        int captured_square = -1;
        PieceType captured_type = PieceType::Pawn;
        bool capture = false;
        bool en_passant = false;
        bool castling = false;
        bool king_side = true;
        PieceType promotion = PieceType::Pawn;
        int rook_from = -1;
        int rook_to = -1;
    };

    bool is_valid_move(const Position &position, const Move &move) noexcept;
    bool apply_move(const Position &position, const Move &move, Position &result) noexcept;
    // Move-oriented delta; requires current to equal apply_move(previous, move).
    FeatureDelta update_features(const Position &previous, const Move &move, const Position &current) noexcept;

} // namespace TAPA::Features
