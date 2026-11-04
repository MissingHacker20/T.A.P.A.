#include "move.h"
#include <cstdlib>
#include <initializer_list>

namespace TAPA::Features {
namespace {
int find_piece(const Position& position, int square, PieceColor color, PieceType type) noexcept {
    for (std::size_t i = 0; i < position.num_pieces; ++i)
        if (position.pieces[i].square == square && position.pieces[i].color == color && position.pieces[i].type == type)
            return static_cast<int>(i);
    return -1;
}
}

bool is_valid_move(const Position& position, const Move& move) noexcept {
    if (!position.is_valid() || !is_valid_color(move.color) ||
        static_cast<unsigned int>(move.moving_type) >= PieceTypes ||
        static_cast<unsigned int>(move.captured_type) >= PieceTypes ||
        static_cast<unsigned int>(move.promotion) >= PieceTypes ||
        !is_valid_square(move.from) || !is_valid_square(move.to) || move.from == move.to) return false;
    if (find_piece(position, move.from, move.color, move.moving_type) < 0) return false;
    if (move.promotion != PieceType::Pawn && move.moving_type != PieceType::Pawn) return false;
    if (move.en_passant && move.castling) return false;
    const int moving_piece_index = find_piece(position, move.from, move.color, move.moving_type);
    const int destination_piece = [&]() {
        for (std::size_t i = 0; i < position.num_pieces; ++i)
            if (position.pieces[i].square == move.to) return static_cast<int>(i);
        return -1;
    }();
    if (move.capture) {
        if (move.captured_type == PieceType::King) return false;
        const int capture_square = move.en_passant ? move.captured_square : move.to;
        if (!is_valid_square(capture_square) || (move.en_passant && capture_square == move.to)) return false;
        const int captured = find_piece(position, capture_square,
            move.color == PieceColor::White ? PieceColor::Black : PieceColor::White, move.captured_type);
        if (captured < 0 || (move.en_passant && destination_piece >= 0) ||
            position.pieces[captured].type == PieceType::King) return false;
    } else if (move.en_passant || (destination_piece >= 0 && destination_piece != moving_piece_index)) {
        return false;
    }
    if (move.castling) {
        if (move.capture || move.moving_type != PieceType::King || !is_valid_square(move.rook_from) || !is_valid_square(move.rook_to)) return false;
        if (move.rook_from == move.to || move.rook_to == move.to || move.rook_to == move.from ||
            (destination_piece >= 0 && destination_piece != moving_piece_index)) return false;
        const bool king_side = square_file(move.to) > square_file(move.from);
        if (king_side != move.king_side ||
            (king_side && (square_file(move.rook_from) != 7 || square_file(move.rook_to) != 5)) ||
            (!king_side && (square_file(move.rook_from) != 0 || square_file(move.rook_to) != 3))) return false;
        if (find_piece(position, move.rook_to, move.color, PieceType::Rook) >= 0) return false;
        if (find_piece(position, move.rook_from, move.color, PieceType::Rook) < 0) return false;
        for (std::size_t i = 0; i < position.num_pieces; ++i) {
            const int square = position.pieces[i].square;
            if (square == move.from || square == move.rook_from) continue;
            const int file = square_file(square);
            const int rank = square_rank(square);
            const int rank_from = square_rank(move.from);
            if (rank == rank_from &&
                ((file > square_file(move.from) && file < square_file(move.to)) ||
                 (file < square_file(move.from) && file > square_file(move.to)))) return false;
        }
        if (square_rank(move.from) != square_rank(move.to) || std::abs(square_file(move.to) - square_file(move.from)) != 2) return false;
        if (move.color == PieceColor::White && square_rank(move.from) != 0) return false;
        if (move.color == PieceColor::Black && square_rank(move.from) != 7) return false;
        if (square_file(move.from) != 4 || square_file(move.to) != (move.king_side ? 6 : 2)) return false;
        if (square_rank(move.rook_from) != square_rank(move.from) || square_rank(move.rook_to) != square_rank(move.from)) return false;
        const int rook_step = move.king_side ? -1 : 1;
        for (int file = square_file(move.rook_from) + rook_step;
             file != square_file(move.rook_to); file += rook_step) {
            const int square = square_rank(move.from) * 8 + file;
            for (std::size_t i = 0; i < position.num_pieces; ++i)
                if (position.pieces[i].square == square) return false;
        }
    }
    if (move.en_passant) {
        if (!move.capture || move.moving_type != PieceType::Pawn || !is_valid_square(move.captured_square) || move.captured_square == move.to) return false;
        const int expected_captured_square = move.to + (move.color == PieceColor::White ? -8 : 8);
        if (move.captured_square != expected_captured_square) return false;
        if (square_file(move.to) == square_file(move.from) || std::abs(square_file(move.to) - square_file(move.from)) != 1) return false;
        if (move.color == PieceColor::White && square_rank(move.to) - square_rank(move.from) != 1) return false;
        if (move.color == PieceColor::Black && square_rank(move.from) - square_rank(move.to) != 1) return false;
    }
    const int from_rank = square_rank(move.from);
    const int to_rank = square_rank(move.to);
    const int forward = move.color == PieceColor::White ? 1 : -1;
    const int file_delta = std::abs(square_file(move.to) - square_file(move.from));
    if (move.moving_type == PieceType::Pawn) {
        if (move.color == PieceColor::White && to_rank == 0) return false;
        if (move.color == PieceColor::Black && to_rank == 7) return false;
        if (move.promotion != PieceType::Pawn && move.promotion != PieceType::Knight &&
            move.promotion != PieceType::Bishop && move.promotion != PieceType::Rook &&
            move.promotion != PieceType::Queen) return false;
        const bool promotion_rank = to_rank == (move.color == PieceColor::White ? 7 : 0);
        if (promotion_rank != (move.promotion != PieceType::Pawn)) return false;
        if (move.capture) {
            if (file_delta != 1 || to_rank - from_rank != forward) return false;
            if (move.en_passant && to_rank != (move.color == PieceColor::White ? 5 : 2)) return false;
        } else if (move.en_passant || file_delta != 0 ||
                   (to_rank - from_rank != forward &&
                    !(to_rank - from_rank == 2 * forward &&
                      from_rank == (move.color == PieceColor::White ? 1 : 6)))) return false;
        if (file_delta == 0 && to_rank - from_rank == 2 * forward) {
            const int passed_square = move.from + forward * 8;
            for (std::size_t i = 0; i < position.num_pieces; ++i)
                if (position.pieces[i].square == passed_square) return false;
        }
    } else if (move.promotion != PieceType::Pawn) {
        return false;
    }
    if (move.moving_type == PieceType::King && !move.castling &&
        (std::abs(square_file(move.to) - square_file(move.from)) > 1 ||
         std::abs(to_rank - from_rank) > 1)) return false;
    if (move.moving_type == PieceType::Knight) {
        const int df = std::abs(square_file(move.to) - square_file(move.from));
        const int dr = std::abs(to_rank - from_rank);
        if (!((df == 1 && dr == 2) || (df == 2 && dr == 1))) return false;
    }
    if (move.moving_type == PieceType::Bishop || move.moving_type == PieceType::Rook || move.moving_type == PieceType::Queen) {
        const int df = std::abs(square_file(move.to) - square_file(move.from));
        const int dr = std::abs(to_rank - from_rank);
        const bool diagonal = df == dr;
        const bool straight = df == 0 || dr == 0;
        if ((move.moving_type == PieceType::Bishop && !diagonal) ||
            (move.moving_type == PieceType::Rook && !straight) ||
            (move.moving_type == PieceType::Queen && !diagonal && !straight)) return false;
        const int step_file = (square_file(move.to) > square_file(move.from)) - (square_file(move.to) < square_file(move.from));
        const int step_rank = (to_rank > from_rank) - (to_rank < from_rank);
        int file = square_file(move.from) + step_file;
        int rank = from_rank + step_rank;
        while (file != square_file(move.to) || rank != to_rank) {
            const int square = rank * 8 + file;
            for (std::size_t i = 0; i < position.num_pieces; ++i)
                if (position.pieces[i].square == square) return false;
            file += step_file;
            rank += step_rank;
        }
    }
    if (move.capture && move.captured_type == PieceType::King) return false;
    return true;
}

bool apply_move(const Position& position, const Move& move, Position& result) noexcept {
    if (!is_valid_move(position, move)) return false;
    if (move.castling && move.to == move.rook_from) return false;
    if (move.castling) {
        for (std::size_t i = 0; i < position.num_pieces; ++i)
            if (position.pieces[i].square == move.rook_to && position.pieces[i].square != move.from)
                return false;
    }
    result = position;
    int moving_index = find_piece(result, move.from, move.color, move.moving_type);
    if (moving_index < 0) return false;
    if (move.capture) {
        const int captured_index = find_piece(result, move.en_passant ? move.captured_square : move.to,
                                              move.color == PieceColor::White ? PieceColor::Black : PieceColor::White,
                                              move.captured_type);
        if (captured_index < 0 || captured_index == moving_index) return false;
        const int last_index = static_cast<int>(result.num_pieces) - 1;
        result.pieces[captured_index] = result.pieces[last_index];
        --result.num_pieces;
        if (moving_index == last_index && captured_index != last_index)
            moving_index = captured_index;
    }
    result.pieces[moving_index].square = move.to;
    if (move.promotion != PieceType::Pawn) result.pieces[moving_index].type = move.promotion;
    if (move.castling) {
        const int rook_index = find_piece(result, move.rook_from, move.color, PieceType::Rook);
        if (rook_index < 0) return false;
        result.pieces[rook_index].square = move.rook_to;
    }
    return result.is_valid();
}

FeatureDelta update_features(const Position& previous, const Move& move, const Position& current) noexcept {
    FeatureDelta delta;
    if (!previous.is_feature_compatible() || !current.is_feature_compatible() ||
        !is_valid_color(move.color) || !is_valid_square(move.from) ||
        !is_valid_square(move.to) || move.from == move.to ||
        static_cast<unsigned int>(move.moving_type) >= PieceTypes ||
        static_cast<unsigned int>(move.promotion) >= PieceTypes ||
        static_cast<unsigned int>(move.captured_type) >= PieceTypes) return delta;
    if (find_piece(previous, move.from, move.color, move.moving_type) < 0) return delta;
    if (move.capture) {
        const int capture_square = move.en_passant ? move.captured_square : move.to;
        if (!is_valid_square(capture_square) ||
            find_piece(previous, capture_square,
                move.color == PieceColor::White ? PieceColor::Black : PieceColor::White,
                move.captured_type) < 0) return delta;
    }
    if (move.castling &&
        (!is_valid_square(move.rook_from) || !is_valid_square(move.rook_to) ||
         find_piece(previous, move.rook_from, move.color, PieceType::Rook) < 0)) return delta;
    delta.previous_pawn_structure = make_pawn_structure_key(previous);
    delta.current_pawn_structure = make_pawn_structure_key(current);

    auto add_piece_feature = [](FeatureSet& set, const Position& position, const Piece& piece, PieceColor perspective) {
        if (piece.type == PieceType::King && piece.color == perspective) return true;
        if (piece.type == PieceType::Pawn && is_structural_pawn(piece.color, piece.square)) return true;
        return set.add(Feature{encode_index(position.king_square(perspective), static_cast<int>(piece.color),
            static_cast<int>(piece.type), piece.square), 1, perspective});
    };

    // King-relative indices for all other pieces change when that king moves.
    for (int perspective_value = 0; perspective_value < PieceColors; ++perspective_value) {
        const auto perspective = static_cast<PieceColor>(perspective_value);
        const int old_king = previous.king_square(perspective);
        const int new_king = current.king_square(perspective);
        if (old_king != new_king) {
            for (std::size_t i = 0; i < previous.num_pieces; ++i)
                if (!add_piece_feature(delta.removed, previous, previous.pieces[i], perspective)) return FeatureDelta{};
            for (std::size_t i = 0; i < current.num_pieces; ++i)
                if (!add_piece_feature(delta.added, current, current.pieces[i], perspective)) return FeatureDelta{};
        }
    }

    const int moving_index = find_piece(previous, move.from, move.color, move.moving_type);
    if (moving_index < 0) return FeatureDelta{};
    const Piece moving_before = previous.pieces[moving_index];
    Piece moving_after = moving_before;
    moving_after.square = move.to;
    if (move.promotion != PieceType::Pawn) moving_after.type = move.promotion;
    const bool moving_king = move.moving_type == PieceType::King;
    for (int perspective_value = 0; perspective_value < PieceColors; ++perspective_value) {
        const auto perspective = static_cast<PieceColor>(perspective_value);
        const bool perspective_king_moved = moving_king && move.color == perspective;
        const bool king_was_rebuilt = previous.king_square(perspective) != current.king_square(perspective);
        if (perspective_king_moved || king_was_rebuilt) continue;
        if (!add_piece_feature(delta.removed, previous, moving_before, perspective) ||
            !add_piece_feature(delta.added, current, moving_after, perspective)) return FeatureDelta{};

        if (move.capture) {
            const int captured_index = find_piece(previous, move.en_passant ? move.captured_square : move.to,
                move.color == PieceColor::White ? PieceColor::Black : PieceColor::White, move.captured_type);
            if (captured_index < 0) return FeatureDelta{};
            if (!add_piece_feature(delta.removed, previous, previous.pieces[captured_index], perspective)) return FeatureDelta{};
        }

        if (move.castling) {
            const int rook_index = find_piece(previous, move.rook_from, move.color, PieceType::Rook);
            if (rook_index < 0) return FeatureDelta{};
            Piece rook_after = previous.pieces[rook_index];
            rook_after.square = move.rook_to;
            if (!add_piece_feature(delta.removed, previous, previous.pieces[rook_index], perspective) ||
                !add_piece_feature(delta.added, current, rook_after, perspective)) return FeatureDelta{};
        }
    }

    delta.valid = true;
    return delta;
}

} // namespace TAPA::Features
