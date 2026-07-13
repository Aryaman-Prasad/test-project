#include "movegen.h"
#include "bitboard.h"

template<Color Us>
static void generate_pawn_moves(const Position& pos, MoveList& list) {
    constexpr Bitboard PromoRank = Us == WHITE ? RANK_8_BB : RANK_1_BB;
    constexpr Bitboard Rank2     = Us == WHITE ? RANK_2_BB : RANK_7_BB;  // rank pawns start on for double push

    Bitboard pawns = pos.pieces(Us, PAWN);
    Bitboard empty = ~pos.occupied();
    Bitboard them  = pos.pieces(Color(Us ^ 1));
    Bitboard ep_bb = pos.ep_square() != SQ_NONE ? square_bb(pos.ep_square()) : 0;

    // Single pushes
    Bitboard single = (Us == WHITE ? pawns << 8 : pawns >> 8) & empty;
    Bitboard promo  = single & PromoRank;
    single &= ~PromoRank;

    while (single) {
        Square to   = pop_lsb(single);
        Square from = Square(int(to) - (Us == WHITE ? 8 : -8));
        list.add(make_move(from, to));
    }
    while (promo) {
        Square to   = pop_lsb(promo);
        Square from = Square(int(to) - (Us == WHITE ? 8 : -8));
        list.add(make_promotion(from, to, QUEEN));
        list.add(make_promotion(from, to, ROOK));
        list.add(make_promotion(from, to, BISHOP));
        list.add(make_promotion(from, to, KNIGHT));
    }

    // Double pushes (pawns on Rank2 can push two squares)
    Bitboard dd = (Us == WHITE ? (pawns & Rank2) << 8 : (pawns & Rank2) >> 8) & empty;
    dd = (Us == WHITE ? dd << 8 : dd >> 8) & empty;
    while (dd) {
        Square to   = pop_lsb(dd);
        Square from = Square(int(to) - (Us == WHITE ? 16 : -16));
        list.add(make_move(from, to));
    }

    // Captures
    Bitboard capt_targets = them | ep_bb;
    Bitboard l_capt = (Us == WHITE ? shift_nw(pawns) : shift_sw(pawns)) & capt_targets;
    Bitboard r_capt = (Us == WHITE ? shift_ne(pawns) : shift_se(pawns)) & capt_targets;

    Bitboard l_promo = l_capt & PromoRank; l_capt &= ~PromoRank;
    Bitboard r_promo = r_capt & PromoRank; r_capt &= ~PromoRank;

    while (l_capt) {
        Square to   = pop_lsb(l_capt);
        Square from = Square(int(to) - (Us == WHITE ? 7 : -9));
        int type = to == pos.ep_square() ? EN_PASSANT : NORMAL;
        list.add(make_move(from, to, type));
    }
    while (r_capt) {
        Square to   = pop_lsb(r_capt);
        Square from = Square(int(to) - (Us == WHITE ? 9 : -7));
        int type = to == pos.ep_square() ? EN_PASSANT : NORMAL;
        list.add(make_move(from, to, type));
    }
    while (l_promo) {
        Square to   = pop_lsb(l_promo);
        Square from = Square(int(to) - (Us == WHITE ? 7 : -9));
        list.add(make_promotion(from, to, QUEEN));
        list.add(make_promotion(from, to, ROOK));
        list.add(make_promotion(from, to, BISHOP));
        list.add(make_promotion(from, to, KNIGHT));
    }
    while (r_promo) {
        Square to   = pop_lsb(r_promo);
        Square from = Square(int(to) - (Us == WHITE ? 9 : -7));
        list.add(make_promotion(from, to, QUEEN));
        list.add(make_promotion(from, to, ROOK));
        list.add(make_promotion(from, to, BISHOP));
        list.add(make_promotion(from, to, KNIGHT));
    }
}

template<Color Us>
static void generate_knight_moves(const Position& pos, MoveList& list) {
    Bitboard knights = pos.pieces(Us, KNIGHT);
    Bitboard not_us  = ~pos.pieces(Us);
    while (knights) {
        Square from = pop_lsb(knights);
        Bitboard targets = KnightAttacks[from] & not_us;
        while (targets) list.add(make_move(from, pop_lsb(targets)));
    }
}

template<Color Us>
static void generate_king_moves(const Position& pos, MoveList& list) {
    constexpr Color Them = Color(Us ^ 1);
    Square from = pos.king_sq(Us);
    Bitboard targets = KingAttacks[from] & ~pos.pieces(Us);
    while (targets) list.add(make_move(from, pop_lsb(targets)));

    // Castling
    int rights = pos.get_castling_rights();
    Bitboard occ = pos.occupied();
    if (Us == WHITE) {
        if ((rights & CASTLE_WK) && !(occ & 0x0000000000000060ULL)
            && !pos.is_attacked(E1, BLACK) && !pos.is_attacked(F1, BLACK))
            list.add(make_move(E1, G1, CASTLING));
        if ((rights & CASTLE_WQ) && !(occ & 0x000000000000000EULL)
            && !pos.is_attacked(E1, BLACK) && !pos.is_attacked(D1, BLACK))
            list.add(make_move(E1, C1, CASTLING));
    } else {
        if ((rights & CASTLE_BK) && !(occ & 0x6000000000000000ULL)
            && !pos.is_attacked(E8, WHITE) && !pos.is_attacked(F8, WHITE))
            list.add(make_move(E8, G8, CASTLING));
        if ((rights & CASTLE_BQ) && !(occ & 0x0E00000000000000ULL)
            && !pos.is_attacked(E8, WHITE) && !pos.is_attacked(D8, WHITE))
            list.add(make_move(E8, C8, CASTLING));
    }
}

template<Color Us>
static void generate_sliding_moves(const Position& pos, MoveList& list) {
    Bitboard occ   = pos.occupied();
    Bitboard not_us = ~pos.pieces(Us);

    for (PieceType pt : {BISHOP, ROOK, QUEEN}) {
        Bitboard pieces = pos.pieces(Us, pt);
        while (pieces) {
            Square from = pop_lsb(pieces);
            Bitboard attacks = attacks_bb(pt, from, occ) & not_us;
            while (attacks) list.add(make_move(from, pop_lsb(attacks)));
        }
    }
}

template<Color Us>
static void generate_all(const Position& pos, MoveList& list) {
    generate_pawn_moves<Us>(pos, list);
    generate_knight_moves<Us>(pos, list);
    generate_king_moves<Us>(pos, list);
    generate_sliding_moves<Us>(pos, list);
}

const MoveList& generate_moves(const Position& pos, MoveList& list) {
    list.clear();
    if (pos.side_to_move() == WHITE) generate_all<WHITE>(pos, list);
    else                             generate_all<BLACK>(pos, list);
    return list;
}
