#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <cassert>

using Bitboard = uint64_t;
using Key      = uint64_t;
using Move     = uint32_t;
using Value    = int32_t;
using Depth    = int32_t;

constexpr int MAX_PLY          = 128;
constexpr Value VALUE_INFINITE = 30001;
constexpr Value VALUE_MATE     = 30000;
constexpr Value VALUE_DRAW     = 0;
constexpr Value VALUE_MATE_IN_MAX_PLY = VALUE_MATE - MAX_PLY;
constexpr Value VALUE_NONE     = 30002;

enum Square : int {
    A1=0, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
    SQ_NONE = 64, NUM_SQUARES = 64
};

enum File : int { FILE_A=0, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_NONE = 8 };
enum Rank : int { RANK_1=0, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_NONE = 8 };

enum Color : int { WHITE = 0, BLACK = 1, COLOR_NONE = 2 };
constexpr Color operator~(Color c) { return Color(c ^ 1); }

enum PieceType : int {
    PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_TYPE_NONE = 6
};

enum Piece : int {
    W_PAWN = 0, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
    B_PAWN = 6, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
    PIECE_NONE = 12, NUM_PIECES = 12
};

enum CastlingRight : int {
    CASTLE_WK = 1,
    CASTLE_WQ = 2,
    CASTLE_BK = 4,
    CASTLE_BQ = 8,
    CASTLE_ALL = 15
};

constexpr int CastleMask[64] = {
    13, 15, 15, 15, 12, 15, 15, 14,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
     7, 15, 15, 15,  3, 15, 15, 11
};

// --- Square utilities ---
constexpr File   file_of(Square sq) { return File(sq & 7); }
constexpr Rank   rank_of(Square sq) { return Rank(sq >> 3); }
constexpr Square make_square(File f, Rank r) { return Square(r * 8 + f); }
constexpr Square relative_square(Color c, Square sq) { return Square(c == WHITE ? sq : sq ^ 56); }
constexpr int    square_distance(Square a, Square b) {
    int fd = file_of(a) - file_of(b); if (fd < 0) fd = -fd;
    int rd = rank_of(a) - rank_of(b); if (rd < 0) rd = -rd;
    return fd > rd ? fd : rd;
}

// --- Piece utilities ---
constexpr PieceType type_of(Piece p) { return PieceType(p % 6); }
constexpr Color     color_of(Piece p) { return Color(p / 6); }
constexpr Piece     make_piece(Color c, PieceType pt) { return Piece(c * 6 + pt); }

// --- Move encoding ---
// bits  0-5:  from square
// bits  6-11: to square
// bits 12-13: move type (0 normal, 1 promotion, 2 en-passant, 3 castling)
// bits 14-16: promotion piece (PieceType, 0-5, only valid for promotion)
// bits 17-31: unused (for move ordering score)
constexpr int FROM_SHIFT  = 0;
constexpr int TO_SHIFT    = 6;
constexpr int TYPE_SHIFT  = 12;
constexpr int PROMO_SHIFT = 14;

enum MoveType : int { NORMAL = 0, PROMOTION = 1, EN_PASSANT = 2, CASTLING = 3 };

inline Square    move_from(Move m)     { return Square(m >> FROM_SHIFT  & 0x3F); }
inline Square    move_to(Move m)       { return Square(m >> TO_SHIFT    & 0x3F); }
inline int       move_type(Move m)     { return      (m >> TYPE_SHIFT  & 0x3); }
inline PieceType move_promo(Move m)    { return PieceType(m >> PROMO_SHIFT & 0x7); }

constexpr Move MOVE_NONE = 0;

inline Move make_move(Square from, Square to, int type = NORMAL, PieceType promo = PIECE_TYPE_NONE) {
    return (from & 0x3F) | ((to & 0x3F) << TO_SHIFT) | ((type & 0x3) << TYPE_SHIFT) | ((promo & 0x7) << PROMO_SHIFT);
}

inline Move make_promotion(Square from, Square to, PieceType promo) {
    return make_move(from, to, PROMOTION, promo);
}

// --- Direction constants ---
enum Direction : int {
    NORTH = 8, SOUTH = -8, EAST = 1, WEST = -1,
    NE = 9, NW = 7, SE = -7, SW = -9
};

#endif
