#ifndef BITBOARD_H
#define BITBOARD_H

#include "types.h"

// --- Bitboard constants ---
constexpr Bitboard FILE_A_BB = 0x0101010101010101ULL;
constexpr Bitboard FILE_B_BB = 0x0202020202020202ULL;
constexpr Bitboard FILE_C_BB = 0x0404040404040404ULL;
constexpr Bitboard FILE_D_BB = 0x0808080808080808ULL;
constexpr Bitboard FILE_E_BB = 0x1010101010101010ULL;
constexpr Bitboard FILE_F_BB = 0x2020202020202020ULL;
constexpr Bitboard FILE_G_BB = 0x4040404040404040ULL;
constexpr Bitboard FILE_H_BB = 0x8080808080808080ULL;

constexpr Bitboard RANK_1_BB = 0x00000000000000FFULL;
constexpr Bitboard RANK_2_BB = 0x000000000000FF00ULL;
constexpr Bitboard RANK_3_BB = 0x0000000000FF0000ULL;
constexpr Bitboard RANK_4_BB = 0x00000000FF000000ULL;
constexpr Bitboard RANK_5_BB = 0x000000FF00000000ULL;
constexpr Bitboard RANK_6_BB = 0x0000FF0000000000ULL;
constexpr Bitboard RANK_7_BB = 0x00FF000000000000ULL;
constexpr Bitboard RANK_8_BB = 0xFF00000000000000ULL;

constexpr Bitboard RankBB[8] = {RANK_1_BB, RANK_2_BB, RANK_3_BB, RANK_4_BB, RANK_5_BB, RANK_6_BB, RANK_7_BB, RANK_8_BB};
constexpr Bitboard FileBB[8] = {FILE_A_BB, FILE_B_BB, FILE_C_BB, FILE_D_BB, FILE_E_BB, FILE_F_BB, FILE_G_BB, FILE_H_BB};

constexpr Bitboard DARK_SQUARES = 0xAA55AA55AA55AA55ULL;

// --- Bitboard utilities ---
constexpr Bitboard square_bb(Square sq) { return 1ULL << sq; }

constexpr Bitboard set_bit(Bitboard b, Square sq)   { return b | square_bb(sq); }
constexpr Bitboard clear_bit(Bitboard b, Square sq) { return b & ~square_bb(sq); }
constexpr bool     test_bit(Bitboard b, Square sq)  { return (b >> sq) & 1ULL; }

inline int     popcount(Bitboard b) { return __builtin_popcountll(b); }
inline Square  lsb(Bitboard b)      { return Square(__builtin_ctzll(b)); }
inline Square  msb(Bitboard b)      { return Square(63 - __builtin_clzll(b)); }
inline Square  pop_lsb(Bitboard& b) { Square s = lsb(b); b &= b - 1; return s; }
inline int     more_than_one(Bitboard b) { return b & (b - 1); }

// --- Direction shifts (safe, no wrap-around) ---
constexpr Bitboard shift_north(Bitboard b) { return b << 8; }
constexpr Bitboard shift_south(Bitboard b) { return b >> 8; }
constexpr Bitboard shift_east(Bitboard b)  { return (b & ~FILE_H_BB) << 1; }
constexpr Bitboard shift_west(Bitboard b)  { return (b & ~FILE_A_BB) >> 1; }
constexpr Bitboard shift_ne(Bitboard b)    { return (b & ~FILE_H_BB) << 9; }
constexpr Bitboard shift_nw(Bitboard b)    { return (b & ~FILE_A_BB) << 7; }
constexpr Bitboard shift_se(Bitboard b)    { return (b & ~FILE_H_BB) >> 7; }
constexpr Bitboard shift_sw(Bitboard b)    { return (b & ~FILE_A_BB) >> 9; }

// --- Precomputed attack tables ---
extern Bitboard PawnAttacks[2][64];
extern Bitboard KnightAttacks[64];
extern Bitboard KingAttacks[64];
extern Bitboard BetweenBB[64][64];
extern Bitboard LineBB[64][64];

// --- Magic bitboard ---
struct Magic {
    Bitboard mask;
    Bitboard magic;
    Bitboard* attacks;
    int shift;

    Bitboard index(Bitboard occ) const {
        return ((occ & mask) * magic) >> shift;
    }
};

extern Magic RookMagics[64];
extern Magic BishopMagics[64];

extern Bitboard RookTable[64][4096];
extern Bitboard BishopTable[64][2048];

// --- Attack getters ---
inline Bitboard pawn_attacks(Color c, Square sq) {
    return PawnAttacks[c][sq];
}

inline Bitboard knight_attacks(Square sq) {
    return KnightAttacks[sq];
}

inline Bitboard king_attacks(Square sq) {
    return KingAttacks[sq];
}

inline Bitboard bishop_attacks(Square sq, Bitboard occ) {
    const Magic& m = BishopMagics[sq];
    return m.attacks[m.index(occ)];
}

inline Bitboard rook_attacks(Square sq, Bitboard occ) {
    const Magic& m = RookMagics[sq];
    return m.attacks[m.index(occ)];
}

inline Bitboard queen_attacks(Square sq, Bitboard occ) {
    return bishop_attacks(sq, occ) | rook_attacks(sq, occ);
}

inline Bitboard attacks_bb(PieceType pt, Square sq, Bitboard occ) {
    switch (pt) {
    case ROOK:   return rook_attacks(sq, occ);
    case BISHOP: return bishop_attacks(sq, occ);
    case QUEEN:  return queen_attacks(sq, occ);
    default:     return 0;
    }
}

// --- Utility ---
inline Bitboard between_bb(Square a, Square b) { return BetweenBB[a][b]; }
inline Bitboard line_bb(Square a, Square b)    { return LineBB[a][b]; }

inline bool aligned(Square a, Square b, Square c) {
    return (line_bb(a, b) & square_bb(c)) != 0;
}

// --- Initialization ---
void init_bitboards();

#endif
