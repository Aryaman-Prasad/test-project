#include "bitboard.h"
#include <cstring>
#include <cstdio>
#include <initializer_list>

// --------------------------------------------------------------------------
// Precomputed tables
// --------------------------------------------------------------------------
Bitboard PawnAttacks[2][64];
Bitboard KnightAttacks[64];
Bitboard KingAttacks[64];
Bitboard BetweenBB[64][64];
Bitboard LineBB[64][64];

Magic RookMagics[64];
Magic BishopMagics[64];

Bitboard RookTable[64][4096];
Bitboard BishopTable[64][2048];

// --------------------------------------------------------------------------
// Magic numbers — generated for our exact mask definitions.
// --------------------------------------------------------------------------
static const Bitboard RookMagicData[64] = {
    0x2080002184400410ULL, 0x2840004010002000ULL, 0x0080082000300085ULL, 0x188010000800810cULL,
    0xc080040080022800ULL, 0x8300010024000802ULL, 0x3400220489100408ULL, 0x8c80004480002100ULL,
    0x0050800240008820ULL, 0x0840804004200080ULL, 0x0102001a00408120ULL, 0x6104801002800800ULL,
    0x0050800800040082ULL, 0x100c011008840020ULL, 0x0202000408020001ULL, 0x0043003100054082ULL,
    0x0008808002a04000ULL, 0x0040008020008840ULL, 0x301103002005c032ULL, 0x0090004008004402ULL,
    0x2410818008000402ULL, 0x0000808002004c00ULL, 0x000080802a000100ULL, 0x0002220012408401ULL,
    0x2000822c80004002ULL, 0x1223008100214008ULL, 0x0000100680200080ULL, 0x000c100080080082ULL,
    0x0048080080800400ULL, 0x8002020080040080ULL, 0x0000020400089019ULL, 0x0043003100054082ULL,
    0x01c0006141800282ULL, 0x0000200048c01000ULL, 0x0001882000801001ULL, 0x6104801002800800ULL,
    0x0010041801001100ULL, 0x2086000402000810ULL, 0x00b5000401010200ULL, 0x4002085482000401ULL,
    0x00e0204002808000ULL, 0xa100200050004000ULL, 0x001004002801a000ULL, 0xc030008038008010ULL,
    0x4000100801010044ULL, 0x0004401020080124ULL, 0x0202000408020001ULL, 0x0080c04284020001ULL,
    0xd080204280090500ULL, 0x2000200080c00880ULL, 0x0000100680200080ULL, 0x4000d80082100080ULL,
    0x0010041801001100ULL, 0x0182011008440200ULL, 0x0442100201180c00ULL, 0x0400030054008200ULL,
    0x6c004100a1128005ULL, 0x0884420054802502ULL, 0x40401a0040208112ULL, 0x0012000420410812ULL,
    0x0082000408102006ULL, 0x0142005108049002ULL, 0x0000421038010284ULL, 0x12410b0824008442ULL
};

static const Bitboard BishopMagicData[64] = {
    0x8034101001102080ULL, 0x0060840400405400ULL, 0x40c3020600440002ULL, 0x8804140a90480823ULL,
    0x0102021000120002ULL, 0x6806080444800065ULL, 0x0101080804a40050ULL, 0x0020422488084000ULL,
    0x0400121001581080ULL, 0x0101080804a40050ULL, 0x0000080214002200ULL, 0x1080040400800080ULL,
    0x6048211040002802ULL, 0x0900010420042008ULL, 0x0800010801504818ULL, 0x000050a404044400ULL,
    0x010800208a240800ULL, 0x0014309011020404ULL, 0x0801023802040230ULL, 0x1704022044108010ULL,
    0x2821000090401080ULL, 0xc088200200942000ULL, 0x0013004a03100210ULL, 0x0002000301012140ULL,
    0x0104100006101001ULL, 0x002910000c044808ULL, 0x1081024084080600ULL, 0x0440040002012044ULL,
    0x4100848004002010ULL, 0x20051d0002004102ULL, 0x0010810104310800ULL, 0x0010810104310800ULL,
    0x0410022040510490ULL, 0x00c1104800304100ULL, 0x88cc544803100484ULL, 0x80000c0400080120ULL,
    0x02400080600a00a0ULL, 0x00300200a0221000ULL, 0x1010212a00110082ULL, 0x000082020000c101ULL,
    0x18080c44200504c8ULL, 0x0200808808042040ULL, 0x0002050043000800ULL, 0x0900000c24002800ULL,
    0x040040110a020100ULL, 0x102a022242000100ULL, 0x10200400aa000582ULL, 0x0810440100505020ULL,
    0x0101080804a40050ULL, 0x0080808410020002ULL, 0x0000142402180000ULL, 0x0800005042020000ULL,
    0x208400301a020000ULL, 0x00403810900080adULL, 0x502002302103000aULL, 0x0060840400405400ULL,
    0x0020422488084000ULL, 0x000050a404044400ULL, 0x8804140a90480823ULL, 0x008000080c208800ULL,
    0x1004000840082200ULL, 0x064a001020211504ULL, 0x0400121001581080ULL, 0x8034101001102080ULL
};

// --------------------------------------------------------------------------
// Raw attack generators (rank/file-based, no wrapping bugs)
// --------------------------------------------------------------------------
static Bitboard rook_attacks_raw(Square sq, Bitboard occ) {
    Bitboard r = 0;
    int f = sq & 7, rk = sq >> 3;
    for (int nf = f + 1; nf < 8; ++nf) { Square s = Square(rk * 8 + nf); r |= 1ULL << s; if (occ >> s & 1) break; }
    for (int nf = f - 1; nf >= 0; --nf) { Square s = Square(rk * 8 + nf); r |= 1ULL << s; if (occ >> s & 1) break; }
    for (int nr = rk + 1; nr < 8; ++nr) { Square s = Square(nr * 8 + f);  r |= 1ULL << s; if (occ >> s & 1) break; }
    for (int nr = rk - 1; nr >= 0; --nr) { Square s = Square(nr * 8 + f);  r |= 1ULL << s; if (occ >> s & 1) break; }
    return r;
}

static Bitboard bishop_attacks_raw(Square sq, Bitboard occ) {
    Bitboard r = 0;
    int f = sq & 7, rk = sq >> 3;
    for (int nf = f+1, nr = rk+1; nf < 8 && nr < 8; ++nf, ++nr) { Square s = Square(nr*8+nf); r |= 1ULL<<s; if (occ>>s&1) break; }
    for (int nf = f-1, nr = rk+1; nf >=0 && nr < 8; --nf, ++nr) { Square s = Square(nr*8+nf); r |= 1ULL<<s; if (occ>>s&1) break; }
    for (int nf = f+1, nr = rk-1; nf < 8 && nr >=0; ++nf, --nr) { Square s = Square(nr*8+nf); r |= 1ULL<<s; if (occ>>s&1) break; }
    for (int nf = f-1, nr = rk-1; nf >=0 && nr >=0; --nf, --nr) { Square s = Square(nr*8+nf); r |= 1ULL<<s; if (occ>>s&1) break; }
    return r;
}

// --------------------------------------------------------------------------
// Masks (exclude edge squares)
// --------------------------------------------------------------------------
static Bitboard rook_mask(Square sq) {
    Bitboard m = 0; int f = sq & 7, rk = sq >> 3;
    for (int nf = f+1; nf < 7; ++nf)        m |= 1ULL << Square(rk*8 + nf);
    for (int nf = f-1; nf > 0; --nf)        m |= 1ULL << Square(rk*8 + nf);
    for (int nr = rk+1; nr < 7; ++nr)       m |= 1ULL << Square(nr*8 + f);
    for (int nr = rk-1; nr > 0; --nr)       m |= 1ULL << Square(nr*8 + f);
    return m;
}

static Bitboard bishop_mask(Square sq) {
    Bitboard m = 0; int f = sq & 7, rk = sq >> 3;
    for (int nf=f+1,nr=rk+1; nf<7&&nr<7; ++nf,++nr) m |= 1ULL<<Square(nr*8+nf);
    for (int nf=f-1,nr=rk+1; nf>0&&nr<7; --nf,++nr) m |= 1ULL<<Square(nr*8+nf);
    for (int nf=f+1,nr=rk-1; nf<7&&nr>0; ++nf,--nr) m |= 1ULL<<Square(nr*8+nf);
    for (int nf=f-1,nr=rk-1; nf>0&&nr>0; --nf,--nr) m |= 1ULL<<Square(nr*8+nf);
    return m;
}

// --------------------------------------------------------------------------
// Table builder
// --------------------------------------------------------------------------
static void init_magic_table(bool is_rook) {
    Magic*          magics = is_rook ? RookMagics : BishopMagics;
    Bitboard*       base   = is_rook ? RookTable[0] : BishopTable[0];
    int             stride = is_rook ? 4096 : 2048;
    const Bitboard* mdata  = is_rook ? RookMagicData : BishopMagicData;

    for (Square sq = A1; sq <= H8; sq = Square(int(sq) + 1)) {
        Bitboard mask = is_rook ? rook_mask(sq) : bishop_mask(sq);
        int n    = popcount(mask);
        int nocc = 1 << n;
        int sh   = 64 - n;

        magics[sq].mask   = mask;
        magics[sq].magic  = mdata[sq];
        magics[sq].shift  = sh;
        magics[sq].attacks = base + sq * stride;

        Bitboard* tbl = magics[sq].attacks;
        __builtin_memset(tbl, 0, nocc * sizeof(Bitboard));

        Bitboard occ = 0;
        do {
            Bitboard attacks = (is_rook ? rook_attacks_raw : bishop_attacks_raw)(sq, occ);
            Bitboard idx     = ((occ & mask) * magics[sq].magic) >> sh;
            tbl[idx] = attacks;
            occ = (occ - mask) & mask;
        } while (occ);
    }
}

// --------------------------------------------------------------------------
// Between & Line tables
// --------------------------------------------------------------------------
static void init_between_line() {
    for (Square a = A1; a <= H8; a = Square(int(a) + 1)) {
        for (Square b = A1; b <= H8; b = Square(int(b) + 1)) {
            int af = a & 7, ar = a >> 3;
            int bf = b & 7, br = b >> 3;
            int df = bf - af, dr = br - ar;

            BetweenBB[a][b] = 0;
            LineBB[a][b]    = 0;

            if (ar == br && df) {
                int step = df > 0 ? 1 : -1;
                for (int f = af + step; f != bf; f += step)  BetweenBB[a][b] |= 1ULL << (ar * 8 + f);
                for (int f = af; f != bf + step; f += step)  LineBB[a][b]    |= 1ULL << (ar * 8 + f);
            }
            else if (af == bf && dr) {
                int step = dr > 0 ? 8 : -8;
                for (int r = ar + step/8; r != br; r += step/8)  BetweenBB[a][b] |= 1ULL << (r * 8 + af);
                for (int r = ar; r != br + step/8; r += step/8)  LineBB[a][b]    |= 1ULL << (r * 8 + af);
            }
            else if (df == dr && df) {
                int step = df > 0 ? 9 : -9;
                for (int s = int(a) + step; s != int(b); s += step)  BetweenBB[a][b] |= 1ULL << s;
                for (int s = int(a); s != int(b) + step; s += step)  LineBB[a][b]    |= 1ULL << s;
            }
            else if (df == -dr && df) {
                int step = dr > 0 ? 7 : -7;
                for (int s = int(a) + step; s != int(b); s += step)  BetweenBB[a][b] |= 1ULL << s;
                for (int s = int(a); s != int(b) + step; s += step)  LineBB[a][b]    |= 1ULL << s;
            }
        }
    }
}

// --------------------------------------------------------------------------
// Top-level initializer
// --------------------------------------------------------------------------
void init_bitboards() {
    for (Square sq = A1; sq <= H8; sq = Square(int(sq) + 1)) {
        Bitboard b = square_bb(sq);
        PawnAttacks[WHITE][sq] = shift_ne(b) | shift_nw(b);
        PawnAttacks[BLACK][sq] = shift_se(b) | shift_sw(b);
    }

    for (Square sq = A1; sq <= H8; sq = Square(int(sq) + 1)) {
        Bitboard k = 0;
        for (int d : {17, 15, 10, 6, -6, -10, -15, -17}) {
            int s = int(sq) + d;
            if (s < 0 || s > 63) continue;
            if (square_distance(sq, Square(s)) > 2) continue;
            k |= 1ULL << s;
        }
        KnightAttacks[sq] = k;
    }

    for (Square sq = A1; sq <= H8; sq = Square(int(sq) + 1)) {
        Bitboard b = square_bb(sq);
        KingAttacks[sq] = shift_north(b) | shift_south(b) | shift_east(b) | shift_west(b)
                        | shift_ne(b) | shift_nw(b) | shift_se(b) | shift_sw(b);
    }

    init_magic_table(true);
    init_magic_table(false);

    init_between_line();
}
