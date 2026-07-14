#include "eval.h"
#include "nnue.h"

constexpr Value PawnValue   = 100;
constexpr Value KnightValue = 320;
constexpr Value BishopValue = 330;
constexpr Value RookValue   = 500;
constexpr Value QueenValue  = 900;
constexpr Value BishopPair  = 30;
constexpr Value Tempo       = 15;

constexpr Value PawnPST[64] = {
    0,  0,  0,  0,  0,  0,  0,  0,
   50, 50, 50, 50, 50, 50, 50, 50,
   10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
};

constexpr Value KnightPST[64] = {
  -50,-40,-30,-30,-30,-30,-40,-50,
  -40,-20,  0,  0,  0,  0,-20,-40,
  -30,  0, 10, 15, 15, 10,  0,-30,
  -30,  5, 15, 20, 20, 15,  5,-30,
  -30,  0, 15, 20, 20, 15,  0,-30,
  -30,  5, 10, 15, 15, 10,  5,-30,
  -40,-20,  0,  5,  5,  0,-20,-40,
  -50,-40,-30,-30,-30,-30,-40,-50
};

constexpr Value BishopPST[64] = {
  -20,-10,-10,-10,-10,-10,-10,-20,
  -10,  0,  0,  0,  0,  0,  0,-10,
  -10,  0,  5, 10, 10,  5,  0,-10,
  -10,  5,  5, 10, 10,  5,  5,-10,
  -10,  0, 10, 10, 10, 10,  0,-10,
  -10, 10, 10, 10, 10, 10, 10,-10,
  -10,  5,  0,  0,  0,  0,  5,-10,
  -20,-10,-10,-10,-10,-10,-10,-20
};

constexpr Value RookPST[64] = {
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
};

constexpr Value QueenPST[64] = {
  -20,-10,-10, -5, -5,-10,-10,-20,
  -10,  0,  0,  0,  0,  0,  0,-10,
  -10,  0,  5,  5,  5,  5,  0,-10,
   -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
  -10,  5,  5,  5,  5,  5,  0,-10,
  -10,  0,  5,  0,  0,  0,  0,-10,
  -20,-10,-10, -5, -5,-10,-10,-20
};

constexpr Value KingMidPST[64] = {
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -20,-30,-30,-40,-40,-30,-30,-20,
  -10,-20,-20,-20,-20,-20,-20,-10,
   20, 20,  0,  0,  0,  0, 20, 20,
   20, 30, 10,  0,  0, 10, 30, 20
};

constexpr Value KingEndPST[64] = {
  -50,-40,-30,-20,-20,-30,-40,-50,
  -30,-20,-10,  0,  0,-10,-20,-30,
  -30,-10, 20, 30, 30, 20,-10,-30,
  -30,-10, 30, 40, 40, 30,-10,-30,
  -30,-10, 30, 40, 40, 30,-10,-30,
  -30,-10, 20, 30, 30, 20,-10,-30,
  -30,-30,  0,  0,  0,  0,-30,-30,
  -50,-30,-30,-30,-30,-30,-30,-50
};

static int game_phase(const Position& pos) {
    int phase = 0;
    phase += popcount(pos.pieces(KNIGHT)) * 1;
    phase += popcount(pos.pieces(BISHOP)) * 1;
    phase += popcount(pos.pieces(ROOK))   * 2;
    phase += popcount(pos.pieces(QUEEN))  * 4;
    return std::min(phase, 24);
}

static Value pawn_structure(const Position& pos, Color c) {
    Value score = 0;
    Bitboard pawns = pos.pieces(c, PAWN);
    Bitboard them = pos.pieces(Color(c ^ 1));
    while (pawns) {
        Square sq = pop_lsb(pawns);
        File f = file_of(sq);
        Rank r = rank_of(sq);
        Bitboard file_mask = FileBB[f];
        Bitboard adj_files = (f > FILE_A ? FileBB[f - 1] : 0) | (f < FILE_H ? FileBB[f + 1] : 0);
        Bitboard same_file_pawns = pos.pieces(c, PAWN) & file_mask;
        if (popcount(same_file_pawns) > 1)
            score -= 15;
        Bitboard neighbors = pos.pieces(c, PAWN) & adj_files;
        bool has_adj = false;
        while (neighbors) {
            Square nsq = pop_lsb(neighbors);
            if (c == WHITE ? (rank_of(nsq) >= r) : (rank_of(nsq) <= r))
                has_adj = true;
        }
        if (!has_adj)
            score -= 10;
    }
    return score;
}

bool eval_is_nnue() {
    return NNUE::loaded;
}

Value evaluate_hce(const Position& pos) {
    Color stm = pos.side_to_move();
    int phase = game_phase(pos);
    int mg_phase = std::min(phase, 24);
    int eg_phase = 24 - mg_phase;

    Value mg[2] = {0, 0};
    Value eg[2] = {0, 0};
    Value mat[2] = {0, 0};

    for (Color c : {WHITE, BLACK}) {
        Square ksq = pos.king_sq(c);
        Bitboard pieces;

        mat[c] += popcount(pos.pieces(c, PAWN)) * PawnValue;
        pieces = pos.pieces(c, PAWN);
        while (pieces) {
            Square sq = pop_lsb(pieces);
            Square rsq = relative_square(c, sq);
            mg[c] += PawnPST[rsq];
        }

        mat[c] += popcount(pos.pieces(c, KNIGHT)) * KnightValue;
        pieces = pos.pieces(c, KNIGHT);
        while (pieces) {
            Square sq = pop_lsb(pieces);
            Square rsq = relative_square(c, sq);
            mg[c] += KnightPST[rsq];
        }

        int bishops = popcount(pos.pieces(c, BISHOP));
        mat[c] += bishops * BishopValue;
        if (bishops >= 2) mat[c] += BishopPair;
        pieces = pos.pieces(c, BISHOP);
        while (pieces) {
            Square sq = pop_lsb(pieces);
            Square rsq = relative_square(c, sq);
            mg[c] += BishopPST[rsq];
        }

        mat[c] += popcount(pos.pieces(c, ROOK)) * RookValue;
        pieces = pos.pieces(c, ROOK);
        while (pieces) {
            Square sq = pop_lsb(pieces);
            Square rsq = relative_square(c, sq);
            mg[c] += RookPST[rsq];
            Bitboard own_pawns = pos.pieces(c, PAWN) & FileBB[file_of(sq)];
            if (!own_pawns)
                mg[c] += 20;
        }

        mat[c] += popcount(pos.pieces(c, QUEEN)) * QueenValue;
        pieces = pos.pieces(c, QUEEN);
        while (pieces) {
            Square sq = pop_lsb(pieces);
            Square rsq = relative_square(c, sq);
            mg[c] += QueenPST[rsq];
        }

        Square rsq = relative_square(c, ksq);
        mg[c] += KingMidPST[rsq];
        eg[c] += KingEndPST[rsq];

        mg[c] += pawn_structure(pos, c);
    }

    Value mg_total = mg[WHITE] - mg[BLACK];
    Value eg_total = eg[WHITE] - eg[BLACK];
    Value total = (mg_total * mg_phase + eg_total * eg_phase) / 24;
    total += mat[WHITE] - mat[BLACK];

    return stm == WHITE ? total + Tempo : -total + Tempo;
}

Value evaluate(const Position& pos) {
    if (NNUE::loaded)
        return NNUE::evaluate(pos);
    return evaluate_hce(pos);
}
