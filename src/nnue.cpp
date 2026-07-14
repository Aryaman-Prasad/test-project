#include "nnue.h"
#include "position.h"
#include <cstring>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <cmath>

namespace NNUE {

bool loaded = false;

alignas(64) int16_t ft_bias[FT_N];
alignas(64) int16_t ft_weights[NUM_FEATURES][FT_N];
alignas(64) int8_t l1_weights[FT_N * 2][L1_N];
alignas(32) int8_t l1_bias[L1_N];
alignas(32) int8_t l2_weights[L1_N];
int16_t l2_bias;

int feature_index(Square king_sq, Square piece_sq, Piece pc) {
    return (king_sq * 64 + piece_sq) * 12 + int(pc);
}

int16_t clamp_i16(int32_t x) {
    if (x < -32768) return -32768;
    if (x > 32767) return 32767;
    return int16_t(x);
}

void add_feature(int16_t acc[FT_N], int f_idx, int delta) {
    int16_t* w = ft_weights[f_idx];
    for (int i = 0; i < FT_N; ++i)
        acc[i] = clamp_i16(int32_t(acc[i]) + delta * int32_t(w[i]));
}

static void acc_add_pieces_for_persp(const Position& pos, int16_t* acc, Square ks) {
    for (Color c : {WHITE, BLACK}) {
        for (PieceType pt : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
            Bitboard bb = pos.pieces(c, pt);
            while (bb) {
                Square sq = pop_lsb(bb);
                Piece pc = make_piece(c, pt);
                int pc_idx = int(pc);
                Piece my_king = (ks == pos.king_sq(WHITE)) ? W_KING : B_KING;
                if (pc == my_king) continue;
                int f_idx = (ks * 64 + sq) * 12 + pc_idx;
                int16_t* w = ft_weights[f_idx];
                for (int i = 0; i < FT_N; ++i)
                    acc[i] = clamp_i16(int32_t(acc[i]) + int32_t(w[i]));
            }
        }
    }
}

void compute_persp_accumulator(const Position& pos, int persp, int16_t* acc) {
    Square ks = (persp == 0) ? pos.king_sq(WHITE) : pos.king_sq(BLACK);
    std::memcpy(acc, ft_bias, sizeof(ft_bias));
    acc_add_pieces_for_persp(pos, acc, ks);
}

void compute_full_accumulator(const Position& pos, int16_t acc[2][FT_N]) {
    std::memcpy(acc[0], ft_bias, sizeof(ft_bias));
    std::memcpy(acc[1], ft_bias, sizeof(ft_bias));

    Square wk_sq = pos.king_sq(WHITE);
    Square bk_sq = pos.king_sq(BLACK);

    for (Color c : {WHITE, BLACK}) {
        for (PieceType pt : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
            Bitboard bb = pos.pieces(c, pt);
            while (bb) {
                Square sq = pop_lsb(bb);
                Piece pc = make_piece(c, pt);
                int pc_idx = int(pc);

                if (!(c == WHITE && pt == KING)) {
                    int f_idx = (wk_sq * 64 + sq) * 12 + pc_idx;
                    int16_t* w = ft_weights[f_idx];
                    for (int i = 0; i < FT_N; ++i)
                        acc[0][i] = clamp_i16(int32_t(acc[0][i]) + int32_t(w[i]));
                }

                if (!(c == BLACK && pt == KING)) {
                    int f_idx = (bk_sq * 64 + sq) * 12 + pc_idx;
                    int16_t* w = ft_weights[f_idx];
                    for (int i = 0; i < FT_N; ++i)
                        acc[1][i] = clamp_i16(int32_t(acc[1][i]) + int32_t(w[i]));
                }
            }
        }
    }
}

static Value forward_from_accumulator(const int16_t acc[2][FT_N]) {
    uint8_t clipped[2][FT_N];
    for (int p = 0; p < 2; ++p)
        for (int i = 0; i < FT_N; ++i)
            clipped[p][i] = uint8_t(std::clamp(int32_t(acc[p][i]), 0, INPUT_ACTIVATION));

    int32_t l1_out[L1_N];
    for (int i = 0; i < L1_N; ++i) {
        int32_t sum = l1_bias[i];
        for (int p = 0; p < 2; ++p)
            for (int j = 0; j < FT_N; ++j)
                sum += int32_t(clipped[p][j]) * int32_t(l1_weights[p * FT_N + j][i]);
        l1_out[i] = std::clamp(sum, 0, HIDDEN_ACTIVATION);
    }

    int32_t output = l2_bias;
    for (int i = 0; i < L1_N; ++i)
        output += l1_out[i] * int32_t(l2_weights[i]);

    return Value(output * 100 / 256);
}

Value evaluate(const Position& pos) {
    if (!loaded)
        return 0;

    return forward_from_accumulator(pos.nnue_accumulator);
}

static bool read_int16(std::ifstream& f, int16_t* buf, size_t n) {
    return f.read(reinterpret_cast<char*>(buf), n * sizeof(int16_t)).gcount() == std::streamsize(n * sizeof(int16_t));
}

static bool read_int8(std::ifstream& f, int8_t* buf, size_t n) {
    return f.read(reinterpret_cast<char*>(buf), n * sizeof(int8_t)).gcount() == std::streamsize(n * sizeof(int8_t));
}

bool init(const char* filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "info string NNUE: could not open " << filename << std::endl;
        loaded = false;
        return false;
    }

    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "NNUE", 4) != 0) {
        std::cerr << "info string NNUE: invalid magic" << std::endl;
        loaded = false;
        return false;
    }

    int32_t version;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        std::cerr << "info string NNUE: unsupported version " << version << std::endl;
        loaded = false;
        return false;
    }

    int32_t ft_n, l1_n;
    f.read(reinterpret_cast<char*>(&ft_n), sizeof(ft_n));
    f.read(reinterpret_cast<char*>(&l1_n), sizeof(l1_n));
    if (ft_n != FT_N || l1_n != L1_N) {
        std::cerr << "info string NNUE: architecture mismatch (ft_n=" << ft_n << " l1_n=" << l1_n << ")" << std::endl;
        loaded = false;
        return false;
    }

    bool ok = true;
    ok = ok && read_int16(f, ft_bias, FT_N);
    ok = ok && read_int16(f, &ft_weights[0][0], NUM_FEATURES * FT_N);
    ok = ok && read_int8(f, &l1_weights[0][0], FT_N * 2 * L1_N);
    ok = ok && read_int8(f, l1_bias, L1_N);
    ok = ok && read_int8(f, l2_weights, L1_N);
    ok = ok && read_int16(f, &l2_bias, 1);

    if (!ok) {
        std::cerr << "info string NNUE: failed to read weights" << std::endl;
        loaded = false;
        return false;
    }

    loaded = true;
    std::cerr << "info string NNUE: loaded weights from " << filename << std::endl;
    return true;
}

} // namespace NNUE
