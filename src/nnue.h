#ifndef NNUE_H
#define NNUE_H

#include "types.h"

class Position;

namespace NNUE {

constexpr int FT_N = 256;
constexpr int L1_N = 32;
constexpr int NUM_FEATURES = 64 * 64 * 12;
constexpr int INPUT_ACTIVATION = 255;
constexpr int HIDDEN_ACTIVATION = 127;

extern bool loaded;

extern int16_t ft_bias[FT_N];
extern int16_t ft_weights[NUM_FEATURES][FT_N];
extern int8_t l1_weights[FT_N * 2][L1_N];
extern int8_t l1_bias[L1_N];
extern int8_t l2_weights[L1_N];
extern int16_t l2_bias;

int16_t clamp_i16(int32_t x);
bool init(const char* filename);
Value evaluate(const Position& pos);
void compute_full_accumulator(const Position& pos, int16_t acc[2][FT_N]);
void compute_persp_accumulator(const Position& pos, int persp, int16_t* acc);

int feature_index(Square king_sq, Square piece_sq, Piece pc);

} // namespace NNUE

#endif
