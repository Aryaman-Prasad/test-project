#ifndef EVAL_H
#define EVAL_H

#include "position.h"

Value evaluate(const Position& pos);
Value evaluate_hce(const Position& pos);
bool eval_is_nnue();

#endif
