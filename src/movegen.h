#ifndef MOVEGEN_H
#define MOVEGEN_H

#include "types.h"
#include "position.h"

struct MoveList {
    Move moves[256];
    int  count = 0;

    void add(Move m) { moves[count++] = m; }
    void clear()     { count = 0; }
};

const MoveList& generate_moves(const Position& pos, MoveList& list);

#endif
