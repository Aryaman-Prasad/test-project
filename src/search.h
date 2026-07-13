#ifndef SEARCH_H
#define SEARCH_H

#include "position.h"
#include <atomic>

struct SearchLimits {
    Depth depth = MAX_PLY;
    int64_t time_left[2] = {0, 0};  // [WHITE], [BLACK]
    int64_t inc[2] = {0, 0};
    int64_t movetime = 0;
    int movestogo = 0;
    bool infinite = false;
};

struct SearchInfo {
    Move best_move = MOVE_NONE;
    Value score = VALUE_NONE;
    Depth depth = 0;
    uint64_t nodes = 0;
    int64_t time_ms = 0;
    bool completed = false;
};

extern std::atomic<bool> SearchAborted;

void init_search();
void search(Position& pos, SearchLimits& limits, SearchInfo& info);
uint64_t perft_root(Position& pos, int depth);

#endif
