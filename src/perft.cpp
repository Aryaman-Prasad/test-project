#include "perft.h"
#include "movegen.h"
#include <cstdio>

uint64_t perft(Position& pos, int depth) {
    if (depth == 0) return 1;

    uint64_t nodes = 0;
    MoveList list;
    generate_moves(pos, list);

    for (int i = 0; i < list.count; ++i) {
        Move m = list.moves[i];

        if (depth <= 2) {
            fprintf(stderr, "  [d%d] trying move %d (total %d): from=%d to=%d type=%d promo=%d\n",
                    depth, i, list.count, move_from(m), move_to(m), move_type(m), move_promo(m));
        }

        pos.make_move(m);
        if (pos.is_legal()) {
            if (depth == 1)
                nodes += 1;
            else
                nodes += perft(pos, depth - 1);
        } else {
            if (depth <= 2)
                fprintf(stderr, "    illegal!\n");
        }
        pos.unmake_move(m);
    }

    return nodes;
}
