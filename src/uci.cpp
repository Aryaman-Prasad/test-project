#include "uci.h"
#include "position.h"
#include "movegen.h"
#include "search.h"
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

static const std::string ENGINE_NAME = "OpencodeChess";
static const std::string ENGINE_VERSION = "1.0";

static void send_bestmove(Move m) {
    if (m == MOVE_NONE) {
        std::cout << "bestmove 0000" << std::endl;
        return;
    }
    Square from = move_from(m);
    Square to = move_to(m);
    int promo = move_promo(m);
    char uci[6];
    uci[0] = 'a' + file_of(from);
    uci[1] = '1' + rank_of(from);
    uci[2] = 'a' + file_of(to);
    uci[3] = '1' + rank_of(to);
    uci[4] = 0;
    uci[5] = 0;
    if (move_type(m) == PROMOTION) {
        uci[4] = "pnbrqk"[promo];
        uci[5] = 0;
    }
    std::cout << "bestmove " << uci << std::endl;
}

static Move parse_move(const std::string& str, const Position& pos) {
    if (str.length() < 4) return MOVE_NONE;
    File from_f = File(str[0] - 'a');
    Rank from_r = Rank(str[1] - '1');
    File to_f = File(str[2] - 'a');
    Rank to_r = Rank(str[3] - '1');
    Square from = make_square(from_f, from_r);
    Square to = make_square(to_f, to_r);

    MoveList list;
    generate_moves(pos, list);
    for (int i = 0; i < list.count; ++i) {
        Move m = list.moves[i];
        if (move_from(m) == from && move_to(m) == to) {
            if (move_type(m) == PROMOTION) {
                if (str.length() < 5) continue;
                char promochar = str[4];
                PieceType promo = PIECE_TYPE_NONE;
                if (promochar == 'q') promo = QUEEN;
                else if (promochar == 'r') promo = ROOK;
                else if (promochar == 'b') promo = BISHOP;
                else if (promochar == 'n') promo = KNIGHT;
                if (move_promo(m) != promo) continue;
            }
            return m;
        }
    }
    return MOVE_NONE;
}

void uci_loop() {
    init_zobrist();
    init_bitboards();

    Position pos;
    std::thread search_thread;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "uci") {
            std::cout << "id name " << ENGINE_NAME << " " << ENGINE_VERSION << std::endl;
            std::cout << "id author OpencodeChess" << std::endl;
            std::cout << "uciok" << std::endl;
        } else if (token == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (token == "ucinewgame") {
            pos.set_fen(Position::START_FEN);
        } else if (token == "position") {
            iss >> token;
            if (token == "startpos") {
                pos.set_fen(Position::START_FEN);
                iss >> token;
            } else if (token == "fen") {
                std::string fen;
                for (int i = 0; i < 6; ++i) {
                    iss >> token;
                    if (i > 0) fen += " ";
                    fen += token;
                }
                pos.set_fen(fen);
                iss >> token;
            }

            if (token == "moves") {
                while (iss >> token) {
                    Move m = parse_move(token, pos);
                    if (m != MOVE_NONE) {
                        pos.make_move(m);
                    }
                }
            }
        } else if (token == "go") {
            SearchLimits limits;
            std::string param;
            while (iss >> param) {
                if (param == "depth") { int d; iss >> d; limits.depth = d; }
                else if (param == "wtime") { int64_t t; iss >> t; limits.time_left[WHITE] = t; }
                else if (param == "btime") { int64_t t; iss >> t; limits.time_left[BLACK] = t; }
                else if (param == "winc") { int64_t t; iss >> t; limits.inc[WHITE] = t; }
                else if (param == "binc") { int64_t t; iss >> t; limits.inc[BLACK] = t; }
                else if (param == "movetime") { int64_t t; iss >> t; limits.movetime = t; }
                else if (param == "movestogo") { int s; iss >> s; limits.movestogo = s; }
                else if (param == "infinite") { limits.infinite = true; }
            }

            if (search_thread.joinable())
                search_thread.join();

            // Copy position for thread safety
            Position pos_copy = pos;
            search_thread = std::thread([pos_copy, limits]() mutable {
                SearchInfo info;
                search(pos_copy, const_cast<SearchLimits&>(limits), info);
                send_bestmove(info.best_move);
            });
        } else if (token == "stop") {
            SearchAborted = true;
            if (search_thread.joinable())
                search_thread.join();
        } else if (token == "quit") {
            SearchAborted = true;
            if (search_thread.joinable())
                search_thread.join();
            break;
        }
    }
}
