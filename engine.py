"""Evaluation and search on bitboards, compiled with numba.

agent.py's search runs on python-chess, which is 74% of its runtime. This is
the replacement: the same algorithms against bitboard.py's representation, with
nothing in the hot loop that numba has to box.
"""

import time

import numpy as np
from numba import njit, objmode

import bitboard as bb

# --- zobrist ---------------------------------------------------------------
# Random keys, drawn once at import from a fixed seed so a position hashes the
# same on every run and the table is reproducible for anyone reading the code.

_rng = np.random.default_rng(0xC0FFEE)
PIECE_KEYS = _rng.integers(0, 1 << 64, size=(12, 64), dtype=np.uint64)
CASTLING_KEYS = _rng.integers(0, 1 << 64, size=16, dtype=np.uint64)
EP_FILE_KEYS = _rng.integers(0, 1 << 64, size=8, dtype=np.uint64)
SIDE_KEY = np.uint64(_rng.integers(0, 1 << 64, dtype=np.uint64))


@njit(cache=False)
def zobrist_jit(bbs, state):
    key = np.uint64(0)
    for piece in range(12):
        pieces = bbs[piece]
        while pieces:
            square = bb.lsb_index(pieces)
            pieces &= pieces - bb.ONE
            key ^= PIECE_KEYS[piece][square]
    key ^= CASTLING_KEYS[state[bb.CASTLING]]
    if state[bb.EP] >= 0:
        key ^= EP_FILE_KEYS[state[bb.EP] % 8]
    if state[bb.SIDE] == 1:
        key ^= SIDE_KEY
    return key


def zobrist(position: bb.Position) -> int:
    return int(zobrist_jit(position[0], position[1]))


def probe_make_unmake_keeps_zobrist(position: bb.Position) -> bool:
    """Play and take back every legal move, checking the key comes home each time."""
    boards, state = position[0].copy(), position[1].copy()
    moves = np.zeros(bb.MAX_MOVES, dtype=np.int64)
    undo = np.zeros(5, dtype=np.int64)
    before = int(zobrist_jit(boards, state))
    count = bb.generate_jit(boards, state, moves)
    for index in range(count):
        move = int(moves[index])
        bb.make_jit(boards, state, move, undo)
        bb.unmake_jit(boards, state, move, undo)
        if int(zobrist_jit(boards, state)) != before:
            return False
    return True


# --- evaluation ------------------------------------------------------------
# Material and piece-square tables, tapered between a middlegame and an endgame
# king table. Same weights agent.py uses, so the two can be compared directly.

PIECE_VALUES = np.array([100, 320, 330, 500, 900, 20000], dtype=np.int64)
PHASE_WEIGHTS = np.array([0, 1, 1, 2, 4, 0], dtype=np.int64)
PHASE_MAX = 24

_PST_SOURCE = {
    0: [0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10, 5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0, 5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    1: [-50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30, -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50],
    2: [-20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10, -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10, -20],
    3: [0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5, 0, 0, 0, 5, 5, 0, 0, 0],
    4: [-20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10, -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20],
    5: [-30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20, -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20, 20, 30, 10, 0, 0, 10, 30, 20],
}
_KING_ENDGAME = [
    -50, -40, -30, -20, -20, -30, -40, -50, -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30, -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30, -50, -30, -30, -30, -30, -30, -30, -50,
]

# Tables are written rank 8 first; square ^ 56 flips them into a1=0 indexing.
PST_WHITE = np.zeros((6, 64), dtype=np.int64)
PST_BLACK = np.zeros((6, 64), dtype=np.int64)
for _piece, _table in _PST_SOURCE.items():
    for _square in range(64):
        PST_WHITE[_piece][_square] = _table[_square ^ 56]
        PST_BLACK[_piece][_square] = _table[_square]
KING_EG_WHITE = np.array([_KING_ENDGAME[sq ^ 56] for sq in range(64)], dtype=np.int64)
KING_EG_BLACK = np.array([_KING_ENDGAME[sq] for sq in range(64)], dtype=np.int64)


@njit(cache=False)
def evaluate_jit(bbs, state):
    """Score from the side to move's point of view, in centipawns."""
    phase_material = 0
    for piece in range(6):
        white_count = 0
        pieces = bbs[piece]
        while pieces:
            pieces &= pieces - bb.ONE
            white_count += 1
        black_count = 0
        pieces = bbs[6 + piece]
        while pieces:
            pieces &= pieces - bb.ONE
            black_count += 1
        phase_material += PHASE_WEIGHTS[piece] * (white_count + black_count)
    phase = phase_material / PHASE_MAX
    if phase > 1.0:
        phase = 1.0

    score = 0
    for piece in range(6):
        pieces = bbs[piece]
        while pieces:
            square = bb.lsb_index(pieces)
            pieces &= pieces - bb.ONE
            if piece == 5:
                table = PST_WHITE[5][square] * phase + KING_EG_WHITE[square] * (1.0 - phase)
                score += PIECE_VALUES[piece] + np.int64(table)
            else:
                score += PIECE_VALUES[piece] + PST_WHITE[piece][square]
        pieces = bbs[6 + piece]
        while pieces:
            square = bb.lsb_index(pieces)
            pieces &= pieces - bb.ONE
            if piece == 5:
                table = PST_BLACK[5][square] * phase + KING_EG_BLACK[square] * (1.0 - phase)
                score -= PIECE_VALUES[piece] + np.int64(table)
            else:
                score -= PIECE_VALUES[piece] + PST_BLACK[piece][square]

    return score if state[bb.SIDE] == 0 else -score


def evaluate(position: bb.Position) -> int:
    return int(evaluate_jit(position[0], position[1]))


# --- search ----------------------------------------------------------------

MATE = 900_000
MATE_BOUND = MATE - 1000
MAX_PLY = 64
TT_BITS = 22
TT_SIZE = 1 << TT_BITS
TT_MASK = np.uint64(TT_SIZE - 1)
EXACT, LOWER, UPPER = 0, 1, 2
NODES_PER_TIME_CHECK = 2048

TT_KEY = np.zeros(TT_SIZE, dtype=np.uint64)
TT_DEPTH = np.zeros(TT_SIZE, dtype=np.int16)
TT_FLAG = np.zeros(TT_SIZE, dtype=np.int8)
TT_MOVE = np.zeros(TT_SIZE, dtype=np.int32)
TT_SCORE = np.zeros(TT_SIZE, dtype=np.int32)

KILLERS = np.zeros((MAX_PLY, 2), dtype=np.int64)
HISTORY = np.zeros((12, 64), dtype=np.int64)


def new_game() -> None:
    TT_KEY[:] = 0
    TT_DEPTH[:] = 0
    TT_FLAG[:] = 0
    TT_MOVE[:] = 0
    TT_SCORE[:] = 0
    KILLERS[:] = 0
    HISTORY[:] = 0


@njit(cache=False)
def piece_on(bbs, square, white):
    offset = 0 if white else 6
    for index in range(6):
        if (bbs[offset + index] >> np.uint64(square)) & bb.ONE:
            return index
    return -1


@njit(cache=False)
def score_moves(bbs, state, moves, count, scores, tt_move, ply, killers,
                history):
    """Cheap ordering: the table's move, then captures by MVV-LVA, then killers."""
    white = state[bb.SIDE] == 0
    for index in range(count):
        move = moves[index]
        origin = move & 63
        target = (move >> 6) & 63
        promotion = (move >> 12) & 7
        if move == tt_move:
            scores[index] = 1_000_000
            continue
        victim = piece_on(bbs, target, not white)
        if victim >= 0:
            attacker = piece_on(bbs, origin, white)
            scores[index] = 900_000 + PIECE_VALUES[victim] * 10 - PIECE_VALUES[attacker]
        elif promotion:
            scores[index] = 880_000 + PIECE_VALUES[promotion - 1]
        elif move == killers[ply][0]:
            scores[index] = 870_000
        elif move == killers[ply][1]:
            scores[index] = 860_000
        else:
            attacker = piece_on(bbs, origin, white)
            offset = 0 if white else 6
            scores[index] = history[offset + attacker][target] if attacker >= 0 else 0


@njit(cache=False)
def pick_move(moves, scores, count, start):
    """Selection sort one step: most searches cut off long before the tail."""
    best = start
    for index in range(start + 1, count):
        if scores[index] > scores[best]:
            best = index
    moves[start], moves[best] = moves[best], moves[start]
    scores[start], scores[best] = scores[best], scores[start]
    return moves[start]


@njit(cache=False)
def in_check_jit(bbs, white):
    king = bbs[(0 if white else 6) + 5]
    if king == bb.ZERO:
        return False
    return bb.is_attacked_jit(bbs, bb.lsb_index(king), not white)


@njit(cache=False)
def quiescence(bbs, state, alpha, beta, ply, move_stack, score_stack, undo_stack,
               counter, killers, history):
    counter[0] += 1
    stand_pat = evaluate_jit(bbs, state)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    if ply >= MAX_PLY - 1:
        return alpha

    moves = move_stack[ply]
    scores = score_stack[ply]
    undo = undo_stack[ply]
    count = bb.generate_jit(bbs, state, moves)
    white = state[bb.SIDE] == 0
    score_moves(bbs, state, moves, count, scores, 0, ply, killers, history)

    for index in range(count):
        move = pick_move(moves, scores, count, index)
        target = (move >> 6) & 63
        flag = (move >> 15) & 3
        promotion = (move >> 12) & 7
        # Captures and promotions only, or the horizon never settles.
        if piece_on(bbs, target, not white) < 0 and flag != bb.FLAG_EP and promotion == 0:
            continue
        bb.make_jit(bbs, state, move, undo)
        if in_check_jit(bbs, white):
            bb.unmake_jit(bbs, state, move, undo)
            continue
        score = -quiescence(bbs, state, -beta, -alpha, ply + 1, move_stack, score_stack,
                            undo_stack, counter, killers, history)
        bb.unmake_jit(bbs, state, move, undo)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


@njit(cache=False)
def negamax(bbs, state, depth, alpha, beta, ply, move_stack, score_stack, undo_stack,
            counter, deadline, timed_out,
            tt_key, tt_depth, tt_flag, tt_moves, tt_score, killers, history):
    counter[0] += 1
    if counter[0] % NODES_PER_TIME_CHECK == 0:
        now = 0.0
        with objmode(now="f8"):
            now = time.perf_counter()
        if now >= deadline:
            timed_out[0] = 1
            return 0
    if timed_out[0]:
        return 0

    if state[bb.HALFMOVE] >= 100 and ply > 0:
        return 0

    key = zobrist_jit(bbs, state)
    slot = np.int64(key & TT_MASK)
    tt_move = 0
    if tt_key[slot] == key:
        tt_move = tt_moves[slot]
        if tt_depth[slot] >= depth and ply > 0:
            score = tt_score[slot]
            if score > MATE_BOUND:
                score -= ply
            elif score < -MATE_BOUND:
                score += ply
            flag = tt_flag[slot]
            if flag == EXACT:
                return score
            if flag == LOWER and score >= beta:
                return score
            if flag == UPPER and score <= alpha:
                return score

    white = state[bb.SIDE] == 0
    checked = in_check_jit(bbs, white)
    if checked:
        depth += 1  # check extension: a forcing line should not be cut short

    if depth <= 0:
        return quiescence(bbs, state, alpha, beta, ply, move_stack, score_stack,
                          undo_stack, counter, killers, history)

    moves = move_stack[ply]
    scores = score_stack[ply]
    undo = undo_stack[ply]
    count = bb.generate_jit(bbs, state, moves)
    score_moves(bbs, state, moves, count, scores, tt_move, ply, killers, history)

    original_alpha = alpha
    best_score = -MATE - 1
    best_move = 0
    legal_moves = 0

    for index in range(count):
        move = pick_move(moves, scores, count, index)
        bb.make_jit(bbs, state, move, undo)
        if in_check_jit(bbs, white):
            bb.unmake_jit(bbs, state, move, undo)
            continue
        legal_moves += 1

        if legal_moves == 1:
            score = -negamax(bbs, state, depth - 1, -beta, -alpha, ply + 1,
                                         move_stack, score_stack, undo_stack, counter,
                                         deadline, timed_out,
                    tt_key, tt_depth, tt_flag, tt_moves, tt_score, killers, history)
        else:
            reduction = 0
            quiet = piece_on(bbs, (move >> 6) & 63, not white) < 0 and (move >> 12) & 7 == 0
            if legal_moves > 4 and depth >= 3 and quiet and not checked:
                reduction = 1
            score = -negamax(bbs, state, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1,
                             move_stack, score_stack, undo_stack, counter, deadline, timed_out,
                             tt_key, tt_depth, tt_flag, tt_moves, tt_score, killers, history)
            if not timed_out[0] and score > alpha and (reduction > 0 or score < beta):
                score = -negamax(bbs, state, depth - 1, -beta, -alpha, ply + 1,
                                             move_stack, score_stack, undo_stack, counter,
                                             deadline, timed_out,
                    tt_key, tt_depth, tt_flag, tt_moves, tt_score, killers, history)
        bb.unmake_jit(bbs, state, move, undo)

        if timed_out[0]:
            return 0
        if score > best_score:
            best_score = score
            best_move = move
            if score > alpha:
                alpha = score
        if alpha >= beta:
            target = (move >> 6) & 63
            if piece_on(bbs, target, not white) < 0 and (move >> 12) & 7 == 0:
                if killers[ply][0] != move:
                    killers[ply][1] = killers[ply][0]
                    killers[ply][0] = move
                attacker = piece_on(bbs, move & 63, white)
                if attacker >= 0:
                    history[(0 if white else 6) + attacker][target] += depth * depth
            break

    if legal_moves == 0:
        return -MATE + ply if checked else 0

    stored = best_score
    if stored > MATE_BOUND:
        stored += ply
    elif stored < -MATE_BOUND:
        stored -= ply
    if tt_key[slot] != key or tt_depth[slot] <= depth:
        tt_key[slot] = key
        tt_depth[slot] = depth
        tt_score[slot] = stored
        tt_moves[slot] = best_move
        if best_score <= original_alpha:
            tt_flag[slot] = UPPER
        elif best_score >= beta:
            tt_flag[slot] = LOWER
        else:
            tt_flag[slot] = EXACT

    return best_score


@njit(cache=False)
def search_root(bbs, state, max_depth, deadline, move_stack, score_stack, undo_stack,
                counter, timed_out,
                tt_key, tt_depth, tt_flag, tt_moves, tt_score, killers, history):
    """Iterative deepening. Keeps the last completed depth's move when time runs out."""
    best_move = 0
    moves = move_stack[0]
    undo = undo_stack[0]
    white = state[bb.SIDE] == 0

    # Something legal to return even if depth 1 does not finish.
    count = bb.generate_jit(bbs, state, moves)
    for index in range(count):
        move = moves[index]
        bb.make_jit(bbs, state, move, undo)
        legal = not in_check_jit(bbs, white)
        bb.unmake_jit(bbs, state, move, undo)
        if legal:
            best_move = move
            break
    if best_move == 0:
        return 0, 0

    completed = 0
    for depth in range(1, max_depth + 1):
        timed_out[0] = 0
        alpha = -MATE - 1
        iteration_best = 0
        count = bb.generate_jit(bbs, state, moves)
        scores = score_stack[0]
        score_moves(bbs, state, moves, count, scores, best_move, 0, killers, history)
        legal_moves = 0

        for index in range(count):
            move = pick_move(moves, scores, count, index)
            bb.make_jit(bbs, state, move, undo)
            if in_check_jit(bbs, white):
                bb.unmake_jit(bbs, state, move, undo)
                continue
            legal_moves += 1
            if legal_moves == 1:
                score = -negamax(bbs, state, depth - 1, -MATE - 1, -alpha, 1, move_stack,
                                 score_stack, undo_stack, counter, deadline, timed_out,
                                 tt_key, tt_depth, tt_flag, tt_moves, tt_score, killers, history)
            else:
                score = -negamax(bbs, state, depth - 1, -alpha - 1, -alpha, 1, move_stack,
                                 score_stack, undo_stack, counter, deadline, timed_out,
                                 tt_key, tt_depth, tt_flag, tt_moves, tt_score, killers, history)
                if not timed_out[0] and score > alpha:
                    score = -negamax(bbs, state, depth - 1, -MATE - 1, -alpha, 1, move_stack,
                                     score_stack, undo_stack, counter, deadline, timed_out,
                    tt_key, tt_depth, tt_flag, tt_moves, tt_score, killers, history)
            bb.unmake_jit(bbs, state, move, undo)
            if timed_out[0]:
                break
            if score > alpha or iteration_best == 0:
                alpha = score
                iteration_best = move

        if timed_out[0]:
            break
        if iteration_best != 0:
            best_move = iteration_best
            completed = depth
        now = 0.0
        with objmode(now="f8"):
            now = time.perf_counter()
        if now >= deadline:
            break

    return best_move, completed


def _stacks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((MAX_PLY + 2, bb.MAX_MOVES), dtype=np.int64),
        np.zeros((MAX_PLY + 2, bb.MAX_MOVES), dtype=np.int64),
        np.zeros((MAX_PLY + 2, 5), dtype=np.int64),
    )


def _run(position: bb.Position, budget_ms: int) -> tuple[int, int, int]:
    boards, state = position[0].copy(), position[1].copy()
    move_stack, score_stack, undo_stack = _stacks()
    counter = np.zeros(1, dtype=np.int64)
    timed_out = np.zeros(1, dtype=np.int64)
    deadline = time.perf_counter() + budget_ms / 1000.0
    move, depth = search_root(
        boards, state, MAX_PLY - 2, deadline, move_stack, score_stack, undo_stack,
        counter, timed_out, TT_KEY, TT_DEPTH, TT_FLAG, TT_MOVE, TT_SCORE, KILLERS, HISTORY,
    )
    return int(move), int(counter[0]), int(depth)


def search(position: bb.Position, budget_ms: int) -> str:
    move, _, _ = _run(position, budget_ms)
    return bb.move_uci(move) if move else ""


def search_nodes(position: bb.Position, budget_ms: int) -> int:
    return _run(position, budget_ms)[1]


def search_verbose(position: bb.Position, budget_ms: int) -> tuple[str, int, int]:
    move, nodes, depth = _run(position, budget_ms)
    return (bb.move_uci(move) if move else ""), nodes, depth
