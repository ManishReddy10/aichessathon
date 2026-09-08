"""The submission entrypoint. The platform imports this file and calls get_move."""

import time

import chess
import chess.polyglot
import numpy as np
from numba import njit

# Import time runs at the game start. 90s to import packages, build tables etc.

MATE = 900_000

_PIECE_VALUE: dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Piece-square tables from White's perspective, rank 8 (a8..h8) first, rank 1 last.
_PST_RAW: dict[int, list[int]] = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0,
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    chess.ROOK: [
          0,  0,  0,  0,  0,  0,  0,  0,
          5, 10, 10, 10, 10, 10, 10,  5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
          0,  0,  0,  5,  5,  0,  0,  0,
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20,
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20,
    ],
}

# sq ^ 56 flips the rank: a1(0)->PST row 7, a8(56)->PST row 0 (White's perspective).
# Black reuses the same table unflipped so rank 1 mirrors White's rank 8.
PST_WHITE: dict[int, list[int]] = {p: [t[sq ^ 56] for sq in range(64)] for p, t in _PST_RAW.items()}
PST_BLACK: dict[int, list[int]] = {p: [t[sq] for sq in range(64)] for p, t in _PST_RAW.items()}

# Passed pawn span: squares in front of a pawn (same + adjacent files) that must be
# free of enemy pawns for the pawn to be "passed".
# _PASSED_SPAN[color_int][square] — color_int: WHITE=1, BLACK=0
_PASSED_SPAN: list[list[int]] = [[0] * 64, [0] * 64]
for _sq in range(64):
    _f, _r = chess.square_file(_sq), chess.square_rank(_sq)
    _span_files = chess.BB_FILES[_f]
    if _f > 0:
        _span_files |= chess.BB_FILES[_f - 1]
    if _f < 7:
        _span_files |= chess.BB_FILES[_f + 1]
    for _nr in range(_r + 1, 8):
        _PASSED_SPAN[1][_sq] |= _span_files & chess.BB_RANKS[_nr]  # white looks up
    for _nr in range(0, _r):
        _PASSED_SPAN[0][_sq] |= _span_files & chess.BB_RANKS[_nr]  # black looks down

# Passed pawn bonus by rank (white perspective: rank 0=rank1, rank 6=rank7)
_PASSED_BONUS = [0, 10, 20, 35, 50, 75, 100, 0]

# ---------------------------------------------------------------------------
# Numpy arrays for the JIT evaluator.  Built once at import time.
# ---------------------------------------------------------------------------

# piece_vals[piece_type] — index 0 unused (empty square)
_NP_PIECE_VALS = np.array([0, 100, 320, 330, 500, 900, 20000], dtype=np.int32)

# PST lookup: _NP_PST_W[piece_type, square], _NP_PST_B[piece_type, square]
_NP_PST_W = np.zeros((7, 64), dtype=np.int32)
_NP_PST_B = np.zeros((7, 64), dtype=np.int32)
for _pt in range(1, 7):
    if _pt in PST_WHITE:
        _NP_PST_W[_pt] = PST_WHITE[_pt]
        _NP_PST_B[_pt] = PST_BLACK[_pt]

# Passed-pawn span masks as int64 bitboards (same bit pattern as uint64)
def _to_i64(x: int) -> np.int64:
    return np.int64(x if x < (1 << 63) else x - (1 << 64))

_NP_PASSED_W  = np.array([_to_i64(_PASSED_SPAN[1][sq]) for sq in range(64)], dtype=np.int64)
_NP_PASSED_B  = np.array([_to_i64(_PASSED_SPAN[0][sq]) for sq in range(64)], dtype=np.int64)
_NP_PASSED_BONUS = np.array(_PASSED_BONUS, dtype=np.int32)

# Square bitmasks and file bitmasks as int64 (bitwise ops ignore the sign)
_NP_BB_SQ    = np.array([_to_i64(1 << sq) for sq in range(64)], dtype=np.int64)
_NP_BB_FILES = np.array([_to_i64(int(chess.BB_FILES[f])) for f in range(8)], dtype=np.int64)

# Pre-allocated encoding buffers (reused every call to avoid heap churn)
_enc_pieces   = np.zeros(64, dtype=np.int32)
_enc_is_white = np.zeros(64, dtype=np.bool_)


@njit(cache=False)
def _jit_eval(
    pieces:       np.ndarray,   # int32[64]  piece type per square (0=empty)
    is_white:     np.ndarray,   # bool[64]   True if the piece is White's
    turn_white:   np.bool_,     # True if White to move
    w_pawns:      np.int64,     # bitboard
    b_pawns:      np.int64,
    w_king:       np.int32,     # square index, -1 if absent
    b_king:       np.int32,
    w_bishops:    np.int32,
    b_bishops:    np.int32,
    piece_vals:   np.ndarray,   # int32[7]
    pst_w:        np.ndarray,   # int32[7, 64]
    pst_b:        np.ndarray,
    passed_w:     np.ndarray,   # int64[64]
    passed_b:     np.ndarray,
    passed_bonus: np.ndarray,   # int32[8]
    bb_sq:        np.ndarray,   # int64[64]
    bb_files:     np.ndarray,   # int64[8]
) -> np.int32:
    score      = np.int32(0)
    pawn_score = np.int32(0)
    w_file     = np.zeros(8, dtype=np.int32)
    b_file     = np.zeros(8, dtype=np.int32)

    # --- Material + PST, and per-file pawn counts + passed-pawn bonus ---
    for sq in range(64):
        pt = pieces[sq]
        if pt == np.int32(0):
            continue
        v   = piece_vals[pt]
        wh  = is_white[sq]
        pst = pst_w[pt, sq] if wh else pst_b[pt, sq]
        val = np.int32(v + pst)
        if wh == turn_white:
            score += val
        else:
            score -= val

        if pt == np.int32(1):          # pawn
            f = sq % 8
            if wh:
                w_file[f] += np.int32(1)
                if (passed_w[sq] & b_pawns) == np.int64(0):
                    pawn_score += passed_bonus[sq >> 3]
            else:
                b_file[f] += np.int32(1)
                if (passed_b[sq] & w_pawns) == np.int64(0):
                    pawn_score -= passed_bonus[np.int32(7) - (sq >> 3)]

    # --- Doubled and isolated pawns ---
    for f in range(8):
        wc = w_file[f]
        bc = b_file[f]
        if wc > np.int32(1):
            pawn_score -= np.int32(20) * (wc - np.int32(1))
        if bc > np.int32(1):
            pawn_score += np.int32(20) * (bc - np.int32(1))
        adj_w = np.int64(0)
        adj_b = np.int64(0)
        if f > 0:
            adj_w |= w_pawns & bb_files[f - 1]
            adj_b |= b_pawns & bb_files[f - 1]
        if f < 7:
            adj_w |= w_pawns & bb_files[f + 1]
            adj_b |= b_pawns & bb_files[f + 1]
        if wc > np.int32(0) and adj_w == np.int64(0):
            pawn_score -= np.int32(15) * wc
        if bc > np.int32(0) and adj_b == np.int64(0):
            pawn_score += np.int32(15) * bc

    score += pawn_score if turn_white else -pawn_score

    # --- King safety (pawn shield + open files) ---
    ks = np.int32(0)

    if w_king >= np.int32(0):
        wkf = w_king % np.int32(8)
        wkr = w_king // np.int32(8)
        for df in range(-1, 2):
            nf = wkf + np.int32(df)
            if nf < np.int32(0) or nf > np.int32(7):
                continue
            if (w_pawns & bb_files[nf]) == np.int64(0):
                ks -= np.int32(10)
            nr1 = wkr + np.int32(1)
            if nr1 <= np.int32(7) and (w_pawns & bb_sq[nr1 * np.int32(8) + nf]) != np.int64(0):
                ks += np.int32(15)
            nr2 = wkr + np.int32(2)
            if nr2 <= np.int32(7) and (w_pawns & bb_sq[nr2 * np.int32(8) + nf]) != np.int64(0):
                ks += np.int32(5)

    if b_king >= np.int32(0):
        bkf = b_king % np.int32(8)
        bkr = b_king // np.int32(8)
        for df in range(-1, 2):
            nf = bkf + np.int32(df)
            if nf < np.int32(0) or nf > np.int32(7):
                continue
            if (b_pawns & bb_files[nf]) == np.int64(0):
                ks += np.int32(10)
            nr1 = bkr - np.int32(1)
            if nr1 >= np.int32(0) and (b_pawns & bb_sq[nr1 * np.int32(8) + nf]) != np.int64(0):
                ks -= np.int32(15)
            nr2 = bkr - np.int32(2)
            if nr2 >= np.int32(0) and (b_pawns & bb_sq[nr2 * np.int32(8) + nf]) != np.int64(0):
                ks -= np.int32(5)

    score += ks if turn_white else -ks

    # --- Bishop pair ---
    bp = (np.int32(30) if w_bishops >= np.int32(2) else np.int32(0)) - \
         (np.int32(30) if b_bishops >= np.int32(2) else np.int32(0))
    score += bp if turn_white else -bp

    return score


EXACT, LOWER, UPPER = 0, 1, 2
# Transposition table: zobrist_hash -> (depth, score, flag, best_move)
TT: dict[int, tuple[int, int, int, chess.Move | None]] = {}
TT_MAX = 1_000_000

_deadline: float = 0.0
_timeout: bool = False
_nodes: int = 0
_NODES_PER_CHECK = 2048


def _evaluate(board: chess.Board) -> int:
    """Encode board in one piece_map pass, then dispatch to the JIT evaluator."""
    _enc_pieces[:] = 0
    _enc_is_white[:] = False
    w_pawns_i = 0
    b_pawns_i = 0
    w_king_sq = -1
    b_king_sq = -1
    w_bishops = 0
    b_bishops = 0

    for sq, piece in board.piece_map().items():
        pt = piece.piece_type
        wh = piece.color == chess.WHITE
        _enc_pieces[sq]   = pt
        _enc_is_white[sq] = wh
        if pt == chess.PAWN:
            if wh: w_pawns_i |= 1 << sq
            else:  b_pawns_i |= 1 << sq
        elif pt == chess.KING:
            if wh: w_king_sq = sq
            else:  b_king_sq = sq
        elif pt == chess.BISHOP:
            if wh: w_bishops += 1
            else:  b_bishops += 1

    return int(_jit_eval(
        _enc_pieces, _enc_is_white, np.bool_(board.turn == chess.WHITE),
        _to_i64(w_pawns_i), _to_i64(b_pawns_i),
        np.int32(w_king_sq), np.int32(b_king_sq),
        np.int32(w_bishops), np.int32(b_bishops),
        _NP_PIECE_VALS, _NP_PST_W, _NP_PST_B,
        _NP_PASSED_W, _NP_PASSED_B, _NP_PASSED_BONUS,
        _NP_BB_SQ, _NP_BB_FILES,
    ))


def _capture_score(board: chess.Board, move: chess.Move) -> int:
    """MVV-LVA: prefer capturing high-value pieces with low-value attackers."""
    attacker = board.piece_type_at(move.from_square) or chess.PAWN
    victim = board.piece_type_at(move.to_square)
    if victim is None:  # en passant
        return 10 * 100 - _PIECE_VALUE[attacker]
    return 10 * _PIECE_VALUE[victim] - _PIECE_VALUE[attacker]


def _order_moves(board: chess.Board, moves: list[chess.Move], tt_move: chess.Move | None) -> None:
    def key(move: chess.Move) -> int:
        if move == tt_move:
            return 30000
        if board.is_capture(move):
            return 20000 + _capture_score(board, move)
        if move.promotion:
            return 18000 + _PIECE_VALUE.get(move.promotion, 0)
        return 0
    moves.sort(key=key, reverse=True)


def _quiescence(board: chess.Board, alpha: int, beta: int) -> int:
    global _timeout, _nodes
    _nodes += 1
    if _nodes % _NODES_PER_CHECK == 0 and time.monotonic() >= _deadline:
        _timeout = True
        return 0

    # Read-only TT probe: if a main-search entry already covers this position, use it
    zh = chess.polyglot.zobrist_hash(board)
    cached = TT.get(zh)
    if cached is not None:
        _, tt_score, tt_flag, _ = cached
        if tt_flag == EXACT:
            return tt_score
        if tt_flag == LOWER and tt_score >= beta:
            return tt_score
        if tt_flag == UPPER and tt_score <= alpha:
            return tt_score

    stand_pat = _evaluate(board)
    if stand_pat >= beta:
        return beta
    alpha = max(alpha, stand_pat)

    captures = sorted(
        (m for m in board.legal_moves if board.is_capture(m) or m.promotion),
        key=lambda m: _capture_score(board, m),
        reverse=True,
    )
    for move in captures:
        board.push(move)
        score = -_quiescence(board, -beta, -alpha)
        board.pop()
        if _timeout:
            return 0
        if score >= beta:
            return beta
        alpha = max(alpha, score)

    return alpha


def _alpha_beta(board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
    global _timeout, _nodes
    _nodes += 1
    if _nodes % _NODES_PER_CHECK == 0 and time.monotonic() >= _deadline:
        _timeout = True
        return 0

    if board.is_repetition(3) or board.is_fifty_moves():
        return 0

    zh = chess.polyglot.zobrist_hash(board)
    tt_move: chess.Move | None = None
    if zh in TT:
        tt_depth, tt_score, tt_flag, tt_move = TT[zh]
        if tt_depth >= depth:
            if tt_flag == EXACT:
                return tt_score
            elif tt_flag == LOWER:
                alpha = max(alpha, tt_score)
            elif tt_flag == UPPER:
                beta = min(beta, tt_score)
            if alpha >= beta:
                return tt_score

    moves = list(board.legal_moves)
    if not moves:
        return (-MATE + ply) if board.is_check() else 0

    if depth == 0:
        return _quiescence(board, alpha, beta)

    _order_moves(board, moves, tt_move)

    orig_alpha = alpha
    best_score = -MATE - 1
    best_move: chess.Move | None = None

    for move in moves:
        board.push(move)
        score = -_alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
        board.pop()
        if _timeout:
            return 0
        if score > best_score:
            best_score = score
            best_move = move
            if score > alpha:
                alpha = score
        if alpha >= beta:
            break

    if not _timeout:
        flag = UPPER if best_score <= orig_alpha else (LOWER if best_score >= beta else EXACT)
        existing = TT.get(zh)
        if existing is None:
            if len(TT) >= TT_MAX:
                TT.clear()  # generation reset: evict everything rather than silently dropping
            TT[zh] = (depth, best_score, flag, best_move)
        elif depth >= existing[0] or flag == EXACT:
            # depth-preferred: replace shallow entries; always keep exact scores
            TT[zh] = (depth, best_score, flag, best_move)

    return best_score


def _search_root(board: chess.Board, depth: int, prev_best: chess.Move | None) -> chess.Move | None:
    zh = chess.polyglot.zobrist_hash(board)
    tt_move = TT.get(zh, (None, None, None, None))[3] or prev_best

    moves = list(board.legal_moves)
    if not moves:
        return None
    _order_moves(board, moves, tt_move)

    alpha = -MATE - 1
    best_move: chess.Move | None = None

    for move in moves:
        board.push(move)
        score = -_alpha_beta(board, depth - 1, -MATE - 1, -alpha, 1)
        board.pop()
        if _timeout:
            return None
        if score > alpha:
            alpha = score
            best_move = move

    return best_move


def get_move(fen: str, time_left_ms: int) -> str:
    global _deadline, _timeout, _nodes

    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if not moves:
        return ""

    # Allocate ~5% of remaining clock + partial increment, min 80ms, max 8s
    if time_left_ms < 2000:
        allotted_ms = 80
    elif time_left_ms < 10000:
        allotted_ms = time_left_ms // 8
    else:
        allotted_ms = min(time_left_ms // 20 + 400, 8000)

    _deadline = time.monotonic() + allotted_ms / 1000.0
    _timeout = False
    _nodes = 0

    best_move: chess.Move | None = None

    for depth in range(1, 64):
        _timeout = False
        candidate = _search_root(board, depth, best_move)
        if not _timeout and candidate is not None:
            best_move = candidate
            print(f"depth {depth} nodes {_nodes} move {best_move}")
        if _timeout:
            break

    return (best_move or moves[0]).uci()


# ---------------------------------------------------------------------------
# JIT warmup — runs inside the 90s init budget before the clock starts.
# Numba compiles per unique argument-type signature; warm every type we'll
# actually pass so the first real call is instant.
# ---------------------------------------------------------------------------
_enc_pieces[0]   = chess.KING;  _enc_is_white[0]  = True
_enc_pieces[63]  = chess.KING;  _enc_is_white[63] = False
_jit_eval(
    _enc_pieces, _enc_is_white, np.bool_(True),
    np.int64(0), np.int64(0), np.int32(0), np.int32(63),
    np.int32(0), np.int32(0),
    _NP_PIECE_VALS, _NP_PST_W, _NP_PST_B,
    _NP_PASSED_W, _NP_PASSED_B, _NP_PASSED_BONUS,
    _NP_BB_SQ, _NP_BB_FILES,
)
_enc_pieces[:]   = 0
_enc_is_white[:] = False
