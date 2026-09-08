"""The submission entrypoint. The platform imports this file and calls get_move."""

import time

import chess
import chess.polyglot

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

EXACT, LOWER, UPPER = 0, 1, 2
# Transposition table: zobrist_hash -> (depth, score, flag, best_move)
TT: dict[int, tuple[int, int, int, chess.Move | None]] = {}
TT_MAX = 500_000

_deadline: float = 0.0
_timeout: bool = False
_nodes: int = 0
_NODES_PER_CHECK = 2048


def _pawn_eval(w_pawns: int, b_pawns: int) -> int:
    """Pawn structure score from White's perspective."""
    score = 0
    for f in range(8):
        file_bb = chess.BB_FILES[f]
        wc = chess.popcount(w_pawns & file_bb)
        bc = chess.popcount(b_pawns & file_bb)

        # Doubled pawns: each extra pawn on the same file is a liability
        if wc > 1:
            score -= 20 * (wc - 1)
        if bc > 1:
            score += 20 * (bc - 1)

        # Isolated pawns: no friendly pawn on an adjacent file
        adj = (chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0)
        if wc and not (w_pawns & adj):
            score -= 15 * wc
        if bc and not (b_pawns & adj):
            score += 15 * bc

    # Passed pawns: no enemy pawn can block or capture it on the way to promotion
    for sq in chess.SquareSet(w_pawns):
        if not (_PASSED_SPAN[1][sq] & b_pawns):
            score += _PASSED_BONUS[chess.square_rank(sq)]
    for sq in chess.SquareSet(b_pawns):
        if not (_PASSED_SPAN[0][sq] & w_pawns):
            score -= _PASSED_BONUS[7 - chess.square_rank(sq)]

    return score


def _king_safety(king_sq: int, friendly_pawns: int, dr: int) -> int:
    """Pawn shield score for one king. dr=+1 for White (shield above), -1 for Black."""
    f = chess.square_file(king_sq)
    r = chess.square_rank(king_sq)
    score = 0
    for df in (-1, 0, 1):
        nf = f + df
        if not (0 <= nf <= 7):
            continue
        # Pawn on the first shield rank is worth more than one on the second
        nr1 = r + dr
        if 0 <= nr1 <= 7 and (friendly_pawns & chess.BB_SQUARES[chess.square(nf, nr1)]):
            score += 15
        nr2 = r + 2 * dr
        if 0 <= nr2 <= 7 and (friendly_pawns & chess.BB_SQUARES[chess.square(nf, nr2)]):
            score += 5
        # Open file next to king is dangerous
        if not (friendly_pawns & chess.BB_FILES[nf]):
            score -= 10
    return score


def _evaluate(board: chess.Board) -> int:
    """Material + PST + pawn structure + king safety + bishop pair, side-to-move perspective."""
    turn = board.turn
    score = 0

    # Material + PST
    for sq, piece in board.piece_map().items():
        v = _PIECE_VALUE[piece.piece_type]
        pst = (PST_WHITE if piece.color == chess.WHITE else PST_BLACK)[piece.piece_type][sq]
        score += (v + pst) if piece.color == turn else -(v + pst)

    # Pawn structure (doubled, isolated, passed)
    w_pawns = int(board.pieces(chess.PAWN, chess.WHITE))
    b_pawns = int(board.pieces(chess.PAWN, chess.BLACK))
    pawn_score = _pawn_eval(w_pawns, b_pawns)
    score += pawn_score if turn == chess.WHITE else -pawn_score

    # King safety: pawn shield and open files near king
    w_king = board.king(chess.WHITE)
    b_king = board.king(chess.BLACK)
    ks = 0
    if w_king is not None:
        ks += _king_safety(w_king, w_pawns, 1)
    if b_king is not None:
        ks -= _king_safety(b_king, b_pawns, -1)
    score += ks if turn == chess.WHITE else -ks

    # Bishop pair: two bishops complement each other well in open positions
    w_bp = 30 if chess.popcount(int(board.pieces(chess.BISHOP, chess.WHITE))) >= 2 else 0
    b_bp = 30 if chess.popcount(int(board.pieces(chess.BISHOP, chess.BLACK))) >= 2 else 0
    bp = w_bp - b_bp
    score += bp if turn == chess.WHITE else -bp

    return score


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

    if not _timeout and len(TT) < TT_MAX:
        if best_score <= orig_alpha:
            flag = UPPER
        elif best_score >= beta:
            flag = LOWER
        else:
            flag = EXACT
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
