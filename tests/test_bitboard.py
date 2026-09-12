"""Bitboard move generation, checked against python-chess at every step.

python-chess is the reference implementation: slow, but correct. Everything
here compares the fast path against it rather than against hardcoded numbers,
so a failure means our generator is wrong.
"""

import chess
import pytest

import bitboard as bb


def as_mask(squares: list[int]) -> int:
    mask = 0
    for square in squares:
        mask |= 1 << square
    return mask


# --- leaper attack tables -------------------------------------------------


@pytest.mark.parametrize("square", range(64))
def test_knight_attacks_match_python_chess(square: int) -> None:
    assert bb.knight_attacks(square) == int(chess.BB_KNIGHT_ATTACKS[square])


@pytest.mark.parametrize("square", range(64))
def test_king_attacks_match_python_chess(square: int) -> None:
    assert bb.king_attacks(square) == int(chess.BB_KING_ATTACKS[square])


@pytest.mark.parametrize("square", range(64))
def test_white_pawn_attacks_match_python_chess(square: int) -> None:
    assert bb.pawn_attacks(chess.WHITE, square) == int(chess.BB_PAWN_ATTACKS[chess.WHITE][square])


@pytest.mark.parametrize("square", range(64))
def test_black_pawn_attacks_match_python_chess(square: int) -> None:
    assert bb.pawn_attacks(chess.BLACK, square) == int(chess.BB_PAWN_ATTACKS[chess.BLACK][square])


# --- sliding attacks, the part magic bitboards exist for ------------------

OCCUPANCIES = [
    0,
    0xFFFF00000000FFFF,           # both back two ranks
    0x0000001818000000,           # four centre squares
    0xAA55AA55AA55AA55,           # checkerboard
    0x00FF00000000FF00,           # both pawn ranks
]


@pytest.mark.parametrize("occupied", OCCUPANCIES)
@pytest.mark.parametrize("square", [0, 7, 27, 28, 35, 36, 56, 63])
def test_rook_attacks_match_python_chess(square: int, occupied: int) -> None:
    expected = int(chess.BB_RANK_ATTACKS[square][occupied & chess.BB_RANK_MASKS[square]]
                   | chess.BB_FILE_ATTACKS[square][occupied & chess.BB_FILE_MASKS[square]])
    assert bb.rook_attacks(square, occupied) == expected


@pytest.mark.parametrize("occupied", OCCUPANCIES)
@pytest.mark.parametrize("square", [0, 7, 27, 28, 35, 36, 56, 63])
def test_bishop_attacks_match_python_chess(square: int, occupied: int) -> None:
    expected = int(chess.BB_DIAG_ATTACKS[square][occupied & chess.BB_DIAG_MASKS[square]])
    assert bb.bishop_attacks(square, occupied) == expected


# --- position parsing -----------------------------------------------------

FENS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # Kiwipete
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",  # en passant available
]


@pytest.mark.parametrize("fen", FENS)
def test_position_bitboards_match_python_chess(fen: str) -> None:
    board = chess.Board(fen)
    pos = bb.from_fen(fen)
    for colour in (chess.WHITE, chess.BLACK):
        for piece_type in range(1, 7):
            expected = int(board.pieces_mask(piece_type, colour))
            assert bb.piece_bitboard(pos, colour == chess.WHITE, piece_type) == expected, (
                fen, colour, piece_type
            )
    assert bb.occupied(pos) == int(board.occupied)


@pytest.mark.parametrize("fen", FENS)
def test_position_state_matches_python_chess(fen: str) -> None:
    board = chess.Board(fen)
    pos = bb.from_fen(fen)
    assert bb.white_to_move(pos) == (board.turn == chess.WHITE)
    assert bb.en_passant_square(pos) == (board.ep_square if board.ep_square is not None else -1)
    assert bb.halfmove_clock(pos) == board.halfmove_clock
    assert bb.can_castle(pos, chess.WHITE, kingside=True) == bool(
        board.castling_rights & chess.BB_H1)
    assert bb.can_castle(pos, chess.WHITE, kingside=False) == bool(
        board.castling_rights & chess.BB_A1)
    assert bb.can_castle(pos, chess.BLACK, kingside=True) == bool(
        board.castling_rights & chess.BB_H8)
    assert bb.can_castle(pos, chess.BLACK, kingside=False) == bool(
        board.castling_rights & chess.BB_A8)


# --- attack detection -----------------------------------------------------


@pytest.mark.parametrize("fen", FENS)
def test_attacked_squares_match_python_chess(fen: str) -> None:
    """Every square, both colours: the legality filter depends entirely on this."""
    board = chess.Board(fen)
    pos = bb.from_fen(fen)
    for colour in (chess.WHITE, chess.BLACK):
        for square in range(64):
            assert bb.is_attacked(pos, square, colour == chess.WHITE) == board.is_attacked_by(
                colour, square
            ), (fen, colour, chess.square_name(square))


# --- move generation ------------------------------------------------------


def python_chess_perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += python_chess_perft(board, depth - 1)
        board.pop()
    return total


@pytest.mark.parametrize("fen", FENS)
def test_legal_moves_match_python_chess(fen: str) -> None:
    """Same set of moves, in any order — the foundation everything else needs."""
    expected = sorted(m.uci() for m in chess.Board(fen).legal_moves)
    assert sorted(bb.legal_move_ucis(bb.from_fen(fen))) == expected, fen


@pytest.mark.parametrize("fen", FENS)
def test_legal_moves_match_after_every_first_move(fen: str) -> None:
    """Catches make() corrupting state: castling rights, ep square, occupancy."""
    board = chess.Board(fen)
    for move in board.legal_moves:
        board.push(move)
        expected = sorted(m.uci() for m in board.legal_moves)
        assert sorted(bb.legal_move_ucis(bb.from_fen(board.fen()))) == expected, (
            fen, move.uci()
        )
        board.pop()


@pytest.mark.parametrize(
    ("fen", "depth"),
    [(FENS[0], 4), (FENS[1], 3), (FENS[2], 4), (FENS[3], 3), (FENS[4], 3), (FENS[5], 3)],
)
def test_perft_matches_python_chess(fen: str, depth: int) -> None:
    """The gate: identical node counts means make/unmake is right too."""
    assert bb.perft(bb.from_fen(fen), depth) == python_chess_perft(chess.Board(fen), depth)


# --- the jitted path ------------------------------------------------------


@pytest.mark.parametrize(
    ("fen", "depth"),
    [(FENS[0], 4), (FENS[1], 3), (FENS[2], 4), (FENS[3], 3), (FENS[4], 3), (FENS[5], 3)],
)
def test_jitted_perft_matches_python_chess(fen: str, depth: int) -> None:
    """The jitted generator must agree with the reference, not merely be fast."""
    assert bb.perft_fast(bb.from_fen(fen), depth) == python_chess_perft(chess.Board(fen), depth)


@pytest.mark.parametrize("fen", FENS)
def test_jitted_move_generation_matches_the_reference(fen: str) -> None:
    position = bb.from_fen(fen)
    assert sorted(bb.legal_move_ucis_fast(position)) == sorted(bb.legal_move_ucis(position)), fen
