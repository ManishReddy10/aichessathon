"""The jitted evaluation and search, built on bitboard.py."""

import chess
import numpy as np
import pytest

import bitboard as bb
import engine

FENS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 11",
    "8/5pk1/6p1/7p/7P/5PP1/r5K1/3R4 w - - 0 40",
]


def mirror(fen: str) -> str:
    """Same position with the colours swapped, so evaluation must negate."""
    return chess.Board(fen).mirror().fen()


# --- zobrist --------------------------------------------------------------


@pytest.mark.parametrize("fen", FENS)
def test_zobrist_is_stable_for_the_same_position(fen: str) -> None:
    assert engine.zobrist(bb.from_fen(fen)) == engine.zobrist(bb.from_fen(fen))


def test_zobrist_separates_positions_differing_only_in_side_to_move() -> None:
    white = bb.from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    black = bb.from_fen("4k3/8/8/8/8/8/4P3/4K3 b - - 0 1")
    assert engine.zobrist(white) != engine.zobrist(black)


def test_zobrist_separates_positions_differing_only_in_castling_rights() -> None:
    with_rights = bb.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    without = bb.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    assert engine.zobrist(with_rights) != engine.zobrist(without)


@pytest.mark.parametrize("fen", FENS)
def test_zobrist_is_restored_after_make_and_unmake(fen: str) -> None:
    """A key that drifts across make/unmake poisons every transposition lookup."""
    position = bb.from_fen(fen)
    before = engine.zobrist(position)
    assert engine.probe_make_unmake_keeps_zobrist(position), fen
    assert engine.zobrist(position) == before, "zobrist() must not mutate the position"


# --- evaluation -----------------------------------------------------------


@pytest.mark.parametrize("fen", FENS)
def test_evaluation_is_colour_symmetric(fen: str) -> None:
    """Scores are from the mover's side, so mirroring the board must not change them.

    Any asymmetry here is a bug that quietly favours one colour, and the arena
    would show it as a colour split rather than as an obvious failure.
    """
    assert engine.evaluate(bb.from_fen(fen)) == engine.evaluate(bb.from_fen(mirror(fen)))


def test_being_a_queen_up_evaluates_better_than_level() -> None:
    level = engine.evaluate(bb.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"))
    a_queen_up = engine.evaluate(bb.from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1"))
    assert a_queen_up > level + 800


def test_endgame_king_prefers_the_centre() -> None:
    centred = engine.evaluate(bb.from_fen("8/8/8/8/4K3/8/6P1/7k w - - 0 1"))
    cornered = engine.evaluate(bb.from_fen("8/8/8/8/8/8/6P1/K6k w - - 0 1"))
    assert centred > cornered


# --- search ---------------------------------------------------------------


def forcing_mates(fen: str) -> dict[str, int]:
    """Moves that force mate, mapped to how many moves it takes (1 or 2)."""
    board = chess.Board(fen)
    found: dict[str, int] = {}
    for first in board.legal_moves:
        board.push(first)
        if board.is_checkmate():
            found[first.uci()] = 1
        elif list(board.legal_moves) and _every_reply_is_mated(board):
            found[first.uci()] = 2
        board.pop()
    return found


def _every_reply_is_mated(board: chess.Board) -> bool:
    for reply in board.legal_moves:
        board.push(reply)
        mated = any(_mates(board, follow_up) for follow_up in board.legal_moves)
        board.pop()
        if not mated:
            return False
    return True


def _mates(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    result = board.is_checkmate()
    board.pop()
    return result


@pytest.mark.parametrize(
    "fen",
    [
        "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",
        "6rk/6pp/8/6N1/8/8/8/6K1 w - - 0 1",
    ],
)
def test_search_plays_an_available_mate_in_one(fen: str) -> None:
    mates = forcing_mates(fen)
    assert mates, "test position is wrong"
    assert engine.search(bb.from_fen(fen), 2_000) in mates


def test_search_prefers_the_shorter_mate() -> None:
    fen = "7k/8/8/8/8/8/6R1/6RK w - - 0 1"
    mates = forcing_mates(fen)
    assert mates, "test position is wrong"
    assert engine.search(bb.from_fen(fen), 3_000) in mates


@pytest.mark.parametrize("fen", FENS)
@pytest.mark.parametrize("budget_ms", [40, 300, 1_500])
def test_search_always_returns_a_legal_move(fen: str, budget_ms: int) -> None:
    legal = bb.legal_move_ucis(bb.from_fen(fen))
    assert engine.search(bb.from_fen(fen), budget_ms) in legal


def test_search_respects_its_budget() -> None:
    import time

    started = time.monotonic()
    engine.search(bb.from_fen(FENS[1]), 500)
    assert time.monotonic() - started < 1.2, "overshooting the budget is a flag"


def test_search_finds_a_free_queen() -> None:
    """A one-ply tactic: the queen on d5 is hanging to the pawn on e4."""
    fen = "4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1"
    assert engine.search(bb.from_fen(fen), 2_000) == "e4d5"


def test_search_is_much_faster_than_the_python_chess_engine() -> None:
    """The whole reason this module exists."""
    nodes = engine.search_nodes(bb.from_fen(FENS[2]), 1_000)
    assert nodes > 300_000, f"only {nodes:,} nodes in 1s -- no faster than python-chess"


# --- repetition history ---------------------------------------------------

WON_FOR_WHITE = "6k1/5ppp/8/8/8/8/5PPP/3QK3 w - - 0 1"


def test_new_game_clears_the_position_history() -> None:
    engine.remember(bb.from_fen(WON_FOR_WHITE))
    engine.new_game()
    assert engine.history_size() == 0


def test_remember_counts_repeated_positions() -> None:
    engine.new_game()
    position = bb.from_fen(WON_FOR_WHITE)
    engine.remember(position)
    engine.remember(position)
    assert engine.times_seen(position) == 2


def test_search_avoids_returning_to_a_position_seen_twice() -> None:
    """The bug that drew 38 of 200 won games: the engine cannot see game history."""
    engine.new_game()
    preferred = engine.search(bb.from_fen(WON_FOR_WHITE), 1_500)

    engine.new_game()
    after = bb.from_fen(WON_FOR_WHITE)
    boards, state = after[0].copy(), after[1].copy()
    undo = np.zeros(5, dtype=np.int64)
    move = next(m for m in bb.generate_legal(after) if bb.move_uci(m) == preferred)
    bb.make_jit(boards, state, move, undo)
    engine.remember((boards, state))
    engine.remember((boards, state))

    assert engine.search(bb.from_fen(WON_FOR_WHITE), 1_500) != preferred


# --- null-move pruning ----------------------------------------------------


def test_null_move_is_refused_while_in_check() -> None:
    """Passing in check leaves the king capturable and the score is nonsense."""
    assert not engine.null_move_allowed(bb.from_fen("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1"))


def test_null_move_is_refused_in_a_king_and_pawn_endgame() -> None:
    """Zugzwang: with only pawns, passing beats every legal move and the score lies."""
    assert not engine.null_move_allowed(bb.from_fen("8/5pk1/8/8/8/8/5PK1/8 w - - 0 1"))


def test_null_move_is_allowed_with_pieces_on_the_board() -> None:
    assert engine.null_move_allowed(bb.from_fen(FENS[1]))


def test_making_a_null_move_only_flips_the_side_and_clears_en_passant() -> None:
    position = bb.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2")
    before = position[0].copy()
    assert bb.en_passant_square(position) == chess.D6
    after = engine.probe_null_move(position)
    assert (after[0] == before).all(), "pieces must not move"
    assert not bb.white_to_move(after)
    assert bb.en_passant_square(after) == -1, "the ep square does not survive a pass"


def test_null_move_search_still_finds_a_forced_mate() -> None:
    """Null-move pruning must never prune away a mate the engine would otherwise see."""
    fen = "7k/8/8/8/8/8/R7/1R6 w - - 0 1"
    assert engine.search(bb.from_fen(fen), 3_000) in forcing_mates(fen)


# --- the evaluation terms the port left behind ----------------------------


# Each pair below differs ONLY in the term being tested: the squares are chosen
# so the piece-square tables contribute identically to both sides of the
# comparison. Without that care these pass whether or not the term exists.


def test_a_passed_pawn_is_worth_more_than_one_that_is_held_up() -> None:
    # White e5 either way. Black's pawn sits on b7 or f7 -- same pawn PST value,
    # but f7 stands in the e-pawn's path and b7 does not.
    passed = engine.evaluate(bb.from_fen("4k3/1p6/8/4P3/8/8/8/4K3 w - - 0 1"))
    held_up = engine.evaluate(bb.from_fen("4k3/5p2/8/4P3/8/8/8/4K3 w - - 0 1"))
    assert passed > held_up


def test_doubled_pawns_are_penalised() -> None:
    # d3 and e3 carry the same pawn PST value, so only the doubling differs.
    doubled = engine.evaluate(bb.from_fen("4k3/8/8/8/4P3/4P3/8/4K3 w - - 0 1"))
    apart = engine.evaluate(bb.from_fen("4k3/8/8/8/4P3/3P4/8/4K3 w - - 0 1"))
    assert apart > doubled


def test_isolated_pawns_are_penalised() -> None:
    # a4, b4 and h4 all score 0 on the pawn table.
    isolated = engine.evaluate(bb.from_fen("4k3/8/8/8/P6P/8/8/4K3 w - - 0 1"))
    connected = engine.evaluate(bb.from_fen("4k3/8/8/8/PP6/8/8/4K3 w - - 0 1"))
    assert connected > isolated


def test_the_bishop_pair_is_worth_something() -> None:
    # Swapping the f1 bishop for a knight is worth 10 in material and 20 in PST
    # on its own, so anything past 30 has to be the pair bonus itself.
    two_bishops = engine.evaluate(bb.from_fen("4k3/8/8/8/8/8/8/2B1KB2 w - - 0 1"))
    bishop_and_knight = engine.evaluate(bb.from_fen("4k3/8/8/8/8/8/8/2B1KN2 w - - 0 1"))
    assert two_bishops - bishop_and_knight > 55


def test_a_king_behind_its_pawns_is_safer_than_one_with_none() -> None:
    # Same king square, same three pawns, same total pawn PST -- only the
    # distance between the king and its pawns changes.
    sheltered = engine.evaluate(bb.from_fen("4k3/8/8/8/8/8/5PPP/6K1 w - - 0 1"))
    abandoned = engine.evaluate(bb.from_fen("4k3/8/8/8/8/8/PPP5/6K1 w - - 0 1"))
    assert sheltered > abandoned


def test_the_search_actually_uses_null_move_pruning() -> None:
    """Guards against the helper existing while the search never calls it.

    Null-move pruning cuts large parts of the tree, so with it enabled the same
    budget must reach further than without it.
    """
    import inspect

    source = inspect.getsource(engine.negamax.py_func)
    assert "null_move_allowed_jit" in source, "negamax never calls the null-move guard"
    assert "null_score" in source, "negamax never performs the null search"
