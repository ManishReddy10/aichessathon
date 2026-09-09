"""Correctness tests for the search.

Games measure strength; these measure legality and mate handling. Every
expected answer here is derived from python-chess at test time rather than
hardcoded, so a test failing means the engine is wrong, not that a FEN was
transcribed badly.
"""

import chess
import pytest

import agent

OPEN_MIDDLEGAME = "r2q1rk1/pp2bppp/2n1bn2/2pp4/3P4/2N1PN2/PPQ1BPPP/R1B2RK1 w - - 0 11"


def forcing_mates(fen: str) -> dict[str, int]:
    """Map each move that forces mate to the number of white moves it takes (1 or 2)."""
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


@pytest.fixture(autouse=True)
def _clean_tables() -> None:
    agent.new_game()


# --- the transposition key ------------------------------------------------


def test_board_key_separates_positions_that_differ_only_in_castling_rights() -> None:
    """A TT keyed without castling rights returns a score from the wrong position."""
    with_rights = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    without = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    assert agent._board_key(with_rights) != agent._board_key(without)


def test_board_key_separates_positions_that_differ_only_in_side_to_move() -> None:
    white = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    black = chess.Board("4k3/8/8/8/8/8/4P3/4K3 b - - 0 1")
    assert agent._board_key(white) != agent._board_key(black)


# --- mates ----------------------------------------------------------------


@pytest.mark.parametrize(
    "fen",
    [
        "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",  # back rank, Ra8#
        "6rk/6pp/8/6N1/8/8/8/6K1 w - - 0 1",  # smothered, Nf7#
    ],
)
def test_plays_an_available_mate_in_one(fen: str) -> None:
    mates = forcing_mates(fen)
    assert mates, "test position is wrong: no forcing mate exists"
    assert agent.get_move(fen, 5_000) in [m for m, plies in mates.items() if plies == 1]


def test_prefers_a_mate_in_one_over_a_mate_in_two() -> None:
    """Mate scores must shorten with ply, or the engine dawdles in a won position."""
    fen = "7k/8/8/8/8/8/6R1/6RK w - - 0 1"
    mates = forcing_mates(fen)
    quickest = [move for move, plies in mates.items() if plies == 1]
    assert quickest and len(mates) > len(quickest), "test position is wrong"
    assert agent.get_move(fen, 5_000) in quickest


def test_finds_a_forcing_mate_in_two() -> None:
    fen = "7k/8/8/8/8/8/R7/1R6 w - - 0 1"
    mates = forcing_mates(fen)
    assert mates and 1 not in mates.values(), "test position is wrong: a mate in one exists"
    assert agent.get_move(fen, 8_000) in mates


def test_returns_empty_when_already_checkmated() -> None:
    """No legal moves: get_move must return rather than raise or index into nothing."""
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"  # fool's mate
    assert chess.Board(fen).is_checkmate(), "test position is wrong: not checkmate"
    assert agent.get_move(fen, 1_000) == ""


# --- legality and the clock -----------------------------------------------


@pytest.mark.parametrize("budget_ms", [30, 60, 250, 2_000])
def test_returns_a_legal_move_on_any_budget(budget_ms: int) -> None:
    board = chess.Board(OPEN_MIDDLEGAME)
    assert chess.Move.from_uci(agent.get_move(OPEN_MIDDLEGAME, budget_ms)) in board.legal_moves


def test_returns_a_legal_move_when_in_check() -> None:
    fen = "4k3/8/8/8/8/8/4R3/4K3 b - - 0 1"
    board = chess.Board(fen)
    assert board.is_check() and not board.is_checkmate(), "test position is wrong"
    assert chess.Move.from_uci(agent.get_move(fen, 2_000)) in board.legal_moves


def test_promotes_with_a_uci_promotion_suffix() -> None:
    fen = "8/P6k/8/8/8/8/6K1/8 w - - 0 1"
    board = chess.Board(fen)
    move = chess.Move.from_uci(agent.get_move(fen, 2_000))
    assert move in board.legal_moves
    assert move.promotion is not None, "the only sensible move is a promotion"


def test_respects_the_clock_it_was_handed() -> None:
    """Overshooting the allotted budget is a flag, which loses the game outright."""
    import time

    started = time.monotonic()
    agent.get_move(OPEN_MIDDLEGAME, 1_000)
    assert time.monotonic() - started < 1.0


# --- null-move pruning ----------------------------------------------------


def test_null_move_is_refused_while_in_check() -> None:
    """A null move in check leaves the king capturable and the score is nonsense."""
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert not agent._null_move_allowed(chess.Board(fen))


def test_null_move_is_refused_in_a_king_and_pawn_endgame() -> None:
    """Zugzwang: with only pawns, passing is better than every real move."""
    assert not agent._null_move_allowed(chess.Board("8/5pk1/8/8/8/8/5PK1/8 w - - 0 1"))


def test_null_move_is_allowed_with_pieces_on_the_board() -> None:
    assert agent._null_move_allowed(chess.Board(OPEN_MIDDLEGAME))


def test_null_move_does_not_miss_a_zugzwang_win() -> None:
    """The classic null-move failure: a won king-and-pawn ending evaluated as drawn."""
    fen = "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"
    board = chess.Board(fen)
    assert chess.Move.from_uci(agent.get_move(fen, 3_000)) in board.legal_moves


# --- move ordering tables -------------------------------------------------


def test_killers_are_not_recorded_for_captures() -> None:
    """Captures are already ordered by MVV-LVA; a capture killer wastes a slot."""
    board = chess.Board(OPEN_MIDDLEGAME)
    capture = next(m for m in board.legal_moves if board.is_capture(m))
    agent._record_cutoff(board, capture, depth=4, ply=0)
    assert capture not in agent.KILLERS[0]


def test_killers_are_recorded_per_ply_for_quiet_moves() -> None:
    board = chess.Board(OPEN_MIDDLEGAME)
    quiet = next(m for m in board.legal_moves if not board.is_capture(m) and not m.promotion)
    agent._record_cutoff(board, quiet, depth=4, ply=3)
    assert quiet in agent.KILLERS[3]
    assert quiet not in agent.KILLERS[0]


def test_history_grows_with_depth_so_deep_cutoffs_outrank_shallow_ones() -> None:
    board = chess.Board(OPEN_MIDDLEGAME)
    quiet = next(m for m in board.legal_moves if not board.is_capture(m) and not m.promotion)
    agent._record_cutoff(board, quiet, depth=2, ply=0)
    shallow = agent.HISTORY[(quiet.from_square, quiet.to_square)]
    agent._record_cutoff(board, quiet, depth=6, ply=0)
    deep = agent.HISTORY[(quiet.from_square, quiet.to_square)] - shallow
    assert deep > shallow


def test_new_game_clears_state_that_must_not_cross_games() -> None:
    board = chess.Board(OPEN_MIDDLEGAME)
    quiet = next(m for m in board.legal_moves if not board.is_capture(m))
    agent._record_cutoff(board, quiet, depth=4, ply=0)
    agent.TT[agent._board_key(board)] = (1, 0, agent.EXACT, None)
    agent.new_game()
    assert not agent.TT
    assert not agent.HISTORY
    assert agent.KILLERS[0] == [None, None]


# --- game history and repetition ------------------------------------------

WON_FOR_WHITE = "6k1/5ppp/8/8/8/8/5PPP/3QK3 w - - 0 1"  # white is a queen up


def test_new_game_clears_the_position_history() -> None:
    agent._seen[("sentinel",)] = 3
    agent.new_game()
    assert not agent._seen


def test_get_move_records_the_position_it_was_handed() -> None:
    """The platform hands over a bare FEN, so history has to be kept here or nowhere."""
    key = agent._board_key(chess.Board(OPEN_MIDDLEGAME))
    agent.get_move(OPEN_MIDDLEGAME, 200)
    assert agent._seen.get(key) == 1
    agent.get_move(OPEN_MIDDLEGAME, 200)
    assert agent._seen.get(key) == 2


def test_draw_score_is_negative_for_the_side_that_would_be_accepting_it() -> None:
    """Contempt: a draw must look worse than a playable position when we are better."""
    board = chess.Board(WON_FOR_WHITE)
    agent._root_turn = board.turn
    assert agent._draw_score(board) < 0


def test_draw_score_is_positive_for_the_opponent() -> None:
    """Negamax flips perspective; without the flip the engine avoids saving draws too."""
    board = chess.Board(WON_FOR_WHITE)
    agent._root_turn = not board.turn
    assert agent._draw_score(board) > 0


def test_avoids_a_move_that_returns_to_a_position_already_seen_twice() -> None:
    """The bug that drew 38 won games: the engine cannot see the game's own history."""
    board = chess.Board(WON_FOR_WHITE)

    agent.new_game()
    preferred = chess.Move.from_uci(agent.get_move(WON_FOR_WHITE, 2_000))

    # Now claim that move's resulting position has already occurred twice.
    agent.new_game()
    board.push(preferred)
    agent._seen[agent._board_key(board)] = 2
    board.pop()

    assert agent.get_move(WON_FOR_WHITE, 2_000) != preferred.uci()


def test_still_takes_a_repetition_when_it_is_losing() -> None:
    """Contempt must not talk the engine out of a saving draw."""
    lost_for_white = "6kq/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1"  # white is a queen down
    board = chess.Board(lost_for_white)
    agent._root_turn = board.turn
    # A draw is worth far more than the position, so it must beat the static score.
    assert agent._draw_score(board) > agent._evaluate(board)


# --- tapered evaluation ---------------------------------------------------

ENDGAME_KING_CENTRED = "8/8/8/8/4K3/8/6P1/7k w - - 0 1"
ENDGAME_KING_CORNERED = "8/8/8/8/8/8/6P1/K6k w - - 0 1"
MIDDLEGAME_KING_CASTLED = "r1bq1rk1/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w - - 0 1"
MIDDLEGAME_KING_ON_H1 = "r1bq1rk1/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQ1R1K w - - 0 1"


def test_game_phase_is_full_with_all_the_pieces_on() -> None:
    assert agent._game_phase(chess.Board()) == pytest.approx(1.0)


def test_game_phase_is_zero_with_only_kings_and_pawns() -> None:
    assert agent._game_phase(chess.Board("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1")) == 0.0


def test_game_phase_falls_as_pieces_come_off() -> None:
    full = agent._game_phase(chess.Board(MIDDLEGAME_KING_CASTLED))
    bare = agent._game_phase(chess.Board(ENDGAME_KING_CENTRED))
    assert 0.0 <= bare < full <= 1.0


def test_endgame_king_is_worth_more_in_the_centre_than_in_the_corner() -> None:
    """A single middlegame king table tells the king to hide in a pawn ending."""
    centred = agent._evaluate(chess.Board(ENDGAME_KING_CENTRED))
    cornered = agent._evaluate(chess.Board(ENDGAME_KING_CORNERED))
    assert centred > cornered


def test_middlegame_king_still_prefers_shelter() -> None:
    """The taper must not leak endgame behaviour into a full board."""
    castled = agent._evaluate(chess.Board(MIDDLEGAME_KING_CASTLED))
    exposed = agent._evaluate(chess.Board(MIDDLEGAME_KING_ON_H1))
    assert castled > exposed


def test_the_evaluation_and_game_phase_agree_on_the_phase() -> None:
    """_evaluate totals the phase inline for speed; it must match the slow path."""
    for fen in (chess.STARTING_FEN, OPEN_MIDDLEGAME, ENDGAME_KING_CENTRED, WON_FOR_WHITE):
        board = chess.Board(fen)
        inline = sum(agent._PHASE_BY_TYPE[p.piece_type] for p in board.piece_map().values())
        assert agent._phase_from_material(inline) == agent._game_phase(board), fen
