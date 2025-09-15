import pytest
from hard_wordle import load_environment, HardWordleEnv, HardModeTextArenaEnv
from verifiers.envs.textarena_env import TextArenaEnv
import textarena as ta
import re


# Helper function to simulate a guess and get feedback/state
def make_guess(env, guess_word):
    action = f"[{guess_word}]"
    is_done, info = env.step(action)
    return is_done, info

# Test fixture for a fresh hard wordle environment
@pytest.fixture
def hard_wordle_env():
    # We need to explicitly set a secret word for deterministic testing
    env = HardWordleEnv(word_length=5, num_guesses=6)
    # Manually set the secret word for testing. In a real scenario, this would be part of env.reset()
    # For this test, let's assume the secret word is 'apple'
    env.reset() # This will generate a random word, we'll override it.
    env.state.game_state["secret_word"] = "apple"
    env.state.game_state["guess_history"] = [] # Clear history in case reset added anything
    return env

def test_initial_env_loading():
    env = load_environment()
    assert isinstance(env, HardModeTextArenaEnv)
    assert env.game == "HardWordle-v0"  # game is the environment ID string

def test_first_guess_valid(hard_wordle_env):
    env = hard_wordle_env
    is_done, info = make_guess(env, "crane")
    assert not is_done
    # The info now contains the observation and reward
    assert "latest_observation" in info
    assert "Feedback:" in info["latest_observation"]["content"]
    assert info["reward"] == 0.0
    assert len(env.state.game_state["guess_history"]) == 1
    assert env.state.game_state["guess_history"][0][0] == "crane"

def test_win_condition(hard_wordle_env):
    env = hard_wordle_env
    is_done, info = make_guess(env, "apple")
    assert is_done
    assert info["reward"] == 1.0
    assert "Congratulations!" in info["reason"]

def test_invalid_word_format(hard_wordle_env):
    env = hard_wordle_env
    is_done, info = env.step("apple") # No brackets - direct call to step
    assert not is_done # Invalid format should NOT end the episode immediately
    assert info["reward"] < 1.0 # Should be less than 1.0 for invalid move
    assert "wrong format" in info["reason"]
    # The guess history should not be updated for invalid format
    assert len(env.state.game_state["guess_history"]) == 0

def test_word_length_violation(hard_wordle_env):
    env = hard_wordle_env
    is_done, info = env.step("[apples]") # Too long - direct call to step
    assert not is_done # Invalid length should NOT end the episode immediately
    assert "Your word must be exactly" in info["reason"]
    assert len(env.state.game_state["guess_history"]) == 0

def test_hard_mode_green_violation(hard_wordle_env):
    env = hard_wordle_env
    make_guess(env, "apply") # Secret: apple. Feedback: A(G) P(P) L(G) Y(X) -> G G G X X
    # First guess 'apply' gives feedback for 'a','p','l' (a at 0, p at 1, l at 2)
    # Green letters: a at 0, p at 1, l at 2

    # Next guess 'album' - violates 'a' at pos 0, 'p' at pos 1, 'l' at pos 2
    is_done, info = make_guess(env, "album")
    assert not is_done # Hard mode violation should NOT end the episode (error_allowance > 1)
    # The hard mode violation should be recorded but the game should continue
    assert len(env.state.game_state["guess_history"]) == 1 # Only the first valid guess is in history

def test_hard_mode_yellow_violation(hard_wordle_env):
    env = hard_wordle_env
    make_guess(env, "rates") # Secret: apple. Feedback: R(X) A(G) T(X) E(Y) S(X) -> X G X Y X
    # First guess 'rates' gives feedback for 'a' (green), 'e' (yellow)
    # Required yellow letter: 'e'

    # Next guess 'plank' - doesn't contain 'e'
    is_done, info = make_guess(env, "plank")
    assert not is_done # Hard mode violation should NOT end the episode (error_allowance > 1)
    # The hard mode violation should be recorded but the game should continue
    assert len(env.state.game_state["guess_history"]) == 1

def test_hard_mode_valid_second_guess(hard_wordle_env):
    env = hard_wordle_env
    make_guess(env, "apply") # Secret: apple. Feedback: G G G G X ('a','p','p','l' are green)

    # Second guess 'apple' - 'a', 'p', 'p', 'l' are in correct positions. This should be a win.
    is_done, info = make_guess(env, "apple")
    assert is_done  # This should be a win
    assert info["reward"] == 1.0
    assert "Congratulations!" in info["reason"]


def test_count_turns_reward_func():
    # Test the count_turns_reward_func directly without using the wrapper
    from hard_wordle import count_turns_reward_func
    from verifiers.parsers.xml_parser import XMLParser
    
    # Create a dummy completion that shows 3 guesses before winning
    dummy_completion = [
        {"role": "assistant", "content": "Welcome to Wordle! A secret 5-letter word has been chosen..."},
        {"role": "assistant", "content": "<guess>[crane]</guess>"},
        {"role": "user", "content": "Feedback: X X Y X Y"},
        {"role": "assistant", "content": "<guess>[great]</guess>"},
        {"role": "user", "content": "Feedback: G G X X Y"},
        {"role": "assistant", "content": "<guess>[grape]</guess>"},
        {"role": "user", "content": "Congratulations! You guessed the word correctly!"},
    ]
    parser = XMLParser(fields=["guess"], answer_field="guess")
    reward = count_turns_reward_func(parser, dummy_completion, "grape")
    # 4 assistant messages, but first is initial prompt, so 3 turns
    # For 3 turns to win: 1.0 / (3 + 1) = 0.25
    assert reward == 0.25


def test_partial_credit_reward_func():
    # Test the partial_credit_reward_func directly without using the wrapper
    from hard_wordle import partial_credit_reward_func
    from verifiers.parsers.xml_parser import XMLParser
    
    # Create a dummy completion with feedback that has 2 greens and 2 yellows
    dummy_completion = [
        {"role": "assistant", "content": "You are playing Wordle..."},
        {"role": "assistant", "content": "<guess>[gnome]</guess>"},
        {"role": "user", "content": "You submitted [GNOME].\nFeedback:\nG X Y X X"},
        {"role": "assistant", "content": "<guess>[goats]</guess>"},
        {"role": "user", "content": "You submitted [GOATS].\nFeedback:\nG O A T S\nG G X Y Y"},
    ]
    parser = XMLParser(fields=["guess"], answer_field="guess")
    reward = partial_credit_reward_func(parser, dummy_completion)
    # For "G G X Y Y": 2 greens, 2 yellows
    # Expected: 0.2 * 2 (greens) + 0.1 * 2 (yellows) = 0.4 + 0.2 = 0.6
    assert abs(reward - 0.6) < 0.001  # Use approximate equality for floating point


def test_partial_credit_reward_func_invalid_move():
    # Test the partial_credit_reward_func with invalid move feedback
    from hard_wordle import partial_credit_reward_func
    from verifiers.parsers.xml_parser import XMLParser
    
    # Create a dummy completion with invalid move feedback (no scoring line)
    dummy_completion = [
        {"role": "assistant", "content": "You are playing Wordle..."},
        {"role": "assistant", "content": "<guess>[notaword]</guess>"},
        {"role": "user", "content": "'notaword' is not an English word."},
    ]
    parser = XMLParser(fields=["guess"], answer_field="guess")
    reward = partial_credit_reward_func(parser, dummy_completion)
    # Should return 0.0 for invalid moves that don't have feedback scoring
    assert reward == 0.0

def test_disallowed_letters_are_not_required(hard_wordle_env):
    env = hard_wordle_env
    make_guess(env, "taser") # Secret: apple. Feedback: T(X) A(Y) S(X) E(Y) R(X)
    # Yellow: A, E (must be included)
    # Disallowed: T, S, R

    # Next guess 'apple' - contains A and E, and no T, S, R. This should be a win.
    is_done, info = make_guess(env, "apple")
    assert is_done  # This should be a win
    assert info["reward"] == 1.0
    assert "Congratulations!" in info["reason"]
