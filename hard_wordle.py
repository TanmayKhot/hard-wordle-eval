

import re
from typing import Optional, Tuple, List, Dict, Any
import verifiers as vf
from verifiers.envs.textarena_env import TextArenaEnv
from textarena.envs.Wordle.env import WordleEnv # This import is crucial for inheriting WordleEnv
import textarena as ta
from textarena.envs.registration import register_with_versions # Import register_with_versions
from datasets import Dataset, Features, Value # Import Dataset, Features, Value
import random # Import random

### prompts
THINK_GUESS_SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.
Remember that from the second turn onwards, you must use all the green and yellow letters identified in the previous round
If the previous round had green letters then your current guess must have the same letters in the same position.
If the previous round had yellow letters then your current guess must contain those letters.

In each turn, think step-by-step inside <think>...</think> tags, \
then follow the instructions inside <guess>...</guess> tags."""

NOTHINK_GUESS_SYSTEM_PROMPT = """You are a competitive game player. \
Make sure you read the game instructions carefully, and always follow the required format.
If the previous round had green letters then your current guess must have the same letters in the same position.
If the previous round had yellow letters then your current guess must contain those letters.

In each turn, give only your guess inside <guess>...</guess> tags."""


### feedback functions
def wordle_feedback_fn(observation: str) -> str:
    if "Feedback:" in observation:
        return observation.split("Feedback:")[-1]
    else:
        return observation


### reward functions
def check_answer_reward_func(parser, completion, answer, **kwargs) -> float:
    guess = parser.parse_answer(completion)
    return 1.0 if guess == "[" + answer + "]" else 0.0


def count_turns_reward_func(parser, completion, answer, **kwargs) -> float:
    # Ensure num_turns accounts for the assistant's initial message if present
    assistant_messages = [x for x in completion if x["role"] == "assistant"]
    num_turns = len(assistant_messages)
    # If the first assistant message is just the initial prompt, don't count it as a turn
    if num_turns > 0 and "Welcome to Wordle!" in assistant_messages[0].get("content", ""):
        num_turns -= 1 
        
    is_correct = check_answer_reward_func(parser, completion, answer, **kwargs)
    return is_correct / (num_turns + 1)


def partial_credit_reward_func(parser, completion, **kwargs) -> float:
    """Reward function that gives partial credit for the correct guess."""
    final_env_response = parser.get_user_messages(completion)[-1]["content"].strip()
    
    # Check if final_env_response contains 'Feedback:' to determine if it's a game message or invalid move message
    if "Feedback:" in final_env_response:
        feedback_line = final_env_response.split("Feedback:")[-1].strip()
        # The feedback line will be the second line of the board, e.g., "W O R D L E\nGYXXX"
        if "\n" in feedback_line:
            scoring = feedback_line.split("\n")[-1].strip()
            num_greens = scoring.count("G")
            num_yellows = scoring.count("Y")
            return 0.2 * num_greens + 0.1 * num_yellows
    return 0.0 # Return 0.0 if no valid feedback is found (e.g., due to an invalid move or initial prompt)


class HardWordleEnv(WordleEnv):
    def __init__(self, word_length: int = 5, num_guesses: int = 6, hardcore: bool = False): # Added hardcore to __init__
        super().__init__(word_length=word_length, num_guesses=num_guesses, hardcore=True)
        # The `hardcore=True` here ensures the larger dictionary is used,
        # but we still need to implement the letter-inclusion rule.
        # Set higher error allowance to allow hard mode violations without ending the game
        # Note: state is created by parent __init__, so we set error_allowance in reset()

    def reset(self, num_players: int = 1, seed: Optional[int] = None):
        """Reset the environment and set higher error allowance for hard mode."""
        super().reset(num_players=num_players, seed=seed)
        # Set higher error allowance to allow hard mode violations without ending the game
        self.state.error_allowance = 10  # Allow up to 10 invalid moves before ending the game

    def step(self, action: str) -> Tuple[bool, ta.Info]: # Changed vf.wrappers.text_arena_wrapper.TextArenaInfo to ta.Info
        player_id = self.state.current_player_id
        self.state.add_observation(message=action, observation_type=ta.ObservationType.PLAYER_ACTION)
        match = re.search(r"\[(\w+)\]", action)

        if match is None:
            self.state.set_invalid_move(reward=self._get_percentage_completion(), reason=f"You tried submitting a word in the wrong format. Please make sure to use squared brackets.")
            is_done, info = self.state.step()
            # Populate info for invalid format
            info["reason"] = "You tried submitting a word in the wrong format. Please make sure to use squared brackets."
            info["reward"] = self.state.rewards.get(0, 0.0) if self.state.rewards else 0.0
            return is_done, info

        word = match.group(1).lower()
        if len(word) != self.state.game_state["word_length"]:
            self.state.set_invalid_move(reward=self._get_percentage_completion(), reason=f"Your word must be exactly {self.state.game_state['word_length']} letters.")
            is_done, info = self.state.step()
            # Populate info for wrong length
            info["reason"] = f"Your word must be exactly {self.state.game_state['word_length']} letters."
            info["reward"] = self.state.rewards.get(0, 0.0) if self.state.rewards else 0.0
            return is_done, info

        # --- Hard Mode Logic ---
        if self.state.game_state["guess_history"]:
            # Get feedback from the previous turn
            last_guess_word, last_feedback = self.state.game_state["guess_history"][-1]

            required_greens = {} # {position: letter}
            required_yellows = set() # {letter}
            
            # Populate required_greens and required_yellows from last feedback
            for i, (letter, feedback_type) in enumerate(zip(last_guess_word, last_feedback)):
                if feedback_type == "G":
                    required_greens[i] = letter
                elif feedback_type == "Y":
                    required_yellows.add(letter)
            
            # Validate current guess against hard mode rules for greens
            for pos, req_letter in required_greens.items():
                if word[pos] != req_letter:
                    self.state.set_invalid_move(reward=self._get_percentage_completion(), reason=f"Hard Mode violation: Letter '{req_letter.upper()}' must be in position {pos + 1}.")
                    is_done, info = self.state.step()
                    # Populate info for hard mode violation
                    info["reason"] = f"Hard Mode violation: Letter '{req_letter.upper()}' must be in position {pos + 1}."
                    info["reward"] = self.state.rewards.get(0, 0.0) if self.state.rewards else 0.0
                    return is_done, info

            # Validate current guess against hard mode rules for yellows
            for req_letter in required_yellows:
                if req_letter not in word:
                    self.state.set_invalid_move(reward=self._get_percentage_completion(), reason=f"Hard Mode violation: Guess must contain letter '{req_letter.upper()}'.")
                    is_done, info = self.state.step()
                    # Populate info for hard mode violation
                    info["reason"] = f"Hard Mode violation: Guess must contain letter '{req_letter.upper()}'."
                    info["reward"] = self.state.rewards.get(0, 0.0) if self.state.rewards else 0.0
                    return is_done, info

        # --- End Hard Mode Logic ---

        # If all hard mode checks pass, or if it's the first turn,
        # proceed with the base WordleEnv's step method
        is_done, info = super().step(action)
        
        # Populate step_info with relevant information for testing
        if not is_done:
            # For normal moves, populate step_info with observation info
            player_id, observation = self.get_observation()
            if observation:
                latest_message = observation[-1][1]  # Last message content
                info["latest_observation"] = {"content": latest_message}
                info["reward"] = self.state.rewards.get(0, 0.0) if self.state.rewards else 0.0
        else:
            # For game-ending moves, populate step_info with outcome info
            if self.state.rewards:
                info["reward"] = self.state.rewards.get(0, 0.0)
            if hasattr(self.state, 'game_info') and 0 in self.state.game_info:
                info["reason"] = self.state.game_info[0].get("reason", "")
        
        return is_done, info

    def ta_to_hf(self) -> Tuple[Dataset, Optional[Dataset]]: # Copy of TextArenaEnv.ta_to_hf
        dataset_rows = []
        eval_dataset_rows = []
        
        # Use self (HardWordleEnv) to get word_list and user_prompt
        # Initialize state if not already done (e.g., if ta_to_hf is called standalone)
        if not hasattr(self, 'state') or self.state is None:
            self.reset(num_players=1) # Resetting HardWordleEnv directly

        # `self._generate_player_prompt` expects a player_id and game_state
        # We can pass dummy values or retrieve from self.state
        # `self.state.game_state` will be populated after `self.reset()`
        user_prompt = self._generate_player_prompt(0, self.state.game_state)
        words = self.word_list

        # set seed
        random.seed(self.seed)
        for i in range(self.num_train_examples + self.num_eval_examples):
            question = user_prompt
            answer = random.choice(words)
            if i < self.num_train_examples:
                dataset_rows.append({"question": question, "answer": answer})
            else:
                eval_dataset_rows.append({"question": question, "answer": answer})
        
        # Explicitly define features to prevent pyarrow type inference issues
        features = Features({"question": Value("string"), "answer": Value("string")})

        dataset = Dataset.from_list(dataset_rows, features=features)
        if self.num_eval_examples > 0:
            eval_dataset = Dataset.from_list(eval_dataset_rows, features=features)
        else:
            eval_dataset = None
        return dataset, eval_dataset

# New class to override TextArenaEnv's ta_to_hf method
class HardModeTextArenaEnv(TextArenaEnv):
    def ta_to_hf(self) -> Tuple[Dataset, Optional[Dataset]]:
        dataset_rows = []
        eval_dataset_rows = []
        
        # The game passed to TextArenaEnv is an ID, so we need to make it
        ta_env_instance = ta.make(env_id=self.game)
        ta_env_instance.reset(num_players=1)
        _, user_prompt = ta_env_instance.get_observation()
        words = ta_env_instance.word_list

        random.seed(self.seed)
        for i in range(self.num_train_examples + self.num_eval_examples):
            question = user_prompt
            answer = random.choice(words)
            if i < self.num_train_examples:
                dataset_rows.append({"question": question, "answer": answer})
            else:
                eval_dataset_rows.append({"question": question, "answer": answer})
        
        features = Features({"question": Value("string"), "answer": Value("string")})
        dataset = Dataset.from_list(dataset_rows, features=features)
        if self.num_eval_examples > 0:
            eval_dataset = Dataset.from_list(eval_dataset_rows, features=features)
        else:
            eval_dataset = None
        return dataset, eval_dataset


# Register HardWordleEnv with textarena
HARD_WORDLE_ENV_ID = "HardWordle-v0"
register_with_versions(
    id=HARD_WORDLE_ENV_ID,
    entry_point=HardWordleEnv,
    wrappers={"default": [], "-train": []},
    hardcore=True, # Ensure hardcore is passed to the underlying WordleEnv
    word_length=5,
    num_guesses=6,
)

### environment loader
def load_environment(
    num_train_examples: int = 2000,
    num_eval_examples: int = 20,
    use_think: bool = True,
):
    if use_think:
        system_prompt = THINK_GUESS_SYSTEM_PROMPT
        parser = vf.XMLParser(fields=["think", "guess"], answer_field="guess")
    else:
        system_prompt = NOTHINK_GUESS_SYSTEM_PROMPT
        parser = vf.XMLParser(fields=["guess"], answer_field="guess")

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(check_answer_reward_func)
    rubric.add_reward_func(partial_credit_reward_func)
    rubric.add_reward_func(count_turns_reward_func)
    rubric.add_reward_func(parser.get_format_reward_func(), weight=0.2)

    # Instantiate HardModeTextArenaEnv instead of TextArenaEnv
    vf_env = HardModeTextArenaEnv(
        game=HARD_WORDLE_ENV_ID, # Pass the registered ID here
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        feedback_fn=wordle_feedback_fn,
        # dataset and eval_dataset are now handled by HardModeTextArenaEnv.ta_to_hf()
    )
    return vf_env
