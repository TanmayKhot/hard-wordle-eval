# Hard Mode Wordle Environment for LLM Evaluation
A challenging Wordle variant that enforces strict letter inclusion rules. The LLM must include all previously identified correct letters (green/yellow) in subsequent guesses, making it significantly harder than standard Wordle. Perfect for testing LLM reasoning and constraint-following abilities with the verifiers framework.
<br>[Link to Prime Intellect environment](https://app.primeintellect.ai/dashboard/environments/tnmy/hard-wordle)

### Overview
- **Environment ID**: `hard-wordle`
- **Short description**: Multi-turn Wordle game environment with hard mode rules; rewards correctness, partial credit, turns, and format.
- **Tags**: games, multi-turn, wordle, xml, feedback, hard-mode

### Datasets
- **Primary dataset(s)**: TextArena `Wordle-v0` (environment provides episodes)
- **Source links**: TextArena
- **Split sizes**: Number of episodes controlled via args

### Task
- **Type**: multi-turn (game interaction)
- **Parser**: `XMLParser` with `think`/`guess` or just `guess` depending on `use_think`
- **Rubric overview**: Exact guess match, partial credit from feedback, turns-based reward, format check, and hard mode rule enforcement.

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval hard-wordle
```

Configure model and sampling:

```bash
uv run vf-eval hard-wordle \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"num_train_examples": 2000, "num_eval_examples": 20, "use_think": true}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `2000` | Number of training episodes |
| `num_eval_examples` | int | `20` | Number of evaluation episodes |
| `use_think` | bool | `true` | Use `<think>` with `guess`; if false, guess-only format |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `check_answer_reward_func` | 1.0 if final guess equals target, else 0.0 |
| `partial_credit_reward_func` | Partial credit from greens/yellows in feedback |
| `count_turns_reward_func` | Higher score for solving in fewer turns |
| `format_reward` | Adherence to expected XML format |

