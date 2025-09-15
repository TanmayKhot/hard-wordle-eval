from verifiers import load_environment
from openai import OpenAI
import os
import json

# Set environment variable (use environment variable or set your own key)
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"


def write_results(filename, results):
    """Write evaluation results to a file in a readable format."""
    with open(filename, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("HARD MODE WORDLE ENVIRONMENT EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Write prompt information
        f.write("PROMPT:\n")
        f.write("-" * 40 + "\n")
        if hasattr(results, 'prompt') and results.prompt:
            if isinstance(results.prompt, list):
                for i, msg in enumerate(results.prompt):
                    if isinstance(msg, dict):
                        f.write(f"Message {i+1} ({msg.get('role', 'unknown')}):\n")
                        f.write(f"{msg.get('content', '')}\n\n")
                    else:
                        f.write(f"Message {i+1}: {str(msg)}\n\n")
            else:
                f.write(f"{results.prompt}\n")
        f.write("\n")
        
        # Write completion information
        f.write("COMPLETION:\n")
        f.write("-" * 40 + "\n")
        if hasattr(results, 'completion') and results.completion:
            if isinstance(results.completion, list):
                for i, msg in enumerate(results.completion):
                    if isinstance(msg, dict):
                        f.write(f"Message {i+1} ({msg.get('role', 'unknown')}):\n")
                        f.write(f"{msg.get('content', '')}\n\n")
                    else:
                        f.write(f"Message {i+1}: {str(msg)}\n\n")
            else:
                f.write(f"{results.completion}\n")
        f.write("\n")
        
        # Write answer information
        f.write("ANSWER:\n")
        f.write("-" * 40 + "\n")
        if hasattr(results, 'answer') and results.answer:
            f.write(f"{results.answer}\n")
        f.write("\n")
        
        # Write reward information
        f.write("REWARD:\n")
        f.write("-" * 40 + "\n")
        if hasattr(results, 'reward') and results.reward is not None:
            if isinstance(results.reward, list):
                f.write(f"Rewards: {results.reward}\n")
                f.write(f"Average Reward: {sum(results.reward) / len(results.reward):.4f}\n")
            else:
                f.write(f"{results.reward}\n")
        f.write("\n")
        
        # Write metrics information
        f.write("METRICS:\n")
        f.write("-" * 40 + "\n")
        if hasattr(results, 'metrics') and results.metrics:
            if isinstance(results.metrics, dict):
                for key, value in results.metrics.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write(f"{results.metrics}\n")
        f.write("\n")
        
        # Write additional info
        f.write("ADDITIONAL INFO:\n")
        f.write("-" * 40 + "\n")
        if hasattr(results, 'info') and results.info:
            if isinstance(results.info, dict):
                for key, value in results.info.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write(f"{results.info}\n")
        f.write("\n")
        
        # Write raw results as JSON for debugging
        f.write("RAW RESULTS (JSON):\n")
        f.write("-" * 40 + "\n")
        try:
            # Convert results to a serializable format
            results_dict = {}
            for attr in ['prompt', 'completion', 'answer', 'reward', 'metrics', 'info']:
                if hasattr(results, attr):
                    value = getattr(results, attr)
                    try:
                        json.dumps(value)  # Test if serializable
                        results_dict[attr] = value
                    except (TypeError, ValueError):
                        results_dict[attr] = str(value)
            
            f.write(json.dumps(results_dict, indent=2, default=str))
        except Exception as e:
            f.write(f"Could not serialize results: {e}\n")
        
        f.close()
    print(f"Results written to {filename}")

# Load the environment
env = load_environment("hard-wordle")

# Use OpenAI client
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="YOUR_API_KEY",
)

# Use the environment for evaluation or training
results = env.evaluate(client=client, model="gpt-4o-mini", examples=1, rollouts_per_example=1)

# Write results to file
write_results("results.txt", results)