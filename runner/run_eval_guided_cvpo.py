import os
from easy_runner import EasyRunner
import textwrap

if __name__ == "__main__":

    exp_name = "eval_guided_cvpo"
    runner = EasyRunner(log_name=exp_name)

    base_dir = "/home/mc/gold/logs/GOLD/SafetyCarButton1Gymnasium-v0-cost-80"

    # Collect all model paths dynamically
    model_paths = []
    for subdir in os.listdir(base_dir):
        model_path = os.path.join(base_dir, subdir)
        if os.path.isdir(model_path) and os.path.exists(model_path):  # Ensure it's a directoryh exists
            model_paths.append(model_path)
        
    print(len(model_paths))

    is_best = [True, False]

    # Define command template using textwrap.dedent for readability
    template = textwrap.dedent(
        """\
        nohup python /home/mc/gold/eval/eval_guided_cvpo.py \
        --eval_episodes 50\
        --model_path '{}' \
        --best '{}' \
        --output_path '{}/eval_result.csv'\
    """
    )

    # Compose training instructions
    train_instructions = runner.compose(template, [model_paths, is_best, [base_dir]])

    # Start tasks in parallel (limit to 3 at a time)
    runner.start(train_instructions, max_parallel=3)

    print("All tasks have started. Check logs for progress.")
