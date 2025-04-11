from easy_runner import EasyRunner
import os

if __name__ == "__main__":

    exp_name = "eval_cvpo"
    runner = EasyRunner(log_name=exp_name)

    base_dir = "/home/mc/gold/logs/SafetyCarButton1Gymnasium-v0-cost-80"

    # Collect all model paths dynamically
    model_paths = []
    for subdir in os.listdir(base_dir):
        model_path = os.path.join(base_dir, subdir)
        if os.path.isdir(model_path) and os.path.exists(
            model_path
        ):  # Ensure it's a directoryh exists
            model_paths.append(model_path)

    print(len(model_paths))
    
    is_best = [True, False]

    # Define command template using textwrap.dedent for readability
    template = f"\
        python /home/mc/gold/eval/eval_cvpo.py \
            --eval_episodes 50\
            --model_path '{{}}' \
            --best '{{}}' \
            --output_path '{base_dir}/eval_result.csv'\
    "

    # Remove all extra whitespace
    template = " ".join(template.split())

    # Compose training instructions
    train_instructions = runner.compose(template, [model_paths, is_best])

    # Start tasks in parallel (limit to 3 at a time)
    runner.start(train_instructions, max_parallel=3)

    print("All tasks have started. Check logs for progress.")
