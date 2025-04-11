import os
from easy_runner import EasyRunner
import textwrap

if __name__ == "__main__":

    exp_name = "eval_bc"
    runner = EasyRunner(log_name=exp_name)

    base_dir = "/home/mc/gold/logs/OfflineCarButton1Gymnasium-v0-cost-80"

    # Collect all model paths dynamically
    model_paths = []
    
    for subdir in os.listdir(base_dir):
        full_subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(full_subdir_path):  # Ensure it's a directory
            model_path = os.path.join(full_subdir_path, subdir)  # Construct full path
            if os.path.exists(model_path):  # Ensure the path exists
                model_paths.append(model_path)

    is_best = [True, False]  # Assuming the same best flag logic

    # Define command template using textwrap.dedent for readability
    template = f"\
        nohup python /home/mc/gold/eval/eval_bc.py \
        --eval_episodes 50 \
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
