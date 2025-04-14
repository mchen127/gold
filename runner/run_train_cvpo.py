import os
import json
import random
from easy_runner import EasyRunner

# Set a master seed
master_seed = 42
rng = random.Random(master_seed)

if __name__ == "__main__":

    exp_name = "train_cvpo"
    runner = EasyRunner(log_name=exp_name)

    config_path = (
        "/home/mc/gold/exp_configs/exp3-cvpo_performance_examination.json"
    )

    with open(config_path, "r") as file:
        config = json.load(file)
    print(config)
    
    # Set hyperparameters
    # non-interable
    project = config["project"]
    logdir = config["logdir"]
    prefix = config["prefix"]
    n_runs = config["n/runs"]
    # interable
    tasks = config["tasks"]
    cost_limits = config["cost_limits"]

    # Generate 30 reproducible random seeds
    seeds = [rng.randint(0, 99999) for _ in range(n_runs)]
    remaining_seeds = seeds.copy()
    remaining_seeds = remaining_seeds[9:]
    

    # Define command template
    template = f"\
        python /home/mc/gold/train/train_cvpo.py \
        --project '{project}' \
        --logdir '{logdir}' \
        --prefix '{prefix}' \
        --task '{{}}' \
        --cost_limit '{{}}' \
        --seed '{{}}' \
        --device 'cuda' \
    "

    # Remove all extra whitespace
    template = " ".join(template.split())

    # Compose training instructions
    train_instructions = runner.compose(template, [tasks, cost_limits, remaining_seeds])

    # Start tasks in parallel (limit to 15 at a time)
    runner.start(train_instructions, max_parallel=4)

    print("All tasks have started. Check logs for progress.")
