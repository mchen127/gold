from easy_runner import EasyRunner
import textwrap
import json
import re

if __name__ == "__main__":

    exp_name = "train_bc_guided_cvpo"
    runner = EasyRunner(log_name=exp_name)

    config_path = "/home/mc/gold/config/temp_cfg.json"
    # config_path = "/home/mc/gold/config/exp-bcexpert_guided_cvpo_cfg.json"

    with open(config_path, "r") as file:
        config = json.load(file)
    print(config)
    
    project = config["project"]
    prefix = config["prefix"]
    
    cost_limits = config["cost_limit"]
    seeds = config["seed"]
    hs = config["h"]
    
    tasks_and_offline_model_paths = config["tasks_and_offline_model_paths"]


    # Define command template
    template = f"\
        python /home/mc/gold/train/train_guided_cvpo.py\
        --project {project}\
        --prefix {prefix}\
        --task {{}}\
        --offline_model_path {{}}\
        --cost_limit {{}}\
        --seed {{}}\
        --h {{}}\
    "

    # Remove all extra whitespace
    template = " ".join(template.split())

    # print(template)
    # # # Compose training instructions
    train_instructions = []
    for i in range(len(tasks_and_offline_model_paths)):
        task = tasks_and_offline_model_paths[i]["task"]
        offline_model_paths = tasks_and_offline_model_paths[i]["offline_model_paths"]
        train_instructions.extend(runner.compose(
            template, [[task], offline_model_paths, cost_limits, seeds, hs]
        ))
        
    
    # print(train_instructions)

    # # Start tasks in parallel (limit to 15 at a time)
    runner.start(train_instructions, max_parallel=2)
    print("All tasks have started. Check logs for progress.")
