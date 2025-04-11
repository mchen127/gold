import os
import textwrap
from easy_runner import EasyRunner

if __name__ == "__main__":

    exp_name = "train_cvpo"
    runner = EasyRunner(log_name=exp_name)

    # config = [
    #     "cvpo_train_config/cvpo_config1.json",
    #     "cvpo_train_config/cvpo_config2.json",
    #     "cvpo_train_config/cvpo_config3.json",
    # ]

    # Define command template
    template = textwrap.dedent(
        """\
        python /home/mc/gold/train/train_cvpo.py \
        --project 'GOLD' \
        --task '{}' \
        --cost_limit {} \
        --seed {} \
    """
    )
    
    # button_tasks = ["SafetyPointButton1Gymnasium-v0", "SafetyCarButton1Gymnasium-v0"]
    goal_tasks = ["SafetyPointGoal1Gymnasium-v0", "SafetyCarGoal1Gymnasium-v0"]
    cost_limits = [80]
    seeds = [10, 20, 30]


    # Compose training instructions
    train_instructions = runner.compose(template, [goal_tasks, cost_limits, seeds])

    # Start tasks in parallel (limit to 15 at a time)
    runner.start(train_instructions, max_parallel=2)

    print("All tasks have started. Check logs for progress.")
