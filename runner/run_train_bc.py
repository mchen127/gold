from easy_runner import EasyRunner
import textwrap

if __name__ == "__main__":

    exp_name = "train_default_bc"
    runner = EasyRunner(log_name=exp_name)
    

    # Define command template
    template = textwrap.dedent("""
        nohup python /home/mc/gold/train/train_bc.py \
        --project 'GOLD' \
        --prefix 'BCExpert' \
        --task '{}' \
        --bc_mode {} \
        --cost_limit {} \
        --seed {} \
    """)

    # button_tasks = ["OfflinePointButton1Gymnasium-v0", "OfflineCarButton1Gymnasium-v0"]
    goal_tasks = ["OfflinePointGoal1Gymnasium-v0", "OfflineCarGoal1Gymnasium-v0"]
    bc_mode = ["expert"]
    cost_limits = [80]
    seed = [10, 20, 30]
    

    # Compose training instructions
    train_instructions = runner.compose(template, [goal_tasks, bc_mode, cost_limits, seed])

    # Start tasks in parallel (limit to 15 at a time)
    runner.start(train_instructions, max_parallel=1)

    print("All tasks have started. Check logs for progress.")
