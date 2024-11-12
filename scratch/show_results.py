import argparse
from glob import glob
import numpy as np

def get_score_from_log_file(args, log_file):
    # look at the last line to get the score, e.g., [2024-11-11 01:57:17][INFO	] Score: 100.0
    # if the last line is [2024-11-11 19:53:40][INFO	] Completed. then we look at the -3 line
    # if the third last line is 'Many recent actions in history are the same -- model is likely in a loop, stopping early.', we skip this log file
    score = None
    with open(log_file, "r") as f:
        lines = f.readlines()
        last_line = lines[-1]
        third_last_line = lines[-3]
        if "Completed." in last_line:
            last_line = third_last_line
        if args.skip_loops and "Many recent actions in history are the same -- model is likely in a loop, stopping early." in third_last_line:
            return None
        try:
            score = float(last_line.split("Score: ")[1])
        except:
            raise ValueError(f"Error getting score for {log_file}")
    return score


def main(args):
    log_files = glob(f"{args.log_dir}/{args.split}_{args.parser_mode}/*.log")
    # filter out the goals to skip
    print(f"There are {len(log_files)} log files")
    log_files = [f for f in log_files if int(f.split("_goal")[-1].split(".")[0]) not in args.goals_to_skip]
    print(f"Goals to skip: {args.goals_to_skip}")
    print(f"After filtering, there are {len(log_files)} log files")
    scores = []
    for log_file in log_files:
        scores.append(get_score_from_log_file(args, log_file))
    total = len(scores)

    # filter out None scores
    scores = [s for s in scores if s is not None]
    filtered = len(scores)

    success_rate = sum([s == 100.0 for s in scores]) / filtered * 100
    print(f"Evaluated on {filtered}/{total} log files")
    print(f"Average score: {np.mean(scores):.2f}")
    print(f"Success rate: {success_rate:.2f}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, default="test")
    parser.add_argument("--parser_mode", type=str, required=True, default="v1")
    parser.add_argument("--goals_to_skip", nargs="+", type=int, default=[])
    parser.add_argument("--skip_loops", action="store_true", default=False)
    args = parser.parse_args()

    print(args.log_dir)
    main(args)
