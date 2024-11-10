import argparse
import os
import time
import torch
import tqdm
from data_convert import compose_webshop_instance_v1, compose_webshop_instance_v2
from utils import load_model, get_model_output, findValidActionNew

import logging
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_file_name(args, goal_idx):
    if (len(args["output_path"]) > 0):
        if not args["output_path"].endswith("/"):
            args["output_path"] += "/"

        # Make path if it doesn't exist
        if not os.path.exists(args['output_path']):
            os.makedirs(args["output_path"])
  
    filenameOutPrefixSeed = args["output_path"] + "goal" + str(goal_idx)

    return filenameOutPrefixSeed
  

# Example user input console, to play through a game.
def eval(args):
    if args["compose_mode"] == "v1":
        compose_instance = compose_webshop_instance_v1
    elif args["compose_mode"] == "v2":
        compose_instance = compose_webshop_instance_v2
    else:
        raise NotImplementedError
    
    # Initialize environment
    from baseline_models.env import WebEnv

    env = WebEnv(args, split=args["set"], id=f'{args["set"]}_seed{args["seed"]}')
    goal_idxs = env.goal_idxs # for test set, there are 500 goals
    lm_model, tokenizer, sbert_model, _ = load_model(args, device)

    scores = []
    for goal_idx in tqdm.tqdm(goal_idxs[10:args["max_num_runs"]]):

        obs, info = env.reset(goal_idx)
        goal = info['goal']
        logger = init_logger(args, goal_idx)
        logger.info(f"Goal: {goal}")

        recent_actions = []
        recent_obs = []

        done = False
        score = 0.0
        step = 0
        prev_obs = None

        while not done:           
            
            logger.info("-"*50+f"Goal: {goal_idx}, Step: {step}"+"-"*50) 

            # Note that the agent is allowed to know the score changes.
            

            split = args["set"]
            logger.info("Split: " + split)
            input_str, _ = compose_instance(step_id=step+1, instruction=goal, curr_action=None, 
                                            curr_obs=obs, recent_actions=recent_actions, recent_obs=recent_obs, test_time=True) 
            ############

            logger.info("InputStr: " + input_str)
            predStrs = get_model_output(args, input_str, tokenizer, lm_model, device, logger)

            # get valid action
            valid_actions = info['valid']
            action = findValidActionNew(predStrs, valid_actions, sbert_model, logger) 
            # we execute the action
            obs, reward, done, info = env.step(action)

            if obs == prev_obs:
                # the action is invalid 
                # obs, reward , done, info = env.step("0")
                pass
            
            prev_obs = obs
            # score = info['score']
            recent_actions.append(action)
            recent_obs.append(obs)

            logger.info(f"Action: {action}")
            logger.info("Obs: " + obs)
            logger.info(f"Score: {score * 10}")

            step += 1
            if done: 
                # the env has step_limit so we stop when done
                score = reward * 10
                break

            logger.info("Recent Actions: " + str(recent_actions))
            logger.info("Recent Observations: " + str(recent_obs))

            # Early stopping if we're in a loop
            if len(recent_actions) >= 5 and len(set(recent_actions[-5:])) == 2:
                logger.info("Many recent actions in history are the same -- model is likely in a loop, stopping early.")
                break

        scores.append(score)
        logger.info("Run completed...")
        logger.info("Score: " + str(score))
 
        time.sleep(2)
    output_path = os.path.join(args["output_path"], "score.txt")
    f = open(output_path, "a")
    f.write("Args: " + str(args) + "\n")
    f.write("Scores: " + str(scores) + "\n")
    f.write("Average Score: " + str(sum(scores) / len(scores)) + "\n")
    f.write("Average Completion Rate: " + str(sum([s == 100.0 for s in scores]) / len(scores) * 100) + "\n")

    f.close()

    logger.info("Shutting down server...")
    logger.info("Completed.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_path", required=True) 
    parser.add_argument("--state_format", default="text_rich", choices=["text_rich", "text", "html"])
    parser.add_argument("--beams", type=int, default=5)
    parser.add_argument("--set", default="test_mini")
    parser.add_argument("--output_path", default="webshop_logs/test_reproduction_v1")
    parser.add_argument("--compose_mode", default="v1")
    parser.add_argument("--model_parallelism_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_len", type=int, default=2048)
    parser.add_argument("--max_output_len", type=int, default=128)
    parser.add_argument("--max_num_runs", type=int, default=500, help="maximum number of runs")
    parser.add_argument("--sbert", action="store_true", default=True)

    # Webshop specific arguments
    parser.add_argument("--human_goals", action="store_true", default=True)
    parser.add_argument("--click_item_name", type=int, default=1)
    parser.add_argument("--num_prev_obs", type=int, default=0)
    parser.add_argument("--num_prev_actions", type=int, default=0)
    parser.add_argument("--extra_search_path", type=str, default="")
    parser.add_argument("--step_limit", type=int, default=100)
    args = parser.parse_args()
    params = vars(args)
    return params


def init_logger(args, goal_idx, log_level=logging.INFO):
    logger = logging.getLogger()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s\t] %(message)s",
                                    datefmt='%Y-%m-%d %H:%M:%S')
    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_dir = args["output_path"]
    if logging_dir:
        os.makedirs(logging_dir, exist_ok=True)
        filename = f"{args['set']}_seed{args['seed']}_goal{goal_idx}.log"
        path = os.path.join(logging_dir, filename)
        fh = logging.FileHandler(path)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(fh)
    return logger

def main():
    args = parse_args()
    print(args)
    args = dotdict(args)
    
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed']) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    eval(args)



        

if __name__ == "__main__":
    main()