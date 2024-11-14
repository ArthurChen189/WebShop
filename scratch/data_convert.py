import json
import random
import os
from argparse import ArgumentParser


IL_TRAJ_PATH = 'data/raw/webshop/il_trajs_finalized_images.jsonl'
HUMAN_GOAL_PATH = 'data/raw/webshop/human_goals.json'


def process_goal(state):
    """Extract the base instruction (goal) without price_text

    Args:
        state (str): the state/look

    Returns:
        _type_: _description_
    """    
    state = state.lower().replace('"', '').replace("'", "")
    state = state.replace('amazon shopping game\ninstruction:', '').replace('webshop\ninstruction:', '')
    state = state.replace('\n[button] search [button_]', '').strip()
    if ', and price lower than' in state:
        state = state.split(', and price lower than')[0]
    return state

def process(s):
    s = s.lower().replace('"', '').replace("'", "").strip()
    s = s.replace('[sep]', '[SEP]')
    return s

def get_data(split='train', filter_long_trajs=20):
    """Get the data from the IL trajs file

    Args:
        split (str, optional): train, eval, test
        filter_long_trajs (int, optional): filter out the trajectories that are longer than the given length

    Returns:
        list[dict]: the data
    """    
    path = IL_TRAJ_PATH
    print('Loading data from {}'.format(path))
    with open(path, 'r') as f:
        json_list = f.readlines()

    human_goals = json.load(open(HUMAN_GOAL_PATH, 'r'))

    random.seed(233)
    random.shuffle(json_list)

    # split by human goal index
    goal_range = range(len(human_goals))
    if split == 'train':
        goal_range = range(1500, len(human_goals))
    elif split == 'eval':
        goal_range = range(500, 1500)
    elif split == 'test':
        goal_range = range(0, 500)

    num_trajs = 0
    traj_list = []
    filtered_traj_count = 0
    for json_str in json_list:
        result = json.loads(json_str)
        # we filter out long trajectories
        if filter_long_trajs:
            if len(result['actions']) > filter_long_trajs:
                filtered_traj_count += 1
                continue
        s = process_goal(result['states'][0])
        assert s in human_goals, s
        goal_idx = human_goals.index(s)

        if goal_idx not in goal_range:
            continue 

        num_trajs += 1
        traj_list.append(result)
    print(f"num of {split} trajs: {num_trajs}")
    if filter_long_trajs:
        print(f"num of filtered trajs: {filtered_traj_count}")
    return traj_list


def preprocess_trajs(trajs, columns_to_remove=None):
    # check if all keys are the same
    keys = trajs[0].keys()
    for traj in trajs:
        assert traj.keys() == keys, "keys are not the same"
    
    # remove useless columns
    if columns_to_remove:
        for traj in trajs:
            for col in columns_to_remove:
                del traj[col]

    # extract instruction before "[button]" and after "Instruction"
    def extract_instruction(sample):
        state = sample['states'][0]
        instruction = state.split("Instruction:")[1].split("[button]")[0].strip()
        assert len(instruction) > 0, "instruction is empty"
        return instruction

    # generate a {instruction: traj} dict
    preprocessed_trajs = {}
    for traj in trajs:
        instr = extract_instruction(traj)
        # make sure the instruction is in the human goals
        if instr in preprocessed_trajs:
            preprocessed_trajs[instr].append(traj)
        else:
            preprocessed_trajs[instr] = [traj]

    print(f"there are {len(preprocessed_trajs)} out of {len(trajs)} unique instructions")
    print(f"columns are {keys}")
    return preprocessed_trajs

def convert_action(action):
    r"""Convert the actions from raw texts (e.g. search[a long clip-in hair extension] or click[\<item id\>]) to dictionary-like format (e.g. {"action": "search",  "ref": "a long clip-in hair extension"} or {"action": "click", "ref": "<item id>"})

    Args:
        action (str): the action in the raw text format

    Raises:
        ValueError: if the action is not in the expected format

    Returns:
        dict: the action in the dictionary format
    """
    if action.startswith("search["):
        return {"action": "search", "ref": action[7:-1].strip()}
    elif action.startswith("click["):
        return {"action": "click", "ref": action[6:-1].strip()}
    else:
        raise ValueError(f"Unknown action: {action}")

def recover_action(action):
    if action["action"] == "search":
        return f"search[{action['ref']}]"
    elif action["action"] == "click":
        return f"click[{action['ref']}]"
    else:
        raise ValueError(f"Unknown action: {action}")

def sanitizeStr(inStr):
    if inStr is None:
        return inStr
    out = inStr.replace("\n", " | ")
    out = out.replace("\n", " | ")
    out = out.replace("Instruction: |", "Instruction: ")
    out = out.replace("Instruction:  |", "Instruction: ")
    out = out.replace("|  |", "|") # double pipe indicates a new line between products
    return out


def remove_instruction(obs, instruction, test_time=False):
    """Remove the instruction from the observation

    Args:
        obs (str): the observation
        instruction (str): the instruction

    Raises:
        ValueError: if the instruction is not found in the observation

    Returns:
        str: the observation without the instruction
    """
    if test_time:
        obs = obs.replace("WebShop\n", "").replace("Amazon Shopping Game\n", "")
    if "Instruction:\n" in obs:
        split = obs.split("Instruction:\n" + instruction)
    elif "Instruction: \n" in obs:
        split = obs.split("Instruction: \n" + instruction)
    else:
        raise ValueError(f"Instruction not found in observation: {obs}")
    return split[1].strip()


# this is modified based on the original SwiftSage code
def compose_webshop_instance(step_id, instruction, curr_action, curr_obs, recent_actions, recent_obs, window_size=10, include_past_obs=False,
                           no_instr_in_past_obs=True, no_instr_in_curr_obs=False, action_to_dict=True, input_instr=True, test_time=False):
    """Composes the input string for WebGUM, which consists of the instruction, the action history, the current observation.
    Args:
        step_id (int): the step id
        instruction (str): the instruction
        curr_action (str): the current action
        curr_obs (str): the current observation
        recent_actions (list[str]): the recent actions
        recent_obs (list[str]): the recent observations
        window_size (int): the sliding window size for the action and observation history
        no_instr_in_past_obs (bool): whether to remove the instruction from the past observations
        no_instr_in_curr_obs (bool): whether to remove the instruction from the current observation
        action_to_dict (bool): whether to convert the action to a dictionary
        input_instr (bool): whether to include the instruction in the input string

    Returns:
        input_str: the input string for the model
        label: sanitized current action
    """
    input_str = ""
    if test_time:
        instruction = instruction.replace("Instruction: ", "")
        label = None
    else:
        label = str(convert_action(curr_action)) if action_to_dict else curr_action
    input_str += "Instruction: " + instruction if input_instr else ""
    input_str += f" </s> Time: {step_id} </s> "
     
    input_str += "Action history: </s>" 
    ind = window_size
    for obs, action in zip(recent_obs[-window_size:], recent_actions[-window_size:]):
        processed_obs = remove_instruction(obs, instruction, test_time) if no_instr_in_past_obs else obs
        processed_action = str(convert_action(action)) if action_to_dict else action
        if include_past_obs:
            input_str += f" <extra_id_{ind}> {processed_action} --> {processed_obs} |"
        else:
            input_str += f" <extra_id_{ind}> {processed_action} |"
        ind -= 1
    input_str += " </s> " 

    # current observation has instruction in it
    curr_obs = remove_instruction(curr_obs, instruction, test_time) if no_instr_in_curr_obs else curr_obs
    input_str += "Current observation: " + curr_obs + " " 
    eos_prompt = '</s> What action should you do next? </s> '
    input_str += eos_prompt
    input_str = sanitizeStr(input_str)
    return input_str, label

def compose_webshop_instance_v1(step_id, instruction, curr_action, curr_obs, recent_actions, recent_obs, test_time=False, window_size=3):
    """Compose the input string for WebGUM, which consists of the instruction, the action history, the observation history, the current observation.
    """
    return compose_webshop_instance(step_id, instruction, curr_action, curr_obs, recent_actions, recent_obs, window_size=window_size,
                              no_instr_in_past_obs=True, no_instr_in_curr_obs=True, action_to_dict=True, input_instr=True,
                              include_past_obs=True, test_time=test_time)


def compose_webshop_instance_v2(step_id, instruction, curr_action, curr_obs, recent_actions, recent_obs, test_time=False, window_size=20):
    """Compose the input string for WebGUM, which consists of the instruction, the action history, the current observation.
    This version does not include the past observations in the input string to ensure the observation is not truncated.
    It has a sliding window size of 20 and does not convert the actions to dictionary format.
    """
    return compose_webshop_instance(step_id, instruction, curr_action, curr_obs, recent_actions, recent_obs, window_size=window_size,
                              no_instr_in_past_obs=True, no_instr_in_curr_obs=True, action_to_dict=False, input_instr=True,
                              include_past_obs=False, test_time=test_time)

def filter_long_input(input_str, tokenizer, max_length=1024):
    """
    Filter out the trajectories where the input string is longer than the max length
    """
    encoded_input = tokenizer(input_str, max_length=max_length, truncation=False, padding="max_length").input_ids
    return len(encoded_input) > max_length


def construct_data(preprocessed_trajs, out_path, parser_mode="v2", window_size=20, filter_func=None, max_length=None, tokenizer=None, verbose=False):
    """
    Construct the data in the SwiftSage format and export it into a .jsonl file
    """
    if parser_mode == "v1":
        compose_webshop_instance = compose_webshop_instance_v1
    elif parser_mode == "v2":
        compose_webshop_instance = compose_webshop_instance_v2
    elif parser_mode == "v2plus":
        compose_webshop_instance = compose_webshop_instance_v2 # we use v2, but with click[item - name] format
    else:
        raise ValueError(f"Unknown parser model: {parser_mode}")
    data = []
    traj_number = 0
    filtered_traj_count = 0
    for instr, trajs in preprocessed_trajs.items():
        for traj in trajs:
            traj_list = []
            no_filter = True

            if parser_mode == "v2plus":
                assert "actions_translate" in traj
                actions = traj["actions_translate"]
            else:
                actions = traj['actions']

            observations = traj['states']

            for step_id, (action, obs) in enumerate(zip(actions, observations), start=1):
                input_str, label = compose_webshop_instance(step_id, instr, action, obs, actions[:step_id-1],
                                                            observations[:step_id-1], window_size=window_size)
                if filter_func and filter_func(input_str, tokenizer, max_length):
                    no_filter = False
                    break
                else:
                    traj_list.append({"input": input_str, "target": label})
            if no_filter:
                data.extend(traj_list)
            else:
                filtered_traj_count += 1
                if verbose:
                    print(f"trajectory {traj_number} is filtered out")
            traj_number += 1
    print(f"there are {filtered_traj_count} out of {traj_number} trajectories filtered out")
    if out_path is not None:
        with open(out_path, 'wt') as f:
            for item in data:   
                f.write(json.dumps(item) + "\n")
    return data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--filter_long_trajs', type=int, default=None,
                        help="filter out the trajectories that are more than the given length of actions")
    parser.add_argument('--parser_mode', type=str, default="v2", choices=["v1", "v2", "v2plus"])
    parser.add_argument('--output_dir', type=str, default="data/preprocessed/webshop/v2")
    parser.add_argument('--filter_long_input', action='store_true', default=False,
                        help="filter out the trajectories where the input string is longer than the max length")
    parser.add_argument('--max_length', type=int, default=None,
                        help="the max length of the input string")
    parser.add_argument('--window_size', type=int, default=20,
                        help="the window size for the action and observation history")
    args = parser.parse_args()

    tokenizer = None
    filter_func = None

    if args.filter_long_input:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "yuchenlin/swift_sw",
            use_fast=True,
            cache_dir="/h/arthur/.cache/huggingface/transformers"
        )
        tokenizer.add_tokens(['[button]', '[button_]', '[clicked button]', '[clicked button_]'], special_tokens=True)
        filter_func = filter_long_input


    # get the data from the IL trajs file
    train_trajs = get_data('train', filter_long_trajs=args.filter_long_trajs)
    dev_trajs = get_data('eval', filter_long_trajs=args.filter_long_trajs)
    test_trajs = get_data('test', filter_long_trajs=args.filter_long_trajs)

    # preprocess the data
    columns = ['images', 'action_idxs', 'available_actions']
    train_preprocessed_dict = preprocess_trajs(train_trajs, columns_to_remove=columns)
    dev_preprocessed_dict = preprocess_trajs(dev_trajs, columns_to_remove=columns)
    test_preprocessed_dict = preprocess_trajs(test_trajs, columns_to_remove=columns)

    # construct the data
    # if the output path directory does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_data = construct_data(train_preprocessed_dict, f"{args.output_dir}/train.jsonl",
                                parser_mode=args.parser_mode, filter_func=filter_func,
                                max_length=args.max_length, tokenizer=tokenizer, window_size=args.window_size)

    dev_data = construct_data(dev_preprocessed_dict, f"{args.output_dir}/val.jsonl",
                              parser_mode=args.parser_mode, filter_func=filter_func,
                              max_length=args.max_length, tokenizer=tokenizer, window_size=args.window_size)

    test_data = construct_data(test_preprocessed_dict, f"{args.output_dir}/test.jsonl",
                              parser_mode=args.parser_mode, filter_func=filter_func,
                              max_length=args.max_length, tokenizer=tokenizer, window_size=args.window_size)
